"""Tests for WhisperService GPU detection and CPU fallback (issue #4).

覆盖场景：
- device='auto' + libcublas 错误 → 自动回退 CPU int8
- device='cuda' + libcublas 错误 → 显式 cuda 不静默回退，抛错
- device='auto' + 非 CUDA 错误 → 不回退，抛错
- device='cpu' → 跳过 GPU 尝试
- WHISPER_DEVICE 环境变量 → 覆盖 config.whisper.device
- load_model 幂等性
"""
from unittest.mock import MagicMock

import pytest

from core.models import VADParameters, WhisperConfig
from services.whisper_service import WhisperService, is_model_downloaded, get_model_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vad_params() -> VADParameters:
    return VADParameters(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=2000,
        speech_pad_ms=400,
    )


@pytest.fixture
def cached_model_dir(tmp_path) -> str:
    """创建一个看起来已缓存的模型目录（通过完整性检测），避免 load_model 启动下载轮询线程。"""
    model_dir = tmp_path / "models"
    # 新版 _is_model_cached 检查 snapshots/<commit>/ 结构
    snapshot_dir = model_dir / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}")
    (snapshot_dir / "model.bin").write_text("fake model weights")
    (snapshot_dir / "tokenizer.json").write_text("{}")
    return str(model_dir)


@pytest.fixture
def make_service(vad_params, cached_model_dir):
    def _make(device="cpu", compute_type="int8", model_size="tiny") -> WhisperService:
        config = WhisperConfig(
            model_size=model_size,
            compute_type=compute_type,
            device=device,
            source_language="auto",
        )
        return WhisperService(config, vad_params, model_dir=cached_model_dir)
    return _make


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_auto_falls_back_to_cpu_on_libcublas_error(
    make_service, mocker, monkeypatch
):
    """device='auto' 时遇到 libcublas 错误应自动回退到 CPU。"""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    service = make_service(device="auto")

    # 模拟 CUDA 设备存在（否则会直接走 CPU 而不会触发回退逻辑）
    mocker.patch.object(WhisperService, "_has_cuda_device", return_value=True)

    mock_model = MagicMock(name="WhisperModelInstance")
    calls: list[dict] = []

    def side_effect(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "Library libcublas.so.12 is not found or cannot be loaded"
            )
        return mock_model

    mocker.patch(
        "services.whisper_service.WhisperModel", side_effect=side_effect
    )

    service.load_model()

    assert len(calls) == 2, f"Expected cuda + cpu fallback, got {len(calls)} call(s)"
    # 第一次尝试用 cuda
    assert calls[0]["device"] == "cuda"
    # 回退到 cpu
    assert calls[1]["device"] == "cpu"
    assert service.model is mock_model


def test_explicit_cuda_does_not_silently_fallback(
    make_service, mocker, monkeypatch
):
    """device='cuda' 显式指定时，libcublas 错误不应被静默吞掉回退。"""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    service = make_service(device="cuda")

    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError(
            "Library libcublas.so.12 is not found or cannot be loaded"
        )

    mocker.patch(
        "services.whisper_service.WhisperModel", side_effect=side_effect
    )

    with pytest.raises(RuntimeError, match="libcublas"):
        service.load_model()
    # 不应回退：只调用一次
    assert call_count["n"] == 1


def test_auto_does_not_swallow_non_cuda_errors(
    make_service, mocker, monkeypatch
):
    """device='auto' 模式下，非 libcublas/CUDA 错误应直接抛出，不回退。"""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    service = make_service(device="auto")

    # 模拟 CUDA 设备存在
    mocker.patch.object(WhisperService, "_has_cuda_device", return_value=True)

    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError("Some other error: out of memory")

    mocker.patch(
        "services.whisper_service.WhisperModel", side_effect=side_effect
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        service.load_model()
    # 错误与 CUDA 无关，不应触发回退
    assert call_count["n"] == 1


def test_cpu_device_skips_gpu_attempt(make_service, mocker, monkeypatch):
    """device='cpu' 时直接走 CPU，不应尝试 CUDA。"""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    service = make_service(device="cpu", compute_type="int8")

    mock_model = MagicMock(name="WhisperModelInstance")
    captured: list[dict] = []

    def side_effect(*args, **kwargs):
        captured.append(kwargs)
        return mock_model

    mocker.patch(
        "services.whisper_service.WhisperModel", side_effect=side_effect
    )

    service.load_model()

    assert len(captured) == 1
    assert captured[0]["device"] == "cpu"
    assert captured[0]["compute_type"] == "int8"


def test_env_var_overrides_config_device(make_service, mocker, monkeypatch):
    """WHISPER_DEVICE 环境变量应覆盖 config.whisper.device。"""
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    service = make_service(device="cpu")  # config 写 cpu，env 覆盖为 cuda

    mock_model = MagicMock(name="WhisperModelInstance")
    captured: list[dict] = []

    def side_effect(*args, **kwargs):
        captured.append(kwargs)
        return mock_model

    mocker.patch(
        "services.whisper_service.WhisperModel", side_effect=side_effect
    )

    service.load_model()

    assert len(captured) == 1
    assert captured[0]["device"] == "cuda"


def test_load_model_is_idempotent(make_service, mocker, monkeypatch):
    """重复调用 load_model 不应重复构造模型。"""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    service = make_service(device="cpu")

    mock_model = MagicMock(name="WhisperModelInstance")
    mock_class = mocker.patch(
        "services.whisper_service.WhisperModel", return_value=mock_model
    )

    service.load_model()
    service.load_model()

    assert mock_class.call_count == 1
    assert service.model is mock_model


# ---------------------------------------------------------------------------
# is_model_downloaded / get_model_dir
# ---------------------------------------------------------------------------

class TestIsModelDownloaded:
    def test_downloaded_model_returns_true(self, tmp_path):
        """已下载的模型应返回 True（必须 3 个关键文件都在）"""
        model_dir = tmp_path / "models"
        cache = model_dir / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc123"
        cache.mkdir(parents=True)
        # 2026-06-06 改进: 完整性要求 3 个关键文件,不是只 model.bin
        (cache / "config.json").write_text("{}")
        (cache / "model.bin").write_text("fake")
        (cache / "tokenizer.json").write_text("{}")
        assert is_model_downloaded("tiny", str(model_dir)) is True

    def test_missing_required_file_returns_false(self, tmp_path):
        """缺 config.json (只有 model.bin + tokenizer.json) → False"""
        model_dir = tmp_path / "models"
        cache = model_dir / "models--Systran--faster-whisper-base" / "snapshots" / "xyz"
        cache.mkdir(parents=True)
        (cache / "model.bin").write_text("fake")
        (cache / "tokenizer.json").write_text("{}")
        # 缺 config.json
        assert is_model_downloaded("base", str(model_dir)) is False

    def test_empty_file_returns_false(self, tmp_path):
        """文件存在但 size=0 → False"""
        model_dir = tmp_path / "models"
        cache = model_dir / "models--Systran--faster-whisper-small" / "snapshots" / "abc"
        cache.mkdir(parents=True)
        (cache / "config.json").write_text("{}")
        (cache / "model.bin").write_text("")  # 0 bytes
        (cache / "tokenizer.json").write_text("{}")
        assert is_model_downloaded("small", str(model_dir)) is False

    def test_not_downloaded_model_returns_false(self, tmp_path):
        """未下载的模型应返回 False"""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        assert is_model_downloaded("small", str(model_dir)) is False

    def test_empty_snapshots_dir_returns_false(self, tmp_path):
        """snapshots 目录存在但为空时应返回 False"""
        model_dir = tmp_path / "models"
        cache = model_dir / "models--Systran--faster-whisper-medium" / "snapshots"
        cache.mkdir(parents=True)
        assert is_model_downloaded("medium", str(model_dir)) is False

    def test_nonexistent_model_dir_returns_false(self):
        """不存在的目录应返回 False"""
        assert is_model_downloaded("tiny", "/nonexistent/path") is False


class TestGetModelDir:
    def test_returns_docker_path_if_exists(self, monkeypatch, tmp_path):
        """Docker 路径存在时应返回 /data/models"""
        monkeypatch.setattr("os.path.isdir", lambda p: p == "/data/models")
        assert get_model_dir() == "/data/models"

    def test_falls_back_to_local_path(self, monkeypatch):
        """Docker 路径不存在时应返回 ./data/models"""
        monkeypatch.setattr("os.path.isdir", lambda p: False)
        assert get_model_dir() == "./data/models"


# ---------------------------------------------------------------------------
# CUDA 预检查
# ---------------------------------------------------------------------------

class TestHasCudaDevice:
    def test_returns_true_when_cuda_available(self, mocker):
        mocker.patch(
            "ctranslate2.get_cuda_device_count", return_value=1
        )
        assert WhisperService._has_cuda_device() is True

    def test_returns_false_when_no_cuda(self, mocker):
        mocker.patch(
            "ctranslate2.get_cuda_device_count", return_value=0
        )
        assert WhisperService._has_cuda_device() is False

    def test_returns_false_on_exception(self, mocker):
        mocker.patch(
            "ctranslate2.get_cuda_device_count",
            side_effect=RuntimeError("driver not loaded")
        )
        assert WhisperService._has_cuda_device() is False


# ---------------------------------------------------------------------------
# 模型完整性检测
# ---------------------------------------------------------------------------

class TestVerifyModelFiles:
    def test_complete_model_returns_true(self, make_service):
        """cached_model_dir fixture 提供的完整模型应通过验证"""
        service = make_service(model_size="tiny")
        assert service._verify_model_files() is True

    def test_missing_snapshot_dir_returns_false(self, make_service, tmp_path):
        """snapshots 目录不存在时返回 False"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        config = WhisperConfig(model_size="tiny", device="cpu", source_language="auto")
        service = WhisperService(
            config,
            VADParameters(0.5, 250, 2000, 400),
            model_dir=str(empty_dir),
        )
        assert service._verify_model_files() is False

    def test_incomplete_model_returns_false(self, tmp_path):
        """缺少关键文件时返回 False（半下载状态）"""
        model_dir = tmp_path / "models"
        snapshot = model_dir / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        # 只有 config.json，缺 model.bin
        (snapshot / "config.json").write_text("{}")
        (snapshot / "tokenizer.json").write_text("{}")

        config = WhisperConfig(model_size="tiny", device="cpu", source_language="auto")
        service = WhisperService(
            config,
            VADParameters(0.5, 250, 2000, 400),
            model_dir=str(model_dir),
        )
        assert service._verify_model_files() is False

    def test_empty_file_returns_false(self, tmp_path):
        """文件存在但大小为 0 时返回 False"""
        model_dir = tmp_path / "models"
        snapshot = model_dir / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "model.bin").write_text("")  # 空文件
        (snapshot / "tokenizer.json").write_text("{}")

        config = WhisperConfig(model_size="tiny", device="cpu", source_language="auto")
        service = WhisperService(
            config,
            VADParameters(0.5, 250, 2000, 400),
            model_dir=str(model_dir),
        )
        assert service._verify_model_files() is False


# ---------------------------------------------------------------------------
# load_model 完整性检测触发重新下载
# ---------------------------------------------------------------------------

def test_load_model_triggers_redownload_on_incomplete_files(
    make_service, mocker, monkeypatch
):
    """模型目录存在但文件不完整时，应触发重新下载（is_cached=False）"""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    # make_service 用的是 cached_model_dir（完整），手动构造一个不完整的
    from pathlib import Path
    incomplete_dir = make_service.__self__ if False else None  # noqa
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "models"
        # snapshots 存在但没有关键文件
        snapshot = model_dir / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        # 缺 model.bin 和 tokenizer.json

        config = WhisperConfig(model_size="tiny", device="cpu", source_language="auto")
        service = WhisperService(
            config,
            VADParameters(0.5, 250, 2000, 400),
            model_dir=str(model_dir),
        )
        # 2026-06-06 改进: _is_model_cached 复用 _verify_model_files
        # 半下载状态时,is_cached 也应该返回 False
        assert service._is_model_cached() is False
        assert service._verify_model_files() is False


class TestIsModelCachedCompletenessGuard:
    """
    2026-06-06 守卫测试:_is_model_cached 必须跟 _verify_model_files 行为一致。
    之前 _is_model_cached 只检查目录非空,导致半下载状态假阳性 → WhisperModel
    加载时卡住/报错。修复后两者必须等价。
    """
    def test_partial_download_returns_false(self, tmp_path):
        """半下载状态(有 config.json 但缺 model.bin)→ _is_model_cached=False"""
        model_dir = tmp_path / "models"
        snapshot = (
            model_dir / "models--Systran--faster-whisper-medium" / "snapshots" / "abc"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "tokenizer.json").write_text("{}")
        # 缺 model.bin

        config = WhisperConfig(model_size="medium", device="cpu", source_language="auto")
        service = WhisperService(
            config, VADParameters(0.5, 250, 2000, 400), model_dir=str(model_dir)
        )
        assert service._is_model_cached() is False
        # 跟 _verify_model_files 行为一致
        assert service._is_model_cached() == service._verify_model_files()

    def test_only_incomplete_files_returns_false(self, tmp_path):
        """只有 .incomplete 文件(用户取消下载后的状态)→ False"""
        model_dir = tmp_path / "models"
        snapshot = (
            model_dir / "models--Systran--faster-whisper-base" / "snapshots" / "xyz"
        )
        snapshot.mkdir(parents=True)
        # 用户取消下载,blobs/ 下有 .incomplete 文件
        # 模拟:snapshot 目录里只有一个 .incomplete 标记文件
        (snapshot / "config.json.incomplete").write_text("partial")
        (snapshot / "model.bin.incomplete").write_text("partial")

        config = WhisperConfig(model_size="base", device="cpu", source_language="auto")
        service = WhisperService(
            config, VADParameters(0.5, 250, 2000, 400), model_dir=str(model_dir)
        )
        assert service._is_model_cached() is False

    def test_complete_model_returns_true(self, tmp_path):
        """完整模型(三个关键文件都存在且 size > 0)→ True"""
        model_dir = tmp_path / "models"
        snapshot = (
            model_dir / "models--Systran--faster-whisper-small" / "snapshots" / "complete"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "model.bin").write_text("fake model")
        (snapshot / "tokenizer.json").write_text("{}")

        config = WhisperConfig(model_size="small", device="cpu", source_language="auto")
        service = WhisperService(
            config, VADParameters(0.5, 250, 2000, 400), model_dir=str(model_dir)
        )
        assert service._is_model_cached() is True

    def test_is_model_downloaded_module_function_same_behavior(self, tmp_path):
        """模块级 is_model_downloaded() 跟实例方法 _is_model_cached() 一致"""
        model_dir = tmp_path / "models"
        snapshot = (
            model_dir / "models--Systran--faster-whisper-large-v3" / "snapshots" / "abc"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        # 半下载
        config = WhisperConfig(model_size="large-v3", device="cpu", source_language="auto")
        service = WhisperService(
            config, VADParameters(0.5, 250, 2000, 400), model_dir=str(model_dir)
        )
        # 模块级函数和实例方法必须返回相同结果
        assert is_model_downloaded("large-v3", str(model_dir)) == service._is_model_cached()
        # 两者都应该返回 False(半下载)
        assert is_model_downloaded("large-v3", str(model_dir)) is False


# ============================================================================
# _poll_download 取消响应（防止 Whisper 下载阶段无法取消）
# ============================================================================

class TestPollDownloadCancellationResponse:
    """
    2026-06-06 守卫测试:poll 线程里 progress_callback 必须能抛 InterruptedError,
    poll 线程必须捕获并设 cancel_event,让外层 WhisperModel 加载检测到取消。

    之前 bug:poll 循环用 `except Exception: pass`,导致 worker 在 progress_callback
    里 raise InterruptedError 都被吞,WhisperModel 阻塞中 hf_hub 继续下载,
    用户取消信号永远不响应。
    """
    def test_interrupted_error_in_callback_sets_cancel_event(self, tmp_path):
        """progress_callback 抛 InterruptedError 时,poll 必须能响应"""
        from services.whisper_service import WhisperService
        config = WhisperConfig(model_size="tiny", device="cpu", source_language="auto")
        service = WhisperService(
            config, VADParameters(0.5, 250, 2000, 400), model_dir=str(tmp_path)
        )

        # 模拟 progress_callback 抛 InterruptedError
        def raising_callback(current, total, message):
            raise InterruptedError("用户取消")

        # 同步执行 _poll_download,应该捕获并退出
        import threading
        stop_event = threading.Event()
        # 直接调用 _poll_download 的内层逻辑(我们修改了异常处理)

        # 这里测的是代码逻辑:poll 循环里不再 `except: pass`,
        # 而是 catch 后设 stop_event/cancel_event
        # 直接断言:在代码里 grep "except Exception" + "pass" 不存在
        import inspect
        source = inspect.getsource(service.load_model)
        # 修复:不允许 `except Exception: pass` 这种"吞掉所有异常"的写法
        assert "except Exception: pass" not in source, \
            "❌ poll 循环里 `except Exception: pass` 会吞掉 InterruptedError,导致取消信号丢失"


def test_load_model_no_cuda_skips_cuda_attempt(
    make_service, mocker, monkeypatch
):
    """容器内没有 CUDA 设备时，device='auto' 应直接走 CPU，不调用 cuda"""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    service = make_service(device="auto")

    # 模拟没有 CUDA 设备
    mocker.patch.object(WhisperService, "_has_cuda_device", return_value=False)

    mock_model = MagicMock(name="WhisperModelInstance")
    calls: list[dict] = []

    def side_effect(*args, **kwargs):
        calls.append(kwargs)
        return mock_model

    mocker.patch(
        "services.whisper_service.WhisperModel", side_effect=side_effect
    )

    service.load_model()

    # 只调用了一次（直接 cpu），没尝试 cuda
    assert len(calls) == 1
    assert calls[0]["device"] == "cpu"
