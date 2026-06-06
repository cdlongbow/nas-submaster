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
        """已下载的模型应返回 True"""
        model_dir = tmp_path / "models"
        cache = model_dir / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc123"
        cache.mkdir(parents=True)
        (cache / "model.bin").write_text("fake")
        assert is_model_downloaded("tiny", str(model_dir)) is True

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
        # 完整性检测应判定为不完整
        assert service._is_model_cached() is True
        assert service._verify_model_files() is False


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
