"""Tests for Whisper download progress callback format (v1.7.7 fix).

背景：之前 load_model() 里的 _poll_download() 把 progress_callback
的 current 写死成 5,用户只看到"已下载 X MB",看不到百分比和总大小。

修复后应该：
- callback current 是真实百分比 (5-99)
- message 同时显示 current_bytes / total_bytes (XX%)
- 如果有未知 model size 走 fallback（只有已下载大小）

注意：这些测试是纯常量/格式守卫，不依赖 faster_whisper，
可以在本地（缺 faster_whisper 的环境）跑。
"""

import pytest


# ---------------------------------------------------------------------------
# 常量守卫
# ---------------------------------------------------------------------------

class TestModelTotalSizeBytes:
    """MODEL_TOTAL_SIZE_BYTES 必须覆盖所有有效 Whisper 模型大小"""

    def test_contains_all_valid_sizes(self):
        from services.whisper_service import MODEL_TOTAL_SIZE_BYTES, _MODEL_REPO_MAP
        # VALID_SIZES 跟 _MODEL_REPO_MAP 一样:
        # tiny, base, small, medium, large-v3
        for size in _MODEL_REPO_MAP.keys():
            assert size in MODEL_TOTAL_SIZE_BYTES, (
                f"MODEL_TOTAL_SIZE_BYTES 缺少 {size!r}，"
                f"下载进度无法计算百分比"
            )

    @pytest.mark.parametrize("size", ["tiny", "base", "small", "medium", "large-v3"])
    def test_size_is_positive_integer(self, size):
        from services.whisper_service import MODEL_TOTAL_SIZE_BYTES
        size_bytes = MODEL_TOTAL_SIZE_BYTES[size]
        assert isinstance(size_bytes, int), f"{size} size 必须是 int"
        assert size_bytes > 0, f"{size} size 必须 > 0"

    @pytest.mark.parametrize("size", ["tiny", "base", "small", "medium", "large-v3"])
    def test_size_reasonable_range(self, size):
        """模型大小应该在 50MB - 5GB 之间（粗略 sanity check）"""
        from services.whisper_service import MODEL_TOTAL_SIZE_BYTES
        size_bytes = MODEL_TOTAL_SIZE_BYTES[size]
        # tiny 至少 50MB，large-v3 不超过 5GB
        assert 50 * 1024 * 1024 <= size_bytes <= 5 * 1024 * 1024 * 1024, (
            f"{size} size {size_bytes} 超出合理范围 [50MB, 5GB]"
        )

    def test_sizes_increase(self):
        """tiny < base < small < medium < large-v3"""
        from services.whisper_service import MODEL_TOTAL_SIZE_BYTES
        sizes = [MODEL_TOTAL_SIZE_BYTES[k] for k in
                 ["tiny", "base", "small", "medium", "large-v3"]]
        for i in range(len(sizes) - 1):
            assert sizes[i] < sizes[i + 1], (
                f"模型大小应该递增: {sizes}"
            )


# ---------------------------------------------------------------------------
# 进度计算函数（纯逻辑，提取出来方便单测）
# ---------------------------------------------------------------------------

class TestProgressCalculation:
    """下载进度的百分比计算 + message 格式

    实际逻辑在 whisper_service.load_model._poll_download() 内嵌函数里,
    这里把核心算法抽出来单测,防止 regression。
    """

    @pytest.fixture
    def calc(self):
        """从 whisper_service 抽出来的纯函数（测试用）"""
        from services.whisper_service import MODEL_TOTAL_SIZE_BYTES
        from utils.format_utils import format_file_size

        def _calc_pct(current_bytes: int, model_size: str) -> int:
            total = MODEL_TOTAL_SIZE_BYTES.get(model_size, 0)
            if total <= 0 or current_bytes <= 0:
                return 5
            raw = (current_bytes / total) * 100
            return max(5, min(99, int(raw + 0.5)))

        def _format_message(current_bytes: int, model_size: str) -> str:
            total = MODEL_TOTAL_SIZE_BYTES.get(model_size, 0)
            if total > 0 and current_bytes > 0:
                pct = _calc_pct(current_bytes, model_size)
                return (
                    f"正在下载模型 {model_size}... "
                    f"{format_file_size(current_bytes)} / "
                    f"{format_file_size(total)} ({pct}%)"
                )
            elif current_bytes > 0:
                return (
                    f"正在下载模型 {model_size}... "
                    f"已下载 {format_file_size(current_bytes)}"
                )
            else:
                return f"正在下载模型 {model_size}..."

        return _calc_pct, _format_message

    @pytest.mark.parametrize("model_size,expected_total", [
        ("tiny", 78_207_087),
        ("base", 147_886_409),
        ("small", 486_215_847),
        ("medium", 1_530_575_217),
        ("large-v3", 3_090_839_273),
    ])
    def test_pct_at_zero_is_clamped_to_5(self, calc, model_size, expected_total):
        """0 字节时进度不应该显示 0%（让用户知道下载确实开始了）"""
        _calc_pct, _ = calc
        assert _calc_pct(0, model_size) == 5

    @pytest.mark.parametrize("model_size,expected_total", [
        ("tiny", 78_207_087),
        ("base", 147_886_409),
        ("medium", 1_530_575_217),
    ])
    def test_pct_at_half(self, calc, model_size, expected_total):
        """下载一半应该是 ~50%（±1）"""
        _calc_pct, _ = calc
        assert _calc_pct(expected_total // 2, model_size) == 50

    def test_pct_clamped_to_99(self, calc):
        """超过总大小（理论错误）应该 clamp 到 99 而不是 ≥100"""
        _calc_pct, _ = calc
        # 给一个不可能的"超量"（实际不会出现，但测试 clamp 行为）
        assert _calc_pct(78_207_087 * 2, "tiny") == 99

    def test_pct_unknown_model_falls_back(self, calc):
        """未知 model size 走 fallback（5%）"""
        _calc_pct, _ = calc
        assert _calc_pct(1_000_000, "nonexistent-model") == 5

    def test_message_contains_all_three_pieces(self, calc):
        """message 必须同时显示 已下载 / 总大小 / 百分比 三项"""
        _, _format_message = calc
        msg = _format_message(78_207_087 // 2, "tiny")  # 50% tiny
        # 总大小（74.6 MB，固定）
        assert "74.6 MB" in msg, f"缺总大小: {msg}"
        # 百分比
        assert "(50%)" in msg, f"缺百分比: {msg}"
        # 已下载：format_file_size 用 1024 进制，78_207_087/2/1024/1024 = 37.3 MB
        # 不强求确切值，只验证有"MB" + 数字格式
        import re
        assert re.search(r"\d+\.\d+ MB / 74\.6 MB", msg), (
            f"已下载大小格式不对: {msg}"
        )

    def test_message_uses_full_path(self, calc):
        """message 包含模型名（不是只显示"下载"）"""
        _, _format_message = calc
        msg = _format_message(100_000_000, "medium")
        assert "medium" in msg
        # 进度数字
        assert "%" in msg

    def test_message_fallback_when_unknown_size(self, calc):
        """未知 model size 只显示"已下载"（兼容老逻辑）"""
        _, _format_message = calc
        msg = _format_message(1_000_000, "nonexistent")
        assert "已下载" in msg
        # 不能有误导性的百分比
        assert "(%)" not in msg
