#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""subtitle_converter.py 单元测试 — 时间解析/格式化与格式转换"""

import tempfile
import os
from services.subtitle_converter import SubtitleConverter, SubtitleEntry


# ============================================================================
# 时间解析
# ============================================================================

class TestParseSrtTime:
    def test_basic(self):
        assert SubtitleConverter.parse_srt_time("00:00:01,000") == 1000

    def test_with_hours(self):
        assert SubtitleConverter.parse_srt_time("01:30:00,000") == 5400000

    def test_with_milliseconds(self):
        assert SubtitleConverter.parse_srt_time("00:00:00,500") == 500

    def test_complex_time(self):
        assert SubtitleConverter.parse_srt_time("02:15:30,123") == 8130123

    def test_invalid_format_raises(self):
        import pytest
        with pytest.raises(ValueError):
            SubtitleConverter.parse_srt_time("invalid")


# ============================================================================
# 时间格式化
# ============================================================================

class TestFormatSrtTime:
    def test_zero(self):
        assert SubtitleConverter.format_srt_time(0) == "00:00:00,000"

    def test_milliseconds(self):
        assert SubtitleConverter.format_srt_time(1500) == "00:00:01,500"

    def test_complex(self):
        assert SubtitleConverter.format_srt_time(8130123) == "02:15:30,123"

    def test_negative_clamped_to_zero(self):
        assert SubtitleConverter.format_srt_time(-100) == "00:00:00,000"


class TestFormatVttTime:
    def test_uses_dot_separator(self):
        assert SubtitleConverter.format_vtt_time(1500) == "00:00:01.500"

    def test_zero(self):
        assert SubtitleConverter.format_vtt_time(0) == "00:00:00.000"


class TestFormatAssTime:
    def test_basic(self):
        result = SubtitleConverter.format_ass_time(1500)
        assert result == "0:00:01.50"

    def test_zero(self):
        assert SubtitleConverter.format_ass_time(0) == "0:00:00.00"

    def test_negative_clamped(self):
        assert SubtitleConverter.format_ass_time(-100) == "0:00:00.00"

    def test_centiseconds_precision(self):
        assert SubtitleConverter.format_ass_time(1234) == "0:00:01.23"


# ============================================================================
# SRT 解析
# ============================================================================

class TestParseSrt:
    def test_basic(self):
        content = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:03,000 --> 00:00:05,000\nWorld"
        entries = SubtitleConverter.parse_srt(content)
        assert len(entries) == 2
        assert entries[0].index == 1
        assert entries[0].text == "Hello"
        assert entries[0].start_ms == 1000
        assert entries[0].end_ms == 3000

    def test_multiline_text(self):
        content = "1\n00:00:01,000 --> 00:00:03,000\nLine one\nLine two"
        entries = SubtitleConverter.parse_srt(content)
        assert len(entries) == 1
        assert entries[0].text == "Line one\nLine two"

    def test_empty_content(self):
        assert SubtitleConverter.parse_srt("") == []

    def test_invalid_block_skipped(self):
        content = "1\nINVALID_TIMECODE\nHello\n\n2\n00:00:01,000 --> 00:00:03,000\nOK"
        entries = SubtitleConverter.parse_srt(content)
        assert len(entries) == 1
        assert entries[0].text == "OK"

    def test_too_few_lines_skipped(self):
        content = "1\n00:00:01,000 --> 00:00:03,000\n\n2\n00:00:03,000 --> 00:00:05,000\nOK"
        entries = SubtitleConverter.parse_srt(content)
        # First block has 3 lines but empty text; second block is OK
        assert len(entries) >= 1


# ============================================================================
# 格式转换输出
# ============================================================================

def _make_entries():
    return [
        SubtitleEntry(index=1, start_ms=1000, end_ms=3000, text="Hello"),
        SubtitleEntry(index=2, start_ms=3000, end_ms=5000, text="World"),
    ]


class TestToSrt:
    def test_basic_output(self):
        entries = _make_entries()
        output = SubtitleConverter.to_srt(entries)
        assert "00:00:01,000 --> 00:00:03,000" in output
        assert "Hello" in output
        assert "World" in output


class TestToVtt:
    def test_has_webvtt_header(self):
        entries = _make_entries()
        output = SubtitleConverter.to_vtt(entries)
        assert output.startswith("WEBVTT")

    def test_uses_dot_separator(self):
        entries = _make_entries()
        output = SubtitleConverter.to_vtt(entries)
        assert "00:00:01.000 --> 00:00:03.000" in output


class TestToAss:
    def test_has_script_info(self):
        entries = _make_entries()
        output = SubtitleConverter.to_ass(entries)
        assert "[Script Info]" in output
        assert "[Events]" in output

    def test_dialogue_lines(self):
        entries = _make_entries()
        output = SubtitleConverter.to_ass(entries)
        assert "Dialogue:" in output
        assert "Hello" in output


class TestToSsa:
    def test_has_script_info_v4(self):
        entries = _make_entries()
        output = SubtitleConverter.to_ssa(entries)
        assert "ScriptType: v4.00" in output
        assert "[V4 Styles]" in output


class TestToSub:
    def test_frame_based_format(self):
        entries = _make_entries()
        output = SubtitleConverter.to_sub(entries, fps=25)
        # 1000ms at 25fps = frame 25, 3000ms = frame 75
        assert "{25}{75}" in output
        assert "Hello" in output

    def test_different_fps(self):
        entries = _make_entries()
        output = SubtitleConverter.to_sub(entries, fps=30)
        # 1000ms at 30fps = frame 30
        assert "{30}" in output


# ============================================================================
# convert_file (端到端，需要临时文件)
# ============================================================================

class TestConvertFile:
    def test_srt_to_vtt(self):
        srt_content = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        fd_in, in_path = tempfile.mkstemp(suffix=".srt")
        with os.fdopen(fd_in, 'w') as f:
            f.write(srt_content)

        fd_out, out_path = tempfile.mkstemp(suffix=".vtt")
        os.close(fd_out)

        try:
            result = SubtitleConverter.convert_file(in_path, 'vtt', out_path)
            assert result == out_path
            with open(out_path, 'r') as f:
                content = f.read()
            assert "WEBVTT" in content
            assert "Hello" in content
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_empty_file_raises(self):
        import pytest
        fd, path = tempfile.mkstemp(suffix=".srt")
        os.close(fd)
        try:
            with pytest.raises(ValueError, match="无法解析"):
                SubtitleConverter.convert_file(path, 'vtt')
        finally:
            os.unlink(path)

    def test_unsupported_format_raises(self):
        import pytest
        srt_content = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        fd, path = tempfile.mkstemp(suffix=".srt")
        with os.fdopen(fd, 'w') as f:
            f.write(srt_content)
        try:
            with pytest.raises(ValueError, match="不支持"):
                SubtitleConverter.convert_file(path, 'xyz')
        finally:
            os.unlink(path)
