#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lang_detection.py 单元测试"""

import tempfile
import os
from utils.lang_detection import (
    detect_language_from_filename,
    get_language_tag,
    detect_language_from_subtitle,
)


# ============================================================================
# detect_language_from_filename
# ============================================================================

class TestDetectLanguageFromFilename:
    def test_chinese_code(self):
        assert detect_language_from_filename("movie.chs.srt") == "chs"

    def test_english_code(self):
        assert detect_language_from_filename("movie.eng.srt") == "en"

    def test_japanese_code(self):
        assert detect_language_from_filename("movie.jpn.srt") == "ja"

    def test_korean_code(self):
        assert detect_language_from_filename("movie.kor.srt") == "ko"

    def test_short_codes(self):
        assert detect_language_from_filename("movie.zh.srt") == "chs"
        assert detect_language_from_filename("movie.en.srt") == "en"
        assert detect_language_from_filename("movie.ja.srt") == "ja"
        assert detect_language_from_filename("movie.ko.srt") == "ko"

    def test_code_at_end(self):
        assert detect_language_from_filename("movie.chs") == "chs"

    def test_traditional_chinese(self):
        assert detect_language_from_filename("movie.cht.srt") == "cht"

    def test_unknown(self):
        assert detect_language_from_filename("movie.srt") == "unknown"
        assert detect_language_from_filename("movie.xx.srt") == "unknown"

    def test_case_insensitive(self):
        assert detect_language_from_filename("movie.CHS.srt") == "chs"
        assert detect_language_from_filename("movie.ENG.srt") == "en"


# ============================================================================
# get_language_tag
# ============================================================================

class TestGetLanguageTag:
    def test_known_codes(self):
        assert get_language_tag('chs') == '简中'
        assert get_language_tag('cht') == '繁中'
        assert get_language_tag('en') == '英语'
        assert get_language_tag('ja') == '日语'
        assert get_language_tag('ko') == '韩语'

    def test_case_insensitive(self):
        assert get_language_tag('CHS') == '简中'
        assert get_language_tag('EN') == '英语'

    def test_unknown(self):
        assert get_language_tag('unknown') == '未知'
        assert get_language_tag('xx') == '未知'


# ============================================================================
# detect_language_from_subtitle (需要临时文件)
# ============================================================================

def _write_temp_srt(content: str) -> str:
    """写入临时 SRT 文件并返回路径"""
    fd, path = tempfile.mkstemp(suffix=".srt")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


class TestDetectLanguageFromSubtitle:
    def test_chinese_content(self):
        srt = _write_temp_srt(
            "1\n00:00:01,000 --> 00:00:03,000\n你好世界欢迎来到测试字幕文件\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\n这是一个用于检测语言功能的字幕\n\n"
            "3\n00:00:05,000 --> 00:00:07,000\n中文字幕内容测试确保有足够的字符数量\n\n"
            "4\n00:00:07,000 --> 00:00:09,000\n语言检测算法需要足够多的样本才能准确判断\n"
        )
        try:
            result = detect_language_from_subtitle(srt)
            assert result in ('chs', 'cht')
        finally:
            os.unlink(srt)

    def test_english_content(self):
        srt = _write_temp_srt(
            "1\n00:00:01,000 --> 00:00:03,000\nHello world this is a test\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nThe quick brown fox jumps over the lazy dog\n\n"
            "3\n00:00:05,000 --> 00:00:07,000\nAnother line of english text for detection\n\n"
            "4\n00:00:07,000 --> 00:00:09,000\nMore words for language detection testing here\n"
        )
        try:
            result = detect_language_from_subtitle(srt)
            assert result == 'en'
        finally:
            os.unlink(srt)

    def test_japanese_content(self):
        srt = _write_temp_srt(
            "1\n00:00:01,000 --> 00:00:03,000\nこんにちは世界ようこそテストへ\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nこれは日本語の字幕テストです\n\n"
            "3\n00:00:05,000 --> 00:00:07,000\n言語検出機能のテストを行います\n\n"
            "4\n00:00:07,000 --> 00:00:09,000\n十分な文字数を確保するために追加のテキスト\n"
        )
        try:
            result = detect_language_from_subtitle(srt)
            assert result == 'ja'
        finally:
            os.unlink(srt)

    def test_too_few_chars_returns_unknown(self):
        srt = _write_temp_srt("1\n00:00:01,000 --> 00:00:03,000\nHi\n")
        try:
            result = detect_language_from_subtitle(srt)
            assert result == 'unknown'
        finally:
            os.unlink(srt)

    def test_nonexistent_file_returns_unknown(self):
        result = detect_language_from_subtitle("/nonexistent/path.srt")
        assert result == 'unknown'
