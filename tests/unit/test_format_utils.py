#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""format_utils.py 单元测试"""

from utils.format_utils import (
    format_file_size,
    format_timestamp,
    get_lang_name,
    format_duration,
    truncate_text,
    format_percentage,
)


# ============================================================================
# format_file_size
# ============================================================================

class TestFormatFileSize:
    def test_bytes(self):
        assert format_file_size(0) == "0.0 B"
        assert format_file_size(512) == "512.0 B"
        assert format_file_size(1023) == "1023.0 B"

    def test_kilobytes(self):
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1536) == "1.5 KB"

    def test_megabytes(self):
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(int(2.5 * 1024 * 1024)) == "2.5 MB"

    def test_gigabytes(self):
        assert format_file_size(1024 ** 3) == "1.0 GB"

    def test_terabytes(self):
        assert format_file_size(1024 ** 4) == "1.0 TB"


# ============================================================================
# format_timestamp
# ============================================================================

class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "00:00:00,000"

    def test_seconds(self):
        assert format_timestamp(1.5) == "00:00:01,500"

    def test_minutes(self):
        assert format_timestamp(90) == "00:01:30,000"

    def test_hours(self):
        assert format_timestamp(3661.123) == "01:01:01,123"

    def test_milliseconds_precision(self):
        assert format_timestamp(0.001) == "00:00:00,001"
        assert format_timestamp(0.999) == "00:00:00,999"


# ============================================================================
# get_lang_name
# ============================================================================

class TestGetLangName:
    def test_known_codes(self):
        assert get_lang_name('zh') == '中文'
        assert get_lang_name('en') == '英语'
        assert get_lang_name('ja') == '日语'

    def test_case_insensitive(self):
        assert get_lang_name('ZH') == '中文'
        assert get_lang_name('EN') == '英语'

    def test_unknown_code_returns_itself(self):
        assert get_lang_name('xx') == 'xx'


# ============================================================================
# format_duration
# ============================================================================

class TestFormatDuration:
    def test_seconds_only(self):
        assert format_duration(30) == "30s"

    def test_minutes_and_seconds(self):
        assert format_duration(90) == "1m 30s"

    def test_hours_and_minutes(self):
        assert format_duration(3661) == "1h 1m"

    def test_exact_minute(self):
        assert format_duration(60) == "1m 0s"

    def test_zero(self):
        assert format_duration(0) == "0s"


# ============================================================================
# truncate_text
# ============================================================================

class TestTruncateText:
    def test_short_text_unchanged(self):
        assert truncate_text("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert truncate_text("hello", 5) == "hello"

    def test_long_text_truncated(self):
        result = truncate_text("hello world", 8)
        assert len(result) == 8
        assert result.endswith("...")

    def test_custom_suffix(self):
        result = truncate_text("hello world", 8, suffix="~")
        assert result.endswith("~")
        assert len(result) == 8

    def test_very_short_max_length(self):
        result = truncate_text("hello", 3)
        assert result == "..."


# ============================================================================
# format_percentage
# ============================================================================

class TestFormatPercentage:
    def test_normal(self):
        assert format_percentage(50, 100) == "50.0%"
        assert format_percentage(75, 100) == "75.0%"

    def test_zero_total(self):
        assert format_percentage(0, 0) == "0%"

    def test_100_percent(self):
        assert format_percentage(100, 100) == "100.0%"

    def test_custom_decimals(self):
        assert format_percentage(1, 3, decimals=2) == "33.33%"
        assert format_percentage(1, 3, decimals=0) == "33%"

    def test_fractional(self):
        assert format_percentage(1, 3) == "33.3%"
