#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""translator.py 单元测试 — 解析逻辑与 SRT I/O"""

import json
import tempfile
import os
from unittest.mock import patch, MagicMock

from services.translator import (
    SubtitleTranslator, TranslationConfig, ParseError,
    parse_srt_file, save_srt_file,
)
from core.models import SubtitleEntry


# ============================================================================
# _parse_translation_response — 正常情况
# ============================================================================

class TestParseTranslationResponse:
    def _make_translator(self):
        config = TranslationConfig(
            api_key="ollama", base_url="http://ollama:11434/v1",
            model_name="test", target_language="zh"
        )
        with patch("services.translator.OpenAI"):
            return SubtitleTranslator(config)

    def test_valid_json(self):
        t = self._make_translator()
        response = json.dumps([{"line": 1, "translation": "你好"}, {"line": 2, "translation": "世界"}])
        result = t._parse_translation_response(response, 2)
        assert result == ["你好", "世界"]

    def test_markdown_code_block_stripped(self):
        t = self._make_translator()
        inner = json.dumps([{"line": 1, "translation": "你好"}])
        response = f"```json\n{inner}\n```"
        result = t._parse_translation_response(response, 1)
        assert result == ["你好"]

    def test_prefix_text_before_json(self):
        t = self._make_translator()
        inner = json.dumps([{"line": 1, "translation": "你好"}])
        response = f"好的，以下是翻译：\n{inner}"
        result = t._parse_translation_response(response, 1)
        assert result == ["你好"]

    def test_trailing_comma_fixed(self):
        t = self._make_translator()
        response = '[{"line": 1, "translation": "你好"},]'
        result = t._parse_translation_response(response, 1)
        assert result == ["你好"]

    def test_missing_closing_bracket_fixed(self):
        t = self._make_translator()
        response = '[{"line": 1, "translation": "你好"}'
        result = t._parse_translation_response(response, 1)
        assert result == ["你好"]

    def test_count_mismatch_raises(self):
        import pytest
        t = self._make_translator()
        response = json.dumps([{"line": 1, "translation": "你好"}])
        with pytest.raises(ParseError, match="数量不匹配"):
            t._parse_translation_response(response, 3)

    def test_ellipsis_detected(self):
        import pytest
        t = self._make_translator()
        response = '[{"line": 1, "translation": "你好"}, ...]'
        with pytest.raises(ParseError):
            t._parse_translation_response(response, 10)

    def test_not_array_raises(self):
        import pytest
        t = self._make_translator()
        response = '{"line": 1, "translation": "你好"}'
        with pytest.raises(ParseError, match="JSON 数组"):
            t._parse_translation_response(response, 1)

    def test_missing_translation_field_raises(self):
        import pytest
        t = self._make_translator()
        response = json.dumps([{"line": 1, "text": "你好"}])
        with pytest.raises(ParseError, match="translation"):
            t._parse_translation_response(response, 1)


# ============================================================================
# _build_translation_prompt
# ============================================================================

class TestBuildTranslationPrompt:
    def _make_translator(self, prompt_template=None):
        config = TranslationConfig(
            api_key="ollama", base_url="http://ollama:11434/v1",
            model_name="test", target_language="zh"
        )
        with patch("services.translator.OpenAI"):
            return SubtitleTranslator(config, prompt_template=prompt_template)

    def test_prompt_contains_entries(self):
        t = self._make_translator()
        entries = [SubtitleEntry("1", "00:00:01,000 --> 00:00:03,000", "Hello")]
        prompt = t._build_translation_prompt(entries)
        assert "Hello" in prompt
        assert "中文" in prompt or "Chinese" in prompt

    def test_prompt_uses_custom_template(self):
        from core.models import PromptTemplate
        tpl = PromptTemplate(role="custom role", rules="custom rules", style_guide="custom style")
        t = self._make_translator(prompt_template=tpl)
        entries = [SubtitleEntry("1", "00:00:01,000 --> 00:00:03,000", "Hello")]
        prompt = t._build_translation_prompt(entries)
        assert "custom role" in prompt
        assert "custom rules" in prompt
        assert "custom style" in prompt

    def test_prompt_contains_context(self):
        t = self._make_translator()
        entries = [SubtitleEntry("1", "00:00:01,000 --> 00:00:03,000", "Hello")]
        prompt = t._build_translation_prompt(entries, context_before="prev line", context_after="next line")
        assert "prev line" in prompt
        assert "next line" in prompt


# ============================================================================
# parse_srt_file / save_srt_file
# ============================================================================

class TestSrtFileIO:
    def test_parse_srt_file(self):
        srt_content = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:03,000 --> 00:00:05,000\nWorld"
        fd, path = tempfile.mkstemp(suffix=".srt")
        with os.fdopen(fd, 'w') as f:
            f.write(srt_content)
        try:
            entries = parse_srt_file(path)
            assert len(entries) == 2
            assert entries[0].text == "Hello"
            assert entries[1].text == "World"
        finally:
            os.unlink(path)

    def test_save_and_reparse(self):
        entries = [
            SubtitleEntry(index="1", timecode="00:00:01,000 --> 00:00:03,000", text="Hello"),
            SubtitleEntry(index="2", timecode="00:00:03,000 --> 00:00:05,000", text="World"),
        ]
        fd, path = tempfile.mkstemp(suffix=".srt")
        os.close(fd)
        try:
            save_srt_file(entries, path)
            with open(path, 'r') as f:
                content = f.read()
            assert "Hello" in content
            assert "World" in content
            assert "00:00:01,000 --> 00:00:03,000" in content
        finally:
            os.unlink(path)

    def test_save_skips_empty_text(self):
        entries = [
            SubtitleEntry(index="1", timecode="00:00:01,000 --> 00:00:03,000", text=""),
            SubtitleEntry(index="2", timecode="00:00:03,000 --> 00:00:05,000", text="World"),
        ]
        fd, path = tempfile.mkstemp(suffix=".srt")
        os.close(fd)
        try:
            save_srt_file(entries, path)
            with open(path, 'r') as f:
                content = f.read()
            assert "World" in content
            # Empty text entry should produce an empty block, not "World"
        finally:
            os.unlink(path)
