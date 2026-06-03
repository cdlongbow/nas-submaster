#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translator 服务单元测试
"""

from unittest.mock import patch, MagicMock
from services.translator import SubtitleTranslator, TranslationConfig


class TestOllamaCredentials:
    """Ollama 不需要 API Key 的场景"""

    @patch("services.translator.OpenAI")
    def test_empty_api_key_defaults_to_ollama(self, mock_openai):
        """api_key 为空时，默认使用 'ollama' 作为占位符"""
        config = TranslationConfig(
            api_key="",
            base_url="http://host.docker.internal:11434/v1",
            model_name="qwen2:7b",
            target_language="zh",
        )
        SubtitleTranslator(config)

        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["api_key"] == "ollama"

    @patch("services.translator.OpenAI")
    def test_ollama_in_url_sets_api_key(self, mock_openai):
        """base_url 包含 'ollama' 时，即使 api_key 非空也应覆盖为 'ollama'"""
        config = TranslationConfig(
            api_key="some-key",
            base_url="http://ollama:11434/v1",
            model_name="qwen2:7b",
            target_language="zh",
        )
        SubtitleTranslator(config)

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["api_key"] == "ollama"

    @patch("services.translator.OpenAI")
    def test_non_ollama_provider_keeps_api_key(self, mock_openai):
        """非 Ollama 服务商保留原始 api_key"""
        config = TranslationConfig(
            api_key="sk-real-key",
            base_url="https://api.deepseek.com/v1",
            model_name="deepseek-chat",
            target_language="zh",
        )
        SubtitleTranslator(config)

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["api_key"] == "sk-real-key"

    @patch("services.translator.OpenAI")
    def test_none_api_key_defaults_to_ollama(self, mock_openai):
        """api_key 为 None 时，默认使用 'ollama'"""
        config = TranslationConfig(
            api_key=None,
            base_url="http://host.docker.internal:11434/v1",
            model_name="qwen2:7b",
            target_language="zh",
        )
        SubtitleTranslator(config)

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["api_key"] == "ollama"


class TestClientTimeout:
    """客户端超时配置"""

    @patch("services.translator.OpenAI")
    def test_client_timeout_set_from_config(self, mock_openai):
        """timeout 应在客户端级别设置"""
        config = TranslationConfig(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model_name="test-model",
            target_language="zh",
            timeout=120,
        )
        SubtitleTranslator(config)

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["timeout"] == 120

    @patch("services.translator.OpenAI")
    def test_default_timeout_is_180(self, mock_openai):
        """默认超时应为 180 秒"""
        config = TranslationConfig(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model_name="test-model",
            target_language="zh",
        )
        SubtitleTranslator(config)

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["timeout"] == 180
