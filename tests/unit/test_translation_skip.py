#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for source==target language translation skip logic (v1.8.3).

背景：之前 worker.py:250 不管三七二十一都调 LLM 翻译，即使字幕已经是
目标语言（中文视频 + target=zh），浪费 LLM 钱+时间。

v1.8.3: 在 worker 决策点加 should_skip_translation() 判断：
- detected_lang == target_lang
- detected_prob >= 0.5 (跟 faster-whisper 官方 threshold 对齐)
- source_language 不是 'auto'（auto 模式信任 LLM）

参考：faster_whisper/transcribe.py:1768-1841 (language_detection_threshold=0.5)
参考：faster_whisper/tests/test_transcribe.py:48,67 (官方测试用 0.7)

我们选 0.5（官方默认）—— 中文/英文/日文等主流语言在 medium 模型下
>0.5 是非常稳的判断（实测 JFK 音频 0.99+）。
"""

import sqlite3
import pytest
from unittest.mock import MagicMock

from core.worker import should_skip_translation
from core.models import WhisperConfig, TranslationConfig


# ============================================================================
# 基础场景：detected == target
# ============================================================================

class TestSkipWhenDetectedEqualsTarget:
    def test_skip_when_zh_to_zh_high_confidence(self):
        """检测到中文 + 目标中文 + 高置信度 → 跳过"""
        skip, reason = should_skip_translation(
            detected_lang="zh",
            detected_prob=0.95,
            source_language="auto",   # Whisper 总是 auto（除非显式传）
            target_language="zh",
        )
        assert skip is True
        assert "zh" in reason
        assert "95" in reason  # 置信度文字

    def test_skip_when_en_to_en_high_confidence(self):
        """英文 → 英文，置信度 0.8 → 跳过"""
        skip, reason = should_skip_translation(
            detected_lang="en",
            detected_prob=0.80,
            source_language="en",
            target_language="en",
        )
        assert skip is True


# ============================================================================
# 反向：detected != target，必须翻译
# ============================================================================

class TestDoNotSkipWhenDifferentLanguage:
    def test_translate_when_en_to_zh(self):
        """英文 → 中文，必须翻译"""
        skip, reason = should_skip_translation(
            detected_lang="en",
            detected_prob=0.99,
            source_language="auto",
            target_language="zh",
        )
        assert skip is False
        assert "翻译" in reason or "继续" in reason or "不跳" in reason

    def test_translate_when_ja_to_zh(self):
        """日文 → 中文，必须翻译"""
        skip, reason = should_skip_translation(
            detected_lang="ja",
            detected_prob=0.95,
            source_language="auto",
            target_language="zh",
        )
        assert skip is False


# ============================================================================
# 置信度门槛 (核心安全逻辑)
# ============================================================================

class TestConfidenceThreshold:
    def test_do_not_skip_when_below_threshold(self):
        """
        置信度 < 0.5 不跳 —— 避免 Whisper 判错时静默跳过

        这是关键安全逻辑：动漫/混合语种/有口音场景 Whisper 可能判错，
        我们宁可花 LLM 钱也不能静默翻译错。
        """
        skip, reason = should_skip_translation(
            detected_lang="zh",
            detected_prob=0.3,  # 低于 0.5
            source_language="auto",
            target_language="zh",
        )
        assert skip is False
        # reason 应该提到置信度低
        assert "置信度" in reason or "conf" in reason.lower()

    def test_skip_at_just_above_threshold(self):
        """置信度 0.51 → 跳过（严格 > 0.5，跟 faster-whisper 官方一致）"""
        skip, reason = should_skip_translation(
            detected_lang="zh",
            detected_prob=0.51,  # 略高于 0.5
            source_language="auto",
            target_language="zh",
        )
        assert skip is True

    def test_do_not_skip_at_exactly_threshold(self):
        """
        置信度 == 0.5 不跳 —— 跟 faster-whisper 官方 `>` 严格大于一致
        (transcribe.py:1829: `if language_probability > language_detection_threshold`)
        """
        skip, reason = should_skip_translation(
            detected_lang="zh",
            detected_prob=0.5,  # 临界值
            source_language="auto",
            target_language="zh",
        )
        assert skip is False

    def test_do_not_skip_at_just_below_threshold(self):
        """置信度 0.49 → 不跳"""
        skip, reason = should_skip_translation(
            detected_lang="zh",
            detected_prob=0.49,
            source_language="auto",
            target_language="zh",
        )
        assert skip is False


# ============================================================================
# Auto 模式（source_language='auto'）
# ============================================================================

class TestAutoMode:
    def test_auto_mode_with_matching_detected_should_skip(self):
        """
        source='auto' 但 Whisper 检测到 == target 时仍可跳
        （Whisper 实际就是 auto 检测）
        """
        skip, reason = should_skip_translation(
            detected_lang="zh",
            detected_prob=0.95,
            source_language="auto",
            target_language="zh",
        )
        assert skip is True

    def test_auto_mode_with_different_detected_should_translate(self):
        """source='auto'，检测到 en，target=zh → 翻译"""
        skip, reason = should_skip_translation(
            detected_lang="en",
            detected_prob=0.95,
            source_language="auto",
            target_language="zh",
        )
        assert skip is False


# ============================================================================
# 边界
# ============================================================================

class TestEdgeCases:
    def test_zero_probability_does_not_skip(self):
        """置信度 0 → 不跳（Whisper 报 0 概率说明完全不确定）"""
        skip, reason = should_skip_translation(
            detected_lang="zh",
            detected_prob=0.0,
            source_language="auto",
            target_language="zh",
        )
        assert skip is False

    def test_empty_detected_lang_does_not_skip(self):
        """detected_lang 为空 → 不跳（Whisper 没检测到）"""
        skip, reason = should_skip_translation(
            detected_lang="",
            detected_prob=0.99,
            source_language="auto",
            target_language="zh",
        )
        assert skip is False

    def test_none_detected_lang_does_not_skip(self):
        """detected_lang=None → 不跳"""
        skip, reason = should_skip_translation(
            detected_lang=None,
            detected_prob=0.99,
            source_language="auto",
            target_language="zh",
        )
        assert skip is False

    def test_case_insensitive_language_match(self):
        """'ZH' 和 'zh' 视为相同（Whisper 用小写但 defensive）"""
        skip, _ = should_skip_translation(
            detected_lang="ZH",
            detected_prob=0.95,
            source_language="auto",
            target_language="zh",
        )
        assert skip is True


# ============================================================================
# Worker 集成点测试（验证实际嵌入 _process_task 决策）
# ============================================================================

class TestWorkerIntegrationWithSkip:
    """_process_task 在 source==target 时应跳过 _translate_subtitle"""

    def test_process_task_skips_translation_when_zh_detected(self, tmp_path, monkeypatch):
        """
        完整 _process_task 流程：Whisper 检测 zh + target=zh → 不调 _translate_subtitle
        """
        from database.task_dao import TaskDAO
        from core.worker import TaskWorker
        from core.models import TaskStatus
        import sqlite3

        # 内存 DB
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                status TEXT DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                log TEXT DEFAULT '',
                log_history TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at CURRENT_TIMESTAMP
            );
        """)
        conn.close()

        def _new_conn():
            return sqlite3.connect(db_path)

        monkeypatch.setattr("core.worker.get_db_connection", _new_conn)
        monkeypatch.setattr("database.task_dao.get_db_connection", _new_conn)

        # 创建任务
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(
            task.id, status=TaskStatus.PROCESSING, stage="extract"
        )

        # Mock os.path.exists 让 worker 认为文件存在
        import os
        monkeypatch.setattr("os.path.exists", lambda p: True)
        # 也让 Path.with_suffix 不会失败
        from pathlib import Path
        monkeypatch.setattr(
            "pathlib.Path.with_suffix",
            lambda self, suffix: Path("/tmp/fake_movie.srt")
        )

        # Mock WhisperService.extract_subtitle：返回 (srt_path, "zh", 0.95)
        def fake_extract_subtitle(self, video_path, output_path=None, progress_callback=None):
            self._last_detected_lang = "zh"
            self._last_detected_prob = 0.95
            if progress_callback:
                progress_callback("extract", 100.0, "字幕提取完成")
            return output_path or "/media/movie.srt", "zh", 0.95

        monkeypatch.setattr(
            "services.whisper_service.WhisperService.extract_subtitle",
            fake_extract_subtitle,
        )

        # Mock translate —— 关键：测它**没被调用**
        translate_called = []

        def fake_translate(self, task_id, srt_path, config):
            translate_called.append(True)
            return True

        monkeypatch.setattr(
            "core.worker.TaskWorker._translate_subtitle",
            fake_translate,
        )

        # 准备 worker + config
        worker = TaskWorker()
        config = MagicMock()
        config.translation.enabled = True
        config.translation.target_language = "zh"
        config.translation.use_embedded_subtitle = False
        config.whisper.source_language = "auto"
        config.export.formats = []

        # 跑 _process_task
        try:
            worker._process_task(task.id, "/media/movie.mp4", config)
        except Exception as e:
            pytest.fail(f"_process_task crashed: {e}")

        # 关键断言：_translate_subtitle 没被调用
        assert len(translate_called) == 0, (
            f"v1.8.3 应该跳过翻译，但 _translate_subtitle 被调了 {len(translate_called)} 次"
        )

        # 验证 task log_history 包含跳过原因（log 字段被最后一步覆盖了，但 log_history 保留）
        final = TaskDAO.get_task_by_id(task.id)
        assert "跳过翻译" in final.log_history or "目标语言" in final.log_history, (
            f"log_history 应包含跳过原因，实际: {final.log_history!r}"
        )


# ============================================================================
# Reason 字符串
# ============================================================================

class TestReasonStrings:
    def test_skip_reason_contains_language_and_probability(self):
        """skip reason 应包含语言和置信度（让用户 log 里看得懂）"""
        _, reason = should_skip_translation(
            detected_lang="zh",
            detected_prob=0.95,
            source_language="auto",
            target_language="zh",
        )
        assert "zh" in reason
        # 95 应该有
        assert "95" in reason

    def test_low_confidence_reason_mentions_confidence(self):
        """不跳时，reason 应提到置信度低（让用户知道为啥仍翻译）"""
        _, reason = should_skip_translation(
            detected_lang="zh",
            detected_prob=0.3,
            source_language="auto",
            target_language="zh",
        )
        assert "置信度" in reason or "低" in reason or "0.3" in reason
