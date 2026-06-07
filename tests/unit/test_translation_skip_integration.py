#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration tests for v1.8.3 translation-skip at the _process_task level.

v1.8.3 在 _process_task 决策点（worker.py:330-345）调用 should_skip_translation，
但当时只写了纯函数单测 + 1 个 Whisper 路径集成测试。**这三个场景当时没测**：

1. **内置字幕路径端到端**：视频只有中文字幕 + target=zh → 走 _try_extract_embedded_subtitle
   路径（不是 Whisper 路径），确认 _last_detected_lang 被正确设置、_translate 没被调
2. **auto 模式端到端**：用户配 source_language='auto' + Whisper 检测 zh + target=zh
   → 必须跳（auto 不影响决策，should_skip_translation 只看 detected vs target）
3. **已存在 SRT 端到端**：视频旁边已有 .srt 文件 → _extract_or_detect_subtitle 走 "已存在" 路径
   → _last_detected_lang=None → 不跳（保守行为，注释里有写"已存在的字幕语言未知，不跳翻译"）

集成测试用 sqlite in-memory + monkeypatch，跟现有 test_translation_skip.py 末尾
的 TestEndToEndSkipTranslation 同一模式。
"""

import os
import sqlite3
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from core.models import TaskStatus, SubtitleTrack


# ============================================================================
# 共用 fixture：in-memory db + get_db_connection patch
# ============================================================================

@pytest.fixture(autouse=True)
def in_memory_db(tmp_path):
    """为每个测试创建数据库，patch DAO 连接（与 test_worker_callbacks.py 同款）"""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            log TEXT DEFAULT '',
            log_history TEXT DEFAULT '',
            stage TEXT,
            stage_progress REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS media_files (
            file_path TEXT PRIMARY KEY,
            subtitles_json TEXT DEFAULT '[]',
            has_translated INTEGER DEFAULT 0
        );
    """)
    conn.close()

    def _new_conn():
        return sqlite3.connect(db_path)

    from unittest.mock import patch
    with patch("database.task_dao.get_db_connection", side_effect=_new_conn):
        with patch("database.media_dao.get_db_connection", side_effect=_new_conn):
            with patch("core.worker.get_db_connection", side_effect=_new_conn):
                yield db_path


def _make_config(target_lang="zh", source_lang="auto",
                 use_embedded=False, translation_enabled=True):
    """构造测试用 AppConfig mock"""
    config = MagicMock()
    config.translation.enabled = translation_enabled
    config.translation.target_language = target_lang
    config.translation.use_embedded_subtitle = use_embedded
    config.translation.max_lines_per_batch = 100
    config.translation.timeout = 60
    config.whisper.source_language = source_lang
    config.whisper.model_size = "base"
    config.export.formats = []  # 不真导出
    config.content_type = "movie"
    # _check_translation_config 调 get_current_provider_config
    config.get_current_provider_config.return_value = MagicMock(
        api_key="x", base_url="http://x", model_name="y"
    )
    return config


# ============================================================================
# 场景 1：内置字幕路径端到端
# ============================================================================
# 视频只有中文字幕 + target=zh → _try_extract_embedded_subtitle 返回
# (srt_path, "zh") → _last_detected_lang="zh" / _last_detected_prob=1.0
# → should_skip_translation 返回 True → _translate_subtitle 不被调

class TestEmbeddedSubtitleSkipIntegration:
    """内置字幕路径（use_embedded_subtitle=True）的端到端测试"""

    def test_embedded_zh_skips_translation_when_target_is_zh(
        self, in_memory_db, monkeypatch, tmp_path
    ):
        """
        场景：用户视频只有中文字幕 + 目标 zh
        期望：走内置字幕路径，LLM 翻译被跳过
        """
        from database.task_dao import TaskDAO
        from core.worker import TaskWorker
        from services.subtitle_extractor import SubtitleExtractor

        # 准备一个真实可写的 srt 文件（worker 要 update_task + 后续 rescan 读它）
        fake_srt = tmp_path / "embedded_zh.srt"
        fake_srt.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\n你好世界\n\n",
            encoding="utf-8"
        )

        # Mock SubtitleExtractor.detect_subtitle_tracks：返回 1 条中文软字幕
        zh_track = SubtitleTrack(
            stream_index=2,
            language="zh",  # 已经 normalize 过
            title="简体中文",
            codec_name="subrip",
            is_soft_subtitle=True,
        )

        def fake_detect(video_path):
            return [zh_track]

        # Mock SubtitleExtractor.select_best_subtitle_track：因为只有 zh，
        # 走兜底分支返回 [zh_track][0]
        def fake_select(tracks, target_language="zh"):
            return tracks[0]

        # Mock SubtitleExtractor.extract_subtitle：返回 fake_srt
        def fake_extract(video_path, track_index, output_path=None,
                         output_format="srt", embedded=False):
            return str(fake_srt)

        monkeypatch.setattr(
            SubtitleExtractor, "detect_subtitle_tracks", staticmethod(fake_detect)
        )
        monkeypatch.setattr(
            SubtitleExtractor, "select_best_subtitle_track",
            staticmethod(fake_select)
        )
        monkeypatch.setattr(
            SubtitleExtractor, "extract_subtitle", staticmethod(fake_extract)
        )

        # Mock os.path.exists：让 worker 看见视频文件存在
        monkeypatch.setattr("os.path.exists", lambda p: True)

        # Mock _translate_subtitle：记录是否被调
        translate_called = []

        def fake_translate(self, task_id, srt_path, config):
            translate_called.append(True)
            return True

        monkeypatch.setattr("core.worker.TaskWorker._translate_subtitle", fake_translate)

        # Mock rescan_video_subtitles（避免读 media_files 表失败）
        monkeypatch.setattr("core.worker.rescan_video_subtitles", lambda p: None)

        # 准备任务
        TaskDAO.add_task("/media/embedded_zh.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(
            task.id, status=TaskStatus.PROCESSING, stage="extract"
        )

        # 跑 _process_task
        worker = TaskWorker()
        config = _make_config(
            target_lang="zh", source_lang="auto", use_embedded=True
        )

        try:
            worker._process_task(task.id, "/media/embedded_zh.mp4", config)
        except Exception as e:
            pytest.fail(f"_process_task crashed: {e}")

        # 关键断言 1：_translate_subtitle 没被调
        assert len(translate_called) == 0, (
            f"内置字幕 zh + target=zh 应跳过翻译，但 _translate_subtitle 被调了 "
            f"{len(translate_called)} 次"
        )

        # 关键断言 2：log_history 含跳过原因
        final = TaskDAO.get_task_by_id(task.id)
        assert "跳过" in final.log_history or "目标语言" in final.log_history, (
            f"log_history 应含跳过原因，实际: {final.log_history!r}"
        )

        # 关键断言 3：任务最终状态是 COMPLETED（跳过翻译不是失败）
        assert final.status == TaskStatus.COMPLETED, (
            f"跳过翻译后任务应 COMPLETED，实际: {final.status}"
        )


# ============================================================================
# 场景 2：auto 模式端到端
# ============================================================================
# Whisper 检测 zh + target=zh + source_language='auto' → 跳
# 验证 should_skip_translation 在 source='auto' 时仍正确决策

class TestAutoSourceModeIntegration:
    """auto 模式（source_language='auto'）的端到端测试"""

    def test_auto_mode_skips_when_whisper_detects_target_language(
        self, in_memory_db, monkeypatch, tmp_path
    ):
        """
        场景：用户配 source='auto' + Whisper 检测 zh + target=zh
        期望：跳过 LLM 翻译（auto 不影响决策）
        """
        from database.task_dao import TaskDAO
        from core.worker import TaskWorker
        from services.whisper_service import WhisperService

        # 准备 fake srt
        fake_srt = tmp_path / "auto_zh.srt"
        fake_srt.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n",
            encoding="utf-8"
        )

        # Mock WhisperService.extract_subtitle：模拟返回 zh + 0.95
        # （虽然在 TestAuto 里我们走更上层的 mock _extract_subtitle，
        #  这个仍作为保险层 —— 万一被调到不会崩）
        def fake_extract(self, video_path, output_path=None, progress_callback=None):
            self._last_detected_lang = "zh"
            self._last_detected_prob = 0.95
            if progress_callback:
                progress_callback("extract", 100.0, "字幕提取完成")
            return str(fake_srt), "zh", 0.95

        # 用字符串路径（v1.8.3 集成测试同款写法）
        monkeypatch.setattr(
            "services.whisper_service.WhisperService.extract_subtitle",
            fake_extract,
        )

        # Mock _translate_subtitle
        translate_called = []

        def fake_translate(self, task_id, srt_path, config):
            translate_called.append(True)
            return True

        monkeypatch.setattr("core.worker.TaskWorker._translate_subtitle", fake_translate)
        monkeypatch.setattr("core.worker.rescan_video_subtitles", lambda p: None)

        # 准备任务
        TaskDAO.add_task("/media/auto_zh.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(
            task.id, status=TaskStatus.PROCESSING, stage="extract"
        )

        # Mock os.path.exists：让 worker 看见视频文件存在
        monkeypatch.setattr("os.path.exists", lambda p: True)

        # 关键：worker._extract_or_detect_subtitle 第一步会检查 srt_path.exists()
        # 若 True 走 fast-return 路径，不调 _extract_subtitle（这样就测不到 skip 逻辑）
        # 我们要测"Whisper 路径端到端"，所以 srt_path 应当**不存在**
        # 方案：直接 mock worker._extract_subtitle（不通过 with_suffix 链路）
        def fake_worker_extract_subtitle(self, task_id, file_path, cfg):
            self._last_detected_lang = "zh"
            self._last_detected_prob = 0.95
            TaskDAO.update_task(task_id, log="字幕提取完成", append_log=True)
            return str(fake_srt)

        monkeypatch.setattr(
            "core.worker.TaskWorker._extract_subtitle",
            fake_worker_extract_subtitle,
        )

        # 跑 _process_task
        worker = TaskWorker()
        config = _make_config(
            target_lang="zh", source_lang="auto",  # ← auto 模式
            use_embedded=False, translation_enabled=True
        )

        try:
            worker._process_task(task.id, "/media/auto_zh.mp4", config)
        except Exception as e:
            pytest.fail(f"_process_task crashed: {e}")

        # 关键断言：auto 模式下仍正确跳过
        assert len(translate_called) == 0, (
            f"auto 模式下 Whisper 检测 zh + target=zh 应跳，但 _translate 被调了 "
            f"{len(translate_called)} 次"
        )

        final = TaskDAO.get_task_by_id(task.id)
        assert final.status == TaskStatus.COMPLETED


# ============================================================================
# 场景 3：已存在 SRT 端到端
# ============================================================================
# 视频旁边有 .srt → _extract_or_detect_subtitle 走 "已存在" 路径
# → _last_detected_lang=None → should_skip_translation 返回 False
# → 走 LLM 翻译（保守行为：已存在的字幕语言未知，不跳）

class TestExistingSrtDoesNotSkipIntegration:
    """已存在 SRT 路径的端到端测试"""

    def test_existing_srt_does_not_skip_translation(
        self, in_memory_db, monkeypatch, tmp_path
    ):
        """
        场景：视频旁边已有 .srt 文件（用户原本就有）
        期望：走翻译流程（保守：语言未知，宁可多花 LLM 钱也别跳错）
        """
        from database.task_dao import TaskDAO
        from core.worker import TaskWorker

        # 准备真实存在的 srt 文件
        existing_srt = tmp_path / "existing.srt"
        existing_srt.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n",
            encoding="utf-8"
        )

        # Mock _translate_subtitle：成功
        translate_called = []

        def fake_translate(self, task_id, srt_path, config):
            translate_called.append(True)
            return True

        monkeypatch.setattr("core.worker.TaskWorker._translate_subtitle", fake_translate)
        monkeypatch.setattr("core.worker.rescan_video_subtitles", lambda p: None)
        monkeypatch.setattr("os.path.exists", lambda p: True)

        # 让 worker 看见 existing_srt：tmp_path 路径用 monkeypatch 替换
        # 策略：mock _extract_or_detect_subtitle 直接返回 srt_path
        def fake_extract_or_detect(self, task_id, file_path, config):
            self._last_detected_lang = None
            self._last_detected_prob = 0.0
            TaskDAO.update_task(task_id, progress=50, log="字幕已存在", append_log=True)
            return str(existing_srt)

        monkeypatch.setattr(
            "core.worker.TaskWorker._extract_or_detect_subtitle",
            fake_extract_or_detect
        )

        # 准备任务
        TaskDAO.add_task("/media/existing.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(
            task.id, status=TaskStatus.PROCESSING, stage="extract"
        )

        # 跑 _process_task
        worker = TaskWorker()
        config = _make_config(target_lang="zh", source_lang="auto")

        try:
            worker._process_task(task.id, "/media/existing.mp4", config)
        except Exception as e:
            pytest.fail(f"_process_task crashed: {e}")

        # 关键断言：_translate_subtitle 被调了 1 次（保守：不跳）
        assert len(translate_called) == 1, (
            f"已存在 SRT + 语言未知应走翻译，但 _translate 被调了 "
            f"{len(translate_called)} 次"
        )

        # 任务最终应 COMPLETED
        final = TaskDAO.get_task_by_id(task.id)
        assert final.status == TaskStatus.COMPLETED
