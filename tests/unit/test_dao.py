#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAO 层集成测试 — 使用内存 SQLite"""

import sqlite3
import json
import pytest
from unittest.mock import patch

from core.models import Task, TaskStatus, MediaFile, SubtitleInfo


# ============================================================================
# 共用 fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def in_memory_db(tmp_path):
    """为每个测试创建数据库并 patch get_db_connection 为工厂函数"""
    db_path = str(tmp_path / "test.db")

    # 初始化 schema
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS media_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            file_name TEXT,
            file_size INTEGER,
            subtitles_json TEXT DEFAULT '[]',
            has_translated INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            log TEXT DEFAULT '',
            log_history TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.close()

    # 工厂函数：每次调用返回新连接（DAO 方法在 finally 中 close）
    def _new_conn():
        return sqlite3.connect(db_path)

    with patch("database.task_dao.get_db_connection", side_effect=_new_conn):
        with patch("database.media_dao.get_db_connection", side_effect=_new_conn):
            yield db_path


# ============================================================================
# TaskDAO
# ============================================================================

class TestTaskDAO:
    def test_add_and_get_task(self):
        from database.task_dao import TaskDAO
        ok, msg = TaskDAO.add_task("/media/movie.mp4")
        assert ok is True
        task = TaskDAO.get_pending_task()
        assert task is not None
        assert task.file_path == "/media/movie.mp4"
        assert task.status == TaskStatus.PENDING

    def test_add_duplicate_task(self):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        ok, msg = TaskDAO.add_task("/media/movie.mp4")
        assert ok is False
        assert "已存在" in msg

    def test_update_task_status(self):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(task.id, status=TaskStatus.PROCESSING, progress=50, log="处理中")
        updated = TaskDAO.get_task_by_id(task.id)
        assert updated.status == TaskStatus.PROCESSING
        assert updated.progress == 50

    def test_update_task_append_log(self):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(task.id, log="第一步", append_log=True)
        TaskDAO.update_task(task.id, log="第二步", append_log=True)
        updated = TaskDAO.get_task_by_id(task.id)
        assert "第一步" in updated.log_history
        assert "第二步" in updated.log_history

    def test_delete_task(self):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.delete_task(task.id)
        assert TaskDAO.get_task_by_id(task.id) is None

    def test_clear_completed_tasks(self):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/a.mp4")
        TaskDAO.add_task("/media/b.mp4")
        tasks = TaskDAO.get_all_tasks()
        TaskDAO.update_task(tasks[0].id, status=TaskStatus.COMPLETED)
        TaskDAO.update_task(tasks[1].id, status=TaskStatus.FAILED)
        TaskDAO.clear_completed_tasks()
        remaining = TaskDAO.get_all_tasks()
        assert len(remaining) == 0

    def test_reset_task(self):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(task.id, status=TaskStatus.FAILED, log="出错")
        TaskDAO.reset_task(task.id)
        reset = TaskDAO.get_task_by_id(task.id)
        assert reset.status == TaskStatus.PENDING
        assert reset.progress == 0

    def test_cancel_task(self):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.cancel_task(task.id)
        cancelled = TaskDAO.get_task_by_id(task.id)
        assert cancelled.status == TaskStatus.CANCELLED

    def test_has_processing_task(self):
        from database.task_dao import TaskDAO
        assert TaskDAO.has_processing_task() is False
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(task.id, status=TaskStatus.PROCESSING)
        assert TaskDAO.has_processing_task() is True

    def test_reset_stale_processing_tasks(self):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(task.id, status=TaskStatus.PROCESSING)
        TaskDAO.reset_stale_processing_tasks()
        reset = TaskDAO.get_task_by_id(task.id)
        assert reset.status == TaskStatus.PENDING

    def test_get_task_count_by_status(self):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/a.mp4")
        TaskDAO.add_task("/media/b.mp4")
        assert TaskDAO.get_task_count_by_status(TaskStatus.PENDING) == 2
        assert TaskDAO.get_task_count_by_status(TaskStatus.COMPLETED) == 0


# ============================================================================
# MediaDAO
# ============================================================================

class TestMediaDAO:
    def test_add_and_get_media(self):
        from database.media_dao import MediaDAO
        MediaDAO.add_or_update_media_file(
            "/media/movie.mp4", "movie.mp4", 1000000,
            [SubtitleInfo(path="/media/movie.srt", lang="zh", source="asr")]
        )
        media = MediaDAO.get_media_by_path("/media/movie.mp4")
        assert media is not None
        assert media.file_name == "movie.mp4"
        assert media.file_size == 1000000
        assert len(media.subtitles) == 1
        assert media.subtitles[0].lang == "zh"

    def test_update_existing_media(self):
        from database.media_dao import MediaDAO
        MediaDAO.add_or_update_media_file("/media/movie.mp4", "movie.mp4", 1000000, [])
        MediaDAO.add_or_update_media_file(
            "/media/movie.mp4", "movie.mp4", 1000000,
            [SubtitleInfo(path="/media/movie.srt", lang="en", source="asr")]
        )
        media = MediaDAO.get_media_by_path("/media/movie.mp4")
        assert len(media.subtitles) == 1
        assert media.subtitles[0].lang == "en"

    def test_delete_media(self):
        from database.media_dao import MediaDAO
        MediaDAO.add_or_update_media_file("/media/movie.mp4", "movie.mp4", 1000000, [])
        MediaDAO.delete_media_file("/media/movie.mp4")
        assert MediaDAO.get_media_by_path("/media/movie.mp4") is None

    def test_get_media_count(self):
        from database.media_dao import MediaDAO
        assert MediaDAO.get_media_count() == 0
        MediaDAO.add_or_update_media_file("/media/a.mp4", "a.mp4", 1000, [])
        MediaDAO.add_or_update_media_file("/media/b.mp4", "b.mp4", 2000, [])
        assert MediaDAO.get_media_count() == 2

    def test_update_media_subtitles(self):
        from database.media_dao import MediaDAO
        MediaDAO.add_or_update_media_file("/media/movie.mp4", "movie.mp4", 1000000, [])
        MediaDAO.update_media_subtitles(
            "/media/movie.mp4",
            [SubtitleInfo(path="/media/movie.zh.srt", lang="zh", source="translated")],
            True
        )
        media = MediaDAO.get_media_by_path("/media/movie.mp4")
        assert media.has_translated is True
        assert media.subtitles[0].source == "translated"

    def test_get_media_files_filtered_with_subtitle(self):
        from database.media_dao import MediaDAO
        MediaDAO.add_or_update_media_file(
            "/media/a.mp4", "a.mp4", 1000,
            [SubtitleInfo(path="/a.srt", lang="zh")]
        )
        MediaDAO.add_or_update_media_file("/media/b.mp4", "b.mp4", 2000, [])

        with_subs = MediaDAO.get_media_files_filtered(has_subtitle=True)
        without_subs = MediaDAO.get_media_files_filtered(has_subtitle=False)
        assert len(with_subs) == 1
        assert len(without_subs) == 1

    def test_get_nonexistent_media(self):
        from database.media_dao import MediaDAO
        assert MediaDAO.get_media_by_path("/nonexistent.mp4") is None
