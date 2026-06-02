"""Tests for ConfigManager prompt_templates persistence.

修复 issue #4 评论中提到的"提示词修改保存不了"bug。
覆盖：
- save() 后重新 load，能读回 prompt_templates
- 空 DB 时 load() 返回默认空 dict（兼容旧 DB）
"""
import sqlite3

import pytest

from core.config import ConfigManager
from core.models import ContentType, PromptTemplate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_db(tmp_path):
    """创建一个临时 SQLite DB（仅含 config 表），返回 get_db 可调用对象。"""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    def get_db():
        # 每次返回新连接，模拟生产中 get_db_connection 的行为
        return sqlite3.connect(str(db_path), check_same_thread=False)

    return get_db


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_save_persists_prompt_templates(fresh_db):
    """save() 后重新 load()，prompt_templates 必须能读回（核心 bug 验证）。"""
    mgr = ConfigManager(fresh_db)
    config = mgr.load()

    # 用户设置一个自定义的电影翻译提示词
    custom_movie = PromptTemplate(
        role="你是一位专业电影翻译",
        rules="1. 保持口语化\n2. 简洁",
        style_guide="电影感"
    )
    config.prompt_templates[ContentType.MOVIE] = custom_movie

    mgr.save(config)

    # 用全新 ConfigManager 模拟下一次会话的 load
    mgr2 = ConfigManager(fresh_db)
    loaded = mgr2.load()

    assert ContentType.MOVIE in loaded.prompt_templates, (
        "prompt_templates 未被持久化到 DB（issue #4 评论中提到的 bug）"
    )
    assert loaded.prompt_templates[ContentType.MOVIE].role == "你是一位专业电影翻译"
    assert loaded.prompt_templates[ContentType.MOVIE].rules == "1. 保持口语化\n2. 简洁"
    assert loaded.prompt_templates[ContentType.MOVIE].style_guide == "电影感"


def test_load_returns_empty_prompt_templates_when_db_empty(fresh_db):
    """空 DB 时 load() 应返回默认空 prompt_templates dict（兼容旧 DB）。"""
    mgr = ConfigManager(fresh_db)
    config = mgr.load()

    assert config.prompt_templates == {}
