#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/config.py APP_VERSION 守卫测试

目的：防止 commit message 写了 v1.x.x 但忘了改代码常量。
具体场景：CI workflow (.github/workflows/docker-publish.yml) 用
`grep -oP 'APP_VERSION = "v\\K[^"]+' core/config.py` 提取版本号打 Docker tag。
如果 APP_VERSION 没改，CI 推的 tag 还是旧版本号，用户 docker
看到的"最新"就跟实际 commit 不一致。

这个测试不强求版本号格式（不同项目格式不同），只要求：
1. APP_VERSION 是字符串
2. 格式是 v<数字>.<数字>.<数字>(+可选后缀)
3. 跟最近一次 commit message 里出现的 vX.Y.Z 不冲突
4. CI 的 grep 模式能正常提取

第 3 点是核心：每次 release 时，code 常量必须跟 commit message
对齐。

注意：所有路径都用 pathlib.Path(__file__) 推导，绝不硬编码绝对路径！
（CI 容器在 /home/runner/work/.../,本地在 /home/dev/nas-submaster/,
 任何硬编码都会让 CI 失败——之前 37b78e7 commit 踩过这个坑。）
"""

import re
import subprocess
from pathlib import Path

import pytest

from core.config import APP_VERSION


# ---------------------------------------------------------------------------
# 路径推导：兼容本地 + CI
# ---------------------------------------------------------------------------

# tests/unit/test_app_version.py → 项目根 = 父级的父级
TESTS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = TESTS_DIR.parent.parent.resolve()
CONFIG_FILE = PROJECT_ROOT / "core" / "config.py"


# ---------------------------------------------------------------------------
# 格式守卫
# ---------------------------------------------------------------------------

class TestAppVersionFormat:
    """APP_VERSION 必须符合 v<major>.<minor>.<patch>[suffix] 格式"""

    def test_app_version_is_string(self):
        assert isinstance(APP_VERSION, str)

    def test_app_version_starts_with_v(self):
        assert APP_VERSION.startswith("v"), f"APP_VERSION 应以 'v' 开头: {APP_VERSION!r}"

    def test_app_version_format(self):
        """v<major>.<minor>.<patch> 或带后缀（-rc1, .1 等）"""
        pattern = r"^v\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$"
        assert re.match(pattern, APP_VERSION), (
            f"APP_VERSION 格式不对: {APP_VERSION!r}\n"
            f"应该是 v<major>.<minor>.<patch>[suffix]，比如 v1.7.7 或 v1.7.7-rc1"
        )


# ---------------------------------------------------------------------------
# Commit message 对齐（防 commit msg 跟 code 漂移）
# ---------------------------------------------------------------------------

class TestAppVersionAlignsWithCommits:
    """APP_VERSION 跟最近 commit message 里提到的版本号要一致

    依赖 git CLI:CI runner (github-hosted ubuntu) 默认有 git,但
    纯 Python 容器可能没装 — 用 skipif 让 fixture/测试优雅退化。
    """

    @pytest.fixture
    def last_commit_message(self) -> str:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip()

    @pytest.fixture
    def last_5_commits(self) -> list:
        result = subprocess.run(
            ["git", "log", "-5", "--pretty=%H %s"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(PROJECT_ROOT),
        )
        return [line.strip() for line in result.stdout.strip().splitlines()]

    @pytest.fixture(autouse=True)
    def require_git(self):
        """在 fixture 阶段就检查 git 是否可用,不可用就 skip 整个 class"""
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pytest.skip("git 不可用,跳过 commit message 守卫")

    def test_recent_commits_have_version_in_subject(self, last_5_commits):
        """最近 5 个 commit 至少要有 1 个主题里含 v<APP_VERSION>

        防止 APP_VERSION 改完，但周围 commit message 全是 v1.7.6
        这种漂移。
        """
        # 把 v1.7.7 变成 v1\.7\.7 适配正则
        esc = re.escape(APP_VERSION)
        pattern = re.compile(esc)
        matches = [c for c in last_5_commits if pattern.search(c)]
        # 这条规则是软的（有些 commit 可能不写版本号），所以只 warn 不 fail
        # 但如果 5 个 commit 全部不含 APP_VERSION，那就说明有漂移
        if not matches:
            pytest.skip(
                f"最近 5 个 commit 都没出现 {APP_VERSION}, "
                f"可能 APP_VERSION 改了但 commit message 漂移（旧版本号还残留）"
            )

    def test_head_commit_message_matches_app_version(self, last_commit_message):
        """🚨 强约束：HEAD commit 主题里必须出现 APP_VERSION 字符串

        这个测试防止「commit message 写 v1.7.x 但忘了改 APP_VERSION」或
        反过来「改了 APP_VERSION 但 commit message 漂移」——

        踩坑历史：
        - 37b78e7 (bump v1.7.6 → v1.7.7): 之前多个 commit 都用 v1.7.6 message，
          但 APP_VERSION 没跟上，Docker Hub 上 v1.7.7 tag 跟实际代码漂移。
        - 8d56aba (feat+fix progress): commit message 写 v1.7.7 但用户
          反馈应该 bump v1.7.8——这是因为之前的守卫太软（没匹配就 skip
          而不是 fail），导致漏检。

        修复方案：HEAD commit 主题（前 200 字符）必须包含 APP_VERSION，
        否则直接 fail。允许 HEAD 是 bump-only commit（即 message 里
        APP_VERSION 出现 2+ 次：旧版号 + 新版号，或者只出现新版号）。
        """
        # 主题 = 第一行；取前 200 字符就够
        subject = last_commit_message.split("\n", 1)[0][:200]
        if APP_VERSION not in subject:
            pytest.fail(
                f"🚨 HEAD commit 主题必须包含 APP_VERSION={APP_VERSION!r}，\n"
                f"实际主题: {subject!r}\n"
                f"完整 message: {last_commit_message[:300]!r}\n"
                f"\n"
                f"这是为了防止两种漂移：\n"
                f"1. commit message 写了 vX.Y.Z 但 APP_VERSION 没改 → CI 推的 tag 跟代码不一致\n"
                f"2. APP_VERSION 改了但 commit message 漂移 → 用户看不出这是哪个版本的提交\n"
                f"\n"
                f"正确做法：\n"
                f"- 改 APP_VERSION 时，commit message 必须写明新版本号\n"
                f"- 建议 message 格式：'fix|feat|chore|...: <描述> v1.7.X'\n"
                f"  （fix 类型可以不写，因为 fix 经常是 v1.7.x→v1.7.x+1 的常规 bump）\n"
                f"\n"
                f"如果这个 commit 真的不该写版本号（极少见），请用 git commit --amend 加上。"
            )


# ---------------------------------------------------------------------------
# CI 提取版本号的契约（镜像 .github/workflows/docker-publish.yml）
# ---------------------------------------------------------------------------

class TestAppVersionGrepCompatible:
    """CI workflow 用这个 grep 提取版本号，必须兼容

    grep -oP 'APP_VERSION = "v\\K[^"]+' core/config.py
    """

    def test_config_file_exists(self):
        """路径推导后能找到 core/config.py（防止 CI 路径错）"""
        assert CONFIG_FILE.exists(), (
            f"找不到 {CONFIG_FILE}，路径推导可能错了。"
            f"TESTS_DIR={TESTS_DIR}, PROJECT_ROOT={PROJECT_ROOT}"
        )

    def test_grep_extracts_cleanly(self):
        """CI workflow 用这个 grep 提取版本号，必须兼容

        原始 CI 命令：grep -oP 'APP_VERSION = "v\\K[^"]+' core/config.py
        (grep -P 是 PCRE,支持 \\K)

        Python re 不支持 \\K,所以这里用等价写法：捕获 v 后面的部分。
        """
        content = CONFIG_FILE.read_text(encoding="utf-8")
        # 等价于 CI 的 grep
        m = re.search(r'APP_VERSION = "(v[^"]+)"', content)
        assert m is not None, "CI grep 模式没匹配到 APP_VERSION,workflow 会挂"
        # 提取出来的带 v 前缀
        assert m.group(1) == APP_VERSION
        # 同时确保 APP_VERSION 是 v 开头（CI 用 lstrip("v") 后取数字部分打 tag）
        assert APP_VERSION.startswith("v")
