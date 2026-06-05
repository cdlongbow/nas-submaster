#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用更新服务
从 GitHub Releases 检查新版本、获取更新日志、执行 Docker 更新
"""

import subprocess
from dataclasses import dataclass
from typing import List, Optional

import requests

from core.config import APP_VERSION

# GitHub 仓库信息
GITHUB_OWNER = "aexachao"
GITHUB_REPO = "nas-submaster"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"


@dataclass
class ReleaseInfo:
    """GitHub Release 信息"""
    tag_name: str       # 如 "v1.7.0"
    name: str           # 如 "v1.7.0 - 新增自动更新"
    body: str           # 更新日志正文
    published_at: str   # 发布时间
    html_url: str       # 链接


def parse_version(version_str: str) -> tuple:
    """
    解析版本号字符串为可比较的元组。

    Args:
        version_str: 如 "v1.7.0" 或 "1.7.0"
    Returns:
        (major, minor, patch) 如 (1, 7, 0)
    """
    v = version_str.lstrip("v")
    parts = v.split(".")
    try:
        return tuple(int(p) for p in parts[:3])
    except (ValueError, IndexError):
        return (0, 0, 0)


def compare_versions(current: str, latest: str) -> int:
    """
    比较两个版本号。

    Returns:
        -1: current < latest（有更新）
         0: current == latest（已是最新）
         1: current > latest（当前版本更新，测试场景）
    """
    c = parse_version(current)
    l = parse_version(latest)
    if c < l:
        return -1
    elif c > l:
        return 1
    return 0


def get_latest_release() -> Optional[ReleaseInfo]:
    """
    从 GitHub API 获取最新 release 信息。

    Returns:
        ReleaseInfo 或 None（网络错误时）
    """
    try:
        resp = requests.get(
            f"{GITHUB_API_BASE}/releases/latest",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return ReleaseInfo(
                tag_name=data.get("tag_name", ""),
                name=data.get("name", ""),
                body=data.get("body", ""),
                published_at=data.get("published_at", ""),
                html_url=data.get("html_url", ""),
            )
    except Exception as e:
        print(f"[Updater] Failed to check latest release: {e}")
    return None


def get_all_releases(limit: int = 10) -> List[ReleaseInfo]:
    """
    获取最近 N 个 release 的信息（用于显示更新日志）。

    Args:
        limit: 最多返回条数
    Returns:
        ReleaseInfo 列表
    """
    try:
        resp = requests.get(
            f"{GITHUB_API_BASE}/releases",
            params={"per_page": limit},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        if resp.status_code == 200:
            return [
                ReleaseInfo(
                    tag_name=r.get("tag_name", ""),
                    name=r.get("name", ""),
                    body=r.get("body", ""),
                    published_at=r.get("published_at", ""),
                    html_url=r.get("html_url", ""),
                )
                for r in resp.json()
            ]
    except Exception as e:
        print(f"[Updater] Failed to fetch releases: {e}")
    return []


def has_update() -> bool:
    """检查是否有新版本可用"""
    latest = get_latest_release()
    if latest is None:
        return False
    return compare_versions(APP_VERSION, latest.tag_name) < 0


def do_update() -> tuple:
    """
    执行 Docker 更新：pull 最新镜像并重建容器。

    Returns:
        (success: bool, message: str)
    """
    try:
        # 拉取最新镜像
        result = subprocess.run(
            ["docker", "compose", "pull"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return False, f"拉取镜像失败: {result.stderr.strip()}"

        # 重建并重启容器
        result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return False, f"重建容器失败: {result.stderr.strip()}"

        return True, "更新成功，容器已重启"

    except subprocess.TimeoutExpired:
        return False, "更新超时（5分钟），请手动执行 docker compose pull && docker compose up -d"
    except FileNotFoundError:
        return False, "未找到 docker 命令，请确认 Docker 已安装"
    except Exception as e:
        return False, f"更新失败: {e}"
