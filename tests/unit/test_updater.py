#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
updater 模块单元测试
"""

import pytest
from unittest.mock import patch, MagicMock
from services.updater import (
    parse_version,
    compare_versions,
    get_latest_release,
    get_all_releases,
    has_update,
    do_update,
    ReleaseInfo,
)


# ============================================================================
# parse_version
# ============================================================================

class TestParseVersion:
    def test_standard_version(self):
        assert parse_version("v1.7.0") == (1, 7, 0)

    def test_without_v_prefix(self):
        assert parse_version("2.3.1") == (2, 3, 1)

    def test_single_digit(self):
        assert parse_version("v1") == (1,)

    def test_two_parts(self):
        assert parse_version("v1.7") == (1, 7)

    def test_invalid_version(self):
        assert parse_version("abc") == (0, 0, 0)

    def test_empty_string(self):
        assert parse_version("") == (0, 0, 0)

    def test_large_numbers(self):
        assert parse_version("v10.20.30") == (10, 20, 30)


# ============================================================================
# compare_versions
# ============================================================================

class TestCompareVersions:
    def test_equal(self):
        assert compare_versions("v1.6.0", "v1.6.0") == 0

    def test_current_older(self):
        assert compare_versions("v1.5.0", "v1.6.0") == -1

    def test_current_newer(self):
        assert compare_versions("v1.7.0", "v1.6.0") == 1

    def test_patch_difference(self):
        assert compare_versions("v1.6.0", "v1.6.1") == -1

    def test_major_difference(self):
        assert compare_versions("v1.0.0", "v2.0.0") == -1

    def test_with_and_without_v(self):
        assert compare_versions("v1.6.0", "1.6.0") == 0

    def test_invalid_versions(self):
        assert compare_versions("abc", "xyz") == 0


# ============================================================================
# ReleaseInfo
# ============================================================================

class TestReleaseInfo:
    def test_creation(self):
        info = ReleaseInfo(
            tag_name="v1.7.0",
            name="v1.7.0 - New Feature",
            body="## What's New\n- Feature A",
            published_at="2026-06-05T12:00:00Z",
            html_url="https://github.com/test/test/releases/tag/v1.7.0"
        )
        assert info.tag_name == "v1.7.0"
        assert "Feature A" in info.body


# ============================================================================
# get_latest_release (mock GitHub API)
# ============================================================================

class TestGetLatestRelease:
    @patch("services.updater.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "tag_name": "v1.7.0",
            "name": "v1.7.0",
            "body": "New release",
            "published_at": "2026-06-05",
            "html_url": "https://github.com/test"
        }
        mock_get.return_value = mock_resp

        result = get_latest_release()
        assert result is not None
        assert result.tag_name == "v1.7.0"

    @patch("services.updater.requests.get")
    def test_api_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = get_latest_release()
        assert result is None

    @patch("services.updater.requests.get")
    def test_network_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = get_latest_release()
        assert result is None


# ============================================================================
# get_all_releases (mock GitHub API)
# ============================================================================

class TestGetAllReleases:
    @patch("services.updater.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "tag_name": "v1.7.0",
                "name": "v1.7.0",
                "body": "Release 1.7",
                "published_at": "2026-06-05",
                "html_url": "https://github.com/test/1.7"
            },
            {
                "tag_name": "v1.6.0",
                "name": "v1.6.0",
                "body": "Release 1.6",
                "published_at": "2026-05-01",
                "html_url": "https://github.com/test/1.6"
            }
        ]
        mock_get.return_value = mock_resp

        result = get_all_releases(limit=2)
        assert len(result) == 2
        assert result[0].tag_name == "v1.7.0"

    @patch("services.updater.requests.get")
    def test_network_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = get_all_releases()
        assert result == []


# ============================================================================
# has_update
# ============================================================================

class TestHasUpdate:
    @patch("services.updater.get_latest_release")
    def test_has_update(self, mock_latest):
        mock_latest.return_value = ReleaseInfo(
            tag_name="v99.0.0",
            name="Future",
            body="",
            published_at="",
            html_url=""
        )
        assert has_update() is True

    @patch("services.updater.get_latest_release")
    def test_no_update(self, mock_latest):
        mock_latest.return_value = ReleaseInfo(
            tag_name="v0.0.1",
            name="Old",
            body="",
            published_at="",
            html_url=""
        )
        assert has_update() is False

    @patch("services.updater.get_latest_release")
    def test_api_failure(self, mock_latest):
        mock_latest.return_value = None
        assert has_update() is False


# ============================================================================
# do_update
# ============================================================================

class TestDoUpdate:
    @patch("services.updater.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        ok, msg = do_update()
        assert ok is True
        assert "成功" in msg

    @patch("services.updater.subprocess.run")
    def test_pull_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="pull failed")
        ok, msg = do_update()
        assert ok is False
        assert "拉取镜像失败" in msg

    @patch("services.updater.subprocess.run")
    def test_up_failure(self, mock_run):
        # pull succeeds, up fails
        mock_run.side_effect = [
            MagicMock(returncode=0, stderr=""),
            MagicMock(returncode=1, stderr="up failed")
        ]
        ok, msg = do_update()
        assert ok is False
        assert "重建容器失败" in msg

    @patch("services.updater.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=300)
        ok, msg = do_update()
        assert ok is False
        assert "超时" in msg

    @patch("services.updater.subprocess.run")
    def test_docker_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        ok, msg = do_update()
        assert ok is False
        assert "docker" in msg.lower()
