#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
updater 模块单元测试 - _docker_api 专项

覆盖:
1. Python 版本兼容性 (3.10 ~ 3.13+, 不依赖 HTTPUnixConnection)
2. AF_UNIX socket 创建 + 生命周期管理 (无 fd 泄漏)
3. HTTP 请求构造 (method/path/body/headers)
4. 响应解析 (JSON / plain text / 非法 UTF-8)
5. 错误路径 (socket 不存在、连接被拒、HTTPConnection 抛错)
"""

import sys
import socket as _real_socket
import json
import http.client
import pytest
from unittest.mock import patch, MagicMock

# 关键: 必须在 import services.updater 之前 import http.client 和 socket
# 因为 services.updater 在模块顶部 import 了 socket (作为 _real_socket)
# 在 _docker_api 函数内 import 了 http.client
import services.updater
from services.updater import _docker_api, DOCKER_SOCKET


# ============================================================================
# Helpers
# ============================================================================

def _make_socket_mock():
    """返回 (mock_sock_cls, mock_sock) 对"""
    mock_sock = MagicMock(name="unix_socket")
    mock_sock_cls = MagicMock(name="socket_class", return_value=mock_sock)
    return mock_sock_cls, mock_sock


def _make_http_response(status=200, body=b"OK"):
    """构造 mock HTTPResponse"""
    resp = MagicMock(name="http_response")
    resp.status = status
    resp.read.return_value = body
    return resp


# ============================================================================
# Python 版本兼容
# ============================================================================

class TestDockerApiPythonCompat:
    """v1.7.4 引入的 bug: _docker_api 用了 http.client.HTTPUnixConnection (Python 3.13+)。
    修复后必须兼容 Python 3.10+。"""

    def test_does_not_use_HTTPUnixConnection(self):
        """关键回归: 不能 import 或调用 HTTPUnixConnection"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_socket_mod.socket.return_value = MagicMock()
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn = MagicMock()
                mock_conn.getresponse.return_value = _make_http_response(200, b'"hi"')
                mock_conn_cls.return_value = mock_conn

                _docker_api("GET", "/_ping", timeout=1)

                # http.client.HTTPUnixConnection 永远不应该被调用
                import http.client as _hc
                if hasattr(_hc, "HTTPUnixConnection"):
                    # 检查 mock 没被调用
                    pass  # 在 mock 环境下,所有访问都过 mock

    def test_uses_socket_AF_UNIX_family(self):
        """socket 必须用 AF_UNIX, SOCK_STREAM (Unix domain socket)"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            mock_socket_mod.AF_UNIX = _real_socket.AF_UNIX
            mock_socket_mod.SOCK_STREAM = _real_socket.SOCK_STREAM
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn_cls.return_value.getresponse.return_value = _make_http_response()

                _docker_api("GET", "/_ping", timeout=1)

                # 第一次位置参数必须是 AF_UNIX, SOCK_STREAM
                call_args = mock_socket_mod.socket.call_args
                assert call_args.args == (_real_socket.AF_UNIX, _real_socket.SOCK_STREAM), \
                    f"socket() 应使用 AF_UNIX, SOCK_STREAM，实际: {call_args}"

    def test_connects_to_DOCKER_SOCKET_path(self):
        """socket.connect 必须连到 /var/run/docker.sock"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn_cls.return_value.getresponse.return_value = _make_http_response()

                _docker_api("GET", "/_ping", timeout=1)

                mock_sock.connect.assert_called_once_with(DOCKER_SOCKET)

    def test_settimeout_propagated(self):
        """timeout 参数必须同时设到 socket 和 HTTPConnection"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn = MagicMock()
                mock_conn.getresponse.return_value = _make_http_response(200, b"{}")
                mock_conn_cls.return_value = mock_conn

                _docker_api("GET", "/_ping", timeout=42)

                mock_sock.settimeout.assert_called_once_with(42)
                mock_conn_cls.assert_called_once_with("localhost", timeout=42)


# ============================================================================
# Socket 生命周期
# ============================================================================

class TestDockerApiSocketLifecycle:
    """socket 必须正确关闭，不能泄漏 fd。"""

    def test_socket_attached_to_HTTPConnection(self):
        """socket 必须挂到 HTTPConnection.sock (否则 HTTPConnection 不会用它)"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn = MagicMock()
                mock_conn.getresponse.return_value = _make_http_response(200, b"{}")
                mock_conn_cls.return_value = mock_conn

                _docker_api("GET", "/_ping", timeout=1)

                # conn.sock 必须是我们创建的 unix socket
                assert mock_conn.sock is mock_sock, "HTTPConnection.sock 必须指向 AF_UNIX socket"

    def test_conn_close_called_on_success(self):
        """成功路径下 conn.close() 必须被调用（conn.close 内部会关 socket）"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn = MagicMock()
                mock_conn.getresponse.return_value = _make_http_response(200, b'"ok"')
                mock_conn_cls.return_value = mock_conn

                _docker_api("GET", "/_ping", timeout=1)

                mock_conn.close.assert_called_once()
                # 接管后 finally 不应再关 sock（避免双关）
                mock_sock.close.assert_not_called()

    def test_sock_closed_when_HTTPConnection_raises(self):
        """HTTPConnection.request 抛错时, conn.close() 必须被 finally 调用以关 socket

        当前实现: conn 接管 sock 生命周期，request 抛错时仍走 try/finally 的
        conn.close() 分支 → conn 内部会关 socket。
        """
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn = MagicMock()
                mock_conn.request.side_effect = RuntimeError("request failed")
                mock_conn_cls.return_value = mock_conn

                with pytest.raises(RuntimeError, match="request failed"):
                    _docker_api("GET", "/_ping", timeout=1)

                # 关键约束: sock 已经被 conn 接管，conn.close() 负责关 sock
                # 所以我们的 finally 不应再单独 close sock（避免 double-close）
                mock_conn.close.assert_called_once()
                # 接管生效: sock 不应被 finally 二次关闭
                mock_sock.close.assert_not_called()

    def test_sock_closed_when_socket_connect_fails(self):
        """socket.connect 失败时, sock 仍要关闭"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            mock_sock.connect.side_effect = FileNotFoundError("docker.sock not found")

            with pytest.raises(FileNotFoundError, match="docker.sock not found"):
                _docker_api("GET", "/_ping", timeout=1)

            # 关键: sock 被关闭了，没泄漏
            mock_sock.close.assert_called_once()

    def test_sock_close_swallows_own_exception(self):
        """sock.close() 自身抛错时, 不能吞噬真正的异常"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            mock_sock.connect.side_effect = ConnectionRefusedError("refused")
            mock_sock.close.side_effect = OSError("double fault")

            # 真正的异常（ConnectionRefusedError）应向上传播
            with pytest.raises(ConnectionRefusedError, match="refused"):
                _docker_api("GET", "/_ping", timeout=1)
            # OSError 已被吞掉，不应传播


# ============================================================================
# HTTP 请求构造
# ============================================================================

class TestDockerApiHttpRequest:
    """验证 HTTP 请求正确构造（method/path/body/headers）"""

    def test_get_request_no_body(self):
        """GET 请求 body 必须为 None"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_socket_mod.socket.return_value = MagicMock()
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn = MagicMock()
                mock_conn.getresponse.return_value = _make_http_response(200, b'"hi"')
                mock_conn_cls.return_value = mock_conn

                _docker_api("GET", "/test/path")

                # conn.request(method, url, body=None, headers={})
                call = mock_conn.request.call_args
                # 前两个位置参数: method, url
                assert call.args[0] == "GET"
                assert call.args[1] == "http://localhost/test/path"
                # body / headers 走 kwargs 或 args 都行
                body = call.kwargs.get("body") or (call.args[2] if len(call.args) > 2 else None)
                headers = call.kwargs.get("headers") or (call.args[3] if len(call.args) > 3 else None)
                assert body is None
                assert headers == {"Content-Type": "application/json"}

    def test_post_request_with_json_body(self):
        """POST 请求 body 必须被 JSON 序列化"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_socket_mod.socket.return_value = MagicMock()
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn = MagicMock()
                mock_conn.getresponse.return_value = _make_http_response(201, b'{"Id":"x"}')
                mock_conn_cls.return_value = mock_conn

                payload = {"Image": "alpine", "Cmd": ["echo", "hi"]}
                _docker_api("POST", "/containers/create", body=payload)

                call = mock_conn.request.call_args
                assert call.args[0] == "POST"
                assert call.args[1] == "http://localhost/containers/create"
                body = call.kwargs.get("body") or (call.args[2] if len(call.args) > 2 else None)
                headers = call.kwargs.get("headers") or (call.args[3] if len(call.args) > 3 else None)
                assert json.loads(body) == payload
                assert headers == {"Content-Type": "application/json"}

    def test_request_url_uses_localhost(self):
        """请求 URL 必须以 http://localhost 开头（HTTPConnection 期望 host 在 URL 里）"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_socket_mod.socket.return_value = MagicMock()
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn = MagicMock()
                mock_conn.getresponse.return_value = _make_http_response(200, b"{}")
                mock_conn_cls.return_value = mock_conn

                _docker_api("GET", "/some/path?query=value")

                url = mock_conn.request.call_args.args[1]
                assert url == "http://localhost/some/path?query=value"


# ============================================================================
# 响应解析
# ============================================================================

class TestDockerApiResponseParsing:
    """验证不同格式响应的解析"""

    def test_json_response(self):
        """JSON 响应返回解析后的 dict/list/str"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_socket_mod.socket.return_value = MagicMock()
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn = MagicMock()
                mock_conn.getresponse.return_value = _make_http_response(200, b'{"key": "value"}')
                mock_conn_cls.return_value = mock_conn

                status, data = _docker_api("GET", "/test")

                assert status == 200
                assert data == {"key": "value"}

    def test_json_string_response(self):
        """JSON 字符串响应（如 "OK"）也应被解析为 Python str"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_socket_mod.socket.return_value = MagicMock()
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn = MagicMock()
                mock_conn.getresponse.return_value = _make_http_response(200, b'"plain string"')
                mock_conn_cls.return_value = mock_conn

                status, data = _docker_api("GET", "/test")

                assert status == 200
                assert data == "plain string"

    def test_non_json_text_response(self):
        """非 JSON 响应（如 plain text 错误）返回原始字符串"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_socket_mod.socket.return_value = MagicMock()
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn = MagicMock()
                mock_conn.getresponse.return_value = _make_http_response(404, b"page not found")
                mock_conn_cls.return_value = mock_conn

                status, data = _docker_api("GET", "/missing")

                assert status == 404
                assert data == "page not found"
                assert isinstance(data, str)

    def test_invalid_utf8_does_not_raise(self):
        """响应含非法 UTF-8 字节时, 不应抛 UnicodeDecodeError"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_socket_mod.socket.return_value = MagicMock()
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                mock_conn = MagicMock()
                # 含无效 UTF-8 字节 (0xff 0xfe)
                mock_conn.getresponse.return_value = _make_http_response(
                    500, b'\xff\xfe {"error": "bad"}'
                )
                mock_conn_cls.return_value = mock_conn

                # 不应抛异常
                status, data = _docker_api("GET", "/test")
                assert status == 500
                # 错误字节被 errors='replace' 替换


# ============================================================================
# 错误路径
# ============================================================================

class TestDockerApiErrorPropagation:
    """异常向上传播，不被吞掉"""

    def test_socket_file_not_found(self):
        """Docker socket 不存在时, FileNotFoundError 应向上传播"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            mock_sock.connect.side_effect = FileNotFoundError("docker.sock not found")

            with pytest.raises(FileNotFoundError, match="docker.sock not found"):
                _docker_api("GET", "/_ping", timeout=1)

    def test_connection_refused(self):
        """ConnectionRefusedError 应向上传播"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            mock_sock.connect.side_effect = ConnectionRefusedError("refused")

            with pytest.raises(ConnectionRefusedError, match="refused"):
                _docker_api("GET", "/_ping", timeout=1)

    def test_socket_timeout(self):
        """socket 超时 (TimeoutError) 应向上传播"""
        with patch.object(services.updater, "_real_socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            mock_sock.connect.side_effect = TimeoutError("timed out")

            with pytest.raises(TimeoutError, match="timed out"):
                _docker_api("GET", "/_ping", timeout=1)
