#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""services/subtitle_extractor.py 单元测试

覆盖：
- normalize_language_code: ISO-639-1 / ISO-639-2 / 全称 / 大小写 / 空值
- detect_subtitle_tracks: 解析 ffprobe 输出 / 多种流类型 / 异常路径
- extract_subtitle: 命令行构造 / 成功 / 失败 / 嵌入式文件命名
- has_embedded_subtitles: 软字幕判断
- select_best_subtitle_track: 非目标语言优先 / 软字幕过滤
- 模块级快捷函数: detect_subtitle_tracks / extract_subtitle / has_embedded_subtitles
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

from core.models import SubtitleTrack
from services.subtitle_extractor import (
    SubtitleExtractor,
    detect_subtitle_tracks,
    extract_subtitle,
    has_embedded_subtitles,
)


# ============================================================================
# normalize_language_code()
# ============================================================================

class TestNormalizeLanguageCode:
    """标准化 ffprobe 返回的各种语言代码"""

    @pytest.mark.parametrize("raw,expected", [
        # ISO 639-1
        ('zh', 'zh'),
        ('en', 'en'),
        ('ja', 'ja'),
        ('ko', 'ko'),
        # ISO 639-2 (ffprobe 常用)
        ('chi', 'zh'),
        ('zho', 'zh'),
        ('eng', 'en'),
        ('jpn', 'ja'),
        ('kor', 'ko'),
        ('fre', 'fr'),
        ('fra', 'fr'),
        ('ger', 'de'),
        ('deu', 'de'),
        ('rus', 'ru'),
        ('spa', 'es'),
        # 全称
        ('chinese', 'zh'),
        ('english', 'en'),
        ('japanese', 'ja'),
        ('korean', 'ko'),
        # 大小写
        ('ZH', 'zh'),
        ('ENG', 'en'),
        ('Chi', 'zh'),
        # 未知语言
        ('und', 'unknown'),
        ('unknown', 'unknown'),
        # 未知但 ASCII（保留原样）
        ('xyz', 'xyz'),
    ])
    def test_known_codes_mapped(self, raw, expected):
        assert SubtitleExtractor.normalize_language_code(raw) == expected

    @pytest.mark.parametrize("empty", ['', None])
    def test_empty_returns_unknown(self, empty):
        assert SubtitleExtractor.normalize_language_code(empty) == 'unknown'

    def test_non_ascii_kept_as_unknown(self):
        """非 ASCII 字符串（不是已知语言）→ unknown"""
        assert SubtitleExtractor.normalize_language_code('中文') == 'unknown'


# ============================================================================
# detect_subtitle_tracks()
# ============================================================================

def _make_ffprobe_result(streams):
    """构造 ffprobe JSON 输出"""
    return json.dumps({'streams': streams})


def _make_stream(index, codec, language=None, title=None, disposition_default=False):
    """构造一个流 dict"""
    stream = {
        'index': index,
        'codec_name': codec,
        'codec_type': 'subtitle',
    }
    tags = {}
    if language:
        tags['language'] = language
    if title:
        tags['title'] = title
    if tags:
        stream['tags'] = tags
    if disposition_default:
        stream['disposition'] = {'default': 1}
    return stream


class TestDetectSubtitleTracks:
    """检测视频字幕轨道（mock ffprobe subprocess.run）"""

    def _run_detect(self, ffprobe_stdout='', ffprobe_returncode=0):
        with patch('services.subtitle_extractor.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=ffprobe_returncode,
                stdout=ffprobe_stdout,
                stderr='',
            )
            return SubtitleExtractor.detect_subtitle_tracks('/fake/video.mkv'), mock_run

    def test_returns_empty_list_on_ffprobe_failure(self):
        tracks, mock_run = self._run_detect(ffprobe_returncode=1)
        assert tracks == []
        mock_run.assert_called_once()

    def test_returns_empty_list_on_invalid_json(self):
        tracks, _ = self._run_detect(ffprobe_stdout='not json')
        assert tracks == []

    def test_no_streams(self):
        stdout = _make_ffprobe_result([])
        tracks, _ = self._run_detect(ffprobe_stdout=stdout)
        assert tracks == []

    def test_single_srt_subtitle_zh(self):
        """一个中文 SRT 软字幕"""
        stdout = _make_ffprobe_result([
            _make_stream(2, 'srt', language='chi', title='简体中文'),
        ])
        tracks, _ = self._run_detect(ffprobe_stdout=stdout)
        assert len(tracks) == 1
        assert tracks[0].codec_name == 'srt'
        assert tracks[0].language == 'zh'  # chi -> zh
        assert tracks[0].title == '简体中文'
        assert tracks[0].is_soft_subtitle is True
        assert tracks[0].stream_index == 0  # enumerate index

    def test_multiple_subtitle_tracks(self):
        """多语言字幕"""
        stdout = _make_ffprobe_result([
            _make_stream(2, 'srt', language='chi'),
            _make_stream(3, 'ass', language='eng', title='English'),
            _make_stream(4, 'srt', language='jpn'),
        ])
        tracks, _ = self._run_detect(ffprobe_stdout=stdout)
        assert len(tracks) == 3
        langs = [t.language for t in tracks]
        assert langs == ['zh', 'en', 'ja']
        codecs = [t.codec_name for t in tracks]
        assert codecs == ['srt', 'ass', 'srt']
        is_soft = [t.is_soft_subtitle for t in tracks]
        assert is_soft == [True, True, True]

    def test_hard_subtitle_movtext_not_soft(self):
        """mov_text (硬字幕) 不算软字幕"""
        stdout = _make_ffprobe_result([
            _make_stream(2, 'mov_text', language='eng'),
        ])
        tracks, _ = self._run_detect(ffprobe_stdout=stdout)
        assert len(tracks) == 1
        assert tracks[0].is_soft_subtitle is False

    def test_unknown_language_fallback(self):
        """没有语言标签 → 'unknown'"""
        stdout = _make_ffprobe_result([
            _make_stream(2, 'srt'),  # 无 language
        ])
        tracks, _ = self._run_detect(ffprobe_stdout=stdout)
        assert tracks[0].language == 'unknown'

    def test_default_disposition_marked_as_default(self):
        """disposition.default=1 时语言标为 'default'"""
        stdout = _make_ffprobe_result([
            _make_stream(2, 'srt', disposition_default=True),
        ])
        tracks, _ = self._run_detect(ffprobe_stdout=stdout)
        # 'default' 不在 LANG_CODE_MAP → 走 fallback 保留原样
        assert tracks[0].language == 'default'

    def test_ffprobe_timeout_returns_empty(self):
        with patch('services.subtitle_extractor.subprocess.run',
                   side_effect=subprocess.TimeoutExpired(cmd='ffprobe', timeout=30)):
            tracks = SubtitleExtractor.detect_subtitle_tracks('/fake/video.mkv')
        assert tracks == []

    def test_ffprobe_command_constructs_correctly(self):
        """验证传给 ffprobe 的命令行参数"""
        with patch('services.subtitle_extractor.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='{"streams":[]}', stderr='')
            SubtitleExtractor.detect_subtitle_tracks('/tmp/movie.mkv')
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == 'ffprobe'
            assert '-v' in cmd and 'quiet' in cmd
            assert '-select_streams' in cmd and 's' in cmd
            assert cmd[-1] == '/tmp/movie.mkv'
            # timeout 必须是 30
            assert mock_run.call_args.kwargs.get('timeout') == 30

    def test_unexpected_exception_returns_empty(self):
        """捕获所有未知异常（防止 worker 主流程崩）"""
        with patch('services.subtitle_extractor.subprocess.run',
                   side_effect=RuntimeError('boom')):
            tracks = SubtitleExtractor.detect_subtitle_tracks('/fake/video.mkv')
        assert tracks == []


# ============================================================================
# extract_subtitle()
# ============================================================================

class TestExtractSubtitle:
    """提取指定轨道字幕（mock ffmpeg subprocess.run）"""

    def _run_extract(self, video_path, track_index, embedded=False,
                     ffmpeg_returncode=0, file_exists=True, **kwargs):
        with patch('services.subtitle_extractor.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=ffmpeg_returncode,
                stdout='',
                stderr='',
            )
            with patch.object(Path, 'exists', return_value=file_exists):
                result = SubtitleExtractor.extract_subtitle(
                    video_path, track_index, embedded=embedded, **kwargs
                )
            return result, mock_run

    def test_successful_extract_default_output_path(self, tmp_path):
        """默认输出路径在视频同目录 + srt 后缀"""
        video = tmp_path / 'movie.mkv'
        video.write_text('fake')
        result, mock_run = self._run_extract(str(video), track_index=0)
        # 文件名应为 movie.srt
        assert result == str(tmp_path / 'movie.srt')
        mock_run.assert_called_once()

    def test_embedded_subtitle_naming(self, tmp_path):
        """embedded=True 时文件名加 .embedded.srt"""
        video = tmp_path / 'movie.mkv'
        video.write_text('fake')
        result, _ = self._run_extract(str(video), track_index=0, embedded=True)
        assert result == str(tmp_path / 'movie.embedded.srt')

    def test_explicit_output_path_respected(self, tmp_path):
        """传 output_path 时优先使用"""
        video = tmp_path / 'movie.mkv'
        video.write_text('fake')
        custom = str(tmp_path / 'custom_path.srt')
        result, _ = self._run_extract(str(video), track_index=0, output_path=custom)
        assert result == custom

    def test_ffmpeg_failure_returns_none(self, tmp_path):
        video = tmp_path / 'movie.mkv'
        video.write_text('fake')
        result, _ = self._run_extract(str(video), track_index=0, ffmpeg_returncode=1)
        assert result is None

    def test_output_file_not_created_returns_none(self, tmp_path):
        """ffmpeg 成功但文件没创建（罕见）→ None"""
        video = tmp_path / 'movie.mkv'
        video.write_text('fake')
        result, _ = self._run_extract(str(video), track_index=0, file_exists=False)
        assert result is None

    def test_ffmpeg_command_constructs_correctly(self, tmp_path):
        """验证传给 ffmpeg 的命令"""
        video = tmp_path / 'movie.mkv'
        video.write_text('fake')
        _, mock_run = self._run_extract(str(video), track_index=2, output_format='ass')
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == 'ffmpeg'
        assert '-y' in cmd
        assert str(video) in cmd
        assert '0:s:2' in cmd  # track index
        assert 'ass' in cmd
        assert mock_run.call_args.kwargs.get('timeout') == 60

    def test_ffmpeg_timeout_returns_none(self, tmp_path):
        video = tmp_path / 'movie.mkv'
        video.write_text('fake')
        with patch('services.subtitle_extractor.subprocess.run',
                   side_effect=subprocess.TimeoutExpired(cmd='ffmpeg', timeout=60)):
            result = SubtitleExtractor.extract_subtitle(str(video), 0)
        assert result is None


# ============================================================================
# has_embedded_subtitles()
# ============================================================================

class TestHasEmbeddedSubtitles:
    """判断是否有软字幕"""

    def test_true_when_soft_subtitle_exists(self):
        with patch.object(
            SubtitleExtractor, 'detect_subtitle_tracks',
            return_value=[SubtitleTrack(stream_index=0, codec_name='srt',
                                         language='en', is_soft_subtitle=True)]
        ):
            assert SubtitleExtractor.has_embedded_subtitles('/fake.mkv') is True

    def test_false_when_no_tracks(self):
        with patch.object(SubtitleExtractor, 'detect_subtitle_tracks', return_value=[]):
            assert SubtitleExtractor.has_embedded_subtitles('/fake.mkv') is False

    def test_false_when_only_hard_subtitles(self):
        """只有 mov_text 等硬字幕 → False"""
        with patch.object(
            SubtitleExtractor, 'detect_subtitle_tracks',
            return_value=[SubtitleTrack(stream_index=0, codec_name='mov_text',
                                         language='en', is_soft_subtitle=False)]
        ):
            assert SubtitleExtractor.has_embedded_subtitles('/fake.mkv') is False

    def test_true_when_mixed(self):
        """硬字幕 + 软字幕 → True（因为 any()）"""
        tracks = [
            SubtitleTrack(stream_index=0, codec_name='mov_text',
                          language='en', is_soft_subtitle=False),
            SubtitleTrack(stream_index=1, codec_name='srt',
                          language='en', is_soft_subtitle=True),
        ]
        with patch.object(SubtitleExtractor, 'detect_subtitle_tracks', return_value=tracks):
            assert SubtitleExtractor.has_embedded_subtitles('/fake.mkv') is True


# ============================================================================
# select_best_subtitle_track()
# ============================================================================

class TestSelectBestSubtitleTrack:
    """选择最佳字幕轨道：优先非目标语言 + 必须是软字幕"""

    def test_empty_tracks_returns_none(self):
        assert SubtitleExtractor.select_best_subtitle_track([]) is None

    def test_only_target_lang_returns_first_soft(self):
        """只有目标语言 → 返回第一条软字幕（worker 仍可尝试）"""
        tracks = [
            SubtitleTrack(stream_index=0, codec_name='srt', language='zh', is_soft_subtitle=True),
        ]
        best = SubtitleExtractor.select_best_subtitle_track(tracks, target_language='zh')
        assert best is not None
        assert best.language == 'zh'

    def test_prefer_non_target_language(self):
        """目标语言和非目标语言同时存在 → 选非目标语言"""
        tracks = [
            SubtitleTrack(stream_index=0, codec_name='srt', language='zh', is_soft_subtitle=True),
            SubtitleTrack(stream_index=1, codec_name='ass', language='en', is_soft_subtitle=True),
        ]
        best = SubtitleExtractor.select_best_subtitle_track(tracks, target_language='zh')
        assert best.language == 'en'

    def test_hard_subtitle_excluded(self):
        """硬字幕被过滤"""
        tracks = [
            SubtitleTrack(stream_index=0, codec_name='mov_text',
                          language='en', is_soft_subtitle=False),
        ]
        best = SubtitleExtractor.select_best_subtitle_track(tracks)
        assert best is None

    def test_mixed_soft_and_hard(self):
        """混合时只用软字幕"""
        tracks = [
            SubtitleTrack(stream_index=0, codec_name='mov_text',
                          language='en', is_soft_subtitle=False),
            SubtitleTrack(stream_index=1, codec_name='srt',
                          language='ja', is_soft_subtitle=True),
        ]
        best = SubtitleExtractor.select_best_subtitle_track(tracks)
        assert best.language == 'ja'

    def test_default_target_lang_zh(self):
        """target_language 默认是 zh"""
        tracks = [
            SubtitleTrack(stream_index=0, codec_name='srt', language='en', is_soft_subtitle=True),
            SubtitleTrack(stream_index=1, codec_name='srt', language='ja', is_soft_subtitle=True),
        ]
        best = SubtitleExtractor.select_best_subtitle_track(tracks)
        # zh 是默认 target, 但列表里没有 → 返回第一条软字幕
        assert best.language == 'en'


# ============================================================================
# 模块级快捷函数
# ============================================================================

class TestModuleShortcuts:
    """测试模块顶层的快捷函数是否正确转发到 SubtitleExtractor"""

    def test_detect_subtitle_tracks_delegates(self):
        with patch.object(SubtitleExtractor, 'detect_subtitle_tracks',
                          return_value=['track']) as mock:
            result = detect_subtitle_tracks('/fake.mkv')
            assert result == ['track']
            mock.assert_called_once_with('/fake.mkv')

    def test_extract_subtitle_delegates(self):
        with patch.object(SubtitleExtractor, 'extract_subtitle',
                          return_value='/tmp/out.srt') as mock:
            result = extract_subtitle('/fake.mkv', 0)
            assert result == '/tmp/out.srt'
            mock.assert_called_once_with('/fake.mkv', 0, None, 'srt', False)

    def test_extract_subtitle_delegates_with_embedded(self):
        """模块级快捷函数必须支持 embedded 参数（bug 守卫）

        之前模块级 extract_subtitle 漏了 embedded 参数,导致外部调用方
        extract_subtitle(path, 0, embedded=True) 直接 TypeError。
        """
        with patch.object(SubtitleExtractor, 'extract_subtitle',
                          return_value='/tmp/out.srt') as mock:
            result = extract_subtitle('/fake.mkv', 0, embedded=True)
            assert result == '/tmp/out.srt'
            mock.assert_called_once_with('/fake.mkv', 0, None, 'srt', True)

    def test_has_embedded_subtitles_delegates(self):
        with patch.object(SubtitleExtractor, 'has_embedded_subtitles',
                          return_value=True) as mock:
            result = has_embedded_subtitles('/fake.mkv')
            assert result is True
            mock.assert_called_once_with('/fake.mkv')
