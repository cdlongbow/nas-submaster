#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/models.py 单元测试 — 数据模型序列化与属性"""

from core.models import (
    TaskStatus, ContentType, SubtitleSource,
    SubtitleInfo, SubtitleEntry, VADParameters, ProviderConfig,
    WhisperConfig, TranslationConfig, ExportConfig, PromptTemplate,
    SubtitleTrack, Task, MediaFile,
    SUPPORTED_VIDEO_EXTENSIONS, SUPPORTED_SUBTITLE_FORMATS,
    SUBTITLE_SOURCE_LABELS,
)


# ============================================================================
# SubtitleInfo
# ============================================================================

class TestSubtitleInfo:
    def test_round_trip(self):
        info = SubtitleInfo(path="/a.srt", lang="zh", source="embedded")
        d = info.to_dict()
        restored = SubtitleInfo.from_dict(d)
        assert restored.path == info.path
        assert restored.lang == info.lang
        assert restored.source == info.source

    def test_from_dict_defaults(self):
        info = SubtitleInfo.from_dict({"path": "/a.srt"})
        assert info.lang == ''
        assert info.source == 'embedded'

    def test_from_dict_tag_fallback(self):
        info = SubtitleInfo.from_dict({"path": "/a.srt", "tag": "chs"})
        assert info.lang == 'chs'

    def test_source_label(self):
        assert SubtitleInfo(path="", lang="", source="embedded").source_label == '内置'
        assert SubtitleInfo(path="", lang="", source="asr").source_label == 'AI提取'
        assert SubtitleInfo(path="", lang="", source="translated").source_label == '已翻译'
        assert SubtitleInfo(path="", lang="", source="unknown").source_label == '内置'

    def test_display_name(self):
        info = SubtitleInfo(path="", lang="zh", source="embedded")
        assert info.display_name == '内置：中文'

        info_en = SubtitleInfo(path="", lang="en", source="asr")
        assert info_en.display_name == 'AI提取：英文'

    def test_display_name_unknown_lang(self):
        info = SubtitleInfo(path="", lang="xx", source="embedded")
        assert '内置' in info.display_name


# ============================================================================
# SubtitleEntry
# ============================================================================

class TestSubtitleEntry:
    def test_round_trip(self):
        entry = SubtitleEntry(index="1", timecode="00:00:01,000 --> 00:00:03,000", text="Hello")
        d = entry.to_dict()
        restored = SubtitleEntry.from_dict(d)
        assert restored.index == "1"
        assert restored.text == "Hello"

    def test_from_dict_defaults(self):
        entry = SubtitleEntry.from_dict({})
        assert entry.index == "1"
        assert entry.text == ""


# ============================================================================
# VADParameters
# ============================================================================

class TestVADParameters:
    def test_to_dict(self):
        vad = VADParameters(threshold=0.5, min_speech_duration_ms=250,
                           min_silence_duration_ms=2000, speech_pad_ms=400)
        d = vad.to_dict()
        assert d['threshold'] == 0.5
        assert d['min_speech_duration_ms'] == 250


# ============================================================================
# ProviderConfig
# ============================================================================

class TestProviderConfig:
    def test_round_trip(self):
        cfg = ProviderConfig(api_key="sk-test", base_url="https://api.example.com", model_name="gpt-4")
        d = cfg.to_dict()
        restored = ProviderConfig.from_dict(d)
        assert restored.api_key == "sk-test"
        assert restored.base_url == "https://api.example.com"
        assert restored.model_name == "gpt-4"

    def test_from_dict_defaults(self):
        cfg = ProviderConfig.from_dict({})
        assert cfg.api_key == ''
        assert cfg.base_url == ''
        assert cfg.model_name == ''


# ============================================================================
# Task
# ============================================================================

class TestTask:
    def test_round_trip(self):
        task = Task(id=1, file_path="/a.mp4", status=TaskStatus.PENDING,
                   progress=0, log="准备中")
        d = task.to_dict()
        assert d['status'] == 'pending'
        restored = Task.from_dict(d)
        assert restored.status == TaskStatus.PENDING
        assert restored.id == 1

    def test_status_enum_from_string(self):
        task = Task.from_dict({'id': 1, 'file_path': '/a', 'status': 'completed'})
        assert task.status == TaskStatus.COMPLETED


# ============================================================================
# MediaFile
# ============================================================================

class TestMediaFile:
    def test_has_subtitle_empty(self):
        mf = MediaFile(id=1, file_path="/a.mp4", file_name="a.mp4", file_size=1000)
        assert mf.has_subtitle is False

    def test_has_subtitle_with_subs(self):
        sub = SubtitleInfo(path="/a.srt", lang="zh")
        mf = MediaFile(id=1, file_path="/a.mp4", file_name="a.mp4",
                      file_size=1000, subtitles=[sub])
        assert mf.has_subtitle is True

    def test_round_trip(self):
        sub = SubtitleInfo(path="/a.srt", lang="zh", source="asr")
        mf = MediaFile(id=1, file_path="/a.mp4", file_name="a.mp4",
                      file_size=1000, subtitles=[sub], has_translated=False)
        d = mf.to_dict()
        restored = MediaFile.from_dict(d)
        assert len(restored.subtitles) == 1
        assert restored.subtitles[0].lang == "zh"
        assert restored.has_subtitle is True

    def test_from_dict_json_string_subtitles(self):
        import json
        data = {
            'id': 1, 'file_path': '/a.mp4', 'file_name': 'a.mp4',
            'file_size': 1000,
            'subtitles': json.dumps([{"path": "/a.srt", "lang": "en", "source": "asr"}]),
            'has_translated': False
        }
        mf = MediaFile.from_dict(data)
        assert len(mf.subtitles) == 1
        assert mf.subtitles[0].lang == "en"


# ============================================================================
# ExportConfig
# ============================================================================

class TestExportConfig:
    def test_round_trip(self):
        cfg = ExportConfig(formats=['srt', 'vtt'])
        d = cfg.to_dict()
        restored = ExportConfig.from_dict(d)
        assert restored.formats == ['srt', 'vtt']

    def test_default_formats(self):
        cfg = ExportConfig()
        assert cfg.formats == ['srt']

    def test_from_dict_default(self):
        cfg = ExportConfig.from_dict({})
        assert cfg.formats == ['srt']


# ============================================================================
# PromptTemplate
# ============================================================================

class TestPromptTemplate:
    def test_round_trip(self):
        tpl = PromptTemplate(role="translator", rules="rule1\nrule2", style_guide="concise")
        d = tpl.to_dict()
        restored = PromptTemplate.from_dict(d)
        assert restored.role == "translator"
        assert restored.rules == "rule1\nrule2"


# ============================================================================
# SubtitleTrack
# ============================================================================

class TestSubtitleTrack:
    def test_display_name_with_language(self):
        track = SubtitleTrack(stream_index=2, codec_name="srt", language="en")
        assert 'English' in track.display_name
        assert 'SRT' in track.display_name

    def test_display_name_with_title(self):
        track = SubtitleTrack(stream_index=2, codec_name="ass", language="zh", title="自定义")
        assert '自定义' in track.display_name

    def test_display_name_unknown_language(self):
        track = SubtitleTrack(stream_index=2, codec_name="srt", language="xx")
        assert 'XX' in track.display_name

    def test_round_trip(self):
        track = SubtitleTrack(stream_index=2, codec_name="srt", language="en", title="subs")
        d = track.to_dict()
        restored = SubtitleTrack.from_dict(d)
        assert restored.stream_index == 2
        assert restored.language == "en"


# ============================================================================
# Constants
# ============================================================================

class TestConstants:
    def test_supported_video_extensions(self):
        assert '.mp4' in SUPPORTED_VIDEO_EXTENSIONS
        assert '.mkv' in SUPPORTED_VIDEO_EXTENSIONS
        assert '.ts' in SUPPORTED_VIDEO_EXTENSIONS

    def test_supported_subtitle_formats(self):
        assert 'srt' in SUPPORTED_SUBTITLE_FORMATS
        assert 'vtt' in SUPPORTED_SUBTITLE_FORMATS
        assert 'ass' in SUPPORTED_SUBTITLE_FORMATS

    def test_subtitle_source_labels(self):
        assert SUBTITLE_SOURCE_LABELS['embedded'] == '内置'
        assert SUBTITLE_SOURCE_LABELS['asr'] == 'AI提取'
        assert SUBTITLE_SOURCE_LABELS['translated'] == '已翻译'
