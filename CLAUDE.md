# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NAS SubMaster (NAS 字幕管家) is a video subtitle extraction and translation tool for home NAS users. It uses Faster-Whisper for speech recognition and LLMs for translation, with a Streamlit web UI.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (development)
streamlit run app.py --server.port 8501

# Build and run with Docker
docker compose up -d --build

# View logs
docker logs -f nas-subtitle
```

## Architecture

### Layer Separation
- **`app.py`**: Streamlit entry point, page routing, UI layout
- **`core/`**: Core business logic (worker, config, models)
- **`services/`**: Service layer - business operations
- **`ui/`**: Streamlit UI components and pages
- **`database/`**: SQLite data access layer
- **`utils/`**: Utility functions

### Task Processing Pipeline (`core/worker.py`)
1. Worker runs in a background daemon thread, polling for pending tasks
2. Whisper model is cached and reused across tasks (reloaded only on config change)
3. Pipeline: Extract SRT → Translate (optional) → Export formats → Update media library → Mark complete
4. Cancellation supported via threading.Event and database status check

### Database Schema (SQLite at `/data/subtitle_manager.db`)
- **`media_files`**: file_path, file_name, file_size, subtitles_json, has_translated
- **`tasks`**: file_path, status (pending/processing/completed/failed/cancelled), progress, log, log_history
- **`config`**: key-value store for app settings

### Key Services
- **`WhisperService`** (`services/whisper_service.py`): Loads Faster-Whisper models, extracts SRT from video
- **`Translator`** (`services/translator.py`): LLM-based subtitle translation with anti-repetition logic
- **`MediaScanner`** (`services/media_scanner.py`): Directory scanning, video/subtitle discovery, 3-level subdirectory navigation
- **`SubtitleConverter`** (`services/subtitle_converter.py`): SRT ↔ VTT/ASS/SSA format conversion

### Configuration (`core/config.py`)
- Stored in SQLite `config` table
- Cached in `ConfigManager._last_saved_config_dict` to avoid redundant writes
- VAD parameters vary by `ContentType` (movie, documentary, variety, animation, lecture, music, custom)

### UI Structure (`ui/`)
- `pages/media_library.py`: Main page with directory selector, file list, batch task submission
- `pages/task_queue.py`: Task monitoring with real-time logs
- `settings_modal.py`: System configuration dialog

## Data Flow
1. User selects subdirectory → `MediaScanner.discover_media_subdirectories()` loads 3-level tree
2. User clicks "扫描" → `MediaScanner.scan_media_directory()` finds videos, stores in `media_files`
3. User selects files and clicks "添加到队列" → Tasks created in `tasks` table
4. Worker picks up task → runs Whisper extraction → optional translation → saves SRT alongside video
5. `rescan_video_subtitles()` called after completion to update `media_files` with new subtitle info
