---
name: "video_highlights_processor"
description: "Processes videos to identify engaging moments, generate transcripts, and create highlight clips with artistic titles and custom cover images. Use when user needs to: extract highlights from long videos or livestreams, clip or cut best moments from videos, process Bilibili/YouTube URLs or local video files, generate transcripts via Whisper, analyze content for engaging moments, create short-form clips with styled titles and covers, adjust cover text position and colors, find and export memorable scenes from recordings, burn subtitles into clips (with optional translation), guide clip selection with user intent, or identify speakers in multi-person conversations."
---

# Video Highlights Processor Skill

Run the video orchestrator to process videos and extract engaging highlights.

## Execution

Run from the project root using `uv` (the project uses `uv` with `pyproject.toml` and `uv.lock`):

```bash
uv run python video_orchestrator.py [options] <source>
```

Where `<source>` is a video URL (Bilibili/YouTube) or local file path (MP4, WebM, AVI, MOV, MKV).

For local files with existing subtitles, place the `.srt` file in the same directory with the same filename (e.g. `video.mp4` → `video.srt`).

## Preflight Checklist

- Run from repository root so relative paths (for example `references/`) resolve correctly
- Set one API key:
  - `QWEN_API_KEY` (default provider: qwen), or
  - `OPENROUTER_API_KEY` (if `--llm-provider openrouter`)
- If using `--speaker-references`: run `uv sync --extra speakers` and set `HUGGINGFACE_TOKEN`
- If using `--burn-subtitles`: ensure `ffmpeg` is installed with `libass`

## CLI Reference

### Required

| Argument | Description |
|---|---|
| `source` | Video URL or local file path |

### Optional

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output <dir>` | `processed_videos` | Output directory |
| `--max-clips <n>` | `5` | Maximum number of highlight clips |
| `--browser <browser>` | `firefox` | Browser for cookies: `chrome`, `firefox`, `edge`, `safari` |
| `--title-style <style>` | `fire_flame` | Title style: `gradient_3d`, `neon_glow`, `metallic_gold`, `rainbow_3d`, `crystal_ice`, `fire_flame`, `metallic_silver`, `glowing_plasma`, `stone_carved`, `glass_transparent` |
| `--title-font-size <size>` | `medium` | Font size preset for artistic titles. Options: small(30px), medium(40px), large(50px), xlarge(60px) |
| `--cover-text-location <loc>` | `center` | Cover text position: `top`, `upper_middle`, `bottom`, `center` |
| `--cover-fill-color <color>` | `yellow` | Cover text fill color: `yellow`, `red`, `white`, `cyan`, `green`, `orange`, `pink`, `purple`, `gold`, `silver` |
| `--cover-outline-color <color>` | `black` | Cover text outline color: `yellow`, `red`, `white`, `cyan`, `green`, `orange`, `pink`, `purple`, `gold`, `silver`, `black` |
| `--language <lang>` | `zh` | Output language: `zh` (Chinese), `en` (English) |
| `--llm-provider <provider>` | `qwen` | LLM provider: `qwen`, `openrouter` |
| `--user-intent <text>` | — | Free-text focus description (e.g. "moments about AI risks"). Steers LLM clip selection toward this topic |
| `--subtitle-translation <lang>` | — | Translate subtitles to this language before burning (e.g. `"Simplified Chinese"`). Requires `--burn-subtitles` and `QWEN_API_KEY` |
| `--speaker-references <dir>` | — | Directory of reference WAV files (one per speaker, filename = speaker name) for speaker diarization. Requires `uv sync --extra speakers` and `HUGGINGFACE_TOKEN` |
| `-f`, `--filename <template>` | — | yt-dlp template: `%(title)s`, `%(uploader)s`, `%(id)s`, etc. |

### Flags

| Flag | Description |
|---|---|
| `--force-whisper` | Ignore platform subtitles, use Whisper |
| `--skip-download` | Use existing downloaded video |
| `--skip-transcript` | Skip transcript generation, use existing transcript file |
| `--skip-analysis` | Skip analysis, use existing analysis file for clip generation |
| `--use-background` | Include background info (streamer names/nicknames) in analysis prompts |
| `--skip-clips` | Skip clip generation |
| `--add-titles` | Add artistic titles to clips (disabled by default) |
| `--skip-cover` | Skip cover image generation |
| `--burn-subtitles` | Burn SRT subtitles into video. Output goes to `clips_post_processed/`. Requires ffmpeg with libass |
| `-v`, `--verbose` | Enable verbose logging |
| `--debug` | Export full prompts sent to LLM (saved to `debug_prompts/`) |

### Custom Filename Template (`-f`)

Uses yt-dlp template syntax. Common variables: `%(title)s`, `%(uploader)s`, `%(upload_date)s`, `%(id)s`, `%(ext)s`, `%(duration)s`.

Example: `-f "%(upload_date)s_%(title)s.%(ext)s"`

### Environment Variables

Set the appropriate API key for the chosen `--llm-provider`:

- `QWEN_API_KEY` — for `--llm-provider qwen`
- `OPENROUTER_API_KEY` — for `--llm-provider openrouter`

## Workflow

The orchestrator runs this pipeline automatically:

1. **Download** video and platform subtitles (Bilibili/YouTube) or accept local file
2. **Split** videos longer than the built-in duration threshold into segments
3. **Transcribe** using platform subtitles or Whisper AI (fallback or `--force-whisper`)
4. **Analyze** transcript for engaging moments via LLM
5. **Generate clips** from identified moments
6. **Add artistic titles** to clips using `--title-style`
7. **Generate cover images** for each highlight

Use `--skip-clips`, `--skip-cover` to skip specific steps. Use `--add-titles` to enable artistic titles. Use `--skip-download` and `--skip-analysis` to resume from intermediate results.

## Output Structure

```
processed_videos/{video_name}/
├── downloads/              # Original video, subtitles, and metadata
├── splits/                 # Split parts and AI analysis results
├── clips/                  # Generated highlight clips + generated cover images
└── clips_post_processed/   # Post-processed clips when using --add-titles and/or --burn-subtitles
```

## Option Selection Guide

**Whisper model** — Default `base` works for clear audio. Use `small` for background noise, multiple speakers, or accents. Use `turbo` for speed + accuracy. Use `large`/`medium` only when transcript quality is critical.

**`--force-whisper`** — Use when platform subtitles are auto-generated (often inaccurate), when "no engaging moments found" occurs (better transcripts improve analysis), or for non-native language content where platform captions are unreliable.

**`--use-background`** — Use for content featuring recurring personalities (streamers, hosts) where nicknames and community references matter. Reads from `prompts/background/background.md`.

**Multi-part analysis** — Videos that get split are analyzed per-segment, then aggregated to the top 5 engaging moments across all segments.

**`--user-intent`** — Steers LLM clip selection at both the per-segment and cross-segment aggregation stages. Useful when you want to find clips about a specific topic (e.g. "AI safety predictions", "funny moments").

**`--burn-subtitles`** — Hardcodes the SRT subtitle into the video frame. Use when you want subtitles always visible (e.g. for social media). Combine with `--subtitle-translation` to add a translated subtitle track below the original.

**`--speaker-references`** — Enables speaker diarization for interviews/podcasts. Provide a directory of 10–30 second clean WAV clips (one per speaker), named after the speaker (e.g. `references/Host.wav`).

## Troubleshooting

| Error | Fix |
|---|---|
| "No API key provided" | Set `QWEN_API_KEY` or `OPENROUTER_API_KEY` env var |
| "Video download failed" | Check network/URL; try different `--browser`; or use local file |
| "Transcript generation failed" | Try `--force-whisper` or check audio quality |
| "No engaging moments found" | Try `--force-whisper` for better transcript accuracy |
| "Clip generation failed" | Ensure analysis completed; check for existing analysis file |
