

<p align="center">
  <img src="./logo.png" alt="OpenClip" width="350"/>
</p>


[English](./README_EN.md) | 简体中文

一个轻量化自动化视频处理流水线，用于识别和提取长视频（特别是口播和直播回放）中最精彩的片段。使用 AI 驱动的分析来发现亮点，生成剪辑，并添加标题和封面。

## 🎯 功能介绍

输入一个视频 URL 或本地文件，自动完成 **下载 → 转录 → 分割 → AI 分析 → 剪辑生成 → 添加标题和封面** 的全流程处理，输出最精彩片段。适合从长直播或视频中快速提取高光时刻。

> 💡 **与 AutoClip 的区别？** 查看[对比说明](#-与-autoclip-的对比)了解 OpenClip 的轻量级设计理念。

## 📢 最新动态

- **2026-03-08**:
  - 新增 `--user-intent` 参数 — 用自然语言告诉 AI 你在找什么（如 `--user-intent "关于 AI 风险的观点"`），LLM 在片段筛选和排名时会优先考虑相关内容
- **2026-03-04**:
  - **Git 历史变更通知**：错误的减小 GitHub size 的尝试导致 Git 历史被重写，对现有用户造成不便，深感抱歉。已有克隆用户需运行 `git fetch origin && git reset --hard origin/main` 以同步最新历史
  - 新增[字幕烧录功能](#subtitle-burning) — 使用 `--burn-subtitles` 将 SRT 字幕直接烧录到剪辑视频中；可选 `--subtitle-translation "Simplified Chinese"` 同时烧录中英双语字幕（需要带 libass 的 ffmpeg）
  - OpenRouter 默认模型从 openrouter/free 切换至 stepfun/step-3.5-flash:free
- **2026-03-01**:
  - Streamlit 界面支持[后台任务处理和并发处理多个视频](#concurrent-processing)
  - 新增[说话人识别功能（预览版）](#speaker-identification)— 使用 `--speaker-references` 为访谈/座谈/播客视频自动标注说话人姓名
  - 优化 AI 提示词，减少时间戳格式混淆（如 `00:01:55` vs `01:55:00`）
- **2025-02-26**:
  - 默认 Qwen 模型从旧版 qwen-turbo 切换至 qwen3.5-flash
  - 优化 AI 提示词，减少时间戳幻觉，提升标题质量

## 🎬 演示

### 网页页面

![OpenClip 演示](demo/demo.gif)

### Agent Skills

<video src="https://github.com/user-attachments/assets/1ddf8318-f6ad-418c-9c4c-bbac0dedc668" controls width="600" height="450"></video>

## ✨ 特性
- **灵活输入**：支持 Bilibili、YouTube URL 或本地视频文件
- **智能转录**：优先使用平台字幕，回退到 Whisper
- **说话人识别**（预览版）：自动识别谁在说话，将真实姓名标注到字幕中，适合访谈、座谈、辩论和播客
- **AI 分析**：基于内容、互动和娱乐价值识别精彩时刻；支持 `--user-intent` 引导 AI 聚焦特定关注点
- **剪辑生成**：提取最精彩时刻为独立视频剪辑，自动生成字幕文件、标题和封面图片
- **字幕烧录**（可选）：将 SRT 字幕硬烧到视频画面中，可选通过 Qwen 翻译成目标语言后烧录双语字幕
- **背景上下文**：可选的添加背景信息（如主播姓名等）以获得更好的分析
- **三界面支持**：Streamlit 网页界面，Agent Skills 和命令行界面，满足不同用户需求
- **Agent Skills**：内置 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 和 [TRAE](https://www.trae.ai/) agent skill，用自然语言即可处理视频

## 📋 前置要求

### 手动安装

- **uv**（Python 包管理器）- [安装指南](https://docs.astral.sh/uv/getting-started/installation/)
- **FFmpeg** - 用于视频处理
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`
  - Windows: 从 [ffmpeg.org](https://ffmpeg.org) 下载

  <details>
  <summary>需要双语字幕烧录？点击查看带 libass 的安装方式</summary>

  默认安装通常不包含 libass：
  - macOS: `brew tap homebrew-ffmpeg/ffmpeg && brew install homebrew-ffmpeg/ffmpeg/ffmpeg`（替换已有的 ffmpeg）
  - Ubuntu: `sudo add-apt-repository ppa:savoury1/ffmpeg4 && sudo apt install ffmpeg`
  - Windows: 从 [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) 下载 **full** 版本
  </details>

- **LLM API Key**（选择其一）
  - **Qwen API Key** - 从[阿里云](https://dashscope.aliyun.com/)获取密钥（默认使用 qwen3.5-flash 模型）
  - **OpenRouter API Key** - 从[OpenRouter](https://openrouter.ai/)获取密钥（默认使用 stepfun/step-3.5-flash:free 模型）

- **Firefox 浏览器** (可选) - 使用浏览器 Cookie 让Bilibili 视频下载更稳定
- **HuggingFace Token** (可选，用于说话人识别) - 从 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 获取，并接受 [pyannote 模型协议](https://huggingface.co/pyannote/speaker-diarization-community-1)

### 由 uv 自动管理

运行 `uv sync` 时会自动安装以下依赖：

- **Python 3.11+** - 如果系统未安装，uv 会自动下载
- **yt-dlp** - 用于从 Bilibili、YouTube 等平台下载视频
- **Whisper** - 用于语音转文字
- 其他 Python 依赖（moviepy、streamlit 等）

## 🚀 快速开始

### 1. 克隆和设置

```bash
# 克隆仓库
git clone https://github.com/linzzzzzz/openclip.git
cd openclip

# 使用 uv 安装依赖
uv sync
```

### 2. 设置 API 密钥（用于 AI 功能）

**使用 Qwen：**
```bash
export QWEN_API_KEY=your_api_key_here
```

**使用 OpenRouter：**
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

### 3. 运行流水线

#### 选项 A：使用 Streamlit 网页界面

**启动 Streamlit 应用：**
```bash
uv run python -m streamlit run streamlit_app.py
```

应用启动后，打开浏览器访问显示的 URL（通常是 `http://localhost:8501`）。

**使用流程：**
1. 在侧边栏选择输入类型（视频 URL 或本地文件）
2. 配置处理选项（LLM 提供商等）
3. 点击「Process Video」按钮开始处理
4. 查看实时进度和最终结果
5. 在结果区域预览生成的剪辑和封面

**优势：** 无需记住命令行参数，提供可视化操作界面，适合所有用户。

<a id="concurrent-processing"></a>
<details>
<summary>🔄 并发处理与后台任务</summary>

Streamlit 界面支持后台任务处理和并发处理多个视频：

**后台任务处理：**
- 视频处理在后台运行，可以关闭浏览器
- 任务持久化保存，重新打开页面可继续查看
- 每个任务独立运行，互不干扰

**并发处理多个视频：**
- 点击「处理视频」启动第一个任务 → 自动跟踪进度
- **打开新标签页**启动第二个任务 → 在新标签页中独立跟踪
- 每个标签页可独立跟踪不同任务

**Watch Progress 功能：**
- 在任务卡片中点击「👁️ Watch Progress」按钮可切换跟踪的任务
- 显示「✓ Watching」表示当前正在跟踪该任务
- 实时查看进度更新和当前处理步骤

**任务管理：**
- 查看所有任务状态（处理中、已完成、失败）
- 取消运行中的任务
- 删除已完成或失败的任务
- 查看任务详情（创建时间、处理时长等）

</details>

#### 选项 B：使用 AI Agent 技能

如果你使用 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 或 [TRAE](https://www.trae.ai/)，可以直接用自然语言处理视频，无需手动输入命令：

```
"帮我从这个视频里提取精彩片段：https://www.bilibili.com/video/BV1234567890"
"处理一下 ~/Downloads/livestream.mp4，用英语作为输出语言"
```

Agent 会自动调用内置技能，完成下载、转录、分析、剪辑和标题添加等全部流程。

技能定义位于 `.claude/skills/` 和 `.trae/skills/` 目录下。

#### 选项 C：使用命令行界面

```bash
# 处理 Bilibili 视频
uv run python video_orchestrator.py "https://www.bilibili.com/video/BV1234567890"

# 处理 YouTube 视频
uv run python video_orchestrator.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 处理本地视频
uv run python video_orchestrator.py "/path/to/video.mp4"
```

> 如需使用已有字幕，请将 `.srt` 文件放在同目录下，文件名保持一致（如 `video.mp4` → `video.srt`）。

<a id="speaker-identification"></a>
<details>
<summary>🎙️ 说话人识别（可选，预览版）</summary>

> ⚠️ **预览功能**：说话人识别功能目前处于预览阶段，后续版本中行为或接口可能有所调整。

适用于访谈、座谈、辩论、播客等多人对话视频。启用后，字幕中每句话前会标注说话人姓名，例如 `[Host] 欢迎来到今天的节目`。这为 AI 的高光分析提供了更丰富的上下文——让它能更准确地识别特定说话人之间最精彩的交流片段，而非将所有语音一视同仁。

**步骤一：安装额外依赖**

```bash
uv sync --extra speakers
```

**步骤二：设置 HuggingFace Token**

```bash
export HUGGINGFACE_TOKEN=hf_your_token_here
```

并在 HuggingFace 上接受 [pyannote 模型协议](https://huggingface.co/pyannote/speaker-diarization-community-1)。

**步骤三：提取参考音频**

从视频中截取每位说话人的参考片段（10–30 秒，单人清晰语音）：

```bash
uv run python tools/extract_reference.py VIDEO 起始时间 结束时间 "references/姓名.wav"

# 示例
uv run python tools/extract_reference.py interview.mp4 00:01:23 00:01:50 "references/Host.wav"
uv run python tools/extract_reference.py interview.mp4 00:03:10 00:03:40 "references/Guest.wav"
```

**步骤四：运行**

```bash
uv run python video_orchestrator.py --speaker-references references/ "VIDEO_URL_OR_PATH"
```

</details>

<a id="subtitle-burning"></a>
<details>
<summary>🔤 字幕烧录（可选）</summary>

将 SRT 字幕文件硬烧到视频画面中（即使没有字幕播放器也能看到字幕）。支持原始字幕烧录，或通过 Qwen 翻译后同时烧录双语字幕。说话人标签（如 `[Sam Altman]`）会自动从画面中移除。

**前提：ffmpeg 需包含 libass**（详见上方安装说明）

**仅烧录原始字幕：**
```bash
uv run python video_orchestrator.py --burn-subtitles "VIDEO_URL"
```

**烧录原始 + 中文翻译字幕：**
```bash
uv run python video_orchestrator.py \
  --burn-subtitles \
  --subtitle-translation "Simplified Chinese" \
  "VIDEO_URL"
```

输出文件在 `clips_post_processed/` 目录，英文字幕显示在底部，中文字幕显示在英文上方。

</details>

## 📖 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `VIDEO_URL_OR_PATH` | 视频 URL 或本地文件路径（位置参数） | 必填 |
| `-o`, `--output` | 自定义输出目录 | `processed_videos` |
| `--llm-provider` | LLM 提供商（`qwen` 或 `openrouter`） | `qwen` |
| `--language` | 输出语言（`zh` 或 `en`） | `zh` |
| `--browser` | 用于 cookie 的浏览器（`chrome`/`firefox`/`edge`/`safari`） | `firefox` |
| `--force-whisper` | 强制使用 Whisper 转录（忽略平台字幕） | 关 |
| `--use-background` | 使用背景信息辅助分析 | 关 |
| `--user-intent` | 用自然语言描述关注重点（如 `"关于 AI 风险的观点"`），引导 AI 优先选取相关片段 | 无 |
| `--max-clips` | 最大精彩片段数量 | `5` |
| `--title-style` | Banner 标题艺术风格（见下方列表） | `fire_flame` |
| `--title-font-size` | 艺术标题字体大小（可选：small=30px, medium=40px, large=50px, xlarge=60px，默认：medium=40px） | `medium` |
| `--cover-text-location` | 封面文字位置（`top`/`upper_middle`/`bottom`/`center`） | `center` |
| `--cover-fill-color` | 封面文字填充颜色（`yellow`/`red`/`white`/`cyan`/`green`/`orange`/`pink`/`purple`/`gold`/`silver`） | `yellow` |
| `--cover-outline-color` | 封面文字描边颜色（`yellow`/`red`/`white`/`cyan`/`green`/`orange`/`pink`/`purple`/`gold`/`silver`/`black`） | `black` |
| `--speaker-references` | 参考音频目录，用于说话人姓名映射（预览版）。文件名即说话人姓名（如 `references/Host.wav`）。需要 `uv sync --extra speakers` 和 `HUGGINGFACE_TOKEN` | 无 |
| `--skip-transcript` | 跳过转录生成（使用已有转录文件） | 关 |
| `--skip-download` | 跳过下载，使用已下载的视频 | 关 |
| `--skip-analysis` | 跳过分析，使用已有分析结果 | 关 |
| `--skip-clips` | 不生成剪辑 | 关 |
| `--add-titles` | 添加艺术标题到剪辑 | 关 |
| `--skip-cover` | 不生成封面图片 | 关 |
| `--burn-subtitles` | 将 SRT 字幕烧录到视频中，输出到 `clips_post_processed/`（需要带 libass 的 ffmpeg） | 关 |
| `--subtitle-translation` | 翻译字幕到指定语言后烧录（例如 `"Simplified Chinese"`），需同时开启 `--burn-subtitles` | 无 |
| `-f`, `--filename` | 自定义输出文件名模板 | 无 |
| `-v`, `--verbose` | 开启详细日志 | 关 |
| `--debug` | 开启调试模式（导出完整 LLM 提示词） | 关 |

<details>
<summary>🎨 Banner 标题艺术风格</summary>

| 风格 | 效果 |
|------|------|
| `fire_flame` | 火焰效果（默认） |
| `gradient_3d` | 渐变3D效果 |
| `neon_glow` | 霓虹发光效果 |
| `metallic_gold` | 金属金色效果 |
| `rainbow_3d` | 彩虹3D效果 |
| `crystal_ice` | 水晶冰效果 |
| `metallic_silver` | 金属银色效果 |
| `glowing_plasma` | 发光等离子效果 |
| `stone_carved` | 石刻效果 |
| `glass_transparent` | 玻璃透明效果 |

</details>

## 🔍 命令行示例

**处理 Bilibili 视频，加载背景信息， 并使用霓虹风格处理Banner标题：**
```bash
uv run python video_orchestrator.py \
  --title-style neon_glow \
  --use-background \
  "https://www.bilibili.com/video/BV1wT6GBBEPp"
```

**仅分析，不生成剪辑：**
```bash
uv run python video_orchestrator.py --skip-clips --skip-cover "VIDEO_URL"
```

**说话人识别（预览版）：**
```bash
uv run python video_orchestrator.py \
  --speaker-references references/ \
  "interview.mp4"
```

**跳过下载，重新处理已有视频：**
```bash
uv run python video_orchestrator.py --skip-download --title-style crystal_ice "VIDEO_URL"
```

## 📁 输出结构

处理后，输出目录结构如下：

```
processed_videos/{video_name}/
├── downloads/                # 原始视频、字幕和元数据
├── splits/                   # 分割片段和 AI 分析结果
├── clips/                    # 生成的精彩剪辑、字幕、摘要和封面图
│   ├── rank_01_xxx.mp4
│   ├── rank_01_xxx.srt
│   ├── engaging_moments_summary.md
│   └── cover_rank_01_xxx.jpg
└── clips_post_processed/     # 后处理剪辑（--add-titles 和/或 --burn-subtitles）
    ├── rank_01_xxx.mp4
    └── ...
```

## 🎨 自定义

### 添加背景信息

创建或编辑 `prompts/background/background.md` 以提供关于主播、昵称或重复主题的上下文：

```markdown
# 背景信息

## 主播信息
- 主播：旭旭宝宝
- 昵称：宝哥
- 游戏：地下城与勇士（DNF）

## 常用术语
- 增幅：装备强化
- 鉴定：物品鉴定
```

然后使用 `--use-background` 标志：
```bash
uv run python video_orchestrator.py --use-background "VIDEO_URL"
```

### 自定义分析提示词

编辑 `prompts/` 中的提示词模板：
- `engaging_moments_part_requirement.md` - 每个片段的分析标准
- `engaging_moments_agg_requirement.md` - 顶级时刻的汇总标准

## 📎 其他

<details>
<summary>🔧 工作流程</summary>

```
输入（URL 或文件）
    ↓
下载/验证视频
    ↓
提取/生成转录
    ↓
检查时长 → 如果 >20分钟则分割
    ↓
AI 分析（每个片段）
    ↓
汇总前5个时刻
    ↓
生成剪辑
    ↓
后期处理（可选）
  ├── 添加艺术标题 (--add-titles)
  └── 烧录字幕 (--burn-subtitles [--subtitle-translation LANG])
    ↓
生成封面图片
    ↓
输出完成！
```

</details>

<details>
<summary>🐛 故障排除</summary>

### 下载失败
**原因**：
- yt-dlp版本过旧。尝试更新依赖版本：`uv sync`。
- Cookie/身份验证问题。尝试 `--browser firefox` 切换浏览器，或先在浏览器中登录 Bilibili。

### 未生成剪辑
**原因**：缺少 API 密钥或分析失败。检查 `echo $QWEN_API_KEY` 或 `echo $OPENROUTER_API_KEY`，并确认分析文件存在。

### FFmpeg 错误
**原因**：FFmpeg 未安装或不在 PATH 中。运行 `ffmpeg -version` 检查，缺失则安装（macOS: `brew install ffmpeg`）。

### 内存问题
**原因**：视频过长。尝试 `--max-duration 10` 缩短分割时长，或不使用 `--add-titles` 分阶段处理。

### 说话人识别不工作

**未找到 WhisperX**：运行 `uv sync --extra speakers` 安装额外依赖。

**HuggingFace Token 错误**：检查 `echo $HUGGINGFACE_TOKEN` 是否已设置，并确认已在 HuggingFace 上接受 [pyannote 模型协议](https://huggingface.co/pyannote/speaker-diarization-community-1)。

**说话人未被识别（显示 SPEAKER_XX 而非姓名）**：参考音频相似度低于阈值（默认 0.7）。尝试使用更长、更清晰的参考片段（推荐 10–30 秒），确保片段中只有一位说话人。

### 中文文本不显示
**原因**：缺少中文字体。macOS 自动检测（STHeiti、PingFang），Windows 需安装宋体或微软雅黑，Linux 安装 `fonts-wqy-zenhei`。

</details>

## 🔄 与 AutoClip 的对比

OpenClip 受 [AutoClip](https://github.com/zhouxiaoka/autoclip) 启发，但采用不同设计理念：

| 特性 | OpenClip | AutoClip |
|------|----------|----------|
| **代码规模** | ~5K 行 | ~2M 行 (含前端依赖) |
| **依赖** | Python + FFmpeg | Docker + Redis + PostgreSQL + Celery |
| **定制性** | 可编辑提示词模板 | 配置文件 |
| **界面** | Web界面+Agent Skills+命令行 | Web界面 |
| **部署** | `uv sync` 即用 | Docker容器化 |

**OpenClip 特点：** 轻量（5K行代码）、快速启动、提示词可定制、易于维护和二次开发

感谢 [AutoClip](https://github.com/zhouxiaoka/autoclip) 为视频自动化处理做出的贡献。

## 🤝 贡献

欢迎PRs！我们希望尽可能保持codebase的轻量化以及可读性：

**改进方向**
- 改进的 AI 分析提示词
- 性能优化
- 多模态分析
- 支持更多视频平台
- 额外的语言支持

## 📞 支持

如有问题或疑问：
1. 查看控制台输出中的错误消息
2. 先用短视频测试
3. 在 GitHub 上提出 issue
4. 加入我们的 [Discord 社区](https://discord.gg/KsC4Keaq) 讨论交流

## ⭐ 喜欢这个项目？

如果这个项目对你有帮助，欢迎在 GitHub 上给我们一个 Star！⭐

你的支持是我们持续改进的动力！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
