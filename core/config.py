"""
Configuration file for LLM clients and other components
"""

import os
from typing import Dict, Any


def _normalize_chat_completions_url(url: str) -> str:
    """Accept either a full endpoint or an OpenAI-compatible API root."""
    normalized = url.rstrip("/")
    if normalized.endswith("/chat/completions") or normalized.endswith("/generation"):
        return normalized
    if normalized.endswith("/v1") or normalized.endswith("/v4") or normalized.endswith("/compatible-mode/v1"):
        return f"{normalized}/chat/completions"
    return normalized


def _env_llm_base_url(provider: str, default: str) -> str:
    env_value = os.getenv(f"{provider.upper()}_BASE_URL")
    if not env_value:
        return default
    return _normalize_chat_completions_url(env_value)


def _env_llm_model(provider: str, default: str) -> str:
    return os.getenv(f"{provider.upper()}_MODEL", default)


# LLM Client configurations
LLM_CONFIG: Dict[str, Dict[str, Any]] = {
    "qwen": {
        # New models (qwen3.5, qwen3, etc.) use OpenAI-compatible endpoint
        "base_url": _env_llm_base_url("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        # Old models (qwen-turbo, qwen-plus, qwen-max) use DashScope endpoint
        "legacy_base_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        "default_model": _env_llm_model("qwen", "qwen3.5-flash"),
        "default_params": {
            "max_tokens": 16384,
            "temperature": 0.7,
            "top_p": 0.8,
            "stream": False
        },
        # Models that use the legacy endpoint
        "legacy_models": ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-long"]
    },
    "openrouter": {
        "base_url": _env_llm_base_url("openrouter", "https://openrouter.ai/api/v1/chat/completions"),
        "default_model": _env_llm_model("openrouter", "stepfun/step-3.5-flash:free"),
        "default_params": {
            "max_tokens": 32768,
            "temperature": 0.7,
            "top_p": 0.8,
            "stream": False
        }
    },
    "glm": {
        "base_url": _env_llm_base_url("glm", "https://open.bigmodel.cn/api/paas/v4/chat/completions"),
        "default_model": _env_llm_model("glm", "glm-4.7"),
        "default_params": {
            "max_tokens": 32768,
            "temperature": 0.7,
            "top_p": 0.8,
            "stream": False
        }
    },
    "minimax": {
        "base_url": _env_llm_base_url("minimax", "https://api.minimaxi.com/v1/chat/completions"),
        "default_model": _env_llm_model("minimax", "MiniMax-M2.7"),
        "default_params": {
            "max_tokens": 32768,
            "temperature": 0.7,
            "top_p": 0.8,
            "stream": False
        }
    }
}


# Environment variable names for API keys
API_KEY_ENV_VARS: Dict[str, str] = {
    "qwen": "QWEN_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "glm": "GLM_API_KEY",
    "minimax": "MINIMAX_API_KEY"
}


# Default LLM provider
DEFAULT_LLM_PROVIDER: str = "qwen"

# Video splitting
MAX_DURATION_MINUTES: float = 20.0

# Whisper model for transcript generation
# Options: tiny, base, small, medium, large, turbo
WHISPER_MODEL: str = "base"

# Lightweight Whisper model used only for transcript language detection.
# Keep this small because the full transcript backend is selected after detection.
TRANSCRIPT_LANGUAGE_DETECT_MODEL: str = os.getenv("TRANSCRIPT_LANGUAGE_DETECT_MODEL", "tiny")

# Local FunASR Paraformer project used for Chinese ASR.
PARAFORMER_PROJECT_DIR: str = os.getenv(
    "PARAFORMER_PROJECT_DIR",
    "/home/tangsipeng/mycode/funasr-paraformer",
)
PARAFORMER_DEVICE: str = os.getenv("PARAFORMER_DEVICE", "auto")

# Title style for artistic text overlay
# Options: gradient_3d, neon_glow, metallic_gold, rainbow_3d, crystal_ice,
#          fire_flame, metallic_silver, glowing_plasma, stone_carved, glass_transparent
DEFAULT_TITLE_STYLE: str = "fire_flame"

# Maximum number of highlight clips to generate
MAX_CLIPS: int = 5

# Skip download by default (use existing files if available)
SKIP_DOWNLOAD: bool = False

# Skip transcript generation (use existing transcript files if available)
SKIP_TRANSCRIPT: bool = False
