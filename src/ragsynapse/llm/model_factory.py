"""
Multi-model LLM factory — RAGSynapse v2
Supports OpenAI, Anthropic Claude, and local Ollama models.
Author: Zeeshan Ibrar
"""

import os
from enum import Enum
from dataclasses import dataclass
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.ollama import Ollama


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    provider: LLMProvider
    model: str
    temperature: float = 0.1
    max_tokens: int = 1024


# Default model for each provider
DEFAULT_MODELS = {
    LLMProvider.OPENAI:    "gpt-4o-mini",
    LLMProvider.ANTHROPIC: "claude-3-5-haiku-20241022",
    LLMProvider.OLLAMA:    "llama3.2",
}

# Cost per 1K tokens (for tracking in MLflow later)
COST_PER_1K_TOKENS = {
    "gpt-4o-mini":                   0.00015,
    "gpt-4o":                        0.005,
    "claude-3-5-haiku-20241022":     0.0008,
    "claude-3-5-sonnet-20241022":    0.003,
    "llama3.2":                      0.0,   # local = free
}


def get_llm(
    provider: str = None,
    model: str = None,
    temperature: float = 0.1,
):
    """
    Factory function — returns a LlamaIndex-compatible LLM.
    Reads LLM_PROVIDER from .env if provider not specified.
    
    Usage:
        llm = get_llm()                          # uses .env default
        llm = get_llm("anthropic")               # Claude Haiku
        llm = get_llm("openai", "gpt-4o")        # GPT-4o
        llm = get_llm("ollama", "mistral")       # local Mistral
    """
    # Read from env if not passed directly
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    
    try:
        p = LLMProvider(provider.lower())
    except ValueError:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {[e.value for e in LLMProvider]}"
        )

    model = model or DEFAULT_MODELS[p]

    if p == LLMProvider.OPENAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in .env")
        return OpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

    elif p == LLMProvider.ANTHROPIC:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")
        return Anthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

    elif p == LLMProvider.OLLAMA:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return Ollama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            request_timeout=180.0,
        )


def get_available_models() -> dict:
    """Returns which models are available based on set API keys."""
    available = {}
    
    if os.getenv("OPENAI_API_KEY"):
        available["openai"] = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    
    if os.getenv("ANTHROPIC_API_KEY"):
        available["anthropic"] = [
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
        ]
    
    # Ollama is always listed — it's local
    available["ollama"] = ["llama3.2", "mistral", "codellama"]
    
    return available