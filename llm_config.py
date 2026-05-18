"""
LLM Configuration Module
=========================
Provides a single `get_llm()` factory function that returns a LangChain
ChatModel configured from environment variables.

Supported providers (all free-tier):
  • gemini  — Google Generative AI  (GOOGLE_API_KEY)
  • groq    — Groq Cloud            (GROQ_API_KEY)
  • ollama  — Local Ollama server    (no key needed)

Usage:
    from llm_config import get_llm
    llm = get_llm()
    response = llm.invoke("Hello!")
"""

import os
import itertools
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Load .env from project root
load_dotenv(override=True)

# ── Global API Key Cycles ──────────────────────────────────────────────
_groq_key_cycle = None
_google_key_cycle = None

def _get_groq_key():
    """Fetches the next Groq API key using a Round-Robin cycle."""
    global _groq_key_cycle
    if _groq_key_cycle is None:
        keys = []
        # Dynamically scan GROQ_API_KEY_1 through GROQ_API_KEY_20
        for i in range(1, 21):
            k = os.getenv(f"GROQ_API_KEY_{i}")
            if k: keys.append(k)
            
        # Also check the default GROQ_API_KEY
        default_k = os.getenv("GROQ_API_KEY")
        if default_k: keys.append(default_k)
            
        if not keys:
            raise ValueError(
                "No GROQ_API_KEY found. Add GROQ_API_KEY or GROQ_API_KEY_1 to your .env file."
            )
            
        # Remove duplicates while preserving order
        keys = list(dict.fromkeys(keys))
        _groq_key_cycle = itertools.cycle(keys)
        print(f"[Groq] Initialized Round-Robin with {len(keys)} API keys.")
        
    return next(_groq_key_cycle)


def _get_google_key():
    """Fetches the next Google API key using a Round-Robin cycle."""
    global _google_key_cycle
    if _google_key_cycle is None:
        keys = []
        # Dynamically scan GOOGLE_API_KEY_1 through GOOGLE_API_KEY_20
        for i in range(1, 21):
            k = os.getenv(f"GOOGLE_API_KEY_{i}")
            if k: keys.append(k)

        # Also check the default GOOGLE_API_KEY / GEMINI_API_KEY
        for env_name in ["GOOGLE_API_KEY", "GEMINI_API_KEY"]:
            k = os.getenv(env_name)
            if k: keys.append(k)

        if not keys:
            raise ValueError(
                "No GOOGLE_API_KEY found. Add GOOGLE_API_KEY_1 to your .env file.\n"
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )

        # Remove duplicates while preserving order
        keys = list(dict.fromkeys(keys))
        _google_key_cycle = itertools.cycle(keys)
        print(f"[Google] Initialized Round-Robin with {len(keys)} API keys.")

    return next(_google_key_cycle)


# ── Default model names per provider ────────────────────────────────────
_DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:7b-instruct-q4_K_M",
}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs
) -> BaseChatModel:
    """
    Factory function to create a LangChain ChatModel.

    Parameters
    ----------
    provider : str, optional
        One of "gemini", "groq", "ollama".
        Falls back to the LLM_PROVIDER env var.
    model : str, optional
        Model name override. Falls back to sensible defaults per provider.
    temperature : float
        Sampling temperature. Default 0.0 for deterministic output.

    Returns
    -------
    BaseChatModel
        A ready-to-use LangChain chat model instance.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower().strip()

    # ── Google Gemini ───────────────────────────────────────────────────
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = _get_google_key()

        model_name = model or _DEFAULT_MODELS["gemini"]
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True,
            **kwargs
        )

    # ── Groq ────────────────────────────────────────────────────────────
    elif provider == "groq":
        from langchain_groq import ChatGroq

        api_key = _get_groq_key()

        model_name = model or _DEFAULT_MODELS["groq"]
        return ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
        )

    # ── Ollama (Local) ──────────────────────────────────────────────────
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        model_name = model or os.getenv("OLLAMA_MODEL", _DEFAULT_MODELS["ollama"])
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported: gemini, groq, ollama"
        )


# ── Quick self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing LLM configuration...")
    llm = get_llm()
    print(f"Provider : {os.getenv('LLM_PROVIDER', 'gemini')}")
    print(f"Model    : {llm.__class__.__name__}")
    response = llm.invoke("Say 'LLM connection successful!' and nothing else.")
    print(f"Response : {response.content}")
    print("\n✔ LLM is configured and working!")
