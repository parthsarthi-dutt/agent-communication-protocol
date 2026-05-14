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
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Load .env from project root
load_dotenv(override=True)

# ── Default model names per provider ────────────────────────────────────
_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3",
}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
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

        # Prefer GEMINI_API_KEY (new standard), fall back to GOOGLE_API_KEY
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Add it to your .env file.\n"
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )

        # Avoid the "Both keys are set" warning by unsetting the conflicting one
        if os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
            os.environ.pop("GOOGLE_API_KEY", None)

        model_name = model or _DEFAULT_MODELS["gemini"]
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True,
        )

    # ── Groq ────────────────────────────────────────────────────────────
    elif provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to your .env file.\n"
                "Get a free key at: https://console.groq.com/keys"
            )

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
