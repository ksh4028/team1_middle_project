# ════════════════════════════════════════
# ▣ LLM Selector Utility
# ════════════════════════════════════════
# Purpose: 용도(purpose)에 따라 OpenAI 또는 Ollama LLM 인스턴스를 팩토리로 생성
# midprj_main.py의 Q&A는 이 모듈을 사용하지 않음 (기존 방식 유지)

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from midprj_defines import (
    OPENAI_SETTINGS,
    OLLAMA_SETTINGS,
    LLM_SELECTOR_SETTINGS,
)
from midprj_func import OpLog


def get_llm(provider):
    """
    용도(purpose)에 따라 OpenAI 또는 Ollama LLM 인스턴스를 반환합니다.

    Args:
        purpose: 용도 키 (eval, reranker, preprocess, pdf_summary)
        provider: LLM 제공자 오버라이드 (None이면 설정에서 자동 선택)
        temperature: LLM temperature

    Returns:
        ChatOpenAI 또는 ChatOllama 인스턴스
    """
    if provider == "ollama":
        resolved_model = OLLAMA_SETTINGS["llm_model_name"]
        resolved_temp = OLLAMA_SETTINGS.get("tempe rature", 0.2)
        return ChatOllama(
            base_url=OLLAMA_SETTINGS["address"],
            model=resolved_model,
            temperature=resolved_temp,
        )
    else:
        resolved_model = OPENAI_SETTINGS["llm_model_name"]
        resolved_temp = OPENAI_SETTINGS.get("temperature", 0.2)
        return ChatOpenAI(model=resolved_model, temperature=resolved_temp)

def get_llm_org(purpose: str,  provider: str = None, temperature: float = 0):
    """
    용도(purpose)에 따라 OpenAI 또는 Ollama LLM 인스턴스를 반환합니다.

    Args:
        purpose: 용도 키 (eval, reranker, preprocess, pdf_summary)
        provider: LLM 제공자 오버라이드 (None이면 설정에서 자동 선택)
        temperature: LLM temperature

    Returns:
        ChatOpenAI 또는 ChatOllama 인스턴스
    """
    provider = provider or LLM_SELECTOR_SETTINGS.get(purpose, "openai")

    if provider == "ollama":
        resolved_model = OLLAMA_SETTINGS["llm_model_name"]
        resolved_temp = OLLAMA_SETTINGS.get("temperature", 0.2)
        OpLog(
            f"[LLMSelector] purpose={purpose}, provider=ollama, "
            f"model={resolved_model}, base_url={OLLAMA_SETTINGS['address']}, temperature={resolved_temp}",
            level="INFO",
        )
        return ChatOllama(
            base_url=OLLAMA_SETTINGS["address"],
            model=resolved_model,
            temperature=resolved_temp,
        )
    else:
        resolved_model = OPENAI_SETTINGS["llm_model_name"]
        resolved_temp = OPENAI_SETTINGS.get("temperature", 0.2)
        OpLog(
            f"[LLMSelector] purpose={purpose}, provider=openai, "
            f"model={resolved_model}, temperature={resolved_temp}",
            level="INFO",
        )
        return ChatOpenAI(model=resolved_model, temperature=resolved_temp)
