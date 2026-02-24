# Standard Library
import os
import configparser
from pathlib import Path
from dataclasses import dataclass, field

# Third-party Libraries
import torch

# Project root directories (based on this file location)
MIDPRJ_DIR = Path(__file__).resolve().parent
BASE_DIR = str(MIDPRJ_DIR.parents[1])
DATA_DIR = os.path.join(BASE_DIR, "data")
ENV_FILE = os.path.join(BASE_DIR, ".env")

CSV_PATH = os.path.join(DATA_DIR, "rfp_files", "data_list.csv")
RFP_DATA_DIR = os.path.join(DATA_DIR, "rfp_files", "files")

SQLITEDB_DIR = os.path.join(DATA_DIR, "dbfile")
SQLITEDB_PATH = os.path.join(SQLITEDB_DIR, "midprj.db") 

LOG_DIR = os.path.join(DATA_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "midprj.log")

INI_PATH = os.path.join(MIDPRJ_DIR, "midprj.ini")

IS_GPU = torch.cuda.is_available()

DEFAULT_QUERY_LIST = [
    "1억 이상 3억 이하의 제안요구서에 대해서 알려 줘.",
    "교육과 IT 관련된 제안요구서를 찾아 줘.",
    "환경에 관련된 사업 제안요구서를 찾아주고 제안서별 요약을 해줘",
    "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘.",
    "콘텐츠 개발 관리 요구 사항에 대해서 더 자세히 알려 줘.",
    "교육이나 학습 관련해서 다른 기관이 발주한 사업은 없나?",
    "기초과학연구원 극저온시스템 사업 요구에서 AI 기반 예측에 대한 요구사항이 있나?",
    "그럼 모니터링 업무에 대한 요청사항이 있는지 찾아보고 알려 줘.",
    "한국 원자력 연구원에서 선량 평가 시스템 고도화 사업을 발주했는데, 이 사업이 왜 추진되는지 목적을 알려 줘.",
    "고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?",
    "고려대학교랑 광주과학기술원 각각 응답 시간에 대한 요구사항이 있나? 문서를 기반으로 정확하게 답변해 줘.",
]

DEFAULT_CORE_FEATURE_KEYWORDS = [
    "핵심 기능",
    "공통적으로",
    "공통",
    "주요 기능",
    "핵심 요구",
    "요구하는 핵심",
]


def load_ollama_settings(ini_path: str) -> dict:
    config = configparser.ConfigParser()
    if not os.path.exists(ini_path):
        return {
            "address": "http://localhost:11434",
            "llm_model_name": "llama3",
            "embedding_model_name": "nomic-embed-text",
        }

    config.read(ini_path, encoding="utf-8")
    if "ollama_model" not in config:
        return {
            "address": "http://localhost:11434",
            "llm_model_name": "llama3",
            "embedding_model_name": "nomic-embed-text",
        }

    address = config["ollama_model"].get("address", "http://localhost:11434").strip()
    llm_model_name = config["ollama_model"].get("llm_model_name", "llama3").strip()
    embedding_model_name = config["ollama_model"].get(
        "embedding_model_name",
        "nomic-embed-text",
    ).strip()
    return {
        "address": address or "http://localhost:11434",
        "llm_model_name": llm_model_name or "llama3",
        "embedding_model_name": embedding_model_name or "nomic-embed-text",
    }


def load_openai_settings(ini_path: str) -> dict:
    config = configparser.ConfigParser()
    if not os.path.exists(ini_path):
        return {
            "llm_model_name": "gpt-5-mini",
            "embedding_model_name": "text-embedding-3-small",
        }

    config.read(ini_path, encoding="utf-8")
    if "openai_model" not in config:
        return {
            "llm_model_name": "gpt-5-mini",
            "embedding_model_name": "text-embedding-3-small",
        }

    section = "openai_model"
    llm_model_name = config[section].get("llm_model_name", "gpt-5-mini").strip()
    embedding_model_name = config[section].get(
        "embedding_model_name",
        "text-embedding-3-small",
    ).strip()
    return {
        "llm_model_name": llm_model_name or "gpt-5-mini",
        "embedding_model_name": embedding_model_name or "text-embedding-3-small",
    }


def load_query_list(ini_path: str, fallback: list[str]) -> list[str]:
    config = configparser.ConfigParser()
    if not os.path.exists(ini_path):
        return fallback

    config.read(ini_path, encoding="utf-8")
    if "query_list" not in config:
        return fallback

    items = config["query_list"].get("items", "")
    lines = [line.strip() for line in items.splitlines() if line.strip()]
    return lines if lines else fallback


def load_list_setting(ini_path: str, section: str, fallback: list[str]) -> list[str]:
    config = configparser.ConfigParser()
    if not os.path.exists(ini_path):
        return fallback

    config.read(ini_path, encoding="utf-8")
    if section not in config:
        return fallback

    items = config[section].get("items", "")
    lines = [line.strip() for line in items.splitlines() if line.strip()]
    return lines if lines else fallback


QUERY_LIST = load_query_list(INI_PATH, DEFAULT_QUERY_LIST)
CORE_FEATURE_KEYWORDS = load_list_setting(
    INI_PATH,
    "core_feature_keywords",
    DEFAULT_CORE_FEATURE_KEYWORDS,
)

OLLAMA_SETTINGS = load_ollama_settings(INI_PATH)
OPENAI_SETTINGS = load_openai_settings(INI_PATH)


def load_llm_selector_settings(ini_path: str) -> dict:
    """용도별 LLM 제공자 설정 로드 (기본값: openai)"""
    config = configparser.ConfigParser()
    defaults = {
        "eval": "openai",
        "reranker": "openai",
        "preprocess": "openai",
        "pdf_summary": "openai",
    }
    if not os.path.exists(ini_path):
        return defaults
    config.read(ini_path, encoding="utf-8")
    if "LLM_SELECTOR" not in config:
        return defaults
    result = {}
    for key, default in defaults.items():
        result[key] = config["LLM_SELECTOR"].get(key, default).strip().lower()
    return result


LLM_SELECTOR_SETTINGS = load_llm_selector_settings(INI_PATH)


def load_history_settings(ini_path: str) -> dict:
    config = configparser.ConfigParser()
    defaults = {
        "max_turns": 3,
        "persist": False,
        "session_id": "",
        "budget_ratio": 0.2,
        "budget_min_tokens": 64,
        "budget_max_tokens": 256,
    }
    if not os.path.exists(ini_path):
        return defaults
    config.read(ini_path, encoding="utf-8")
    if "HISTORY" not in config:
        return defaults

    section = config["HISTORY"]
    max_turns = section.get("max_turns", str(defaults["max_turns"]))
    persist = section.get("persist", str(defaults["persist"]))
    session_id = section.get("session_id", defaults["session_id"]).strip()
    budget_ratio = section.get("budget_ratio", str(defaults["budget_ratio"]))
    budget_min_tokens = section.get("budget_min_tokens", str(defaults["budget_min_tokens"]))
    budget_max_tokens = section.get("budget_max_tokens", str(defaults["budget_max_tokens"]))

    try:
        max_turns = int(max_turns)
    except (TypeError, ValueError):
        max_turns = defaults["max_turns"]

    persist = str(persist).strip().lower() in {"1", "true", "yes", "y"}

    try:
        budget_ratio = float(budget_ratio)
    except (TypeError, ValueError):
        budget_ratio = defaults["budget_ratio"]
    budget_ratio = max(0.0, min(budget_ratio, 1.0))

    try:
        budget_min_tokens = int(budget_min_tokens)
    except (TypeError, ValueError):
        budget_min_tokens = defaults["budget_min_tokens"]

    try:
        budget_max_tokens = int(budget_max_tokens)
    except (TypeError, ValueError):
        budget_max_tokens = defaults["budget_max_tokens"]

    if budget_min_tokens > budget_max_tokens:
        budget_min_tokens = budget_max_tokens

    return {
        "max_turns": max_turns,
        "persist": persist,
        "session_id": session_id,
        "budget_ratio": budget_ratio,
        "budget_min_tokens": budget_min_tokens,
        "budget_max_tokens": budget_max_tokens,
    }


HISTORY_SETTINGS = load_history_settings(INI_PATH)

    # --execute_index "$execute_index" \
    # --whatmodelItem "$whatmodelItem" \
    # --embedding_model "$embedding_model" \
    # --llm_model "$llm_model" \
    # --temperature "$temperature" \
    # --repetition_penalty "$repetition_penalty" \
    # --chunk_size "$chunk_size" \
    # --chunk_overlap "$chunk_overlap" \
    # --k "$k" \
    # --llm_selector "$llm_sel" \
    # --reranker_type "$reranker" \
    # --do_retriever "$retriever"


@dataclass
class PARAMVAR:
    execute_index: int = 0
    embedding_model: str = OPENAI_SETTINGS["embedding_model_name"]
    llm_model: str = OPENAI_SETTINGS["llm_model_name"]
    chunk_size: int = 500
    chunk_overlap: int = 50
    temperature: float = 0.2
    repetition_penalty: float = 1.2
    query: str = ""
    answer: str = ""
    start_time: str = "2000-01-01 00:00:00"
    end_time: str = "2001-01-01 00:00:00"
    is_gpu: bool = IS_GPU
    csv_path: str = CSV_PATH
    rfp_data_dir: str = RFP_DATA_DIR
    k: int = 10
    whatmodelItem: str = "openai"
    newCreate: bool = False
    retriever_llm_type: str = "openai"  

    hybrid_alpha: float = 0.5
    reranker_type: str = "openai"  # 또는 "ollama"
    reranker_types: list = field(default_factory=lambda: [
        "dense",
        "openai",
        "ollama",
        "hybrid"
    ])
    max_input_tokens_hf: int = 2048
    store_ver: str = "V15"
    ollama_address: str = OLLAMA_SETTINGS["address"]
    ollama_model_name: str = OLLAMA_SETTINGS["llm_model_name"]
    ollama_embedding_model_name: str = OLLAMA_SETTINGS["embedding_model_name"]
    history_max_turns: int = HISTORY_SETTINGS["max_turns"]
    history_persist: bool = HISTORY_SETTINGS["persist"]
    history_session_id: str = HISTORY_SETTINGS["session_id"]
    history_budget_ratio: float = HISTORY_SETTINGS["budget_ratio"]
    history_budget_min_tokens: int = HISTORY_SETTINGS["budget_min_tokens"]
    history_budget_max_tokens: int = HISTORY_SETTINGS["budget_max_tokens"]
    llm_selector: str = "openai"
    do_retriever: str = "hybrid"  # 또는 "vector", "hybrid", "hybrid_llm"
    do_eval : str = "optimize"
    

