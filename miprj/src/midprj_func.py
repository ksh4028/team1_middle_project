
def filter_metadata(meta, filters):
    """메타데이터 필터링: 예산, 기관 등"""
    budget = meta.get("budget", 0)
    try:
        if isinstance(budget, str):
            budget = float(budget.replace(',', '').replace('원', ''))
        else:
            budget = float(budget)
    except (ValueError, TypeError):
        budget = None
    if hasattr(filters, 'budget_max') and filters.budget_max and budget is not None and budget > filters.budget_max:
        return False
    if hasattr(filters, 'budget_min') and filters.budget_min and budget is not None and budget < filters.budget_min:
        return False
    if hasattr(filters, 'agency') and filters.agency:
        doc_agency = meta.get("agency", "").replace(" ", "")
        filter_agency = filters.agency.replace(" ", "")
        if filter_agency not in doc_agency:
            return False
    return True
# Standard Library
import datetime
import sys

# Third-party Libraries
from filelock import FileLock

# Project Modules
from midprj_defines import LOG_FILE


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def Lines(text=None, count=100):
    print("═" * count)
    if text is not None:
        print(text)
        print("═" * count)


def OpLog(log, bLines=False, level="INFO"):
    if bLines:
        Lines(log)
    try:
        frame = sys._getframe(1)
        caller_name = frame.f_code.co_name
        caller_line = frame.f_lineno
    except Exception:
        caller_name = "UnknownFunction"
        caller_line = 0

    log_lock_filename = LOG_FILE + ".lock"
    safe_level = str(level).upper() if level else "INFO"
    log_content = f"[{now_str()}] [{safe_level}] {caller_name}:{caller_line}: {log}\n"
    try:
        with FileLock(log_lock_filename, timeout=10):
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_content)
    except Exception as e:
        print(f"Log write error: {e}")


def filter_clean_for_rfp(text):
    """RFP 문서용 텍스트 전처리 (HTML 제거, 반복 축소, 중요 기호 보존)"""
    import re
    if not text:
        return ""

    # 1. HTML 태그 및 URL 제거 (Garbage removal)
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"(http|https)://\S+", " ", text)

    # 2. 반복 문자 정규화 (3회 이상 반복은 2회로 축소)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 3. 특수문자 필터링 (RFP용 허용 범위 확대 - 기술/문서 기호 보존)
    # 보존: . , ? ! ~ ; : - _ ( ) [ ] { } < > / % & = + " ' ` * #
    # 제거: 이모지, 제어 문자 등
    text = re.sub(r"[^\w\s.,?!~;:()\[\]{}<>/%&=+\"'`*#-]", " ", text)

    # 4. 다중 공백 -> 단일 공백
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_model_item(value: str | None) -> str:
    """모델 타입 정규화: openai, huggingface, ollama"""
    if not value:
        return "huggingface"
    normalized = str(value).strip().lower()
    if normalized in {"openai", "huggingface", "ollama"}:
        return normalized
    return "huggingface"



