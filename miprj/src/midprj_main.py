
# ===============================
# ▣ Import 정리 (최상단으로 이동)
# ===============================

# 표준 라이브러리
import os
import sys
import shutil
import pickle
import struct
import zlib
import unicodedata
import re
import gc
from pathlib import Path
from typing import Optional

from midprj_history import truncate_text_by_tokens

# 외부 라이브러리
import pandas as pd
import numpy as np
import olefile
import torch
import faiss
from dotenv import load_dotenv

# transformers 관련
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# pydantic
from pydantic import BaseModel, Field

# LangChain Core
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

# LangChain Text & Document Processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

# LangChain Vector Stores & Retrievers
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# LangChain Embeddings & Models
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings

try:
    from langchain_ollama import ChatOllama
except (ModuleNotFoundError, ImportError):
    from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# 프로젝트 내부 모듈
from midprj_defines import (
    BASE_DIR,
    DATA_DIR,
    ENV_FILE,
    CSV_PATH,
    RFP_DATA_DIR,
    SQLITEDB_DIR,
    SQLITEDB_PATH,
    LOG_DIR,
    LOG_FILE,
    IS_GPU,
    OPENAI_SETTINGS,
    PARAMVAR,
    QUERY_LIST,
    CORE_FEATURE_KEYWORDS,
)
from midprj_func import Lines, OpLog, now_str, filter_clean_for_rfp, normalize_model_item
from midprj_preprocessor import BidMatePreprocessor
from midprj_history import HistoryManager
from midprj_sqlite import SQLiteDB
from midprj_retriever import build_retriever_from_param, build_retriever_from_param
from midprj_reranker import Reranker


def init_Env():
    print("환경 변수 초기화 및 디렉토리 생성")
    OpLog("환경 변수 초기화 시작", level="INFO")
    load_dotenv(ENV_FILE)
    if not os.path.exists(SQLITEDB_DIR):
        os.makedirs(SQLITEDB_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    OpLog("환경 변수 초기화 완료", level="INFO")

init_Env()


def model_item_label(value: str) -> str:
    normalized = normalize_model_item(value)
    if normalized == "openai":
        return "OpenAI"
    if normalized == "ollama":
        return "Ollama"
    return "HuggingFace"


def normalize_reranker_type(value: str | None) -> str:
    if not value:
        return "openai"
    if value == "openai":
        return "openai"
    normalized = str(value).strip().lower()
    if normalized in {"openai", "ollama"}:
        return normalized
    return "openai"



def is_huggingface(value: str) -> bool:
    return normalize_model_item(value) == "huggingface"

# ════════════════════════════════════════
# ▣ 메타데이터 로드 및 전처리
# ════════════════════════════════════════
class SearchFilter(BaseModel):
    """사업 공고 검색을 위한 필터 구조"""
    notice_no: Optional[str] = Field(None, description="공고 번호")
    notice_round: Optional[str] = Field(None, description="공고 차수")
    project_name: Optional[str] = Field(None, description="사업명 키워드")
    agency: Optional[str] = Field(None, description="발주 기관명 (예: 서울특별시, 한국지능정보사회진흥원)")
    budget_min: Optional[int] = Field(None, description="최소 사업 금액 (원 단위)")
    budget_max: Optional[int] = Field(None, description="최대 사업 금액 (원 단위)")
    publish_date: Optional[str] = Field(None, description="공개 일자 (YYYY-MM-DD)")
    start_date: Optional[str] = Field(None, description="입찰 참여 시작일 (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="입찰 참여 마감일 (YYYY-MM-DD)")
    project_summary: Optional[str] = Field(None, description="사업 요약 키워드")
    file_type: Optional[str] = Field(None, description="파일형식 (pdf, hwp 등)")
    file_name: Optional[str] = Field(None, description="파일명")
    deadline_before: Optional[str] = Field(None, description="마감 기한 (YYYY-MM-DD 형식)")
    domain: Optional[str] = Field(None, description="사업 분야 (과학, IT, 건설, 의료, 교육, 환경, 문화, 복지, 국방, 통신, 행정, 기타)")
    keywords: Optional[str] = Field(None, description="핵심 키워드")
    region: Optional[str] = Field(None, description="관련 지역")

# ════════════════════════════════════════
# ▣ 메타데이터 필터 추출기 (LLM)
# ════════════════════════════════════════
class QueryFilterExtractor:
    _domain_keywords_cache = None

    def __init__(self, llm):
        self.parser = PydanticOutputParser(pydantic_object=SearchFilter)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "사용자의 질문에서 검색 필터를 추출하세요. 정보가 없으면 null로 설정하세요.\n{format_instructions}",
                ),
                ("human", "{query}"),
            ]
        ).partial(format_instructions=self.parser.get_format_instructions())
        self.chain = self.prompt | llm | self.parser

    def _parse_korean_number(self, text):
        """한국어 숫자 표현을 숫자로 변환 (예: 1억 -> 100000000, 3천만 -> 30000000)"""
        import re

        # 오타 수정
        text = text.replace("에산", "예산").replace("사업비", "예산").replace("금액", "예산")

        result = text
        # 억 단위 변환
        for match in re.finditer(r"(\d+(?:\.\d+)?)억", text):
            num = float(match.group(1))
            result = result.replace(match.group(0), str(int(num * 100000000)))
        # 천만 단위 변환
        for match in re.finditer(r"(\d+(?:\.\d+)?)천만", text):
            num = float(match.group(1))
            result = result.replace(match.group(0), str(int(num * 10000000)))
        # 만 단위 변환
        for match in re.finditer(r"(\d+(?:\.\d+)?)만", text):
            num = float(match.group(1))
            result = result.replace(match.group(0), str(int(num * 10000)))

        return result

    def _manual_extract_domain(self, query: str):
        """질문에서 도메인/분야 키워드를 추출 (DB 연동, 캐시 적용)"""
        if QueryFilterExtractor._domain_keywords_cache is None:
            db = SQLiteDB()
            QueryFilterExtractor._domain_keywords_cache = db.get_domain_keywords()
        domain_keywords = QueryFilterExtractor._domain_keywords_cache

        for category, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query.lower():
                    Lines(f"[DEBUG] 도메인 키워드 감지: '{keyword}' (카테고리: {category})")
                    return category  # 키워드 대신 카테고리 반환
        return None

    def _manual_extract_budget(self, query: str):
        """질문에서 예산 범위를 정규식으로 추출"""
        import re

        normalized = self._parse_korean_number(query)
        Lines(f"[DEBUG] QueryFilterExtractor - 정규화된 질문: {normalized}")

        budget_min = None
        budget_max = None

        # "X 이상 Y 이하" 패턴
        pattern1 = (
            r"(\d+(?:,\d{3})*)\s*(?:원)?\s*이상\s*(?:~)?\s*(\d+(?:,\d{3})*)\s*(?:원)?\s*이하"
        )
        match = re.search(pattern1, normalized)
        if match:
            budget_min = int(match.group(1).replace(",", ""))
            budget_max = int(match.group(2).replace(",", ""))
            Lines(f"[DEBUG] 패턴 매칭 성공 (이상~이하): min={budget_min}, max={budget_max}")
            return budget_min, budget_max

        # "X 이상" 패턴
        pattern_min = r"(\d+(?:,\d{3})*)\s*(?:원)?\s*이상"
        match_min = re.search(pattern_min, normalized)
        if match_min:
            budget_min = int(match_min.group(1).replace(",", ""))
        # "X 이하" 패턴
        pattern_max = r"(\d+(?:,\d{3})*)\s*(?:원)?\s*이하"
        match_max = re.search(pattern_max, normalized)
        if match_max:
            budget_max = int(match_max.group(1).replace(",", ""))

        if budget_min or budget_max:
            Lines(f"[DEBUG] 패턴 매칭 성공: min={budget_min}, max={budget_max}")

        return budget_min, budget_max

    def extract(self, query: str) -> SearchFilter:
        # 먼저 수동 추출 시도 (정규식 기반)
        budget_min, budget_max = self._manual_extract_budget(query)
        domain_keyword = self._manual_extract_domain(query)

        # LLM 추출 시도
        try:
            filters = self.chain.invoke({"query": query})

            # 수동 추출 결과로 LLM 결과 보완
            if budget_min and not filters.budget_min:
                filters.budget_min = budget_min
                Lines(f"[DEBUG] 수동 추출한 budget_min 적용: {budget_min}")
            if budget_max and not filters.budget_max:
                filters.budget_max = budget_max
                Lines(f"[DEBUG] 수동 추출한 budget_max 적용: {budget_max}")
            if domain_keyword and not filters.domain:
                filters.domain = domain_keyword
                Lines(f"[DEBUG] 수동 추출한 도메인 키워드 적용: {domain_keyword}")

            if filters.agency:
                agency_text = str(filters.agency).strip()
                if agency_text in {"다른 기관", "타기관", "기타 기관", "외부 기관"}:
                    filters.agency = None
                    Lines("[DEBUG] 비특정 기관 표현 감지 - agency 필터 제거")
            OpLog("필터 추출 완료", level="INFO")
            return filters
        except Exception as e:
            Lines(f"[DEBUG] LLM 필터 추출 실패: {e}")
            OpLog(f"필터 추출 실패: {e}", level="WARN")
            # LLM 실패 시 수동 추출 결과만 사용
            return SearchFilter(
                budget_min=budget_min,
                budget_max=budget_max,
                domain=domain_keyword,
            )

# ════════════════════════════════════════
# ▣ 텍스트 자르기 및 컨텍스트 빌드
# ════════════════════════════════════════

# 컨텍스트 빌드
def build_budgeted_context(sql_results, context_result, max_context_tokens, tokenizer):
    if max_context_tokens <= 0:
        return ""

    sql_header = "### [데이터베이스 직접 조회 결과]\n"
    doc_header = "### [관련 문서 내용]\n"

    has_sql = bool(sql_results)
    has_docs = bool(context_result)

    if has_sql and has_docs:
        sql_budget = int(max_context_tokens * 0.4)
        sql_budget = min(sql_budget, max_context_tokens)
        doc_budget = max_context_tokens - sql_budget

        min_budget = 64
        if sql_budget < min_budget:
            sql_budget = min(min_budget, max_context_tokens)
            doc_budget = max_context_tokens - sql_budget
        if doc_budget < min_budget:
            doc_budget = min(min_budget, max_context_tokens)
            sql_budget = max_context_tokens - doc_budget
    else:
        sql_budget = max_context_tokens
        doc_budget = max_context_tokens

    parts = []
    if has_sql:
        sql_text = f"{sql_header}{sql_results}"
        parts.append(truncate_text_by_tokens(sql_text, sql_budget, tokenizer))
    if has_docs:
        doc_text = f"{doc_header}{context_result}"
        parts.append(truncate_text_by_tokens(doc_text, doc_budget, tokenizer))
    return "\n\n".join(part for part in parts if part)


# ════════════════════════════════════════
# ▣ 베이스 모델 클래스 
# ════════════════════════════════════════
class BaseLLMModel():
    _SYSTEM_CONSTRAINTS = (
        "1. 사용자가 요청한 금액(예산)과 기관 조건을 반드시 엄격히 준수하여 답변하세요.\n"
        "2. 아래 [데이터베이스 직접 조회 결과]에 항목이 있다면, 이는 검색 엔진이 정확하게 찾아낸 것이므로 **단 하나도 빠뜨리지 말고 모두** 목록에 포함하여 답변하세요.\n"
        "3. 만약 [데이터베이스 직접 조회 결과]와 [관련 문서 내용]의 정보가 충돌한다면 데이터베이스 직접 조회 결과를 우선하세요.\n"
    )
    _PROMPT_TEMPLATE = (
        "{basequery}\n\n시스템 제약조건:\n{constraints}\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )

    def __init__(self,param:PARAMVAR):
        self._my_name = f"embedding:{param.embedding_model}_llm:{param.llm_model}"
        self._param = param
        self._vector_store = None
        self._llm = None
        self._reranker = Reranker(self._param)
        self._retriever = None
        self._ensemble_retriever = None
        self._tokenizer = None
        self._max_input_tokens = None
        self._basequery = "다음 메타데이터와 문서 내용을 참고하여 질문에 답변해주세요."
        self._processor = None
        self._last_context = ""
        self._history_manager = HistoryManager(param)
        # 임베딩 모델 자동 생성
        model_item = normalize_model_item(param.whatmodelItem)
        if model_item == "openai":
            from langchain_openai import OpenAIEmbeddings
            self._embedder = OpenAIEmbeddings(model=param.embedding_model)
        elif model_item == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings
            self._embedder = OllamaEmbeddings(model=param.ollama_embedding_model_name)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embedder = HuggingFaceEmbeddings(model_name=param.embedding_model)
        self._my_llm_type = None

    def rag_search(self, question):
        # LLM이 없으면 make_model() 호출
        if self._llm is None:
            self.make_model()
        if self._ensemble_retriever is None:
            self._ensemble_retriever = build_retriever_from_param(vector_store=self._vector_store, 
                                                                  all_docs=self._processor.get_all_docs(), 
                                                                  param=self._param, 
                                                                  llm_selector= self._param.retriever_llm_type)

        # OpenAI와 동일한 _perform_rag_search 사용 (SQL DB 메타데이터 검색 포함)
        return self._perform_rag_search(question, self._ensemble_retriever)

    def get_all_docs(self):
        """BidMatePreprocessor를 통해 모든 문서 로드"""
        if not hasattr(self, "_processor") or self._processor is None:
            raise ValueError("BidMatePreprocessor가 초기화되지 않았습니다.")
        return self._processor.get_all_docs()

    ## 결과 저장

    def save_result(self, query_index, query, answer, start_time, end_time, context=None, do_retriever=None, retriever_llm_type=None, reranker_type=None):
        Lines(f"Query 시작: query_index={query_index}, query={query[:50]}...")
        OpLog(f"결과 저장 시작: query_index={query_index}", level="INFO")

        db = SQLiteDB()
        model_item = model_item_label(self._param.whatmodelItem)
        Lines(f"Query 완료: query_index={query_index}, 답변 길이={len(answer)}")
        context_text = context if context is not None else self._last_context
        # reranker_type 우선순위: 인자 → myparam.reranker_type → self._param.reranker_type
        reranker_type_to_save = reranker_type if reranker_type is not None else getattr(self._param, 'reranker_type', None)
        db.save_results(
            execute_index=self._param.execute_index,
            model_item=model_item,
            embedding_model=self._param.embedding_model,
            llm_model=self._param.llm_model,
            retriever_llm_type=retriever_llm_type if retriever_llm_type is not None else self._param.retriever_llm_type,
            reranker_type=reranker_type_to_save,
            chunk_size=self._param.chunk_size,
            chunk_overlap=self._param.chunk_overlap,
            k=self._param.k,
            is_openai=int(normalize_model_item(self._param.whatmodelItem) == "openai"),
            is_gpu=int(self._param.is_gpu),
            store_ver=self._param.store_ver,
            temperature=self._param.temperature,
            repetition_penalty=self._param.repetition_penalty,
            query_index=query_index,
            query=query,
            answer=answer,
            context=context_text or "",
            start_time=start_time,
            end_time=end_time,
            do_retriever=do_retriever,
        )
        OpLog(f"결과 저장 완료: query_index={query_index}", level="INFO")

    

    ## 질의응답 수행 및 결과 기록.
    def eval_queries_answers_Ex(self):
        """retriever_types × reranker_types 조합별로 질의응답 및 결과 저장"""
        if self._llm is None:
            self.make_model()
        if self._vector_store is None:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        all_docs = self._processor.get_all_docs()
        from dataclasses import replace
        myparam = replace(self._param)
        what_llm = normalize_reranker_type(self._param.whatmodelItem)
        retriever_llm_types = [what_llm]
        # LLM 기반만 허용
        reranker_types = ["bge","dense","ollama","hybrid"] # openai 토큰 비용 문제로 제외
        querylist = QUERY_LIST
        do_treivers =[ "bm25", "vector", "hybrid", "llm", "bm25_llm", "vector_llm", "hybrid_llm"]

        for retr_llm_type in retriever_llm_types:
            myparam.retriever_llm_type = retr_llm_type
            for do_retriever in do_treivers:
                retriever = build_retriever_from_param(self._vector_store, 
                                                       all_docs, myparam, 
                                                       llm_selector=retr_llm_type,
                                                       do_retriever=do_retriever)
                for rer_type in reranker_types:
                    myparam.reranker_type = rer_type
                    self._reranker = Reranker(myparam)
                    self._history_manager.clear() ## 평가를 위해서 매 조합별로 히스토리 초기화
                    for index, query in enumerate(querylist):
                        # 평가 파라미터 정보도 함께 출력
                        OpLog(
                            f"실험 파라미터: whatmodelItem={self._param.whatmodelItem}, "
                            f"temperature={self._param.temperature}, repetition_penalty={self._param.repetition_penalty}, "
                            f"chunk_size={self._param.chunk_size}, chunk_overlap={self._param.chunk_overlap}, k_values={self._param.k}, "
                            f"질의응답 시작: Index={index}, Retriever LLM={retr_llm_type}, Reranker={rer_type}, Retriever Type={do_retriever}",
                            level="INFO"
                        )
                        start_time = now_str()
                        answer = self._perform_rag_search(query, retriever)
                        end_time = now_str()
                        self.save_result(index, query, answer, start_time, end_time, self._last_context, do_retriever=do_retriever, retriever_llm_type=myparam.retriever_llm_type, reranker_type=rer_type)

    ## 질의응답 수행 및 결과 기록.
    def eval_queries_answers(self):
        """retriever_types × reranker_types 조합별로 질의응답 및 결과 저장"""
        if self._llm is None:
            self.make_model()
        if self._vector_store is None:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        all_docs = self._processor.get_all_docs()
        from dataclasses import replace
        myparam = replace(self._param)
        what_llm = normalize_reranker_type(self._param.whatmodelItem)
        retriever_llm_types = [what_llm]
        reranker_types = ["hybrid"]
        querylist = QUERY_LIST
        do_treivers = ["vector","hybrid","hybrid_llm"]
        print(f"=== 평가 시작: Retriever LLM Types={retriever_llm_types}, Reranker Types={reranker_types}, Retriever Types={do_treivers}, ===")

        for retr_llm_type in retriever_llm_types:
            myparam.retriever_llm_type = retr_llm_type
            for do_retriever in do_treivers:
                retriever = build_retriever_from_param(self._vector_store, 
                                                       all_docs, myparam, 
                                                       llm_selector=retr_llm_type,
                                                       do_retriever=do_retriever)
                for rer_type in reranker_types:
                    myparam.reranker_type = rer_type
                    self._reranker = Reranker(myparam)
                    self._history_manager.clear() ## 평가를 위해서 매 조합별로 히스토리 초기화
                    for index, query in enumerate(querylist):
                        # 평가 파라미터 정보도 함께 출력
                        OpLog(
                            f"평가 실행 완료 - 실험 파라미터: "
                            f"whatmodelItem={myparam.whatmodelItem}, "
                            f"temperature={myparam.temperature}, repetition_penalty={myparam.repetition_penalty}, "
                            f"chunk_size={myparam.chunk_size}, chunk_overlap={myparam.chunk_overlap}, k_values={myparam.k}",
                            level="INFO",
                            bLines=True,
                        )
                        Lines(f"\n=== 질의응답 시작 (Index: {index}, Retriever LLM: {retr_llm_type}, Reranker: {rer_type}, Retriever Type: {do_retriever}) ===")
                        start_time = now_str()
                        answer = self._perform_rag_search(query, retriever)
                        end_time = now_str()
                        self.save_result(index, query, answer, start_time, end_time, self._last_context, do_retriever=do_retriever, retriever_llm_type=myparam.retriever_llm_type)

    ## 질의응답 수행 및 결과 기록.
    def eval_queries_answers_optimize(self):
        """retriever_types × reranker_types 조합별로 질의응답 및 결과 저장"""
        if self._llm is None:
            self.make_model()
        if self._vector_store is None:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        all_docs = self._processor.get_all_docs()
        from dataclasses import replace
        myparam = replace(self._param)
        print(f"=== 평가 시작: Retriever LLM Types={myparam.retriever_llm_types}, Reranker Types={myparam.reranker_types}, Retriever Types={myparam.do_treivers}, ===")
        self._reranker = Reranker(myparam)
        retriever = build_retriever_from_param(vector_store=self._vector_store, all_docs=all_docs, myparam=myparam)
        querylist = QUERY_LIST
        for index, query in enumerate(querylist):
            # 평가 파라미터 정보도 함께 출력
            OpLog(
                f"평가 실행 완료 - 실험 파라미터: "
                f"whatmodelItem={myparam.whatmodelItem}, "
                f"temperature={myparam.temperature}, repetition_penalty={myparam.repetition_penalty}, "
                f"chunk_size={myparam.chunk_size}, chunk_overlap={myparam.chunk_overlap}, k_values={myparam.k}",
                level="INFO",
                bLines=True,
            )
            Lines(f"\n=== 질의응답 시작 (Index: {index}, Retriever LLM: {myparam.retriever_llm_type}, Reranker: {myparam.reranker_type}, Retriever Type: {myparam.do_retriever}) ===")
            start_time = now_str()
            answer = self._perform_rag_search(query, retriever)
            end_time = now_str()
            self.save_result(index, query, answer, start_time, end_time, self._last_context, do_retriever=myparam.do_retriever, retriever_llm_type=myparam.retriever_llm_type)


    def clear_mem(self):
        if self._vector_store is not None:
            del self._vector_store
            self._vector_store = None
        torch.cuda.empty_cache()
        gc.collect()

    def _apply_metadata_filters(self, all_docs, filters):
        """메타데이터 필터(예산, 기관) 적용하여 필터링된 문서 반환"""
        filtered_docs = []
        for doc in all_docs:
            meta = doc.metadata
            doc_budget = meta.get("budget", 0)
            try:
                if isinstance(doc_budget, str):
                    doc_budget = float(doc_budget.replace(',', '').replace('원', ''))
                else:
                    doc_budget = float(doc_budget)
            except (ValueError, TypeError):
                doc_budget = None

            # 예산 필터 (최대)
            if filters.budget_max and doc_budget is not None:
                if doc_budget > filters.budget_max:
                    continue
            
            # 예산 필터 (최소)
            if filters.budget_min and doc_budget is not None:
                if doc_budget < filters.budget_min:
                    continue

            # 기관 필터
            if filters.agency:
                doc_agency = meta.get("agency", "").replace(" ", "")
                filter_agency = filters.agency.replace(" ", "")
                if filter_agency not in doc_agency:
                    continue
            filtered_docs.append(doc)
        
        return filtered_docs

    def _is_core_features_query(self, question: str) -> bool:
        return any(keyword in question for keyword in CORE_FEATURE_KEYWORDS)

    def _rerank_and_get_context(self, question, filtered_docs, embedder=None):
        return self._reranker.rerank_and_get_context(question, filtered_docs, embedder)


    def _perform_rag_search(self, question, retriever):
        """통합 RAG 검색: 필터링 → 리랭킹 → LLM 답변 생성"""
        OpLog("RAG 검색 시작", level="INFO")
        from midprj_sqlresult import SQLSearchHelper
        try:
            # 1. 질문에서 필터 추출
            extractor = QueryFilterExtractor(self._llm)
            filters = extractor.extract(question)
            OpLog(
                "필터 추출 완료: "
                f"budget_min={filters.budget_min}, budget_max={filters.budget_max}, "
                f"agency={filters.agency}",
                level="INFO",
            )
            
            # 2. SQL 기반 직접 조회 (리스트 요청 등 정형화된 검색이 필요할 때)
            sql_results = ""
            sql_items = []
            if SQLSearchHelper.detect_search_intent(question, filters):
                OpLog("검색 의도 감지됨 - SQL 검색 수행", level="INFO")
                if (filters.budget_max or filters.budget_min or filters.agency or filters.notice_no or filters.project_name or filters.domain or filters.project_summary):
                    sql_items = SQLSearchHelper.sql_metadata_search(filters)
                    OpLog(f"SQL 검색 결과: {len(sql_items)}개 항목", level="INFO")
                    if sql_items:
                        sql_results = SQLSearchHelper.format_metadata_items(sql_items)
                        Lines(f"[DEBUG] SQL 결과 샘플 (처음 200자):\n{sql_results[:200]}...")
                else:
                    OpLog("필터 조건 없음 - SQL 검색 건너뜀", level="INFO")
            else:
                OpLog("검색 의도 감지 안됨 - SQL 검색 건너뜀", level="INFO")
            
            # 3. 메타데이터 필터 함수 정의 (retriever 레벨 필터링용)
            def metadata_filter(metadata):
                try:
                    budget = metadata.get("budget", 0)
                    if isinstance(budget, str):
                        budget = float(budget.replace(',', '').replace('원', ''))
                    else:
                        budget = float(budget)
                    # NaN 방지: budget이 NaN이면 필터 조건에서 제외
                    if isinstance(budget, float) and (budget != budget or budget is None):
                        return True
                    if filters.budget_max and budget > filters.budget_max:
                        return False
                    if filters.budget_min and budget < filters.budget_min:
                        return False
                    if filters.agency:
                        input_agency = filters.agency.replace(" ", "")
                        meta_agency = metadata.get("agency", "").replace(" ", "")
                        if input_agency not in meta_agency:
                            return False
                    return True
                except:
                    return True

            # 4. 문서 검색 (필터 적용 및 k 상향)
            all_retrieved_docs = retriever.invoke(question, filter=metadata_filter)
            OpLog(f"Retriever 검색 결과: {len(all_retrieved_docs)}개 문서", level="INFO")
            
            # 5. 필터 재검증 (강제 필터링)
            filtered_docs = self._apply_metadata_filters(all_retrieved_docs, filters)
            OpLog(f"필터링 후 문서 수: {len(filtered_docs)}개", level="INFO")

            # 5-1. 필터링 결과가 비어 있으면 완화 후 재검색
            if not filtered_docs:
                relaxed_filters = SearchFilter(**filters.model_dump())
                if relaxed_filters.agency:
                    relaxed_filters.agency = None
                    OpLog("필터 완화: agency 제거 후 재검색", level="INFO")

                if relaxed_filters.domain:
                    relaxed_filters.domain = None
                    OpLog("필터 완화: domain 제거 후 재검색", level="INFO")

                def relaxed_metadata_filter(metadata):
                    try:
                        budget = metadata.get("budget", 0)
                        if isinstance(budget, str):
                            budget = float(budget.replace(',', '').replace('원', ''))
                        else:
                            budget = float(budget)

                        if relaxed_filters.budget_max and budget > relaxed_filters.budget_max:
                            return False
                        if relaxed_filters.budget_min and budget < relaxed_filters.budget_min:
                            return False

                        if relaxed_filters.agency:
                            input_agency = relaxed_filters.agency.replace(" ", "")
                            meta_agency = metadata.get("agency", "").replace(" ", "")
                            if input_agency not in meta_agency:
                                return False
                        return True
                    except Exception:
                        return True

                all_retrieved_docs = retriever.invoke(question, filter=relaxed_metadata_filter)
                filtered_docs = self._apply_metadata_filters(all_retrieved_docs, relaxed_filters)
                OpLog(f"완화 후 문서 수: {len(filtered_docs)}개", level="INFO")
            
            # 6. 리랭킹 및 context 생성
            final_docs, context_result = self._rerank_and_get_context(question, filtered_docs, self._embedder)
            OpLog(
                f"최종 리랭킹 후 문서 수: {len(final_docs) if final_docs else 0}개",
                level="INFO",
            )

            is_core_query = self._is_core_features_query(question)
            doc_items = SQLSearchHelper.collect_metadata_from_docs(final_docs)
            metadata_items = sql_items or doc_items
            meta_list_text = SQLSearchHelper.format_metadata_items(metadata_items)
            project_names_text = SQLSearchHelper.format_project_names(metadata_items)
            
            # 7. SQL 결과와 Retrieval 결과 병합
            combined_context = ""
            if sql_results:
                combined_context += f"### [데이터베이스 직접 조회 결과]\n{sql_results}\n\n"
                OpLog("SQL 결과가 context에 추가됨", level="INFO")
            else:
                OpLog("SQL 결과 없음", level="INFO")
            
            if final_docs:
                combined_context += f"### [관련 문서 내용]\n{context_result}"
                OpLog("문서 내용이 context에 추가됨", level="INFO")
            else:
                OpLog("검색된 문서 없음", level="WARN")
            
            if not combined_context:
                OpLog("Context가 비어있음 - 관련 정보 없음", level="WARN")
                self._last_context = ""
                return "관련된 정보를 찾을 수 없습니다."
            
            self._last_context = combined_context
            OpLog(f"최종 Context 길이: {len(combined_context)} 문자", level="INFO")
            Lines(f"[DEBUG] Context 미리보기 (처음 400자):\n{'='*80}\n{combined_context[:400]}...\n{'='*80}")
            
            # 8. LLM을 통한 답변 생성
            history_block = self._history_manager.format_block()
            if is_core_query:
                prompt_prefix_base = (
                    f"{self._basequery}\n\n시스템 제약조건:\n{self._SYSTEM_CONSTRAINTS}\n\n"
                    "질문이 '핵심 기능' 유형일 경우 아래 형식으로 답하세요:\n"
                    "1) 핵심 기능: 핵심 기능을 불릿으로 나열\n"
                    "2) 해당 사업명: 아래 목록에서 해당하는 사업명만 불릿으로 나열\n\n"
                    f"사업명 목록:\n{project_names_text or '정보 없음'}\n\n"
                )
            else:
                prompt_prefix_base = (
                    f"{self._basequery}\n\n시스템 제약조건:\n{self._SYSTEM_CONSTRAINTS}\n\n"
                )
            prompt_prefix = f"{prompt_prefix_base}{history_block}Context:\n"
            prompt_suffix = f"\n\nQuestion: {question}"

            # HuggingFace/Ollama 모델에서 입력 길이 초과 방지
            if (is_huggingface(self._param.whatmodelItem) or self._param.whatmodelItem == "ollama") and self._tokenizer and self._max_input_tokens:
                generation_buffer = 1000 + 128 # 1000 for generation + 128 safety
                max_input_tokens = max(self._max_input_tokens - generation_buffer, 256)
                prompt = prompt_prefix + combined_context + prompt_suffix
                prompt_tokens = self._tokenizer.encode(prompt, add_special_tokens=False)
                OpLog(
                    f"프롬프트 토큰 수: {len(prompt_tokens)}, 최대 허용: {max_input_tokens}",
                    level="INFO",
                )
                if len(prompt_tokens) > max_input_tokens:
                    OpLog("토큰 수 초과 - 예산 기반 Context 구성 시작", level="WARN")
                    context_tokens = self._tokenizer.encode(combined_context, add_special_tokens=False)
                    prefix_tokens = self._tokenizer.encode(prompt_prefix, add_special_tokens=False)
                    suffix_tokens = self._tokenizer.encode(prompt_suffix, add_special_tokens=False)
                    max_available_tokens = max(
                        max_input_tokens - len(prefix_tokens) - len(suffix_tokens),
                        0,
                    )
                    budgeted_history_block = self._history_manager.build_budgeted_block(
                        self._tokenizer,
                        max_available_tokens,
                    )
                    prompt_prefix = f"{prompt_prefix_base}{budgeted_history_block}Context:\n"
                    prefix_tokens = self._tokenizer.encode(prompt_prefix, add_special_tokens=False)
                    max_context_tokens = max(
                        max_input_tokens - len(prefix_tokens) - len(suffix_tokens),
                        0,
                    )
                    truncated_context = build_budgeted_context(
                        sql_results,
                        context_result,
                        max_context_tokens,
                        self._tokenizer,
                    )
                    truncated_tokens = self._tokenizer.encode(
                        truncated_context,
                        add_special_tokens=False,
                    )
                    OpLog(
                        f"Context 토큰 {len(context_tokens)} -> {len(truncated_tokens)}로 축소",
                        level="INFO",
                    )
                    prompt = prompt_prefix + truncated_context + prompt_suffix
                    combined_context = truncated_context
                    self._last_context = truncated_context
                else:
                    OpLog("토큰 수 정상 - 잘라내기 불필요", level="INFO")
            else:
                prompt = prompt_prefix + combined_context + prompt_suffix
            OpLog(f"LLM에 전송할 최종 프롬프트 길이: {len(prompt)} 문자", level="INFO")
            Lines(f"[DEBUG] 최종 프롬프트 미리보기 (처음 600자):\n{'='*80}\n{prompt[:600]}...\n{'='*80}")
            response = self._llm.invoke([HumanMessage(content=prompt)])
            OpLog(f"LLM 응답 받음: {len(response.content)} 문자", level="INFO")
            OpLog("RAG 검색 완료", level="INFO")

            self._history_manager.append("user", question)
            self._history_manager.append("assistant", response.content)

            if is_core_query and project_names_text:
                return f"{response.content}\n\n해당 사업명:\n{project_names_text}"
            if (not is_core_query) and meta_list_text:
                return f"{meta_list_text}\n\n{response.content}"
            return response.content
        except Exception as e:
            Lines(f"[ERROR] 검색 중 오류: {str(e)}")
            OpLog(f"검색 오류: {str(e)}", level="ERROR")
            import traceback
            Lines(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            self._last_context = ""
            return f"검색 중 오류가 발생했습니다: {str(e)}"


# ════════════════════════════════════════
# ▣ OpenAI 및 HuggingFace 모델 클래스
# ════════════════════════════════════════
class OpenAIModel(BaseLLMModel):
    def __init__(self, param: PARAMVAR):
        super().__init__(param)
        self._processor = BidMatePreprocessor(self._param)
        self._vector_store = self.get_vector_store(self._param.newCreate)
        self._param.whatmodelItem = "openai"
        
    def get_vector_store(self, newCreate: bool):
        faiss_name = self._processor.make_faiss_name()
        if not newCreate:
            vector_store = self._processor._check_vector_store_exists(faiss_name)
            if vector_store is not None:
                return vector_store
        OpLog(f"Vector DB 새로 생성: {faiss_name}", level="INFO")
        return self._processor.get_openai_vector_store(faiss_name)
     
    def make_model(self):
        OpLog(f"OpenAI Make Model :: embedding_model_name:{self._param.embedding_model},llm_model:{self._param.llm_model},model_name:{self._my_name},temperature:{self._param.temperature},repetition_penalty:{self._param.repetition_penalty}", level="INFO")
        self._llm = ChatOpenAI(model=self._param.llm_model)
        
class HugginFaceModel(BaseLLMModel):
    def __init__(self,param:PARAMVAR):
        super().__init__(param)
        self._param.whatmodelItem = "huggingface"
        self._processor = BidMatePreprocessor(param)
        self._vector_store = self.get_vector_store(self._param.newCreate)

    def get_vector_store(self, newCreate: bool):
        faiss_name = self._processor.make_faiss_name()
        if not newCreate:
            vector_store = self._processor._check_vector_store_exists(faiss_name)
            if vector_store is not None:
                return vector_store
        OpLog(f"Vector DB 새로 생성: {faiss_name}", level="INFO")
        return self._processor.get_hugging_vector_store(faiss_name)
    
    def make_model(self):
        OpLog(f"HuggingFace Make Model :: embedding_model_name:{self._param.embedding_model},llm_model:{self._param.llm_model},model_name:{self._my_name},temperature:{self._param.temperature},repetition_penalty:{self._param.repetition_penalty}", level="INFO")
        from transformers import AutoModelForCausalLM, AutoTokenizer        
        
        if self._param.is_gpu:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            device_map = "auto"
        else:
            bnb_config = None
            device_map = None

        OpLog(f"AutoModelForCausalLM 로드 시작: {self._param.llm_model}",bLines = True, level="INFO")
        model = AutoModelForCausalLM.from_pretrained(
            self._param.llm_model,
            quantization_config= bnb_config if self._param.is_gpu else None,
            device_map=device_map,
            trust_remote_code=True,
        )
        OpLog("AutoModelForCausalLM 로드 완료", level="INFO",bLines = True)
        OpLog(f"Tokenizer 로드 시작: {self._param.llm_model}", level="INFO",bLines=True)
        tokenizer = AutoTokenizer.from_pretrained(self._param.llm_model)
        OpLog("Tokenizer 로드 완료", level="INFO")
        
        # Tokenizer 설정 (SQL DB 검색 및 토큰 길이 제한에 사용)
        self._tokenizer = tokenizer
        model_max_length = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and model_max_length < 100000:
            self._max_input_tokens = min(self._param.max_input_tokens_hf, model_max_length)
        else:
            self._max_input_tokens = self._param.max_input_tokens_hf
        
        from transformers import pipeline
        Lines("Make LLM pipeline")
        OpLog("LLM Pipeline 생성 시작", level="INFO")
        llm_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=self._param.temperature,
            repetition_penalty=self._param.repetition_penalty,
            return_full_text=False,
            max_new_tokens=1000,
        )
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        self._llm = ChatHuggingFace(llm=llm)
        OpLog("HuggingFace LLM 초기화 완료", level="INFO")


class OllamaModel(BaseLLMModel):
    def __init__(self, param: PARAMVAR):
        super().__init__(param)
        self._param.whatmodelItem = "ollama"
        self._processor = BidMatePreprocessor(self._param)
        self._vector_store = self.get_vector_store(self._param.newCreate)
        # Ollama 입력 토큰 제한 설정
        self._max_input_tokens = getattr(self._param, "max_input_tokens_ollama", 4096)
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._param.llm_model)
        except Exception:
            self._tokenizer = None

    def get_vector_store(self, newCreate: bool):
        faiss_name = self._processor.make_faiss_name()
        if not newCreate:
            vector_store = self._processor._check_vector_store_exists(faiss_name)
            if vector_store is not None:
                return vector_store
        OpLog(f"Vector DB 새로 생성: {faiss_name}", level="INFO")
        return self._processor.get_ollama_vector_store(faiss_name)

    def make_model(self):
        model_name = self._param.ollama_model_name or self._param.llm_model
        OpLog(
            "Ollama Make Model :: "
            f"embedding_model_name:{self._param.embedding_model},"
            f"llm_model:{model_name},"
            f"model_name:{self._my_name},"
            f"temperature:{self._param.temperature}",
            level="INFO",
        )
        self._llm = ChatOllama(
            base_url=self._param.ollama_address,
            model=model_name,
            temperature=self._param.temperature,
        )

def Execute_Model(param: PARAMVAR):
    # 대화형 RAG 모델 실행
    param.whatmodelItem = normalize_model_item(param.whatmodelItem)
    param.reranker_type = normalize_reranker_type(param.reranker_type)
    if not hasattr(param, "execute_index"):
        param.execute_index = 0
    
    # 모델 초기화
    model_item = normalize_model_item(param.whatmodelItem)
    if model_item == "openai":
        model = OpenAIModel(param)
    elif model_item == "ollama":
        model = OllamaModel(param)
    else:
        model = HugginFaceModel(param)
    model.make_model()
    OpLog(f"모델 초기화 완료: {type(model).__name__}", level="INFO")
    
    # 대화형 루프
    try:
        while True: 
            query = input("질문을 입력하세요 (종료하려면 Ctrl+C): ")
            OpLog("질문 입력됨", level="INFO")
            answer = model.rag_search(query)
            print("답변:")
            Lines(answer)
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        OpLog("대화형 실행 종료", level="INFO")
        model.clear_mem()

## 평가 실행 함수 (모든 조합)
def Execute_eval( param: PARAMVAR):
    param.reranker_type = normalize_reranker_type(param.reranker_type)
    model_item = normalize_model_item(param.whatmodelItem)
    if model_item == "openai":
        model = OpenAIModel(param)
    elif model_item == "ollama":
        model = OllamaModel(param)
    else:
        model = HugginFaceModel(param)
    model.make_model()
    OpLog(f"모델 초기화 완료: {type(model).__name__}", level="INFO")
    
    # 질의응답 평가 수행
    OpLog("평가 실행 시작", level="INFO")
    model.eval_queries_answers()
    OpLog(
        f"평가 실행 완료 - 실험 파라미터: "
        f"whatmodelItem={param.whatmodelItem}, "
        f"temperature={param.temperature}, repetition_penalty={param.repetition_penalty}, "
        f"chunk_size={param.chunk_size}, chunk_overlap={param.chunk_overlap}, k_values={param.k}",
        level="INFO",
        bLines=True,
    )
    model.clear_mem()


## 평가 실행 함수 (모든 조합)
def Execute_eval_optimize( param: PARAMVAR):
    param.reranker_type = normalize_reranker_type(param.reranker_type)
    model_item = normalize_model_item(param.whatmodelItem)
    if model_item == "openai":
        model = OpenAIModel(param)
    elif model_item == "ollama":
        model = OllamaModel(param)
    else:
        model = HugginFaceModel(param)
    model.make_model()
    OpLog(f"모델 초기화 완료: {type(model).__name__}", level="INFO")
    
    # 질의응답 평가 수행
    OpLog("평가 실행 시작", level="INFO")
    model.eval_queries_answers()
    OpLog(
        f"평가 실행 완료 - 실험 파라미터: "
        f"whatmodelItem={param.whatmodelItem}, "
        f"temperature={param.temperature}, repetition_penalty={param.repetition_penalty}, "
        f"chunk_size={param.chunk_size}, chunk_overlap={param.chunk_overlap}, k_values={param.k}",
        level="INFO",
        bLines=True,
    )
    model.clear_mem()



## 평가 실행 함수 (retriever_types × reranker_types 모든 조합)
def Execute_evalEx( param: PARAMVAR):
    # 대화형 RAG 모델 실행
    param.reranker_type = normalize_reranker_type(param.reranker_type)
    model_item = normalize_model_item(param.whatmodelItem)
    if model_item == "openai":
        model = OpenAIModel(param)
    elif model_item == "ollama":
        model = OllamaModel(param)
    else:
        model = HugginFaceModel(param)
    model.make_model()
    OpLog(f"모델 초기화 완료: {type(model).__name__}", level="INFO")
    
    # 질의응답 평가 수행
    OpLog("평가 실행 시작", level="INFO")
    model.eval_queries_answers_Ex()
    OpLog(
        f"평가 실행 완료 - 실험 파라미터: "
        f"whatmodelItem={param.whatmodelItem}, "
        f"temperature={param.temperature}, repetition_penalty={param.repetition_penalty}, "
        f"chunk_size={param.chunk_size}, chunk_overlap={param.chunk_overlap}, k_values={param.k}",
        level="INFO",
        bLines=True,
    )

    model.clear_mem()

# 	"0.2|1.2"

if __name__ == "__main__":
    param = PARAMVAR()
    param.execute_index = 0
    param.whatmodelItem = "openai"
    param.chunk_size = 500
    param.chunk_overlap = 50
    param.temperature = 0.2
    param.repetition_penalty = 1.2
    param.newCreate =  False
    param.k = 10 
    param.max_input_tokens_hf = 4096
    param.retriever_llm_type = "openai"
    model_item = normalize_model_item(param.whatmodelItem)
    if model_item == "openai":
        param.embedding_model = OPENAI_SETTINGS["embedding_model_name"]
        param.llm_model = OPENAI_SETTINGS["llm_model_name"]
    elif model_item == "ollama":
        param.embedding_model = param.ollama_embedding_model_name
        param.llm_model = param.ollama_model_name
    else:
        param.embedding_model ="nlpai-lab/KoE5" # "BAAI/bge-m3"
        param.llm_model = "nlpai-lab/KULLM3"
    Execute_Model(param)
    