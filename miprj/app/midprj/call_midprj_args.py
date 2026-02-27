# -*- coding: utf-8 -*-
# ════════════════════════════════════════
# ▣ CLI 공통 인자 정의 및 PARAMVAR 빌드
# ════════════════════════════════════════
# call_midprj.py, call_midprj_eval.py 등에서 공통으로 사용하는
# argparse 정의와 PARAMVAR 매핑을 하나로 통합합니다.

import argparse
from . import midprj_main as prj
from .midprj_defines import OPENAI_SETTINGS, PARAMVAR


def build_argument_parser(description: str = "MidPrj RAG 실행 스크립트") -> argparse.ArgumentParser:
    """모든 CLI 스크립트에서 공유하는 argparse 파서를 생성합니다."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--llm_selector", type=str, default=None, help="LLM selector override (openai/ollama, 기본: None, ini 설정 우선)")

    # 실행 설정
    parser.add_argument("--execute_index", type=int, default=0, help="실행 인덱스 (기본: 0)")

    # 임베딩/LLM 모델
    parser.add_argument("--embedding_model", type=str, default="nlpai-lab/KoE5", help="임베딩 모델 (기본: nlpai-lab/KoE5)")
    parser.add_argument("--llm_model", type=str, default="nlpai-lab/KULLM3", help="LLM 모델 (기본: nlpai-lab/KULLM3)")

    # 청크/토큰 설정
    parser.add_argument("--chunk_size", type=int, default=1000, help="청크 크기 (기본: 1000)")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="청크 오버랩 (기본: 100)")

    # 모델 파라미터
    parser.add_argument("--temperature", type=float, default=0.7, help="온도 값 (기본: 0.7)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="반복 패널티 (기본: 1.2)")
    parser.add_argument("--k", type=int, default=5, help="검색 개수 k (기본: 5)")

    # 모델 종류 및 DB 생성
    parser.add_argument("--whatmodelItem", type=str, default="huggingface", help="모델 종류 (openai/huggingface/ollama, 기본: huggingface)")
    parser.add_argument("--newCreate", type=str, default="False", help="새로운 DB 생성 여부 (기본: False, True/False)")

    # HuggingFace/Adobe 설정
    parser.add_argument("--hf_max_input_tokens", type=int, default=4096, help="HuggingFace 최대 입력 토큰 (기본: 4096)")


    # 리랭커
    parser.add_argument("--retriever_llm_type", type=str, default="openai", help="retriever LLM 타입 (기본: openai)")
    parser.add_argument("--reranker_type", type=str, default="flashrank", help="리랭커 타입 (기본: flashrank, 옵션: flashrank/openai)")

    # 입출력 경로
    parser.add_argument("--csv_path", type=str, default=prj.CSV_PATH, help="CSV 경로 (기본: 프로젝트 설정)")
    parser.add_argument("--rfp_data_dir", type=str, default=prj.RFP_DATA_DIR, help="RFP 데이터 디렉토리 (기본: 프로젝트 설정)")
    parser.add_argument("--is_gpu", type=str, default=str(prj.IS_GPU), help="GPU 사용 여부 (기본: 자동, True/False)")

    # 질의/시간
    parser.add_argument("--query", type=str, default="", help="초기 질문 텍스트 (기본: 빈 문자열)")
    parser.add_argument("--answer", type=str, default="", help="초기 답변 텍스트 (기본: 빈 문자열)")
    parser.add_argument("--start_time", type=str, default="2000-01-01 00:00:00", help="시작 시간 (기본: 2000-01-01 00:00:00)")
    parser.add_argument("--end_time", type=str, default="2001-01-01 00:00:00", help="종료 시간 (기본: 2001-01-01 00:00:00)")

    # do_retriever, do_eval 인자 추가
    parser.add_argument("--do_retriever", type=str, default="hybrid", help="retriever 방식 (vector/hybrid/hybrid_llm, 기본: hybrid)")
    parser.add_argument("--do_eval", type=str, default="optimize", help="평가 방식 (기본: optimize)")

    return parser


def build_param_from_args(args) -> PARAMVAR:
    """파싱된 args를 PARAMVAR 인스턴스로 매핑합니다."""
    DEFAULT_HF_EMBEDDING = "nlpai-lab/KoE5"
    DEFAULT_HF_LLM = "nlpai-lab/KULLM3"

    param = PARAMVAR()
    param.whatmodelItem = args.whatmodelItem.lower()
    param.execute_index = args.execute_index
    param.chunk_size = args.chunk_size
    param.chunk_overlap = args.chunk_overlap
    param.temperature = args.temperature
    param.repetition_penalty = args.repetition_penalty
    param.k = args.k
    param.newCreate = args.newCreate.lower() == "true"

    # OpenAI 선택 시 기본 모델을 OPENAI 모델로 교체
    if param.whatmodelItem == "openai":
        if args.embedding_model == DEFAULT_HF_EMBEDDING:
            args.embedding_model = OPENAI_SETTINGS["embedding_model_name"]
        if args.llm_model == DEFAULT_HF_LLM:
            args.llm_model = OPENAI_SETTINGS["llm_model_name"]

    param.embedding_model = args.embedding_model
    param.llm_model = args.llm_model
    param.max_input_tokens_hf = args.hf_max_input_tokens
    param.retriever_llm_type = args.retriever_llm_type
    param.reranker_type = prj.normalize_reranker_type(args.reranker_type)
    param.csv_path = args.csv_path
    param.rfp_data_dir = args.rfp_data_dir
    param.query = args.query
    param.answer = args.answer
    param.start_time = args.start_time
    param.end_time = args.end_time

    # is_gpu가 있으면 적용
    if hasattr(args, "is_gpu"):
        param.is_gpu = args.is_gpu.lower() == "true"

    # llm_selector 인자 적용
    if hasattr(args, "llm_selector") and args.llm_selector:
        param.llm_selector = args.llm_selector.lower()

    # do_retriever, do_eval 인자 적용
    if hasattr(args, "do_retriever"):
        param.do_retriever = args.do_retriever
    if hasattr(args, "do_eval"):
        param.do_eval = args.do_eval

    return param
