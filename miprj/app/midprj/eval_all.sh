#!/usr/bin/env bash

# 최적의 성능을 위한 RAG 파라미터 튜닝 버전
# execute_index 400번대에 실험 데이터를 축적합니다.

set -euo pipefail

execute_index=400
python3 util_sqlite.py
python3 util_del_executeindex.py --execute_index "$execute_index"

# ==========================================================
# 1. OpenAI (gpt-5-mini) - "The Precision Sniper"
# ==========================================================
# 특징: 추론 능력이 매우 뛰어나므로 노이즈가 없는 정제된 정보 전달이 핵심
# ----------------------------------------------------------
whatmodelItem="openai"
embedding_model="text-embedding-3-small"
llm_model="gpt-5-mini"
temperature="0.0"          # 완전한 사실 기반 (0.1보다 더 엄격하게 설정 가능)
repetition_penalty="1.1"   # 답변의 다양성보다 정확성 강조
chunk_size="400"           # 300보다 약간 늘려 의미가 끊기지 않게 조정
chunk_overlap="40"         # 청크 간 연결성 확보
k="4"                      # 상위 4개 문서만으로도 충분히 답변 가능
llm_sel="openai"
reranker="bge"             # BGE-Reranker로 가짜 유사 문서 사전 차단
retriever="hybrid_llm"     # Multi-Query + Hybrid로 검색 누락 방지
do_eval="all"

echo ">>> OpenAI 최적화 모드 실행 중..."
python3 call_midprj_eval.py \
    --execute_index "$execute_index" \
    --whatmodelItem "$whatmodelItem" \
    --embedding_model "$embedding_model" \
    --llm_model "$llm_model" \
    --temperature "$temperature" \
    --repetition_penalty "$repetition_penalty" \
    --chunk_size "$chunk_size" \
    --chunk_overlap "$chunk_overlap" \
    --k "$k" \
    --llm_selector "$llm_sel" \
    --reranker_type "$reranker" \
    --do_retriever "$retriever" \
    --do_eval "$do_eval"

# ==========================================================
# 2. Ollama (exaone3.5:7.8b) - "The Context Weaver"
# ==========================================================
# 특징: 문맥 보존이 중요하며, 많은 정보를 주되 Reranker로 질서를 잡아줘야 함
# ----------------------------------------------------------
whatmodelItem="ollama"
embedding_model="bge-m3"
llm_model="exaone3.5:7.8b"
temperature="0.2"          # 문장 흐름을 위해 약간의 유연성 부여
repetition_penalty="1.05"  # 한국어 답변에서 문장이 딱딱해지는 것 방지
chunk_size="1000"          # 1200보다 살짝 줄여 컨텍스트 부하 최적화
chunk_overlap="150"        # 충분한 오버랩으로 문맥 단절 방지
k="13"                     # 풍부한 정보량 제공
llm_sel="ollama"
reranker="bge"             # 필수! 많은 정보 중 핵심만 LLM 상단에 배치
retriever="hybrid_llm"     # 키워드(BM25)와 의미(Vector)를 모두 잡음
do_eval="all"

echo ">>> Ollama 최적화 모드 실행 중..."
python3 call_midprj_eval.py \
    --execute_index "$execute_index" \
    --whatmodelItem "$whatmodelItem" \
    --embedding_model "$embedding_model" \
    --llm_model "$llm_model" \
    --temperature "$temperature" \
    --repetition_penalty "$repetition_penalty" \
    --chunk_size "$chunk_size" \
    --chunk_overlap "$chunk_overlap" \
    --k "$k" \
    --llm_selector "$llm_sel" \
    --reranker_type "$reranker" \
    --do_retriever "$retriever" \
    --do_eval "$do_eval"

# 결과 분석
python3 midprj_eval.py --execute_index "$execute_index"