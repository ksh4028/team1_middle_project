#!/usr/bin/env bash

# 최적의 성능을 위한 RAG 파라미터 튜닝 버전
# execute_index 500번대에 실험 데이터를 축적합니다.

set -euo pipefail

execute_index=500
python3 util_sqlite.py
python3 util_del_executeindex.py --execute_index "$execute_index"

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
