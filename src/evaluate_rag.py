"""
RAG 시스템 성능 평가

검색(Retrieval) 및 생성(Generation) 성능을 평가합니다.
- 검색: MRR, Recall@K, Precision@K (LLM-as-Judge로 관련 문서 판정)
- 생성: Faithfulness, Relevance (LLM-as-Judge)
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import re

import faiss
import numpy as np
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# src 디렉토리를 path에 추가 (프로젝트 루트에서 실행 시)
_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# rag_qa_system 모듈에서 함수 import
from rag_qa_system import (
    load_faiss_index,
    search_similar_documents,
    rag_qa,
)


def get_base_dir_eval() -> Path:
    """프로젝트 루트 경로 반환"""
    script_path = Path(__file__).resolve()
    if script_path.parent.name == "src":
        return script_path.parent.parent
    _cwd = Path.cwd()
    if (_cwd / "src").exists() or (_cwd / "data").exists():
        return _cwd
    return _cwd.parent


def load_questions(questions_path: Path) -> List[str]:
    """질문 파일 로드 (JSON 또는 TXT)"""
    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"질문 파일을 찾을 수 없습니다: {path}")

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # generate_questions.py 출력 형식: {"questions": [...], "questions_by_document": {...}}
        if isinstance(data, dict) and "questions" in data:
            raw = data["questions"]
            return [q if isinstance(q, str) else q.get("question", str(q)) for q in raw]
        if isinstance(data, list):
            return [q if isinstance(q, str) else q.get("question", str(q)) for q in data]
        return []
    elif path.suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        questions = []
        for line in lines:
            # "1. 질문내용" 형식 제거
            m = re.match(r"^\d+\.\s*(.+)$", line)
            questions.append(m.group(1) if m else line)
        return questions
    else:
        raise ValueError("지원 형식: .json, .txt")


def judge_chunk_relevance(
    question: str,
    chunk_text: str,
    llm: ChatOpenAI,
) -> bool:
    """LLM으로 청크의 질문 관련성 판정"""
    chunk_preview = chunk_text[:800] + ("..." if len(chunk_text) > 800 else "")
    prompt = f"""다음 질문에 답하는 데 아래 문서 청크가 도움이 됩니까?

질문: {question}

문서 청크:
{chunk_preview}

도움이 되면 '예', 아니면 '아니오'만 출력하세요."""

    try:
        response = llm.invoke(prompt)
        ans = (response.content if hasattr(response, "content") else str(response)).strip()
        return "예" in ans or "yes" in ans.lower() or "1" in ans
    except Exception:
        return False


def judge_faithfulness(
    context: str,
    answer: str,
    llm: ChatOpenAI,
) -> float:
    """답변이 컨텍스트에 충실한지 1~5 점수"""
    context_preview = context[:2000] + ("..." if len(context) > 2000 else "")
    prompt = f"""다음 문서 내용을 바탕으로 생성된 답변이 문서에 근거해 있습니까?

문서 내용:
{context_preview}

생성된 답변:
{answer}

1~5 점수로 평가하세요. (1=전혀 무관, 5=완전히 문서 기반)
숫자만 출력하세요."""

    try:
        response = llm.invoke(prompt)
        txt = (response.content if hasattr(response, "content") else str(response)).strip()
        score = float(re.search(r"[1-5]", txt).group()) if re.search(r"[1-5]", txt) else 3.0
        return max(1.0, min(5.0, score))
    except Exception:
        return 3.0


def judge_relevance(question: str, answer: str, llm: ChatOpenAI) -> float:
    """답변이 질문에 적절한지 1~5 점수"""
    prompt = f"""다음 질문에 대한 답변이 적절합니까?

질문: {question}

답변:
{answer}

1~5 점수로 평가하세요. (1=전혀 관련 없음, 5=매우 적절)
숫자만 출력하세요."""

    try:
        response = llm.invoke(prompt)
        txt = (response.content if hasattr(response, "content") else str(response)).strip()
        score = float(re.search(r"[1-5]", txt).group()) if re.search(r"[1-5]", txt) else 3.0
        return max(1.0, min(5.0, score))
    except Exception:
        return 3.0


def evaluate_retrieval(
    question: str,
    retrieved_docs: List[Dict],
    llm: ChatOpenAI,
    k: int = 5,
) -> Dict:
    """검색 성능 평가 (MRR, Recall@K, Precision@K)"""
    if not retrieved_docs:
        return {"mrr": 0.0, f"recall@{k}": 0.0, f"precision@{k}": 0.0}

    # 상위 k개만 판정 (API 호출 절약)
    to_judge = retrieved_docs[:k]
    relevant_ranks = []

    for rank, doc in enumerate(to_judge, 1):
        if judge_chunk_relevance(question, doc["text"], llm):
            relevant_ranks.append(rank)

    # MRR: 첫 관련 문서 순위의 역수
    mrr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0

    # Precision@K: 상위 K개 중 관련 문서 비율
    precision_at_k = len(relevant_ranks) / k

    # Recall@K: 상위 K개 내 관련 문서 수 / (여기서는 상위 K개만 판정하므로 precision과 동일)
    recall_at_k = precision_at_k

    return {
        "mrr": mrr,
        f"recall@{k}": recall_at_k,
        f"precision@{k}": precision_at_k,
        "num_relevant": len(relevant_ranks),
    }


def evaluate_generation(
    question: str,
    answer: str,
    context: str,
    llm: ChatOpenAI,
) -> Dict:
    """생성 성능 평가 (Faithfulness, Relevance)"""
    faithfulness = judge_faithfulness(context, answer, llm)
    relevance = judge_relevance(question, answer, llm)
    return {"faithfulness": faithfulness, "relevance": relevance}


def main(
    questions_path: Optional[Path] = None,
    vector_store_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    embedding_model_name: str = "jhgan/ko-sroberta-multitask",
    llm_model: str = "gpt-4o-mini",
    top_k: int = 5,
    retrieval_k: int = 5,
    max_questions: Optional[int] = None,
):
    """RAG 성능 평가 실행"""
    BASE_DIR = get_base_dir_eval()

    # 경로 설정
    if questions_path is None:
        questions_path = BASE_DIR / "data" / "processed" / "generated_questions.json"
    if vector_store_dir is None:
        vector_store_dir = BASE_DIR / "data" / "processed" / "vector_store"
    if output_path is None:
        output_path = BASE_DIR / "data" / "processed" / "evaluation_results.json"

    questions_path = Path(questions_path)
    vector_store_dir = Path(vector_store_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    # 질문 파일 존재 확인 (기본 경로일 때 생성 스크립트 안내)
    default_questions_path = BASE_DIR / "data" / "processed" / "generated_questions.json"
    if not questions_path.exists():
        if questions_path == default_questions_path:
            print("오류: 생성된 질문 파일을 찾을 수 없습니다.")
            print(f"  경로: {questions_path}")
            print("  먼저 질문을 생성해주세요:")
            print("    python src/generate_questions.py")
            print("  또는 --questions 옵션으로 질문 파일 경로를 지정하세요.")
        else:
            print(f"오류: 질문 파일을 찾을 수 없습니다: {questions_path}")
        sys.exit(1)

    print("=" * 60)
    print("RAG 시스템 성능 평가")
    print("=" * 60)
    print(f"질문 파일: {questions_path} (generate_questions.py 출력)")
    print(f"벡터 스토어: {vector_store_dir}")
    print(f"결과 저장: {output_path}")
    print(f"평가 질문 수: {max_questions or '전체'}")
    print()

    # 질문 로드 (generate_questions.py JSON 형식: questions 리스트)
    questions = load_questions(questions_path)
    if not questions:
        print("평가할 질문이 없습니다. 질문 파일 형식을 확인하세요.")
        sys.exit(1)

    if max_questions:
        questions = questions[:max_questions]
    print(f"평가할 질문: {len(questions)}개")

    # 모델 로드
    print("\n모델 로딩 중...")
    index, metadata = load_faiss_index(vector_store_dir)
    embedding_model = SentenceTransformer(embedding_model_name)
    llm = ChatOpenAI(model_name=llm_model, temperature=0.0, max_tokens=1000)
    print("로딩 완료.\n")

    # 평가 결과 저장
    results = []
    retrieval_scores = []
    faithfulness_scores = []
    relevance_scores = []

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] 평가 중: {question[:50]}...")

        # 검색
        retrieved = search_similar_documents(
            question, index, metadata, embedding_model, k=retrieval_k
        )

        # 검색 평가
        ret_metrics = evaluate_retrieval(question, retrieved, llm, k=retrieval_k)
        retrieval_scores.append(ret_metrics)

        # RAG 답변 생성
        context_parts = []
        current_len = 0
        max_ctx = 3000
        for doc in retrieved[:top_k]:
            t = f"[파일: {doc['source']}, 페이지: {doc['page']}]\n{doc['text']}"
            if current_len + len(t) > max_ctx:
                break
            context_parts.append(t)
            current_len += len(t)
        context = "\n\n".join(context_parts)

        rag_result = rag_qa(
            question=question,
            index=index,
            metadata=metadata,
            embedding_model=embedding_model,
            llm=llm,
            top_k=top_k,
        )
        answer = rag_result["answer"]

        # 생성 평가
        gen_metrics = evaluate_generation(question, answer, context, llm)
        faithfulness_scores.append(gen_metrics["faithfulness"])
        relevance_scores.append(gen_metrics["relevance"])

        results.append({
            "question": question,
            "answer": answer[:300] + "..." if len(answer) > 300 else answer,
            "retrieval": ret_metrics,
            "generation": gen_metrics,
        })

    # 전체 지표 집계
    avg_mrr = np.mean([r["mrr"] for r in retrieval_scores])
    avg_precision = np.mean([r[f"precision@{retrieval_k}"] for r in retrieval_scores])
    avg_faithfulness = np.mean(faithfulness_scores)
    avg_relevance = np.mean(relevance_scores)

    summary = {
        "retrieval": {
            "MRR": round(avg_mrr, 4),
            f"Precision@{retrieval_k}": round(avg_precision, 4),
            f"Recall@{retrieval_k}": round(avg_precision, 4),
        },
        "generation": {
            "Faithfulness (1-5)": round(avg_faithfulness, 2),
            "Relevance (1-5)": round(avg_relevance, 2),
        },
        "num_questions": len(questions),
    }

    # 결과 저장
    output_data = {
        "summary": summary,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 결과 출력
    print("\n" + "=" * 60)
    print("평가 결과 요약")
    print("=" * 60)
    print("\n[검색 성능]")
    print(f"  MRR: {summary['retrieval']['MRR']}")
    print(f"  Precision@{retrieval_k}: {summary['retrieval'][f'Precision@{retrieval_k}']}")
    print("\n[생성 성능]")
    print(f"  Faithfulness (1-5): {summary['generation']['Faithfulness (1-5)']}")
    print(f"  Relevance (1-5): {summary['generation']['Relevance (1-5)']}")
    print(f"\n결과 저장: {output_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 시스템 성능 평가")
    parser.add_argument("--questions", "-q", type=str, help="질문 파일 경로 (기본: data/processed/generated_questions.json, generate_questions.py 출력)")
    parser.add_argument("--vector-store-dir", type=str, help="벡터 스토어 디렉토리")
    parser.add_argument("--output", "-o", type=str, help="평가 결과 저장 경로")
    parser.add_argument("--embedding-model", type=str, default="jhgan/ko-sroberta-multitask")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=5, help="RAG에 사용할 문서 개수")
    parser.add_argument("--retrieval-k", type=int, default=5, help="검색 평가 시 판정할 문서 개수")
    parser.add_argument("--max-questions", type=int, default=None, help="평가할 최대 질문 수 (None=전체)")

    args = parser.parse_args()

    main(
        questions_path=Path(args.questions) if args.questions else None,
        vector_store_dir=Path(args.vector_store_dir) if args.vector_store_dir else None,
        output_path=Path(args.output) if args.output else None,
        embedding_model_name=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k,
        retrieval_k=args.retrieval_k,
        max_questions=args.max_questions,
    )
