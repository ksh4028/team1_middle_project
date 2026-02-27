"""
RAG 기반 Q&A 시스템

FAISS 벡터화된 RFP 문서를 활용한 질의응답 시스템
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv가 없어도 환경변수로 직접 설정 가능


def get_base_dir() -> Path:
    """프로젝트 루트 경로 반환"""
    # 스크립트 파일의 위치 기준으로 프로젝트 루트 찾기
    script_path = Path(__file__).resolve()
    # src/rag_qa_system.py -> team1_middle_project/
    if script_path.parent.name == "src":
        return script_path.parent.parent
    # 현재 작업 디렉토리 확인
    _cwd = Path.cwd()
    if (_cwd / "src").exists() or (_cwd / "data").exists():
        return _cwd
    return _cwd.parent


def load_faiss_index(vector_store_dir: Path) -> tuple[faiss.Index, List[Dict]]:
    """FAISS 인덱스 및 메타데이터 로드 (Windows 한글 경로 문제 해결)"""
    vector_store_dir = Path(vector_store_dir)
    index_path = vector_store_dir / "faiss_index.index"
    metadata_path = vector_store_dir / "metadata.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
    
    # FAISS 인덱스 로드
    if sys.platform == "win32" and any(ord(c) > 127 for c in str(index_path)):
        # Windows 한글 경로: 임시 영문 경로로 복사 후 로드
        temp_dir = Path(tempfile.gettempdir()) / "faiss_temp"
        temp_dir.mkdir(exist_ok=True)
        temp_index_path = temp_dir / "faiss_index_load.index"
        
        shutil.copy2(index_path, temp_index_path)
        index = faiss.read_index(str(temp_index_path))
    else:
        # Linux 또는 영문 경로: 직접 로드
        index = faiss.read_index(str(index_path))
    
    # 메타데이터 로드
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    return index, metadata


def search_similar_documents(
    query: str,
    index: faiss.Index,
    metadata: List[Dict],
    embedding_model: SentenceTransformer,
    k: int = 5
) -> List[Dict]:
    """
    쿼리와 유사한 문서 검색
    
    Args:
        query: 검색 쿼리
        index: FAISS 인덱스
        metadata: 청크 메타데이터 리스트
        embedding_model: 임베딩 모델
        k: 반환할 상위 k개
    
    Returns:
        유사 문서 리스트 (거리, 텍스트, 메타데이터 포함)
    """
    # 쿼리 임베딩 생성
    query_embedding = embedding_model.encode([query])
    query_vector = np.array(query_embedding).astype("float32")
    
    # FAISS 검색
    distances, indices = index.search(query_vector, k)
    
    # 결과 구성
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = metadata[idx]
        results.append({
            "distance": float(dist),
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "file_path": chunk["file_path"]
        })
    
    return results


def rag_qa(
    question: str,
    index: faiss.Index,
    metadata: List[Dict],
    embedding_model: SentenceTransformer,
    llm: ChatOpenAI,
    top_k: int = 5,
    max_context_length: int = 3000
) -> Dict:
    """
    RAG 기반 질의응답
    
    Args:
        question: 질문
        index: FAISS 인덱스
        metadata: 청크 메타데이터
        embedding_model: 임베딩 모델
        llm: LLM 모델
        top_k: 검색할 상위 k개 문서
        max_context_length: 컨텍스트 최대 길이 (문자 수)
    
    Returns:
        답변 및 참조 문서 정보
    """
    # 1. 유사 문서 검색
    similar_docs = search_similar_documents(
        question, index, metadata, embedding_model, k=top_k
    )
    
    # 2. 컨텍스트 구성 (길이 제한)
    context_parts = []
    current_length = 0
    
    for doc in similar_docs:
        doc_text = f"[파일: {doc['source']}, 페이지: {doc['page']}]\n{doc['text']}"
        if current_length + len(doc_text) > max_context_length:
            break
        context_parts.append(doc_text)
        current_length += len(doc_text)
    
    context = "\n\n".join(context_parts)
    
    # 3. 프롬프트 구성
    prompt = f"""다음은 기업 및 정부 제안요청서(RFP) 문서의 일부입니다.

문서 내용:
{context}

위 문서 내용을 바탕으로 다음 질문에 답변해주세요. 문서에 명시되지 않은 내용은 추측하지 말고, 문서에 있는 정보만 사용하여 답변하세요.

질문: {question}

답변:"""
    
    # 4. LLM 호출
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    # 5. 결과 반환
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "file": doc["source"],
                "page": doc["page"],
                "distance": doc["distance"],
                "text_preview": doc["text"][:200] + "..."
            }
            for doc in similar_docs[:top_k]
        ],
        "context_length": len(context)
    }


def main(
    question: str,
    vector_store_dir: Path | None = None,
    embedding_model_name: str = "jhgan/ko-sroberta-multitask",
    llm_model: str = "gpt-4o-mini",
    top_k: int = 5,
    index: faiss.Index | None = None,
    metadata: List[Dict] | None = None,
    embedding_model: SentenceTransformer | None = None
):
    """
    RAG Q&A 실행
    
    Args:
        question: 질문
        vector_store_dir: 벡터 스토어 디렉토리 (None이면 기본 경로 사용)
        embedding_model_name: 임베딩 모델명
        llm_model: LLM 모델명
        top_k: 검색할 상위 k개 문서
    """
    BASE_DIR = get_base_dir()
    
    # 경로 설정
    if vector_store_dir is None:
        vector_store_dir = BASE_DIR / "data" / "processed" / "vector_store"
    
    vector_store_dir = Path(vector_store_dir)
    index_path = vector_store_dir / "faiss_index.index"
    metadata_path = vector_store_dir / "metadata.json"
    
    # FAISS 인덱스 로드 (없으면 새로 로드)
    if index is None:
        print("FAISS 인덱스 로딩 중...")
        index, metadata_result = load_faiss_index(vector_store_dir)
        print(f"인덱스 로드 완료: {index.ntotal}개 벡터")
    else:
        if metadata is None:
            # 메타데이터만 로드
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        metadata_result = metadata
    
    faiss_index = index
    chunks_metadata = metadata_result
    
    if metadata is None:
        print(f"메타데이터 로드 완료: {len(chunks_metadata)}개 청크")
    
    # 임베딩 모델 로드 (없으면 새로 로드)
    if embedding_model is None:
        print(f"\n임베딩 모델 로딩: {embedding_model_name}...")
        embedding_model = SentenceTransformer(embedding_model_name)
    
    # LLM 초기화
    print(f"\nLLM 초기화: {llm_model}")
    llm = ChatOpenAI(
        model_name=llm_model,
        temperature=0.0,
        max_tokens=1000
    )
    
    # 질의응답 실행
    print(f"\n질문: {question}")
    print("검색 및 답변 생성 중...\n")
    
    result = rag_qa(
        question=question,
        index=faiss_index,
        metadata=chunks_metadata,
        embedding_model=embedding_model,
        llm=llm,
        top_k=top_k
    )
    
    # 결과 출력
    print("=" * 60)
    print("답변:")
    print(result["answer"])
    print("\n" + "=" * 60)
    print("\n참조 문서:")
    for i, source in enumerate(result["sources"], 1):
        print(f"\n[{i}] {source['file']} (페이지 {source['page']}, 거리: {source['distance']:.4f})")
        print(f"    미리보기: {source['text_preview']}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 기반 Q&A 시스템")
    parser.add_argument("question", type=str, nargs="?", help="질문 (생략 시 대화형 모드)")
    parser.add_argument("--vector-store-dir", type=str, help="벡터 스토어 디렉토리")
    parser.add_argument("--embedding-model", type=str, default="jhgan/ko-sroberta-multitask", help="임베딩 모델명")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM 모델명")
    parser.add_argument("--top-k", type=int, default=5, help="검색할 상위 k개 문서")
    parser.add_argument("--interactive", "-i", action="store_true", help="대화형 모드로 실행")
    
    args = parser.parse_args()
    
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("환경변수 설정: export OPENAI_API_KEY='your_key'")
        sys.exit(1)
    
    # 벡터 스토어 및 임베딩 모델 초기화 (한 번만 로드)
    BASE_DIR = get_base_dir()
    vector_store_dir = Path(args.vector_store_dir) if args.vector_store_dir else BASE_DIR / "data" / "processed" / "vector_store"
    
    print("FAISS 인덱스 및 임베딩 모델 로딩 중...")
    index, metadata = load_faiss_index(vector_store_dir)
    embedding_model = SentenceTransformer(args.embedding_model)
    print("로딩 완료!\n")
    
    # 질문이 제공되었거나 대화형 모드가 아닌 경우
    if args.question:
        main(
            question=args.question,
            vector_store_dir=vector_store_dir,
            embedding_model_name=args.embedding_model,
            llm_model=args.llm_model,
            top_k=args.top_k,
            index=index,
            metadata=metadata,
            embedding_model=embedding_model
        )
    else:
        # 대화형 모드
        print("=" * 60)
        print("RAG 기반 Q&A 시스템 (대화형 모드)")
        print("종료하려면 'quit', 'exit', 'q'를 입력하세요.")
        print("=" * 60)
        print()
        
        while True:
            try:
                question = input("\n질문을 입력하세요: ").strip()
                
                if question.lower() in ["quit", "exit", "q", ""]:
                    print("\n종료합니다.")
                    break
                
                if not question:
                    continue
                
                print("\n답변 생성 중...\n")
                main(
                    question=question,
                    vector_store_dir=vector_store_dir,
                    embedding_model_name=args.embedding_model,
                    llm_model=args.llm_model,
                    top_k=args.top_k,
                    index=index,
                    metadata=metadata,
                    embedding_model=embedding_model
                )
            except KeyboardInterrupt:
                print("\n\n종료합니다.")
                break
            except Exception as e:
                print(f"\n오류 발생: {e}")
                import traceback
                traceback.print_exc()
