"""
RFP 문서 FAISS 벡터화 스크립트

제안요청서(RFP) PDF 파일들을 FAISS로 벡터화하여 RAG 시스템 구축을 위한 인덱스 생성
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer


def get_base_dir() -> Path:
    """프로젝트 루트 경로 반환"""
    # 스크립트 파일의 위치 기준으로 프로젝트 루트 찾기
    script_path = Path(__file__).resolve()
    # src/rag_vectorization.py -> team1_middle_project/
    if script_path.parent.name == "src":
        return script_path.parent.parent
    # 현재 작업 디렉토리 확인
    _cwd = Path.cwd()
    if (_cwd / "src").exists() or (_cwd / "data").exists():
        return _cwd
    return _cwd.parent


def load_pdf_documents(pdf_dir: Path) -> List[Dict]:
    """PDF 디렉토리에서 모든 PDF 파일을 로드하고 텍스트 추출"""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    documents = []
    
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            for i, page in enumerate(pages):
                documents.append({
                    "content": page.page_content,
                    "source": pdf_path.name,
                    "page": i + 1,
                    "file_path": str(pdf_path)
                })
            print(f"[OK] {pdf_path.name}: {len(pages)}페이지")
        except Exception as e:
            print(f"[WARN] {pdf_path.name} 로드 실패: {e}")
    
    return documents


def save_faiss_index(index: faiss.Index, index_path: Path):
    """FAISS 인덱스 저장 (Windows 한글 경로 문제 해결)"""
    if sys.platform == "win32" and any(ord(c) > 127 for c in str(index_path)):
        # Windows 한글 경로: 임시 영문 경로에 저장 후 복사
        temp_dir = Path(tempfile.gettempdir()) / "faiss_temp"
        temp_dir.mkdir(exist_ok=True)
        temp_index_path = temp_dir / "faiss_index.index"
        
        print(f"임시 경로에 저장 중: {temp_index_path}")
        faiss.write_index(index, str(temp_index_path))
        
        # 원래 경로로 복사
        index_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temp_index_path, index_path)
        print(f"인덱스 저장 완료: {index_path}")
    else:
        # Linux 또는 영문 경로: 직접 저장
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        print(f"인덱스 저장 완료: {index_path}")


def main(
    pdf_dir: Path | None = None,
    vector_store_dir: Path | None = None,
    model_name: str = "jhgan/ko-sroberta-multitask",
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """
    PDF 문서를 FAISS로 벡터화
    
    Args:
        pdf_dir: PDF 파일 디렉토리 (None이면 기본 경로 사용)
        vector_store_dir: 벡터 저장 디렉토리 (None이면 기본 경로 사용)
        model_name: 임베딩 모델명
        chunk_size: 청크 크기
        chunk_overlap: 청크 간 겹치는 부분
    """
    BASE_DIR = get_base_dir()
    
    # 경로 설정
    if pdf_dir is None:
        # 기본 경로: team1_middle_project/data/raw/원본 데이터-20260206T023549Z-1-001/원본 데이터/files
        default_pdf_dir = BASE_DIR / "data" / "raw" / "원본 데이터-20260206T023549Z-1-001" / "원본 데이터" / "files"
        # 경로가 없으면 상위 디렉토리에서 찾기 시도
        if not default_pdf_dir.exists():
            # data/raw 안의 모든 디렉토리 확인
            raw_dir = BASE_DIR / "data" / "raw"
            if raw_dir.exists():
                subdirs = [d for d in raw_dir.iterdir() if d.is_dir()]
                if subdirs:
                    # 첫 번째 하위 디렉토리에서 files 폴더 찾기
                    for subdir in subdirs:
                        files_dir = subdir / "원본 데이터" / "files"
                        if files_dir.exists():
                            default_pdf_dir = files_dir
                            break
                        # 또는 직접 files 폴더가 있는지 확인
                        files_dir_alt = subdir / "files"
                        if files_dir_alt.exists():
                            default_pdf_dir = files_dir_alt
                            break
        pdf_dir = default_pdf_dir
    if vector_store_dir is None:
        vector_store_dir = BASE_DIR / "data" / "processed" / "vector_store"
    
    pdf_dir = Path(pdf_dir)
    vector_store_dir = Path(vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"PDF 디렉토리: {pdf_dir}")
    print(f"벡터 저장 디렉토리: {vector_store_dir}")
    
    # 1. 임베딩 모델 초기화
    print(f"\n임베딩 모델 로딩: {model_name}...")
    embedding_model = SentenceTransformer(model_name)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    print(f"임베딩 차원: {embedding_dim}")
    
    # 2. PDF 문서 로드
    print("\nPDF 문서 로딩 중...")
    print(f"확인 중인 경로: {pdf_dir}")
    print(f"경로 존재 여부: {pdf_dir.exists()}")
    
    if not pdf_dir.exists():
        # 대안 경로 찾기
        raw_dir = BASE_DIR / "data" / "raw"
        print(f"\n대안 경로 탐색 중: {raw_dir}")
        if raw_dir.exists():
            print(f"  - raw 디렉토리 존재: {raw_dir.exists()}")
            subdirs = [d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            print(f"  - 하위 디렉토리: {[d.name for d in subdirs]}")
        
        raise FileNotFoundError(
            f"PDF 디렉토리가 없습니다: {pdf_dir}\n"
            f"프로젝트 루트: {BASE_DIR}\n"
            f"data/raw 디렉토리에 PDF 파일이 있는 폴더를 배치하거나 --pdf-dir 옵션으로 경로를 지정해주세요."
        )
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"발견된 PDF 파일: {len(pdf_files)}개")
    if pdf_files:
        print(f"  예시: {pdf_files[0].name}")
    
    if not pdf_files:
        raise FileNotFoundError(
            f"PDF 파일을 찾을 수 없습니다: {pdf_dir}\n"
            f"해당 디렉토리에 .pdf 파일을 배치해주세요.\n"
            f"현재 디렉토리 내용: {list(pdf_dir.iterdir())[:5] if pdf_dir.exists() else '디렉토리 없음'}"
        )
    
    documents = load_pdf_documents(pdf_dir)
    print(f"총 {len(documents)}개 문서 청크 로드 완료")
    
    if len(documents) == 0:
        raise ValueError("로드된 문서가 없습니다. PDF 파일을 확인해주세요.")
    
    # 3. 텍스트 청킹
    print("\n텍스트 청킹 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        text_chunks = text_splitter.split_text(doc["content"])
        for chunk_text in text_chunks:
            chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "page": doc["page"],
                "file_path": doc["file_path"]
            })
    
    print(f"총 {len(chunks)}개 청크 생성")
    
    if len(chunks) == 0:
        raise ValueError("생성된 청크가 없습니다.")
    
    # 4. 벡터 임베딩 생성
    print("\n벡터 임베딩 생성 중...")
    texts = [chunk["text"] for chunk in chunks]
    
    if len(texts) == 0:
        raise ValueError("임베딩할 텍스트가 없습니다.")
    
    embeddings = embedding_model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings)
    
    # 빈 배열 체크
    if embeddings.size == 0 or len(embeddings.shape) == 0:
        raise ValueError("임베딩 생성 실패: 빈 배열")
    
    print(f"임베딩 완료: {embeddings.shape}")
    
    # 5. FAISS 인덱스 생성 및 저장
    print("\nFAISS 인덱스 생성 중...")
    # shape가 (0,)인 경우 처리
    if len(embeddings.shape) == 1:
        if embeddings.shape[0] == 0:
            raise ValueError("임베딩 생성 실패: 빈 배열")
        # 1차원 배열인 경우 (단일 텍스트) 2차원으로 변환
        dimension = embedding_model.get_sentence_embedding_dimension()
        embeddings = embeddings.reshape(1, -1)
    else:
        dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    embeddings_np = np.array(embeddings).astype("float32")
    index.add(embeddings_np)
    print(f"FAISS 인덱스 생성 완료: {index.ntotal}개 벡터")
    
    # 인덱스 저장
    index_path = vector_store_dir / "faiss_index.index"
    save_faiss_index(index, index_path)
    
    # 메타데이터 저장
    metadata_path = vector_store_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n메타데이터 저장: {metadata_path}")
    print(f"총 {len(chunks)}개 청크, {len(set(c['source'] for c in chunks))}개 PDF 파일")
    
    return index, chunks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RFP 문서 FAISS 벡터화")
    parser.add_argument("--pdf-dir", type=str, help="PDF 파일 디렉토리")
    parser.add_argument("--output-dir", type=str, help="벡터 저장 디렉토리")
    parser.add_argument("--model", type=str, default="jhgan/ko-sroberta-multitask", help="임베딩 모델명")
    parser.add_argument("--chunk-size", type=int, default=500, help="청크 크기")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="청크 간 겹치는 부분")
    
    args = parser.parse_args()
    
    main(
        pdf_dir=Path(args.pdf_dir) if args.pdf_dir else None,
        vector_store_dir=Path(args.output_dir) if args.output_dir else None,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
