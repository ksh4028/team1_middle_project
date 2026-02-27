"""
제안요청서 요약 서비스

기업/정부 제안요청서 내용을 NLP/LLM을 활용하여 요약
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict


def get_base_dir() -> Path:
    """프로젝트 루트 경로 반환"""
    _cwd = Path.cwd()
    if (_cwd / "notebooks").exists() or (_cwd / "src").exists():
        return _cwd
    return _cwd.parent


def load_docx(file_path: Path) -> str:
    """.docx 파일에서 텍스트 추출"""
    from docx import Document
    doc = Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)


def load_pdf(file_path: Path) -> str:
    """.pdf 파일에서 텍스트 추출"""
    import PyPDF2
    text_parts = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def load_document(file_path: str | Path) -> str:
    """문서 로드 - .docx, .pdf 지원"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return load_docx(path)
    elif suffix == ".pdf":
        return load_pdf(path)
    else:
        raise ValueError(f"지원하지 않는 형식: {suffix}. 지원: .docx, .pdf")


@dataclass
class SummaryResult:
    """요약 결과"""
    summary: str
    key_points: list
    original_length: int
    summary_length: int


def summarize_proposal(text: str, model: str = "gpt-4o-mini") -> SummaryResult:
    """
    제안요청서 텍스트 요약
    
    TODO: OpenAI 등 LLM API 연동 구현
    """
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.chat.completions.create(...)
    
    return SummaryResult(
        summary="[요약 구현 예정]",
        key_points=["핵심 포인트 1", "핵심 포인트 2"],
        original_length=len(text),
        summary_length=0,
    )


def main(document_path: str | Path | None = None):
    """
    제안요청서 요약 실행
    
    Args:
        document_path: 문서 경로 (None이면 기본 경로에서 첫 번째 파일 사용)
    """
    BASE_DIR = get_base_dir()
    DATA_DIR = BASE_DIR / "data" / "raw"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 문서 경로 설정
    if document_path is None:
        doc_folder = DATA_DIR / "원본 데이터-20260206T023549Z-1-001" / "원본 데이터" / "files"
        doc_files = list(doc_folder.glob("*.pdf")) + list(doc_folder.glob("*.docx")) if doc_folder.exists() else []
        document_path = doc_files[0] if doc_files else DATA_DIR / "sample.pdf"
    
    document_path = Path(document_path)
    
    if document_path.exists():
        print(f"문서 로드: {document_path}")
        text = load_document(document_path)
        result = summarize_proposal(text)
        
        print(f"\n원본 길이: {result.original_length}자")
        print(f"\n요약:")
        print(result.summary)
        print(f"\n핵심 포인트:")
        for point in result.key_points:
            print(f"  - {point}")
    else:
        print(f"문서 경로를 확인해주세요. (data/raw/ 아래 폴더·파일명은 자유롭게 사용 가능)")
        print(f"현재 시도 경로: {document_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="제안요청서 요약 서비스")
    parser.add_argument("--document", type=str, help="문서 경로")
    
    args = parser.parse_args()
    
    main(document_path=args.document)
