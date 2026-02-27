"""
벡터화된 문서에서 사용자 질문 생성

FAISS로 벡터화된 RFP 문서를 분석하여 사용자가 물어볼만한 질문들을 GPT API로 생성
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict
import random

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
    # src/generate_questions.py -> team1_middle_project/
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


def group_metadata_by_document(metadata: List[Dict]) -> Dict[str, List[Dict]]:
    """메타데이터를 문서(source)별로 그룹화"""
    file_groups: Dict[str, List[Dict]] = {}
    for chunk in metadata:
        # source(파일명)를 키로 사용
        source = chunk.get("source", chunk.get("file_path", "unknown"))
        if isinstance(source, str) and "/" in source:
            source = source.split("/")[-1]  # 경로에서 파일명만 추출
        if source not in file_groups:
            file_groups[source] = []
        file_groups[source].append(chunk)
    return file_groups


def chunks_to_document_text(chunks: List[Dict], max_length_per_chunk: int = 2000) -> str:
    """청크 리스트를 문서 텍스트로 변환"""
    texts = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if len(text) > max_length_per_chunk:
            text = text[:max_length_per_chunk] + "..."
        texts.append(f"[페이지: {chunk.get('page', '?')}]\n{text}")
    return "\n\n---\n\n".join(texts)


def sample_documents(metadata: List[Dict], num_samples: int = 50, max_length_per_doc: int = 2000) -> str:
    """
    메타데이터에서 문서 샘플링하여 컨텍스트 생성
    
    Args:
        metadata: 청크 메타데이터 리스트
        num_samples: 샘플링할 청크 개수
        max_length_per_doc: 문서당 최대 길이
    
    Returns:
        샘플링된 문서 텍스트
    """
    file_groups = group_metadata_by_document(metadata)
    
    # 각 파일에서 샘플링
    sampled_chunks = []
    files = list(file_groups.keys())
    random.shuffle(files)
    
    for file_path in files[:min(10, len(files))]:  # 최대 10개 파일
        chunks = file_groups[file_path]
        sampled = random.sample(chunks, min(5, len(chunks)))
        sampled_chunks.extend(sampled)
        
        if len(sampled_chunks) >= num_samples:
            break
    
    # 텍스트 조합
    texts = []
    for chunk in sampled_chunks[:num_samples]:
        text = chunk.get("text", "")
        if len(text) > max_length_per_doc:
            text = text[:max_length_per_doc] + "..."
        texts.append(f"[파일: {chunk.get('source', 'unknown')}, 페이지: {chunk.get('page', '?')}]\n{text}")
    
    return "\n\n---\n\n".join(texts)


def generate_questions(
    documents_text: str,
    num_questions: int = 30,
    llm_model: str = "gpt-4o-mini"
) -> List[str]:
    """
    GPT API를 사용하여 문서 기반 질문 생성
    
    Args:
        documents_text: 샘플링된 문서 텍스트
        num_questions: 생성할 질문 개수
        llm_model: 사용할 LLM 모델
    
    Returns:
        생성된 질문 리스트
    """
    # LLM 초기화
    llm = ChatOpenAI(
        model_name=llm_model,
        temperature=0.7,
        max_tokens=2000
    )
    
    # 프롬프트 구성
    prompt = f"""다음은 기업 및 정부 제안요청서(RFP) 문서의 일부입니다.

문서 내용:
{documents_text}

위 문서 내용을 바탕으로 사용자가 물어볼만한 질문을 {num_questions}개 생성해주세요.

요구사항:
1. 문서 내용을 기반으로 한 구체적인 질문이어야 합니다
2. 다양한 관점에서 질문을 생성하세요 (요구사항, 일정, 예산, 기술사양, 평가기준 등)
3. 각 질문은 한 문장으로 명확하게 작성하세요
4. 질문만 나열하세요 (번호나 설명 없이)
5. 질문은 실제 사용자가 궁금해할만한 내용이어야 합니다

생성된 질문:
"""
    
    print("질문 생성 중...")
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    # 질문 파싱
    questions = []
    for line in answer.strip().split('\n'):
        line = line.strip()
        # 번호 제거 (1. 2. 등)
        if line and (line[0].isdigit() or line.startswith('-')):
            # 번호나 불릿 제거
            line = line.split('.', 1)[-1].strip()
            if line.startswith('-'):
                line = line[1:].strip()
        
        if line and len(line) > 10:  # 너무 짧은 것은 제외
            # 질문 형식 확인 (한국어 질문은 보통 '?', '인가요', '인지' 등으로 끝남)
            if '?' in line or '인가요' in line or '인지' in line or '무엇' in line or '어떻게' in line or '언제' in line or '어디' in line or '누구' in line or '왜' in line:
                questions.append(line)
            elif len(line) > 20:  # 질문 마커가 없어도 충분히 긴 경우 포함
                questions.append(line)
    
    return questions[:num_questions]


def generate_questions_per_document(
    metadata: List[Dict],
    questions_per_doc: int = 20,
    max_chunks_per_doc: int = 15,
    max_length_per_chunk: int = 2000,
    llm_model: str = "gpt-4o-mini",
) -> tuple[List[str], Dict[str, List[str]]]:
    """
    문서별로 질문 생성 (각 문서당 questions_per_doc개)
    
    Args:
        metadata: 청크 메타데이터 리스트
        questions_per_doc: 문서당 생성할 질문 개수
        max_chunks_per_doc: 문서당 최대 청크 개수 (토큰 제한)
        max_length_per_chunk: 청크당 최대 길이
        llm_model: 사용할 LLM 모델
    
    Returns:
        (전체 질문 리스트, {문서명: [질문 리스트]} 딕셔너리)
    """
    file_groups = group_metadata_by_document(metadata)
    all_questions: List[str] = []
    questions_by_doc: Dict[str, List[str]] = {}
    
    llm = ChatOpenAI(model_name=llm_model, temperature=0.7, max_tokens=2000)
    
    for doc_idx, (source, chunks) in enumerate(file_groups.items(), 1):
        # 각 문서에서 청크 샘플링 (최대 max_chunks_per_doc개)
        sampled = chunks[:max_chunks_per_doc]
        if len(chunks) > max_chunks_per_doc:
            sampled = random.sample(chunks, max_chunks_per_doc)
        
        doc_text = chunks_to_document_text(sampled, max_length_per_chunk)
        
        if len(doc_text.strip()) < 100:
            print(f"  [건너뜀] {source}: 내용이 너무 적음")
            continue
        
        print(f"  [{doc_idx}/{len(file_groups)}] {source}: 질문 {questions_per_doc}개 생성 중...")
        
        prompt = f"""다음은 기업 및 정부 제안요청서(RFP) 문서의 일부입니다.

문서 내용:
{doc_text}

위 문서 내용을 바탕으로 사용자가 물어볼만한 질문을 {questions_per_doc}개 생성해주세요.

요구사항:
1. 문서 내용을 기반으로 한 구체적인 질문이어야 합니다
2. 다양한 관점에서 질문을 생성하세요 (요구사항, 일정, 예산, 기술사양, 평가기준 등)
3. 각 질문은 한 문장으로 명확하게 작성하세요
4. 질문만 나열하세요 (번호나 설명 없이)
5. 질문은 실제 사용자가 궁금해할만한 내용이어야 합니다

생성된 질문:
"""
        try:
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            
            parsed = []
            for line in answer.strip().split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    line = line.split(".", 1)[-1].strip()
                    if line.startswith("-"):
                        line = line[1:].strip()
                if line and len(line) > 10:
                    if "?" in line or "인가요" in line or "인지" in line or "무엇" in line or "어떻게" in line or "언제" in line or "어디" in line or "누구" in line or "왜" in line:
                        parsed.append(line)
                    elif len(line) > 20:
                        parsed.append(line)
            
            questions = parsed[:questions_per_doc]
            questions_by_doc[source] = questions
            all_questions.extend(questions)
            print(f"    -> {len(questions)}개 생성 완료")
            
        except Exception as e:
            print(f"    [오류] {e}")
    
    return all_questions, questions_by_doc


def main(
    vector_store_dir: Path | None = None,
    num_questions: int = 30,
    num_samples: int = 50,
    questions_per_doc: int | None = None,
    max_chunks_per_doc: int = 15,
    llm_model: str = "gpt-4o-mini",
    output_file: Path | None = None
):
    """
    문서 기반 질문 생성 실행
    
    Args:
        vector_store_dir: 벡터 스토어 디렉토리 (None이면 기본 경로 사용)
        num_questions: 생성할 총 질문 개수 (questions_per_doc 미사용 시)
        num_samples: 샘플링할 문서 청크 개수 (questions_per_doc 미사용 시)
        questions_per_doc: 문서당 질문 개수 (None이면 기존 방식, 20 등 지정 시 문서별 생성)
        max_chunks_per_doc: 문서당 최대 청크 개수 (questions_per_doc 사용 시)
        llm_model: 사용할 LLM 모델
        output_file: 질문 저장 파일 경로 (None이면 출력만)
    """
    BASE_DIR = get_base_dir()
    
    # 경로 설정
    if vector_store_dir is None:
        vector_store_dir = BASE_DIR / "data" / "processed" / "vector_store"
    
    vector_store_dir = Path(vector_store_dir)
    
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("환경변수 설정: export OPENAI_API_KEY='your_key'")
        print("또는 .env 파일에 OPENAI_API_KEY를 설정하세요.")
        sys.exit(1)
    
    print("=" * 60)
    print("문서 기반 질문 생성")
    print("=" * 60)
    print(f"벡터 스토어: {vector_store_dir}")
    
    # FAISS 인덱스 및 메타데이터 로드
    print("\nFAISS 인덱스 로딩 중...")
    index, metadata = load_faiss_index(vector_store_dir)
    print(f"인덱스 로드 완료: {index.ntotal}개 벡터")
    print(f"메타데이터 로드 완료: {len(metadata)}개 청크")
    print()
    
    if questions_per_doc is not None:
        # 문서별 N개씩 질문 생성
        print(f"문서별 {questions_per_doc}개씩 질문 생성 모드")
        print(f"문서당 최대 청크: {max_chunks_per_doc}개\n")
        
        all_questions, questions_by_doc = generate_questions_per_document(
            metadata=metadata,
            questions_per_doc=questions_per_doc,
            max_chunks_per_doc=max_chunks_per_doc,
            llm_model=llm_model,
        )
        questions = all_questions
        
        print(f"\n총 {len(questions_by_doc)}개 문서, {len(questions)}개 질문 생성 완료")
    else:
        # 기존 방식: 전체에서 num_questions개 생성
        print(f"생성할 질문 개수: {num_questions}")
        print(f"\n문서 샘플링 중... (최대 {num_samples}개 청크)")
        documents_text = sample_documents(metadata, num_samples=num_samples)
        print(f"샘플링 완료: {len(documents_text)}자\n")
        
        questions = generate_questions(
            documents_text=documents_text,
            num_questions=num_questions,
            llm_model=llm_model
        )
        questions_by_doc = None
    
    # 결과 출력
    print("\n" + "=" * 60)
    print(f"생성된 질문 ({len(questions)}개):")
    print("=" * 60)
    for i, q in enumerate(questions[:50], 1):  # 처음 50개만 출력
        print(f"{i}. {q}")
    if len(questions) > 50:
        print(f"... 외 {len(questions) - 50}개")
    print()
    
    # 파일 저장
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "total_questions": len(questions),
            "questions": questions,
            "metadata": {
                "vector_store_dir": str(vector_store_dir),
                "llm_model": llm_model,
                "questions_per_doc": questions_per_doc,
                "num_questions": num_questions if questions_per_doc is None else None,
                "num_samples": num_samples if questions_per_doc is None else None,
            }
        }
        if questions_by_doc is not None:
            output_data["questions_by_document"] = questions_by_doc
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"질문이 저장되었습니다: {output_file}")
        
        # 텍스트 파일로도 저장
        txt_file = output_file.with_suffix('.txt')
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"생성된 질문 ({len(questions)}개)\n")
            f.write("=" * 60 + "\n\n")
            for i, q in enumerate(questions, 1):
                f.write(f"{i}. {q}\n")
        
        print(f"텍스트 파일도 저장되었습니다: {txt_file}")
    
    return questions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="벡터화된 문서에서 사용자 질문 생성")
    parser.add_argument("--vector-store-dir", type=str, help="벡터 스토어 디렉토리")
    parser.add_argument("--questions-per-doc", type=int, default=20, help="문서당 생성할 질문 개수 (기본: 20)")
    parser.add_argument("--num-questions", type=int, default=30, help="생성할 총 질문 개수 (--questions-per-doc 미사용 시)")
    parser.add_argument("--num-samples", type=int, default=50, help="샘플링할 문서 청크 개수 (--questions-per-doc 미사용 시)")
    parser.add_argument("--max-chunks-per-doc", type=int, default=15, help="문서당 최대 청크 개수 (문서별 생성 시)")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM 모델명")
    parser.add_argument("--output", "-o", type=str, help="질문 저장 파일 경로 (JSON 및 TXT 형식)")
    parser.add_argument("--legacy", action="store_true", help="기존 방식 사용 (전체에서 num-questions개만 생성)")
    
    args = parser.parse_args()
    
    BASE_DIR = get_base_dir()
    
    # 출력 파일 경로 설정
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = BASE_DIR / "data" / "processed" / "generated_questions.json"
    
    # legacy 모드: questions_per_doc=None
    questions_per_doc = None if args.legacy else args.questions_per_doc
    
    main(
        vector_store_dir=Path(args.vector_store_dir) if args.vector_store_dir else None,
        num_questions=args.num_questions,
        num_samples=args.num_samples,
        questions_per_doc=questions_per_doc,
        max_chunks_per_doc=args.max_chunks_per_doc,
        llm_model=args.llm_model,
        output_file=output_file
    )
