# 제안요청서 요약 서비스

기업 및 정부 제안요청서 내용을 NLP/LLM을 활용하여 요약하는 서비스입니다.

## 프로젝트 개요

- **프로젝트명**: 자연어처리 모델 프로젝트
- **내용**: 제안요청서 내용 요약 서비스
- **기술**: 자연어처리(NLP), 대규모언어모델(LLM)

## 프로젝트 구조

```
team1_middle_project/
├── .ssh-config.example   # GCP VM SSH 접속 설정 예시
├── data/
│   ├── raw/         # 원본 제안요청서 문서 (.docx, .pdf)
│   └── processed/   # 처리·요약 결과
├── scripts/
│   └── hwp_to_pdf.py   # HWP → PDF 변환 스크립트
├── src/            # Python 스크립트 (권장)
│   ├── rag_vectorization.py   # PDF → FAISS 벡터화
│   ├── rag_qa_system.py        # RAG Q&A 시스템
│   └── proposal_summarizer.py # 제안요청서 요약
├── requirements.txt
└── README.md
```

## SSH 접속 (VS Code / Cursor)

로컬 에디터에서 GCP VM에 SSH로 접속하려면:

1. **Remote - SSH** 확장 설치 (Ctrl+Shift+X → "Remote - SSH" 검색)
2. `.ssh-config.example` 내용을 `~/.ssh/config` (Windows: `C:\Users\<사용자명>\.ssh\config`)에 복사
3. **F1** → `Remote-SSH: Connect to Host` → `gcp-vm` 선택
4. 비밀번호 입력 (접속 시 한 번)

## 데이터 준비

각자 로컬에서 원본 데이터를 받아와 `data/raw/` 디렉토리에 넣어주세요.  
제안요청서 문서(.docx, .pdf)를 해당 폴더에 배치한 후 노트북에서 로드하여 팀 프로젝트를 진행합니다.

- **디렉토리 구조**: 예시는 `data/raw/원본 데이터-20260206T023549Z-1-001/원본 데이터/files/`  
  → `files` 안에 .pdf, .docx를 두면 노트북이 자동으로 첫 번째 파일을 로드합니다.
- **HWP 파일**: `scripts/hwp_to_pdf.py`로 PDF 변환 후 요약 가능. (요약은 .pdf, .docx만 지원)
- 노트북 4단계 셀에서 `doc_folder` 경로만 실제 구조에 맞게 수정하면 됩니다.

### HWP → PDF 변환

한글 파일을 PDF로 바꾸려면 **LibreOffice**가 필요합니다.

- **Windows**: [LibreOffice](https://www.libreoffice.org/) 설치
- **Linux (JupyterHub VM 등)**: `sudo apt install libreoffice` (Ubuntu/Debian)

프로젝트 루트에서 실행:

```bash
# 기본 경로(data/raw/원본 데이터-.../원본 데이터/files) 변환
python scripts/hwp_to_pdf.py

# 다른 디렉토리 지정
python scripts/hwp_to_pdf.py "data/raw/원본 데이터-20260206T023549Z-1-001/원본 데이터/files"

# 출력 디렉토리 지정 (기본: 입력과 동일)
python scripts/hwp_to_pdf.py "입력경로" "출력경로"
```

변환된 PDF는 같은 `files` 폴더에 생성되며, 노트북에서 그대로 로드할 수 있습니다.

> `data/` 디렉토리는 .gitkeep으로 구조만 저장되어 있으며, 실제 데이터 파일은 Git에 포함되지 않습니다.

## 실행 방법

### 방법 1: Python 스크립트 (권장)

```bash
# 1. PDF 벡터화
python src/rag_vectorization.py

# 2. RAG Q&A (질문 직접 입력)
python src/rag_qa_system.py "이 제안요청서의 주요 요구사항은 무엇인가요?"

# 3. 제안요청서 요약
python src/proposal_summarizer.py --document "data/raw/.../files/sample.pdf"
```


### RAG 시스템 워크플로우

1. **PDF 벡터화**: `python src/rag_vectorization.py`
   - PDF 파일들을 로드하고 텍스트 추출
   - 텍스트 청킹 (500자 단위, 50자 overlap)
   - 한국어 임베딩 모델(`jhgan/ko-sroberta-multitask`)로 벡터화
   - FAISS 인덱스 생성 및 저장 (`data/processed/vector_store/`)

2. **RAG Q&A 시스템**: `python src/rag_qa_system.py "질문"`
   - FAISS 인덱스 로드
   - 질문 임베딩 생성 및 유사 문서 검색
   - 검색된 문서를 컨텍스트로 LLM에 전달하여 답변 생성

## TODO

- [x] PDF → FAISS 벡터화 파이프라인
- [x] RAG 기반 Q&A 시스템 구축
- [ ] 평가 지표 및 성능 측정
- [ ] 다양한 임베딩 모델 실험
- [ ] 프롬프트 최적화
- [ ] 섹션별 요약 기능
