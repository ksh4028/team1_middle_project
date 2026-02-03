# 제안요청서 요약 서비스

기업 및 정부 제안요청서 내용을 NLP/LLM을 활용하여 요약하는 서비스입니다.

## 프로젝트 개요

- **프로젝트명**: 자연어처리 모델 프로젝트
- **내용**: 제안요청서 내용 요약 서비스
- **기술**: 자연어처리(NLP), 대규모언어모델(LLM)

## 프로젝트 구조

```
team1_middle_project/
├── data/
│   ├── raw/         # 원본 제안요청서 문서 (.docx, .pdf)
│   └── processed/   # 처리·요약 결과
├── notebooks/
│   └── proposal_summarizer.ipynb   # 메인 노트북
├── requirements.txt
└── README.md
```

## 데이터 준비

각자 로컬에서 원본 데이터를 받아와 `data/raw/` 디렉토리에 넣어주세요.  
제안요청서 문서(.docx, .pdf)를 해당 폴더에 배치한 후 노트북에서 로드하여 팀 프로젝트를 진행합니다.

> `data/` 디렉토리는 .gitkeep으로 구조만 저장되어 있으며, 실제 데이터 파일은 Git에 포함되지 않습니다.

## 접속 및 실행

1. **JupyterHub 접속**: http://34.45.125.252:8000/
2. 로그인 후 `notebooks/proposal_summarizer.ipynb` 열기
3. 첫 셀에서 `!pip install openai python-docx PyPDF2` 실행 (필요시)
4. 셀 순서대로 실행 (Shift+Enter)

## TODO

- [ ] LLM API 연동 (OpenAI 등)
- [ ] 제안요청서 특화 프롬프트 설계
- [ ] 섹션별 요약 기능
