# 이 스크립트는 프로젝트의 모든 .py 파일을 검사하여 추출된 라이브러리를 설치합니다.
# 가상환경이 활성화된 상태에서 실행하세요.

Write-Host "Installing all project dependencies..." -ForegroundColor Cyan

# 핵심 프레임워크 및 데이터 처리
pip install reflex `
            pandas `
            numpy `
            olefile `
            torch `
            torchvision `
            torchaudio `
            faiss-cpu `
            python-dotenv `
            pydantic `
            tqdm `
            matplotlib `
            seaborn `
            filelock `
            openpyxl

# AI / ML / NLP 및 모델 관련
pip install transformers `
            sentence-transformers `
            bitsandbytes `
            accelerate `
            peft `
            nltk `
            gensim `
            FlagEmbedding `
            scikit-learn `
            ultralytics `
            torchmetrics `
            torch-fidelity

# LangChain 기반 RAG 구성 요소
pip install langchain `
            langchain-core `
            langchain-community `
            langchain-openai `
            langchain-huggingface `
            langchain-ollama `
            langchain-text-splitters `
            langgraph `
            langsmith

# 문서 처리 및 추출
pip install pdfplumber `
            pypdf `
            unstructured[pdf] `
            pdf2image `
            pikepdf `
            pdfminer.six `
            beautifulsoup4 `
            lxml

# 유틸리티 및 기타
pip install httpx `
            aiohttp `
            tenacity `
            backoff `
            tiktoken `
            kaggle

Write-Host "All dependencies installed successfully." -ForegroundColor Green
