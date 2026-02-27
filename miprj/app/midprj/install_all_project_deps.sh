#!/bin/bash

# 텍스트 색상 정의
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color (색상 초기화)

echo -e "${CYAN}Installing all project dependencies...${NC}"

# pip 최신 버전 업데이트
python3 -m pip install --upgrade pip

# 핵심 프레임워크 및 데이터 처리
pip install reflex \
            pandas \
            numpy \
            olefile \
            torch \
            torchvision \
            torchaudio \
            faiss-cpu \
            python-dotenv \
            pydantic \
            tqdm \
            matplotlib \
            seaborn \
            filelock \
            openpyxl

# AI / ML / NLP 및 모델 관련
pip install transformers \
            sentence-transformers \
            bitsandbytes \
            accelerate \
            peft \
            nltk \
            gensim \
            FlagEmbedding \
            scikit-learn \
            ultralytics \
            torchmetrics \
            torch-fidelity

# LangChain 기반 RAG 구성 요소
pip install langchain \
            langchain-core \
            langchain-community \
            langchain-openai \
            langchain-huggingface \
            langchain-ollama \
            langchain-text-splitters \
            langgraph \
            langsmith

# 문서 처리 및 추출
pip install pdfplumber \
            pypdf \
            "unstructured[pdf]" \
            pdf2image \
            pikepdf \
            pdfminer.six \
            beautifulsoup4 \
            lxml

# 유틸리티 및 기타
pip install httpx \
            aiohttp \
            tenacity \
            backoff \
            tiktoken \
            kaggle

echo -e "${GREEN}All dependencies installed successfully.${NC}"
