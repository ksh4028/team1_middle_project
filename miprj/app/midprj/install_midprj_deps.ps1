# midprj_main.py 의 import를 기반으로 필요한 라이브러리를 설치하는 스크립트입니다.
# 가상환경이 활성화된 상태에서 실행하세요.

Write-Host "Installing dependencies for midprj_main.py..." -ForegroundColor Cyan

pip install pandas `
            numpy `
            olefile `
            torch `
            faiss-cpu `
            python-dotenv `
            transformers `
            pydantic `
            langchain-core `
            langchain-text-splitters `
            langchain-community `
            langchain-openai `
            langchain-huggingface `
            langchain-ollama `
            bitsandbytes `
            accelerate `
            pypdf `
            unstructured

Write-Host "Installation script completed." -ForegroundColor Green
