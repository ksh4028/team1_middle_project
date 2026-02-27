import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# 현재 파일의 부모 디렉토리를 sys.path에 추가하여 모듈 임포트 가능하게 함
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from .midprj_defines import BASE_DIR, OPENAI_SETTINGS
from .util_summary_pdf_gpt import PDFTableSummarizer

def main():
    # 기본 PDF 경로 설정 (테스트용)
    default_pdf = os.path.join(
        BASE_DIR,
        "data",
        "rfp_files",
        "files",
        "(사)벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업 .pdf"
    )

    parser = argparse.ArgumentParser(description="PDF 표/그림 추출 및 LLM 요약 (Local Logic)")
    parser.add_argument("--pdf", default=default_pdf, help="분석할 PDF 파일 경로")
    parser.add_argument("--model", default=OPENAI_SETTINGS["llm_model_name"], help="사용할 LLM 모델 이름")

    args = parser.parse_args()
    pdf_path = args.pdf

    if not os.path.exists(pdf_path):
        print(f"[-] 파일을 찾을 수 없습니다: {pdf_path}")
        return

    print(f"[INFO] PDF 분석 시작: {pdf_path}")
    print(f"[INFO] 모델: {args.model}")

    # 1. 요약기 초기화
    summarizer = PDFTableSummarizer(model_name=args.model)

    # 2. 표/그림 추출
    # util_summary_pdf_gpt.py의 extract_tables_and_figures 사용
    elements = summarizer.extract_tables_and_figures(pdf_path)

    if not elements:
        print("[-] 추출된 표나 그림 요소가 없습니다.")
        return

    # 3. 요약 생성
    summary = summarizer.summarize_content(elements)

    # 4. 결과 출력
    print("\n" + "=" * 60)
    print("🚀 PDF 표/그림 추출 및 요약 결과")
    print("=" * 60)
    print(summary)
    print("=" * 60)

if __name__ == "__main__":
    main()
