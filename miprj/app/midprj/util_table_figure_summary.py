import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pdfplumber
from langchain_core.prompts import ChatPromptTemplate
import re

# 경로 및 환경설정은 midprj_defines.py에서 import
from .midprj_defines import DATA_DIR, ENV_FILE

# PDF2TEXT_DIR은 DATA_DIR로 대체 (문자열로 사용)
PDF2TEXT_DIR = os.path.join(DATA_DIR, "rfp_files", "files")

# Load environment variables (override=True ensures file values take precedence over system vars)

class PDFSummarizer:
    def __init__(self, model_name="gpt-5-mini"):
        """Initialize the PDF summarizer with OpenAI LLM and fallback models."""
        self.primary_model = model_name
        self.fallbacks = ["openai-5-nano", "openai-4o-mini"]
        self.llm = self._init_llm(self.primary_model)

    def _init_llm(self, model_name):
        try:
            print(f"[*] Initializing LLM ({model_name})...")
            # Ensure model name is valid
            return ChatOpenAI(model=model_name, temperature=0, max_retries=1)
        except Exception as e:
            print(f"[-] LLM initialization failed for {model_name}: {e}")
            return None

    def process_pdf(self, pdf_path):
        """Extract tables and figures for a single PDF using pdfplumber."""
        print(f"\n[*] Processing: {pdf_path.name}")
        
        tables = []
        figures = []
        all_text = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                print(f"[*] Total pages: {len(pdf.pages)}")
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    # Extract tables
                    page_tables = page.find_tables()
                    if page_tables:
                        if page_num % 10 == 0 or len(pdf.pages) < 20:
                            print(f"  [+] Page {page_num}: Found {len(page_tables)} table(s)")
                        
                        for table in page_tables:
                            cells = table.extract()
                            if cells and len(cells) > 0:
                                # Create strict Markdown table format
                                headers = [str(c).replace('\n', ' ') if c else "" for c in cells[0]]
                                separator = ["---" for _ in headers]
                                rows = []
                                for row in cells[1:]:
                                    rows.append([str(c).replace('\n', ' ') if c else "" for c in row])
                                
                                md_table = f"| {' | '.join(headers)} |\n| {' | '.join(separator)} |\n"
                                for row in rows:
                                    md_table += f"| {' | '.join(row)} |\n"
                                
                                tables.append(f"Page {page_num} Table:\n{md_table}")
                    
                    # Detect images/figures
                    text = page.extract_text() or ""
                    all_text.append(text)
                    
                    # If page has images or explicit "그림/도표" text
                    has_images = len(page.images) > 0
                    has_figure_text = any(kw in text for kw in ["그림", "도표", "Figure", "Fig."])
                    
                    if has_images or has_figure_text:
                        # Use regex to find specific captions like "그림 1-1", "Fig 1"
                        captions = re.findall(r"(?:그림|도표|Figure|Fig\.?)\s*\d+[-.]?\d*", text)
                        context = f"Page {page_num} "
                        if captions:
                            context += f"Captions: {', '.join(captions)} "
                        context += f"Context: {text[:1000]}"
                        figures.append(context)

            # Generate summary if elements found
            if not tables and not figures:
                print(f"[-] No tables or figures found in {pdf_path.name}")
                return "이 파일에서 추출된 표나 그림이 없습니다."

            return self.generate_summary_with_fallback(tables, figures)
            
        except Exception as e:
            print(f"[-] Error processing {pdf_path.name}: {e}")
            return f"오류 발생: {e}"

    def generate_summary_with_fallback(self, tables, figures):
        """Try generating structured data with primary model, then fallbacks if access fails."""
        models_to_try = [self.primary_model] + self.fallbacks
        last_error = ""
        
        for model in models_to_try:
            self.llm = self._init_llm(model)
            if not self.llm:
                continue
                
            result = self._call_llm(tables, figures)
            if "Error code: 403" not in result and "model_not_found" not in result and "데이터 추출 중 오류" not in result:
                return result
            
            last_error = result
            print(f"[-] Model {model} failed: {result[:100]}...")
            print(f"[*] Trying next fallback...")
            
        return f"모든 협력 모델 실패. 마지막 오류: {last_error}"

    def _call_llm(self, tables, figures):
        """Internal method to call LLM for structured extraction."""
        table_context = "\n\n".join(tables[:15]) if tables else "없음" # Increased limit but fewer tables
        figure_context = "\n\n".join(figures[:10]) if figures else "없음"

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "당신은 문서 데이터 구조화 전문가입니다. 당신의 목표는 제공된 원시 데이터를 "
                "최대한 손실 없이 구조화된 텍스트 형태로 복원하는 것입니다.\n\n"
                "규칙:\n"
                "1. 표는 반드시 마크다운(Markdown) 표 형식을 유지하세요.\n"
                "2. 그림/도표 컨텍스트는 다음 두 가지 방식으로 기술하세요:\n"
                "   - [시각적 구조]: 요소 간의 관계를 나타내는 '텍스트 다이어그램'을 <visual_diagram> 태그 내에 작성하세요. (예: <visual_diagram>\n+-- A\n|   +-- B\n</visual_diagram>)\n"
                "   - [RAG 요약]: 각 요소와 그들 사이의 관계, 도표의 핵심 의미를 자연어 문장으로 상세히 기술하세요. "
                "이는 벡터 검색 성능을 최적화하기 위함입니다.\n"
                "3. 수치, 고유 명사, 기술 용어는 절대 생략하거나 변경하지 마세요."
            )),
            ("human", (
                "다음 데이터를 바탕으로 [표 데이터]와 [그림/도표 구조]를 정확하게 복원해 주세요.\n\n"
                "### 제공된 표 데이터 ###\n{table_context}\n\n"
                "### 제공된 그림/도표 컨텍스트 ###\n{figure_context}"
            ))
        ])

        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "table_context": table_context,
                "figure_context": figure_context
            })
            return response.content
        except Exception as e:
            return f"데이터 추출 중 오류 발생: {e}"

    def generate_summary(self, tables, figures):
        # Kept for compatibility, redirects to fallback version
        return self.generate_summary_with_fallback(tables, figures)

def main():
    # Simple help check
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("Usage: python pdf_summarizer.py")
        print("Processes all PDF files in the directory and saves structured text to .txt files.")
        return


    if not Path(PDF2TEXT_DIR).exists():
        print(f"[-] Target directory not found: {PDF2TEXT_DIR}")
        return

    summarizer = PDFSummarizer()

    pdf_files = list(Path(PDF2TEXT_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print(f"[-] No PDF files found in {PDF2TEXT_DIR}")
        return

    print(f"[*] Found {len(pdf_files)} PDF files. Starting processing...")

    for pdf_path in pdf_files:
        output_path = pdf_path.with_suffix(".txt")
        
        result = summarizer.process_pdf(pdf_path)
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"[+] Structured data saved to: {output_path.name}")
        except Exception as e:
            print(f"[-] Failed to save result for {pdf_path.name}: {e}")

if __name__ == "__main__":
    env_path = Path(ENV_FILE)
    load_dotenv(env_path, override=True)
    main()
