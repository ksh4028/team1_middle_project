
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# LangChain 관련 라이브러리
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from util_llmselector import get_llm
from langchain_community.document_loaders import UnstructuredPDFLoader

from midprj_defines import OPENAI_SETTINGS

class PDFTableSummarizer:
    def __init__(self,param):
        self.param = param
        # PDF 표/그림 요약 클래스 초기화
        # OpenAI API 키 유효성 확인 및 LLM 초기화
        try:
            provider = None
            if hasattr(self, 'param') and hasattr(self.param, 'llm_selector') and self.param.llm_selector:
                provider = self.param.llm_selector
            self.llm = get_llm(provider=provider)
            print(f"[*] LLM 초기화 시도 중...")
            # 간단한 테스트로 유효성 확인
            self.llm.invoke("Hi")
            print("[+] LLM 초기화 완료")
        except Exception as e:
            print(f"[-] LLM 초기화 실패: {e}")
            self.llm = None

    def extract_tables_and_figures(self, pdf_path):
        # PDF에서 표와 그림 요소를 정교하게 추출합니다. 
        # 여러 전략(hi_res, auto, ocr_only)을 시도하여 성공 확률을 높입니다.
        print(f"\n[*] 표/그림 추출 시작: {os.path.basename(pdf_path)}")
        strategies = [
            ("hi_res", "고해상도 분석"),
            ("auto", "자동 선택"),
            ("ocr_only", "OCR 전용")
        ]
        
        for strategy, label in strategies:
            print(f"[*] 추출 전략 시도: {label} ({strategy})...")
            extracted_elements = []
            try:
                loader = UnstructuredPDFLoader(
                    str(pdf_path),
                    strategy=strategy,
                    infer_table_structure=True,
                    chunking_strategy="by_title",
                    languages=["ko", "en"]
                )
                docs = loader.load()
                
                for doc in docs:
                    metadata = doc.metadata or {}
                    category = str(metadata.get("category", "")).lower()
                    
                    # HTML 형태의 표 데이터 탐색
                    html_table = metadata.get("text_as_html") or metadata.get("table_as_html") or ""
                    
                    # 1. 명시적 카테고리 확인
                    is_table_or_fig = any(key in category for key in ("table", "figure", "image"))
                    # 2. 내용 기반 확인 (HTML 태그 또는 특정 패턴)
                    content = doc.page_content.strip()
                    has_html_tag = "<table>" in content.lower() or "<table>" in html_table.lower()
                    # 3. 보수적으로 내용이 표와 같아 보이는 경우 (탭이나 여러 공백이 반복되는 경우 등)
                    is_likely_table = "|" in content or "  " in content and len(content) > 10
                    
                    if has_html_tag:
                        table_content = html_table if "<table>" in html_table.lower() else content
                        extracted_elements.append({"type": "Table (HTML)", "content": table_content})
                    elif is_table_or_fig:
                        extracted_elements.append({"type": category.capitalize(), "content": content})
                    elif is_likely_table and strategy == "ocr_only":
                        # OCR 모드에서는 카테고리가 텍스트로만 나올 수 있으므로 구조적 특징 확인
                        extracted_elements.append({"type": "Potential Table/Data", "content": content})

                if extracted_elements:
                    print(f"[+] {label} 전략으로 {len(extracted_elements)}개의 요소를 찾았습니다.")
                    return extracted_elements
                else:
                    print(f"[-] {label} 전략에서 추출된 요소가 없습니다.")

            except ImportError:
                print("[-] 'unstructured' 패키지가 필요합니다. (pip install unstructured[pdf])")
                break # 임베딩 패키지가 없으면 다음 전략도 실패할 것이므로 중단
            except Exception as e:
                print(f"[-] {label} 전략 실행 중 오류 발생: {e}")
                continue

        return []

    def summarize_content(self, elements):
        """
        추출된 요소를 분석하고 LLM을 통해 요약합니다.
        LLM 실패 시 추출된 원문 일부를 보여줍니다.
        """
        if not elements:
            return "추출된 표나 그림 요소가 없어 요약을 생성할 수 없습니다."
        
        # 분석용 컨텍스트 구축
        context = ""
        for i, el in enumerate(elements):
            # 너무 길면 요약에 방해되므로 일정 길이로 제한하거나 중요 정보 위주로 구성
            text_snippet = el['content'][:2000] 
            context += f"\n--- Element {i+1} ({el['type']}) ---\n{text_snippet}\n"

        if not self.llm:
            header = "LLM 요약을 생성할 수 없습니다 (API Key 확인 필요). 추출된 원문 데이터는 다음과 같습니다:\n"
            return header + context[:3000] + "\n...(이하 생략)..."

        print("[2/2] LLM을 통한 표/그림 내용 요약 및 해석 중...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 문서 분석 전문가입니다. 제공된 PDF의 표(HTML 포함)와 그림/이미지 내용을 종합적으로 해석하여 주요 정보를 풍부하게 한글로 요약해 주세요."),
            ("human", "다음은 PDF에서 추출된 표와 그림 요소들입니다. 이 내용들을 바탕으로 핵심 내용을 요약해 주세요:\n\n{context}")
        ])

        try:
            chain = prompt | self.llm
            response = chain.invoke({"context": context})
            return response.content
        except Exception as e:
            fallback_msg = f"\n[!] 요약 생성 중 오류가 발생했습니다: {e}\n\n추출된 원문 데이터 일부:\n"
            return fallback_msg + context[:2000]
    
    def summarize_pdf(self, pdf_file):
        elements = self.extract_tables_and_figures(pdf_file)
        summary = self.summarize_content(elements)
        return summary
    