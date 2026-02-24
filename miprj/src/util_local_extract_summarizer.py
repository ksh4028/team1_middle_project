import os
import json
import csv
import configparser
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 기존 프로젝트 설정 임포트
from midprj_defines import BASE_DIR, OPENAI_SETTINGS, INI_PATH
from midprj_sqlite import SQLiteDB
from util_llmselector import get_llm
from langchain_core.prompts import ChatPromptTemplate

# 로컬 추출 라이브러리
import pdfplumber
from unstructured.partition.pdf import partition_pdf

load_dotenv(os.path.join(BASE_DIR, ".env"))

class LocalPdfExtractSummarizer:
    def __init__(self):
        # ini 파일에서 [adobe] 섹션 로드 (사용자 요청 반영)
        config = configparser.ConfigParser()
        config.read(INI_PATH, encoding="utf-8")
        
        adobe_section = config["adobe"] if "adobe" in config else {}
        
        self.model_name = adobe_section.get("model_name", OPENAI_SETTINGS.get("llm_model_name", "gpt-4o-mini"))
        self.max_elements = int(adobe_section.get("max_elements", 10000))
        self.chunk_size = int(adobe_section.get("chunk_size", 150))

        try:
            # param.llm_selector가 있으면 provider로 전달
            from midprj_defines import PARAMVAR
            param = getattr(self, 'param', None)
            provider = None
            if param and hasattr(param, 'llm_selector') and param.llm_selector:
                provider = param.llm_selector
            self.llm = get_llm(provider=provider)
            print(f"[+] LLM 초기화 완료 ({self.model_name})")
            print(f"[INFO] 설정 로드: max_elements={self.max_elements}, chunk_size={self.chunk_size}")
        except Exception as e:
            print(f"[-] LLM 초기화 실패: {e}")
            self.llm = None

        self.output_base_dir = Path(BASE_DIR) / "data" / "extract_output"
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def extract_local(self, pdf_path):
        if not os.path.exists(pdf_path):
            print(f"[-] 파일을 찾을 수 없습니다: {pdf_path}")
            return []

        pdf_name = Path(pdf_path).stem.strip()
        save_dir = self.output_base_dir / pdf_name
        image_dir = save_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        extracted_data = []
        print(f"\n[*] 로컬 하이브리드 분석 시작: {os.path.basename(pdf_path)}")

        try:
            # 1. Unstructured를 사용한 그림(Figure) 추출 및 텍스트 매핑
            print("[1/2] 그림 및 구조 분석 중 (unstructured, hi_res)...")
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",                    # OCR 강제 (hi_res)
                extract_images_in_pdf=True,           # 물리적 파일 저장
                infer_table_structure=False,          # 표는 pdfplumber 담당
                chunking_strategy="by_title",         # 문맥 파악을 위한 청킹
                extract_image_block_output_dir=str(image_dir),
                languages=["kor", "eng"],             # OCR 언어 지정
            )   
            print(f"[+] 총 {len(elements)}개의 요소 그림에서 추출 완료.")
            
            # 디버그: 어떤 카테고리들이 나왔는지 확인
            categories = {}
            for el in elements:
                cat = str(el.category)
                categories[cat] = categories.get(cat, 0) + 1
            print(f"[ DEBUG ] 추출된 카테고리 분포: {categories}")

            # 이미 처리된 이미지 경로 추적
            processed_images = set()



            # 페이지별 텍스트 요소 미리 수집 (CompositeElement, Text, Title, Paragraph 등 모두 포함)
            page_text_blocks = {}
            for el in elements:
                meta = el.metadata.to_dict()
                page_num = meta.get('page_number', None)
                cat = str(el.category)
                if page_num is not None and cat in ["Text", "Title", "Paragraph", "CompositeElement"] and el.text:
                    if page_num not in page_text_blocks:
                        page_text_blocks[page_num] = []
                    page_text_blocks[page_num].append({
                        "text": el.text,
                        "coords": meta.get('coordinates', None),
                        "el": el,
                        "cat": cat
                    })


            for el in elements:
                meta = el.metadata.to_dict()
                image_path = meta.get("image_path")
                is_image_category = el.category == "Image" or "Figure" in str(el.category)
                if is_image_category or image_path:
                    caption = el.text if el.text else "설명 없음"
                    page_num = meta.get('page_number', '알수없음')
                    coords = meta.get('coordinates', None)

                    # 같은 페이지 내 텍스트 중 가장 가까운 블록 찾기 (좌표 기반 + category별)
                    nearest_text = None
                    min_dist = None
                    if page_num in page_text_blocks and coords:
                        x0, y0, x1, y1 = coords if isinstance(coords, (list, tuple)) and len(coords) == 4 else (None, None, None, None)
                        cy = (y0 + y1) / 2 if y0 is not None and y1 is not None else None
                        for tb in page_text_blocks[page_num]:
                            tcoords = tb["coords"]
                            if tcoords and cy is not None:
                                tx0, ty0, tx1, ty1 = tcoords if isinstance(tcoords, (list, tuple)) and len(tcoords) == 4 else (None, None, None, None)
                                tcy = (ty0 + ty1) / 2 if ty0 is not None and ty1 is not None else None
                                if tcy is not None:
                                    dist = abs(tcy - cy)
                                    # category별로 CompositeElement, Text, Title, Paragraph 우선
                                    if min_dist is None or dist < min_dist:
                                        min_dist = dist
                                        nearest_text = tb["text"]
                    # 좌표 정보가 없거나 매칭 실패 시, 같은 페이지의 첫 번째 텍스트 블록 연결
                    if not nearest_text and page_num in page_text_blocks:
                        nearest_text = page_text_blocks[page_num][0]["text"]
                    # 연결 텍스트가 있으면 추가
                    if nearest_text:
                        caption += f"\n[그림 주변 본문] {nearest_text}"

                    content_str = f"[그림 정보]\n- 페이지: {page_num}\n- 캡션/텍스트: {caption}"
                    if image_path:
                        content_str += f"\n- 파일 경로: {image_path}"
                        processed_images.add(os.path.basename(image_path))
                    extracted_data.append({
                        "type": "Figure/Image",
                        "content": content_str
                    })

            # 2. Unstructured가 파일을 생성했으나 요소로 매핑되지 않은 경우 확인
            if os.path.exists(image_dir):
                all_images = os.listdir(image_dir)
                for img_file in all_images:
                    if img_file not in processed_images:
                        full_path = os.path.join(image_dir, img_file)
                        extracted_data.append({
                            "type": "Figure/Image (File Only)",
                            "content": f"[그림 파일 발견]\n- 파일 경로: {full_path}\n- (문서 내 위치 매핑 실패)"
                        })

            print(f"[+] 그림 요소 중 {len(extracted_data)}개를 텍스트화했습니다.")
            # 2. pdfplumber를 사용한 표(Table) 추출 (정밀도 우선)
            print("[2/2] 표 데이터 정밀 추출 중 (pdfplumber)...")
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for j, table in enumerate(tables):
                        if not table: continue
                        
                        # 표 데이터를 텍스트(마크다운 스타일)로 변환
                        rows = []
                        for row in table:
                            clean_row = [cell.replace('\n', ' ') if cell else "" for cell in row]
                            rows.append(" | ".join(clean_row))
                        
                        table_as_text = "\n".join(rows)
                        extracted_data.append({
                            "type": "Table",
                            "content": f"[페이지 {i+1} 표 데이터]\n{table_as_text}"
                        })

            print(f"[+] 추출 성공: 총 {len(extracted_data)}개의 유효 요소를 텍스트화했습니다.")
            return extracted_data

        except Exception as e:
            print(f"[-] 로컬 추출 실패: {e}")
            return []

    def summarize_content(self, elements, max_elements=None):
        """
        추출된 텍스트 리스트를 기반으로 요약 생성
        """
        if max_elements is None:
            max_elements = self.max_elements

        if not elements:
            return "추출된 요소가 없어 요약을 생성할 수 없습니다."

        context = ""
        for i, el in enumerate(elements[:max_elements]):
            context += f"\n--- Element {i + 1} ({el['type']}) ---\n{el['content']}\n"

        if not self.llm:
            return f"LLM 요약 불가. 추출 데이터:\n{context[:500]}"

        # 기존 Adobe 코드의 프롬프트 로직 활용
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 PDF의 표와 그림 정보를 요약하는 전문가입니다. "
                       "표는 수치 위주로, 그림은 캡션과 주변 정보를 바탕으로 그 의미를 한국어로 설명하세요."),
            ("human", "다음 추출된 요소들을 분석하여 요약해 주세요:\n\n{context}")
        ])

        try:
            chain = prompt | self.llm
            response = chain.invoke({"context": context})
            return response.content
        except Exception as e:
            return f"요약 중 오류 발생: {e}"


    def get_summary_cache(self, pdf_name, file_size):
        db = SQLiteDB()
        return db.get_adobe_summary_cache(
            pdf_name,
            file_size,
            self.model_name, 
            self.max_elements, 
            self.chunk_size
        )
    def get_extract_data(self, pdf_path, file_size):
         
        pdf_name = Path(pdf_path).stem.strip()
        summary_text = self.get_summary_cache(pdf_name, file_size)
        if summary_text is not None:
            print("[*] 요약 캐시가 DB에서 발견되었습니다.")
            return summary_text
        elements = self.extract_local(pdf_path)
        print(f"elements size:{len(elements)}")
        
        # 2. 요약
        summary = self.summarize_content(elements)
        
        print("\n[요약 결과]")
        print(summary)

        # 3. DB 저장
        try:
            db = SQLiteDB()
            stat = os.stat(pdf_path)
            file_mtime = stat.st_mtime
            file_size = stat.st_size
            
            db.save_adobe_summary_cache(
                pdf_name,
                file_mtime,
                file_size,
                self.model_name,
                self.max_elements,
                self.chunk_size,
                summary
            )
            print("[+] DB (adobe_summary_cache) 저장 완료")
            return summary
        except Exception as e:
            print(f"[!] DB 저장 실패: {e}")     
        return ""   

def main():
    # 저장 경로 설정
    pdf_dir = os.path.join(BASE_DIR, "data", "rfp_files", "files")
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    
    summarizer = LocalPdfExtractSummarizer()
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        summarizer.get_extract_data(pdf_path, os.stat(pdf_path).st_size)

if __name__ == "__main__":
    main()
    