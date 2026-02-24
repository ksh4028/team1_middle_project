import argparse
import csv
import io
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv


from midprj_defines import BASE_DIR, OPENAI_SETTINGS
from util_llmselector import get_llm
from langchain_core.prompts import ChatPromptTemplate


_SDK_IMPORT_ERROR = None

try:
	from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
	from adobe.pdfservices.operation.pdf_services import PDFServices
	from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
	from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
	from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
	from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
	from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType
	from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
except ImportError as e:
	_SDK_IMPORT_ERROR = e
	ServicePrincipalCredentials = None

# Adobe PDF Extract API 을 이용해서 PDF 문서에서 표와 그림을 추출하고,
# 추출된 내용을 LLM을 통해 요약하는 예제 코드입니다.

load_dotenv(os.path.join(BASE_DIR, ".env"))

#파리미터로 받아 오던 것을 
# ini 파일로 변경 하여 읽어서 사용하도록 변경 
# 또한 .env는 에서 가져오는 정보는 이제 필요 없으므로 삭제.
# [adobe]
# model_name = openai-5-mini
# max_elements = 30
# chunk_size = 150



import configparser

class AdobePdfExtractSummarizer:
	def __init__(self):
		"""
		Adobe PDF Extract API 기반 PDF 표/그림 요약 클래스 초기화
		ini 파일의 [adobe] 섹션에서 model_name, max_elements, chunk_size를 읽어 사용
		"""
		# ini 파일에서 파라미터 로드
		ini_path = os.path.join(os.path.dirname(__file__), "midPrj.ini")
		config = configparser.ConfigParser()
		config.read(ini_path, encoding="utf-8")
		section = config["adobe"] if "adobe" in config else {}
		self.model_name = section.get("model_name", OPENAI_SETTINGS["llm_model_name"])
		self.max_elements = int(section.get("max_elements", 20))
		self.chunk_size = int(section.get("chunk_size", 120))

		try:
			provider = None
			if hasattr(self, 'param') and hasattr(self.param, 'llm_selector') and self.param.llm_selector:
				provider = self.param.llm_selector
			self.llm = get_llm(provider=provider)
			print(f"[*] LLM 초기화 시도 중...")
			self.llm.invoke("Hi")
			print("[+] LLM 초기화 완료")
		except Exception as e:
			print(f"[-] LLM 초기화 실패 (API Key 확인 필요): {e}")
			self.llm = None

		# adobecredentials.json에서 자격증명 로드
		cred_path = os.path.join(BASE_DIR, "parangnabang.json")
		self.client_id = None
		self.client_secret = None
		self.client_token = None
		if os.path.exists(cred_path):
			try:
				with open(cred_path, "r", encoding="utf-8") as f:
					cred = json.load(f)
				self.client_id = cred.get("CLIENT_ID")
				# CLIENT_SECRETS가 리스트로 저장되어 있음
				secrets = cred.get("CLIENT_SECRETS")
				if isinstance(secrets, list) and secrets:
					self.client_secret = secrets[0]
				else:
					self.client_secret = cred.get("CLIENT_SECRET")
				# 토큰이 있으면 추가로 할당
				self.client_token = cred.get("CLIENT_TOKEN")
			except Exception as e:
				print(f"[-] adobecredentials.json 로드 실패: {e}")
		else:
			self.client_id = os.getenv("ADOBE_PDF_SERVICES_CLIENT_ID")
			self.client_secret = os.getenv("ADOBE_PDF_SERVICES_CLIENT_SECRET")
			self.client_token = os.getenv("ADOBE_PDF_SERVICES_CLIENT_TOKEN")
		self.has_sdk = ServicePrincipalCredentials is not None

		if not self.has_sdk:
			print("[-] Adobe PDF Services SDK가 설치되지 않았습니다.")
			if _SDK_IMPORT_ERROR:
				print(f"[*] ImportError: {_SDK_IMPORT_ERROR}")
			print("[*] 아래 명령을 사용해 현재 인터프리터에 설치하세요.")
			print(f"    {sys.executable} -m pip install pdfservices-sdk")
			print(f"[*] Python 실행 경로: {sys.executable}")
		elif not self.client_id or not self.client_secret:
			print("[-] Adobe PDF Services API 자격 증명이 없습니다. .env에 ID/SECRET을 설정하세요.")
		else:
			print("[+] Adobe PDF Services SDK 준비 완료")

	def extract_with_adobe(self, pdf_path, output_dir=None, save_zip=False):
		"""
		Adobe PDF Extract API를 사용하여 PDF에서 표와 그림을 추출합니다.
		"""
		if not self.has_sdk:
			return []

		if not self.client_id or not self.client_secret:
			return []

		if not os.path.exists(pdf_path):
			print(f"[-] 파일을 찾을 수 없습니다: {pdf_path}")
			return []

		print(f"\n[*] Adobe Extract 시작: {os.path.basename(pdf_path)}")

		try:
			credentials = ServicePrincipalCredentials(
				client_id=self.client_id,
				client_secret=self.client_secret
			)
			pdf_services = PDFServices(credentials=credentials)

			with open(pdf_path, "rb") as f:
				input_stream = f.read()

			input_asset = pdf_services.upload(
				input_stream=input_stream,
				mime_type=PDFServicesMediaType.PDF
			)
			extract_pdf_params = ExtractPDFParams(

					
    			# TEXT를 추가하여 본문 내용도 가져오게 변경
    			elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES], 
    			elements_to_extract_renditions=[
        		ExtractRenditionsElementType.TABLES,
        		ExtractRenditionsElementType.FIGURES
    			]
			)
			extract_pdf_job = ExtractPDFJob(
				input_asset=input_asset,
				extract_pdf_params=extract_pdf_params
			)

			location = pdf_services.submit(extract_pdf_job)
			pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)
			result_asset = pdf_services_response.get_result().get_resource()
			stream_asset = pdf_services.get_content(result_asset)

			content = stream_asset.get_input_stream()
			if hasattr(content, "read"):
				content = content.read()
			if save_zip:
				output_dir = output_dir or os.path.join(BASE_DIR, "data", "extract_output")
				os.makedirs(output_dir, exist_ok=True)
				timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
				zip_path = os.path.join(output_dir, f"extract_{timestamp}.zip")
				with open(zip_path, "wb") as f:
					f.write(content)
				print(f"[+] 추출 결과 저장: {zip_path}")

			return self._parse_extract_zip_bytes(content)

		except Exception as e:
			print(f"[-] Adobe Extract 실패: {e}")
			return []

	def _format_element_summary(self, element):
		"""
		추출 요소의 핵심 정보를 문자열로 정리합니다.
		"""
		path = element.get("Path", "")
		page = element.get("Page")
		bounds = element.get("Bounds")
		text = element.get("Text") or element.get("Alt") or ""

		attributes = element.get("Attributes") or element.get("attributes") or {}
		caption = attributes.get("Caption") or attributes.get("caption") or ""

		parts = []
		if caption:
			parts.append(f"Caption: {caption}")
		if text:
			parts.append(f"Text: {text}")
		if path:
			parts.append(f"Path: {path}")
		if page is not None:
			parts.append(f"Page: {page}")
		if bounds:
			parts.append(f"Bounds: {bounds}")

		return "\n".join(parts).strip()

	def _add_element_to_extracted_data(self, element, extracted_data):
		path = element.get("Path", "")
		el_type = element.get("Type", "") # 대소문자 주의 (보통 API는 'Text', 'Table' 등으로 반환)
		summary = self._format_element_summary(element)

		# --- 수정 시작 ---
		# 텍스트가 너무 짧은 것(페이지 번호 등)은 제외하고 본문 텍스트 추가
		if el_type == "Text" and len(summary) > 10: 
			extracted_data.append({
				"type": "Text",
				"content": summary
			})
		# --- 기존 코드 유지 ---
		elif "Table" in path or el_type == "Table":
			extracted_data.append({
				"type": "Table",
				"content": summary or path
			})
		elif "Figure" in path or el_type == "Figure":
			extracted_data.append({
				"type": "Figure/Image",
				"content": summary or path
			})

	def _parse_extract_zip(self, zip_path):
		"""
		Adobe Extract 결과 ZIP에서 structuredData.json을 파싱합니다.
		"""
		extracted_data = []

		try:
			with zipfile.ZipFile(zip_path, "r") as zip_ref:
				json_files = [name for name in zip_ref.namelist() if name.endswith("structuredData.json")]
				if not json_files:
					print("[-] structuredData.json을 찾을 수 없습니다.")
					return []

				with zip_ref.open(json_files[0]) as f:
					data = json.load(f)

			for element in data.get("elements", []):
				self._add_element_to_extracted_data(element, extracted_data)

			if not extracted_data:
				elements_preview = json.dumps(data.get("elements", [])[:5], ensure_ascii=True)
				extracted_data.append({
					"type": "Extract Metadata",
					"content": elements_preview
				})

			print(f"[+] 추출 성공: 총 {len(extracted_data)}개의 요소를 찾았습니다.")

		except Exception as e:
			print(f"[-] ZIP 파싱 실패: {e}")

		return extracted_data

	def _parse_extract_zip_bytes(self, zip_bytes):
		"""
		메모리 상의 ZIP bytes에서 structuredData.json을 파싱합니다.
		"""
		extracted_data = []

		if not zip_bytes:
			return []

		try:
			with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_ref:
				json_files = [name for name in zip_ref.namelist() if name.endswith("structuredData.json")]
				if not json_files:
					print("[-] structuredData.json을 찾을 수 없습니다.")
					return []

				with zip_ref.open(json_files[0]) as f:
					data = json.load(f)

			for element in data.get("elements", []):
				self._add_element_to_extracted_data(element, extracted_data)

			if not extracted_data:
				elements_preview = json.dumps(data.get("elements", [])[:5], ensure_ascii=True)
				extracted_data.append({
					"type": "Extract Metadata",
					"content": elements_preview
				})

			print(f"[+] 추출 성공: 총 {len(extracted_data)}개의 요소를 찾았습니다.")

		except Exception as e:
			print(f"[-] ZIP 파싱 실패: {e}")

		return extracted_data

	def summarize_content(self, elements, max_elements=20):
		"""
		추출된 데이터를 LLM으로 요약합니다.
		"""

		if not elements:
			return "추출된 요소가 없어 요약을 생성할 수 없습니다."

		# max_elements가 명시적으로 주어지지 않으면 self.max_elements 사용
		if max_elements is None:
			max_elements = self.max_elements

		context = ""
		for i, el in enumerate(elements[:max_elements]):
			context += f"\n--- Element {i + 1} ({el['type']}) ---\n{el['content'][:2000]}\n"

		if not self.llm:
			return f"LLM 요약 불가 (API Key 확인).\n데이터 샘플:\n{context[:1000]}"

		print("[*] Adobe Extract 데이터 기반 LLM 요약 중...")

		prompt = ChatPromptTemplate.from_messages([
			(
				"system",
				"당신은 PDF의 표/그림 정보를 벡터 검색에 적합한 평문 요약으로 정리하는 전문가입니다. "
				"표는 문장형으로 핵심 수치/지표만 요약하고, 그림은 핵심 메시지를 짧게 설명하세요. "
				"불필요한 서론 없이 한국어로 간결히 작성하세요."
			),
			("human", "다음은 추출된 PDF 요소들입니다. 핵심 내용을 요약해 주세요:\n\n{context}")
		])

		try:
			chain = prompt | self.llm
			response = chain.invoke({"context": context})
			return response.content
		except Exception as e:
			return f"요약 중 오류 발생: {e}\n\n데이터 샘플:\n{context[:500]}"

	def summarize_by_type(self, elements, max_elements=20):
		"""
		요소 타입별로 개별 요약을 생성합니다.
		"""
		tables = [el for el in elements if el.get("type") == "Table"]
		figures = [el for el in elements if el.get("type") == "Figure/Image"]

		# max_elements가 명시적으로 주어지지 않으면 self.max_elements 사용
		if max_elements is None:
			max_elements = self.max_elements

		summaries = {}
		if tables:
			summaries["Table"] = self.summarize_content(tables, max_elements=max_elements)
		if figures:
			summaries["Figure/Image"] = self.summarize_content(figures, max_elements=max_elements)

		return summaries

	def _chunk_elements(self, elements, chunk_size):
		if not elements:
			return []
		chunk_size = max(1, int(chunk_size))
		return [elements[i:i + chunk_size] for i in range(0, len(elements), chunk_size)]

	def _summarize_elements_chunked(self, elements, chunk_size=100, max_elements=20):
		"""
		요소를 여러 덩어리로 나눠 요약 후, 다시 통합 요약합니다.
		"""
		if not elements:
			return ""

		# chunk_size, max_elements가 명시적으로 주어지지 않으면 self.chunk_size, self.max_elements 사용
		if chunk_size is None:
			chunk_size = self.chunk_size
		if max_elements is None:
			max_elements = self.max_elements

		chunks = self._chunk_elements(elements, chunk_size)
		partial_summaries = []
		for idx, chunk in enumerate(chunks, start=1):
			print(f"[*] Adobe 요약 분할 진행: {idx}/{len(chunks)}")
			partial = self.summarize_content(chunk, max_elements=max_elements)
			if partial:
				partial_summaries.append(partial)

		if not partial_summaries:
			return ""

		merged_context = "\n\n".join(
			[f"--- Summary Part {i + 1} ---\n{text}" for i, text in enumerate(partial_summaries)]
		)
		merged_elements = [{"type": "Summary", "content": merged_context}]
		return self.summarize_content(merged_elements, max_elements=1)

	def summarize_pdf_elements(self, pdf_path, max_elements=20, chunk_size=120):
		"""
		PDF의 표/그림 요소를 요약 텍스트로 반환합니다.
		"""
		elements = self.extract_with_adobe(pdf_path, save_zip=False)
		if not elements:
			return ""

		# max_elements, chunk_size가 명시적으로 주어지지 않으면 self.max_elements, self.chunk_size 사용
		if max_elements is None:
			max_elements = self.max_elements
		if chunk_size is None:
			chunk_size = self.chunk_size

		summaries = self.summarize_by_type(elements, max_elements=max_elements)
		if summaries:
			ordered = []
			for label in ("Table", "Figure/Image"):
				text = summaries.get(label)
				if text:
					ordered.append(f"[{label}]\n{text}")
			return "\n\n".join(ordered)

		return self._summarize_elements_chunked(
			elements,
			chunk_size=chunk_size,
			max_elements=max_elements,
		)



def build_arg_parser():
	default_pdf = os.path.join(
		BASE_DIR,
		"data",
		"rfp_files",
		"files",
		"(사)벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업 .pdf"
	)

	parser = argparse.ArgumentParser(
		description="Adobe PDF Extract API로 표/그림을 추출하고 LLM 요약을 생성합니다."
	)
	parser.add_argument("--pdf", default=default_pdf, help="추출할 PDF 파일 경로")
	parser.add_argument("--output-dir", default=None, help="추출 결과 ZIP 저장 폴더")
	parser.add_argument(
		"--model",
		default=OPENAI_SETTINGS["llm_model_name"],
		help="OpenAI 모델 이름 (ini의 openai_model 사용)",
	)
	parser.add_argument("--max-elements", type=int, default=20, help="요약에 포함할 최대 요소 수")
	parser.add_argument("--save-md", action="store_true", help="요약 결과를 Markdown으로 저장")
	parser.add_argument("--save-csv", action="store_true", help="추출 요소를 CSV로 저장")
	parser.add_argument("--separate-summary", action="store_true", help="표/그림 요약을 분리 출력")

	return parser


def main_org(pdf_path, output_dir=None):
	"""
	PDF 요약 실행 (ini 기반 Summarizer)
	:param pdf_path: 분석할 PDF 파일 경로
	:param output_dir: 결과 저장 폴더 (기본값: BASE_DIR/data/extract_output)
	"""
	output_dir = output_dir or os.path.join(BASE_DIR, "data", "extract_output")
	os.makedirs(output_dir, exist_ok=True)

	summarizer = AdobePdfExtractSummarizer()
	pdf_dir = os.path.join(BASE_DIR, "data", "rfp_files", "files")
	pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
	print(f"[INFO] {len(pdf_files)}개 PDF 파일 분석 시작: {pdf_dir}")
	for pdf_file in pdf_files:
		pdf_path = os.path.join(pdf_dir, pdf_file)
		print(f"\n{'='*60}\n[PDF] {pdf_file}")
		elements = summarizer.extract_with_adobe(pdf_path, output_dir=output_dir)
		summary = summarizer.summarize_content(elements)
		separate_summaries = summarizer.summarize_by_type(elements)

		base_name = os.path.splitext(os.path.basename(pdf_path))[0]
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

		print("\n" + "=" * 60)
		print("🚀 Adobe PDF Extract 기반 분석 결과 요약")
		print("=" * 60)
		print(summary)
		print("=" * 60)

		if separate_summaries:
			print("\n" + "=" * 60)
			print("🚀 표/그림 분리 요약")
			print("=" * 60)
			for label, text in separate_summaries.items():
				print(f"\n[{label}]\n{text}")
			print("=" * 60)

		# 저장 옵션은 Summarizer의 ini 설정을 따르거나, 필요시 별도 함수로 분리 가능
		csv_path = os.path.join(output_dir, f"{base_name}_elements_{timestamp}.csv")
		with open(csv_path, "w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(["type", "content"])
			for el in elements:
				writer.writerow([el.get("type"), el.get("content")])
		print(f"[+] CSV 저장 완료: {csv_path}")

		md_path = os.path.join(output_dir, f"{base_name}_summary_{timestamp}.md")
		with open(md_path, "w", encoding="utf-8") as f:
			f.write("# Adobe PDF Extract 요약\n\n")
			f.write("## 통합 요약\n\n")
			f.write(summary + "\n\n")
			if separate_summaries:
				f.write("## 표/그림 분리 요약\n\n")
				for label, text in separate_summaries.items():
					f.write(f"### {label}\n\n{text}\n\n")
		print(f"[+] Markdown 저장 완료: {md_path}")


# 새 main: 폴더 내 모든 PDF 분석 후 adobe_summary_cache에 저장
def main(output_dir=None):
	from midprj_sqlite import SQLiteDB
	output_dir = output_dir or os.path.join(BASE_DIR, "data", "extract_output")
	os.makedirs(output_dir, exist_ok=True)

	summarizer = AdobePdfExtractSummarizer()
	pdf_dir = os.path.join(BASE_DIR, "data", "rfp_files", "files")
	pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
	print(f"[INFO] {len(pdf_files)}개 PDF 파일 분석 및 DB 저장 시작: {pdf_dir}")
	db = SQLiteDB()
	for pdf_file in pdf_files:
		pdf_path = os.path.join(pdf_dir, pdf_file)
		print(f"\n{'='*60}\n[PDF] {pdf_file}")
		elements = summarizer.extract_with_adobe(pdf_path, output_dir=output_dir)
		summary = summarizer.summarize_content(elements)
		# 캐시 저장
		try:
			stat = os.stat(pdf_path)
			file_mtime = stat.st_mtime
			file_size = stat.st_size
			db.save_adobe_summary_cache(
				str(pdf_path), file_mtime, file_size,
				summarizer.model_name, summarizer.max_elements, summarizer.chunk_size,
				summary
			)
			print("[+] DB 캐시 저장 완료")
		except Exception as e:
			print(f"[!] DB 저장 실패: {e}")


if __name__ == "__main__":
	
	parser = build_arg_parser()
	args = parser.parse_args()
	main(output_dir=args.output_dir)