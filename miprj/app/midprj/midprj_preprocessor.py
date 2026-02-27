from .util_llmselector import get_llm
# ════════════════════════════════════════
# ▣ BidMatePreprocessor - 문서 전처리 모듈
# ════════════════════════════════════════
# 문서 로드(HWP/PDF), 청킹, 임베딩, 벡터스토어(FAISS) 관리

import os
import pickle
import re
import struct
import unicodedata
import zlib
from pathlib import Path

import numpy as np
import faiss
import olefile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from .midprj_defines import PARAMVAR
from .midprj_func import Lines, OpLog, filter_clean_for_rfp, normalize_model_item
from .midprj_sqlite import SQLiteDB
from .midprj_retriever import build_retriever_from_param

from .util_local_extract_summarizer import LocalPdfExtractSummarizer

class BidMatePreprocessor:

    def call_llm_summarize(text, param=None):
        """
        LLM을 이용해 입력 텍스트를 요약합니다.
        param: PARAMVAR 객체 (옵션, 없으면 기본 LLM 사용)
        """
        llm = get_llm(param) if param is not None else get_llm()
        prompt = f"아래 내용을 간결하게 요약해 주세요.\n\n{text}"
        try:
            result = llm.invoke(prompt)
            if isinstance(result, dict) and 'content' in result:
                return result['content']
            return str(result)
        except Exception as e:
            return f"[LLM 요약 실패] {e}"
    """입찰 문서 전처리: 로드, 청킹, 임베딩, 벡터스토어"""

    def __init__(self, param: PARAMVAR):
        self.myVer = "01"
        self.param = param
        self.data_dir = Path(self.param.rfp_data_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=param.chunk_size,
            chunk_overlap=param.chunk_overlap,
        )
        self.embeddings = None
        model_item = normalize_model_item(param.whatmodelItem)
        if model_item == "openai":
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model=param.embedding_model,
            )
        elif model_item == "ollama":
            ollama_embedding_model = (
                param.ollama_embedding_model_name
                if getattr(param, "ollama_embedding_model_name", "")
                else param.embedding_model
            )
            self.embeddings = OllamaEmbeddings(
                base_url=param.ollama_address,
                model=ollama_embedding_model,
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=param.embedding_model)

        self.vector_store = None
        self._pdfextract = LocalPdfExtractSummarizer()
        db = SQLiteDB()
        self.metadata_df = db.load_metadata()
        OpLog(f"메타데이터 로드 완료 (DB): {len(self.metadata_df)}개 레코드", True, "INFO")

    def normalize_filenames(self, directory):
        """파일명 NFC 정규화 (한글 자음/모음 분리 해결)"""
        target_dir = Path(directory)
        if not target_dir.exists():
            print(f"Directory not found: {target_dir}")
            return

        print(f"Scanning directory: {target_dir}")
        count = 0

        for file_path in target_dir.iterdir():
            if file_path.is_file():
                original_name = file_path.name
                normalized_name = unicodedata.normalize("NFC", original_name)
                if original_name != normalized_name:
                    new_path = target_dir / normalized_name
                    try:
                        file_path.rename(new_path)
                        print(f"Renamed: {original_name} -> {normalized_name}")
                        count += 1
                    except Exception as e:
                        print(f"Error renaming {original_name}: {e}")

        print(f"Normalization complete. {count} files renamed.")

    def _extract_hwp_text(self, file_path):
        """HWP 파일(v5) Record 구조 파싱, 텍스트(Tag ID 67) 추출"""
        try:
            f = olefile.OleFileIO(file_path)
            dirs = f.listdir()
            text = ""
            for d in dirs:
                if "BodyText" in d:
                    section = f.openstream(d).read()
                    data = None

                    if section[:2] in [b"\x78\x9c", b"\x78\xda", b"\x78\x01", b"\x78\x5e"]:
                        for wbits in [-15, 15, -zlib.MAX_WBITS, zlib.MAX_WBITS]:
                            try:
                                data = zlib.decompress(section, wbits)
                                Lines(f"✓ HWP BodyText zlib 압축 해제 성공 [{os.path.basename(file_path)}]")
                                break
                            except zlib.error:
                                continue

                        if data is None:
                            file_name = os.path.basename(file_path)
                            file_size = len(section)
                            OpLog(f"⚠️ HWP 압축 해제 실패 [{file_name}] (크기: {file_size} bytes) - 처리 스킵", level="WARN")
                            print(f"⚠️ HWP 압축 해제 실패, 건너뜀: {file_path}")
                            continue
                    else:
                        file_name = os.path.basename(file_path)
                        magic = section[:2].hex()
                        Lines(f"✓ HWP BodyText 비압축 형식 감지 [{file_name}] (매직: 0x{magic}) - 직접 파싱")
                        data = section

                    pos = 0
                    size_mp = len(data)

                    while pos < size_mp:
                        if pos + 4 > size_mp:
                            break

                        header = struct.unpack("<I", data[pos : pos + 4])[0]
                        pos += 4

                        tag_id = header & 0x3FF
                        rec_len = (header >> 20) & 0xFFF

                        if rec_len == 0xFFF:
                            if pos + 4 > size_mp:
                                break
                            rec_len = struct.unpack("<I", data[pos : pos + 4])[0]
                            pos += 4

                        if pos + rec_len > size_mp:
                            break

                        if tag_id == 67:
                            text_bytes = data[pos : pos + rec_len]
                            try:
                                decoded = text_bytes.decode("utf-16", errors="ignore")
                                clean = re.sub(r"[^가-힣a-zA-Z0-9\s.,()\-\[\]]", " ", decoded)
                                clean = re.sub(r"\s+", " ", clean).strip()
                                if len(clean) > 0:
                                    text += clean + " "
                            except Exception:
                                pass

                        pos += rec_len

            content = re.sub(r"\s+", " ", text).strip()
            content = filter_clean_for_rfp(content)
            if not content:
                return []

            doc = Document(
                page_content=content,
                metadata={"file_name": Path(file_path).name},
            )
            splits = self.text_splitter.split_documents([doc])
            return splits

        except Exception as e:
            print(f"❌ HWP 추출 오류 ({file_path}): {e}")
            return []

    def _get_adobe_table_figure_summary(self, pdf_name):
        """PDF 표/그림 요약 (Adobe PDF Extract API, SQLite 캐시)"""
        table_figure_summary = ""
        try:
            full_path = os.path.join(self.data_dir, pdf_name)
            stat = os.stat(full_path)
            file_mtime = stat.st_mtime
            file_size = stat.st_size
            cached_summary = self._pdfextract.get_extract_data(
                full_path, file_size)
            if cached_summary:
                table_figure_summary = cached_summary
                Lines("   ▣ Adobe 요약 캐시 HIT")
            else:
                Lines("   ▣ Adobe 요약 캐시 없음 (main에서 일괄 저장 필요)")
        except Exception as e:
            OpLog(f"   ⚠️  Adobe 요약 실패: {e}", level="WARN",bLines=True)
        return table_figure_summary

    def read_pdf_table_figure(self, pdf_path):
        """
        PDF에서 추출된 표/그림 요약 파일(.txt)을 읽어,
        - 표는 마크다운(또는 원본 텍스트) 그대로 반환
        - 그림(다이어그램/도식)은 LLM 기반 자연어 요약문으로 변환하여 반환
        """
        txt_path = str(pdf_path.with_suffix(".txt"))
        summary_text = ""
        if not os.path.exists(txt_path):
            Lines(f"⚠️ 표/그림 요약 파일 없음: {txt_path}")
            return ""

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                summary_text = f.readlines()
        except Exception as e:
            Lines(f"⚠️ 표/그림 요약 읽기 실패: {e}")
            summary_text = ""

        # summary_text가 list이면 문자열로 변환
        if isinstance(summary_text, list):
            summary_text = "".join(summary_text)

        # <visual_diagram> 태그로 감싸진 부분 제거
        summary_text = re.sub(r"<visual_diagram>.*?</visual_diagram>", "", summary_text, flags=re.DOTALL)

        return summary_text


    def _extract_pdf_text(self, pdf_file):
        """단일 PDF 텍스트/표/그림 추출 및 청크 분할"""
        pdf_path = Path(pdf_file)
        base_text = ""
        try:
            text_loader = PyPDFLoader(str(pdf_path))
            pages = text_loader.load()
            base_text = "\n".join([p.page_content for p in pages])
        except Exception as e:
            OpLog(f"   ⚠️  텍스트 추출 실패: {str(e)}", level="WARN",bLines= True)

        if base_text:
            base_text = filter_clean_for_rfp(base_text)

        table_figure_summary = self.read_pdf_table_figure(pdf_path)

        if not base_text and not table_figure_summary:
            print("❌ 오류: 추출된 내용이 없습니다.")
            return []

        combined_sections = []
        if base_text:
            combined_sections.append(base_text)
        if table_figure_summary:
            combined_sections.append("표/그림 요약:\n" + table_figure_summary)

        Lines(f"   ▣ 표/그림 요약 포함: {bool(table_figure_summary)} (길이: {len(table_figure_summary)})")

        combined_content = "\n\n".join(combined_sections).strip()
        all_docs = [
            Document(page_content=combined_content, metadata={"file_name": pdf_path.name})
        ]

        Lines(f"\n📊 총 {len(all_docs)}개 문서 추출 완료")
        Lines(f"✂️  청크 분할 중..chunk_size={self.param.chunk_size}, chunk_overlap={self.param.chunk_overlap}")

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.param.chunk_size,
                chunk_overlap=self.param.chunk_overlap,
            )
            all_splits = text_splitter.split_documents(all_docs)
            if not all_splits:
                print("❌ 오류: 청크 분할 후 all_splits가 비어있습니다.")
                return []
            Lines(f"✓ 청크 분할 완료: 총 {len(all_splits)}개 청크 생성")
            return all_splits
        except Exception as e:
            print(f"❌ 청크 분할 실패: {str(e)}")
            OpLog(f"❌ 청크 분할 실패: {str(e)}", level="ERROR")
            return []

    def _exist_all_docs(self):
        """캐시된 all_docs 로드"""
        all_docs_names = (
            f"all_docs_cs_{self.param.chunk_size}_co_{self.param.chunk_overlap}_"
            f"{self.param.store_ver}_{self.myVer}"
        )
        if self.param.newCreate:
            return None, all_docs_names
        db = SQLiteDB()
        try:
            blob_bytes = db.load_blob(all_docs_names)
            if blob_bytes:
                all_docs = pickle.loads(blob_bytes)
                OpLog(f"✅ 캐시된 all_docs 로드 완료: {all_docs_names} ({len(all_docs)}개)", level="INFO")
                return all_docs, all_docs_names
            OpLog(f"⚠️ 캐시된 all_docs 없음: {all_docs_names}", level="WARN")
            return None, all_docs_names
        except Exception as e:
            OpLog(f"⚠️ all_docs 로드 실패: {e}", level="WARN")
            return None, all_docs_names

    def get_all_docs(self):
        """전체 문서 로드 및 청킹"""
        OpLog("전체 문서 로드 시작", level="INFO")
        re_all_docs, all_docs_names = self._exist_all_docs()
        if re_all_docs is not None:
            OpLog("전체 문서 캐시 사용", level="INFO")
            return re_all_docs

        all_docs = []

        def get_field(row, kor_key, eng_key, default=""):
            value = row.get(eng_key, None)
            if value is None:
                value = row.get(kor_key, default)
            return value

        def change_hwpext_to_pdfext(file_path):
            path = Path(file_path)
            suffix = path.suffix.lower()
            if suffix not in {".hwp", ".hwpx"}:
                Lines(f"   📁 한글 확장자 아님: {path.name} (확장자: {suffix})")
                return str(path)
            pdf_path = path.with_suffix(".pdf")
            if pdf_path.exists():
                Lines(f"   📁 한글 확장자 -> PDF로 변경: {path.name} -> {pdf_path.name}")
                return str(pdf_path)
            return str(path)

        for _, row in self.metadata_df.iterrows():
            file_name = unicodedata.normalize("NFC", str(get_field(row, "파일명", "file_name", "")))
            file_path = os.path.join(self.param.rfp_data_dir, file_name)
            file_path = change_hwpext_to_pdfext(file_path)
            if not os.path.exists(file_path):
                print(f"⚠️ 파일 없음: {file_name}")
                continue

            print(f"📄 처리 중: {file_name}")
            content = ""
            file_splits = None
            effective_suffix = Path(file_path).suffix.lower()
            if effective_suffix == ".pdf":
                file_splits = self._extract_pdf_text(file_path)
            elif effective_suffix in {".hwp", ".hwpx"}:
                file_splits = self._extract_hwp_text(file_path)
            else:
                print(f"   ⚠️ 지원하지 않는 파일 형식: {file_name} (확장자: {effective_suffix})")
                continue

            notice_no = get_field(row, "공고 번호", "Notice_no")
            notice_round = get_field(row, "공고 차수", "Notice_round")
            project_name = get_field(row, "사업명", "project_name")
            budget = get_field(row, "사업 금액", "budget")
            agency = get_field(row, "발주 기관", "agency")
            publish_date = get_field(row, "공개 일자", "publish_date")
            participation_start_date = get_field(row, "입찰 참여 시작일", "participation_start_date")
            participation_end_date = get_field(row, "입찰 참여 마감일", "participation_end_date")
            project_summary = get_field(row, "사업 요약", "project_summary")
            file_type = get_field(row, "파일형식", "file_type")
            keywords = get_field(row, "keywords", "keywords")
            region = get_field(row, "region", "region")

            add_content = (
                f"파일이름:{file_name}, 공고번호:{notice_no}, 공고차수:{notice_round}, "
                f"사업명:{project_name}, 예산:{budget}, 발주기관:{agency}, 공개일자:{publish_date}, "
                f"입찰시작일:{participation_start_date}, 입찰마감일:{participation_end_date}, "
                f"사업요약:{project_summary}, 파일형식:{file_type}, 파일명:{file_name}"
            )
            Lines(add_content)
            content += "\n" + add_content
            base_metadata = {
                "Notice_no": notice_no,
                "Notice_round": notice_round,
                "project_name": project_name,
                "budget": budget,
                "agency": agency,
                "publish_date": publish_date,
                "participation_start_date": participation_start_date,
                "participation_end_date": participation_end_date,
                "project_summary": project_summary,
                "file_type": file_type,
                "file_name": file_name,
                "keywords": keywords,
                "region": region,
            }

            if file_splits:
                for split_doc in file_splits:
                    split_doc.page_content = split_doc.page_content + "\n" + add_content
                    split_doc.metadata.update(base_metadata)
                all_docs.extend(file_splits)
            else:
                doc = Document(page_content=content, metadata=base_metadata)
                splits = self.text_splitter.split_documents([doc])
                all_docs.extend(splits)

            summary_doc = Document(
                page_content=f"사업 요약: {project_summary}\n핵심 목표: {project_name}",
                metadata={
                    "type": "summary",
                    "Notice_no": notice_no,
                    "project_name": project_name,
                    "budget": budget,
                    "agency": agency,
                },
            )
            all_docs.append(summary_doc)

        try:
            blob_bytes = pickle.dumps(all_docs)
            db = SQLiteDB()
            db.save_blob(all_docs_names, blob_bytes)
            OpLog(f"✅ all_docs 캐시 저장 완료: {all_docs_names} ({len(all_docs)}개)", level="INFO")
        except Exception as e:
            OpLog(f"⚠️ all_docs 캐시 저장 실패: {e}", level="WARN")
        OpLog(f"전체 문서 로드 완료: {len(all_docs)}개", level="INFO")
        return all_docs

    def get_ensemble_retriever(self, vector_store, llm=None):
        """Ensemble Retriever 생성"""
        all_docs = self.get_all_docs()
        return build_retriever_from_param(vector_store, all_docs, self.param, llm=llm)

    def make_faiss_name(self):
        """FAISS 벡터스토어 캐시 이름 생성"""
        vector_name = f"{self.param.embedding_model.replace('/', '_')}_{self.param.llm_model.replace('/', '')}"
        return f"faiss_store_{vector_name}_cs_{self.param.chunk_size}_co_{self.param.chunk_overlap}_{self.param.store_ver}"

    def _check_vector_store_exists(self, faiss_name):
        """기존 벡터 스토어 로드"""
        vector_store = None
        db = SQLiteDB()
        blob_bytes = db.load_blob(faiss_name)
        if blob_bytes:
            try:
                vector_store = FAISS.deserialize_from_bytes(
                    blob_bytes,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"✅ 기존 Vector DB 로드 완료: {faiss_name}")
                OpLog(f"Vector DB 캐시 로드 완료: {faiss_name}", level="INFO")
            except Exception as e:
                print(f"⚠️ Vector DB 역직렬화 실패: {e}")
                OpLog(f"Vector DB 역직렬화 실패: {e}", level="WARN")
        else:
            print(f"⚠️ 기존 Vector DB 없음: {faiss_name}")
            OpLog(f"Vector DB 캐시 없음: {faiss_name}", level="INFO")
        return vector_store

    def get_hugging_vector_store(self, faiss_name):
        """HuggingFace 임베딩으로 FAISS 벡터스토어 생성"""
        OpLog("HuggingFace Vector DB 생성 시작", level="INFO")
        all_docs = self.get_all_docs()
        embedding_dim = len(self.embeddings.embed_query("hello world"))
        index = faiss.IndexFlatIP(embedding_dim)
        texts = [doc.page_content for doc in all_docs]
        vectors = [self.embeddings.embed_query(text) for text in texts]
        vectors = np.array(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-10)
        docstore_dict = {str(i): doc for i, doc in enumerate(all_docs)}
        index_to_docstore_id = {i: str(i) for i in range(len(all_docs))}
        self._vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(docstore_dict),
            index_to_docstore_id=index_to_docstore_id,
        )
        self._vector_store.index.add(vectors)
        print(f"🚀 [huggingface] 총 {len(all_docs)}개 청크를 벡터화하여 저장합니다...")
        self.vector_store = FAISS.from_documents(all_docs, self.embeddings)
        blob_bytes = self.vector_store.serialize_to_bytes()
        db = SQLiteDB()
        db.save_blob(faiss_name, blob_bytes)
        OpLog(f"HuggingFace Vector DB 저장 완료: {faiss_name}", level="INFO")
        return self.vector_store

    def get_openai_vector_store(self, faiss_name):
        """OpenAI 임베딩으로 FAISS 벡터스토어 생성"""
        OpLog("OpenAI Vector DB 생성 시작", level="INFO")
        all_docs = self.get_all_docs()
        print(f"🚀 [openai] 총 {len(all_docs)}개 청크를 벡터화하여 저장합니다...")
        self.vector_store = FAISS.from_documents(all_docs, self.embeddings)
        blob_bytes = self.vector_store.serialize_to_bytes()
        db = SQLiteDB()
        db.save_blob(faiss_name, blob_bytes)
        OpLog(f"OpenAI Vector DB 저장 완료: {faiss_name}", level="INFO")
        return self.vector_store

    def get_ollama_vector_store(self, faiss_name):
        """Ollama 임베딩으로 FAISS 벡터스토어 생성"""
        OpLog("Ollama Vector DB 생성 시작", level="INFO")
        all_docs = self.get_all_docs()
        print(f"🚀 [ollama] 총 {len(all_docs)}개 청크를 벡터화하여 저장합니다...")
        self.vector_store = FAISS.from_documents(all_docs, self.embeddings)
        blob_bytes = self.vector_store.serialize_to_bytes()
        db = SQLiteDB()
        db.save_blob(faiss_name, blob_bytes)
        OpLog(f"Ollama Vector DB 저장 완료: {faiss_name}", level="INFO")
        return self.vector_store
