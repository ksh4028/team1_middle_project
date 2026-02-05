import os
import pandas as pd
import olefile
import zlib
import unicodedata
import struct
import pickle
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import shutil
from pathlib import Path
import unicodedata
import sqlite3
import datetime
import sys
from filelock import FileLock
import torch
from dataclasses import dataclass

# LangChain - Vector Stores & Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

# LangChain - Core
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LangChain - Document Loaders
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

# LangChain - Vector Stores & Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
import numpy as np
import faiss
# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# 환경 변수
from dotenv import load_dotenv

BASE_DIR =  r"D:\project\TodoPrj" if os.path.exists(r"D:\project\TodoPrj") else "/home/nabi/project"
DATA_DIR =  os.path.join(BASE_DIR, "data")
# .env 파일 로드
ENV_FILE = os.path.join(BASE_DIR, ".env")

CSV_PATH = os.path.join(DATA_DIR,"rfp_files", "data_list.csv")
RFP_DATA_DIR = os.path.join(DATA_DIR, "rfp_files","files")
MY_FILE_VER = f"midprj_01"
#_{os.path.basename(__file__).split('.')[0]}.01"

SQLITEDB_DIR = os.path.join(DATA_DIR, "dbfile")
SQLITEDB_PATH = os.path.join(SQLITEDB_DIR, f"{MY_FILE_VER}.db")

LOG_DIR = os.path.join(DATA_DIR, "log")
LOG_FILE = os.path.join(LOG_DIR, f"{MY_FILE_VER}.txt")
IS_GPU = torch.cuda.is_available()

STORE_VER = "V05"

def init_Env():
    print("환경 변수 초기화 및 디렉토리 생성")
    load_dotenv(ENV_FILE)
    if not os.path.exists(SQLITEDB_DIR):
        os.makedirs(SQLITEDB_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

init_Env()


# ════════════════════════════════════════
# ▣ 유틸리티 함수 
# ════════════════════════════════════════
## 시간 함수
def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 문자 구분선 및 메시지 출력 함수 복구
def Lines(text=None, count=100):
    print("═" * count)
    if text is not None:
        print(text)
        print("═" * count)


## 로그 함수
def OpLog(log,bLines = False):
    if bLines:
        Lines(log)
    try:
        frame = sys._getframe(1)
        caller_name = frame.f_code.co_name
        caller_line = frame.f_lineno
    except Exception:
        caller_name = "UnknownFunction"
        caller_line = 0

    log_lock_filename = LOG_FILE + ".lock"
    log_content = f"[{now_str()}] {caller_name}:{caller_line}: {log}\n"
    try:
        with FileLock(log_lock_filename, timeout=10):
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(log_content)
    except Exception as e:
        print(f"Log write error: {e}")



@dataclass
class PARAMVAR: 
    embedding_model:str = "text-embedding-3-small"
    llm_model:str = "gpt-4o-mini"
    chunk_size:int = 1000
    chunk_overlap:int = 100
    temperature:float = 0.2
    repetition_penalty:float = 1.2
    query:str = ""
    answer:str = ""
    start_time:str = "2000-01-01 00:00:00"
    end_time:str = "2001-01-01 00:00:00"
    is_gpu:bool = IS_GPU
    csv_path:str =  CSV_PATH
    rfp_data_dir:str = RFP_DATA_DIR
    k : int = 5
    is_openai : bool = True
    newCreate : bool = False


# ════════════════════════════════════════
# ▣ SQLite 데이터베이스 핸들러 클래스
# ════════════════════════════════════════
class SQLiteDB:
## SQLiteDB 초기화 및 테이블 생성
    def __init__(self):
        Lines("SQLiteDB 초기화 시작")
        self._db_path = SQLITEDB_PATH
        self.connection = None
        self.cursor = None
        self.create_table()
        Lines("SQLiteDB 초기화 완료")

## 데이터베이스 연결 생성
    def _connect(self):
        self._close()
        self.connection = sqlite3.connect(self._db_path)
        self.cursor = self.connection.cursor()

## 데이터베이스 연결 종료
    def _close(self):
        if self.cursor is not None:
            self.cursor.close()
            self.cursor = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    ## 필요한 데이터 테이블 생성 (청크 기반 저장)
    def create_table(self):
        OpLog("테이블 생성 시작")
        # 청크 단위 저장용 테이블 (모든 데이터를 청크로 저장)
        self.execute('''CREATE TABLE IF NOT EXISTS blob_data (
            blob_name TEXT ,
            blob_index INTEGER,
            blob_content BLOB,
            PRIMARY KEY (blob_name, blob_index)
            )
            ''')
        self.execute('''CREATE TABLE IF NOT EXISTS result_data (
            execute_index integer,
            model_item TEXT, -- OpenAI, HuggingFace 
            embedding_model TEXT,
            llm_model TEXT,
            temperature REAL,
            repetition_penalty REAL,          
            query_index INTEGER,
            query TEXT,
            answer TEXT,
            start_time TEXT,
            end_time TEXT,
            PRIMARY KEY (execute_index,model_item,embedding_model,llm_model,temperature,repetition_penalty,query_index)
            )
            ''')
        
        self.execute('''CREATE TABLE IF NOT EXISTS rfp_metadata (
            Notice_no TEXT, 
            Notice_round TEXT,
            project_name TEXT,
            budget REAL,
            agency TEXT,
            publish_date TEXT,
            participation_start_date TEXT,
            participation_end_date TEXT,
            project_summary TEXT,
            file_type TEXT,
            file_name TEXT,
            text_content TEXT,
             PRIMARY KEY (Notice_no)
            )
            ''')
    

    ### 일반 SQL 쿼리 실행
    def execute( self, query):
        self._connect()
        # Lines(f"SQL:{query}") # 로그 너무 많으면 주석 처리
        self.cursor.execute(query)
        self.connection.commit()
        self._close()

    ## SELECT 쿼리 실행 결과 반환
    def select(self, sql):
        self._connect()
        cursor = self.connection.execute(sql)
        rows = cursor.fetchall()
        self._close()
        return rows
    
    ## 파라미터화된 SELECT 쿼리 실행 결과 반환 (특수문자 안전)
    def select_with_params(self, sql, params):
        self._connect()
        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()
        self._close()
        return rows

    ## 데이터베이스 초기화 (테이블 삭제 후 재생성)
    def clear_db(self):
        OpLog("데이터베이스 초기화 시작")
        self.execute('DROP TABLE IF EXISTS blob_data')
        self.execute('DROP TABLE IF EXISTS result_data')
        self.execute('DROP TABLE IF EXISTS rfp_metadata')
        self.create_table()
        OpLog("데이터베이스 초기화 완료")

    ## BLOB 데이터 청크 단위 저장 
    def save_blob(self, blob_name: str, blob_content: bytes):
        OpLog(f"Blob 저장 시작: {blob_name} (크기: {len(blob_content) / (1024**3):.2f} GB)")
        
        # 2GB 단위로 분할 (2GB = 2 * 1024 * 1024 * 1024 바이트)
        CHUNK_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
        chunks = []
        for i in range(0, len(blob_content), CHUNK_SIZE):
            chunks.append(blob_content[i:i+CHUNK_SIZE])
        
        OpLog(f"총 {len(chunks)}개의 청크로 분할됨")
        
        self._connect()
        # 기존 데이터 삭제 (UPDATE 대신 DELETE)
        sql_delete = 'DELETE FROM blob_data WHERE blob_name = ?'
        self.cursor.execute(sql_delete, (blob_name,))
        
        # 청크 단위로 저장
        sql_insert = '''
            INSERT INTO blob_data (blob_name, blob_index, blob_content)
            VALUES (?, ?, ?)
        '''
        for index, chunk in enumerate(chunks):
            self.cursor.execute(sql_insert, (blob_name, index, chunk))
            OpLog(f"청크 {index}/{len(chunks)-1} 저장 완료 (크기: {len(chunk) / (1024**3):.2f} GB)")
        
        self.connection.commit()
        self._close()
        OpLog(f"Blob 저장 완료: {blob_name} ({len(chunks)}개 청크)")
    
    ## BLOB 데이터 청크 단위 로드 및 병합
    def load_blob(self, blob_name: str) -> bytes:
        OpLog(f"Blob 로드 시작: {blob_name}")
        self._connect()
        
        # 모든 청크를 blob_index 순서대로 로드
        sql = '''
            SELECT blob_index, blob_content FROM blob_data 
            WHERE blob_name = ? 
            ORDER BY blob_index ASC
        '''
        self.cursor.execute(sql, (blob_name,))
        rows = self.cursor.fetchall()
        self._close()
        
        if rows:
            # 청크들을 순서대로 합치기
            combined_content = b''
            for index, (blob_index, chunk_content) in enumerate(rows):
                combined_content += chunk_content
                OpLog(f"청크 {blob_index} 로드 완료 (누적 크기: {len(combined_content) / (1024**3):.2f} GB)")
            
            OpLog(f"Blob 로드 완료: {blob_name} ({len(rows)}개 청크 병합)")
            return combined_content
        else:
            OpLog(f"Blob 없음: {blob_name}")
            return None
    
    ## 결과 데이터 로드
    def load_results(self, execute_index:int)-> bool:
        sql = '''
            SELECT * FROM result_data 
            WHERE execute_index=?
        '''
        params = (execute_index,)
        rows = self.select_with_params(sql, params)
        if rows:
            return True
        else:
            return False 
    
    ## 결과 데이터 저장
    def save_results(self, execute_index:int, model_item:str, embedding_model:str, llm_model:str, temperature:float, repetition_penalty:float, query_index:int, query:str, answer:str, start_time:str, end_time:str):
        sql = '''
            INSERT OR REPLACE INTO result_data 
            (execute_index, model_item, embedding_model, llm_model, temperature, repetition_penalty, query_index, query, answer, start_time, end_time)
            VALUES 
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (execute_index, model_item, embedding_model, llm_model, temperature, repetition_penalty, query_index, query, answer, start_time, end_time)
        self._connect()
        self.cursor.execute(sql, params)
        self.connection.commit()
        self._close()
        Lines(f"Result 저장 완료: execute_index={execute_index}, query_index={query_index}, model_item={model_item}, embedding_model={embedding_model}, llm_model={llm_model}\ntemperature={temperature}, repetition_penalty={repetition_penalty}\nonlocal query={query}\nanswer={answer}\nstart_time={start_time}, end_time={end_time}")
        OpLog(f"Result 저장 완료: execute_index={execute_index}, query_index={query_index}")
     
    ## 메타데이터 저장
    def save_metadata(self, metadata:pd.DataFrame):
        sql = "DELETE FROM rfp_metadata"
        self.execute(sql)
        self._connect()
        sql = '''
            INSERT OR REPLACE INTO rfp_metadata (Notice_no, Notice_round, project_name, budget, agency, publish_date, participation_start_date, participation_end_date, project_summary, file_type, file_name, text_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        # 공고 번호	공고 차수	사업명	사업 금액	발주 기관	공개 일자	입찰 참여 시작일	입찰 참여 마감일	사업 요약	파일형식	파일명	텍스트

        params_list = [(
            row['공고 번호'],
            row['공고 차수'],
            row['사업명'],
            row['사업 금액'],
            row['발주 기관'],
            row['공개 일자'],
            row['입찰 참여 시작일'],
            row['입찰 참여 마감일'],
            row['사업 요약'],
            row['파일형식'],
            row['파일명'],
            row['텍스트']
        ) for _, row in metadata.iterrows()]
        self.cursor.executemany(sql, params_list)
        self.connection.commit()
        self._close()
        OpLog(f"메타데이터 저장 완료: {len(metadata)}개 레코드")
    
    ## 메타데이터 로드
    def load_metadata(self)-> pd.DataFrame:
        sql = '''
            SELECT * FROM rfp_metadata
        '''
        rows = self.select(sql)
        columns = ['Notice_no', 'Notice_round', 'project_name', 'budget', 'agency', 'publish_date', 'participation_start_date', 'participation_end_date', 'project_summary', 'file_type', 'file_name', 'text_content']
        df = pd.DataFrame(rows, columns=columns)
        OpLog(f"메타데이터 로드 완료: {len(df)}개 레코드")
        return df

# ════════════════════════════════════════
# ▣ 메타데이터 로드 및 전처리
# ════════════════════════════════════════
class BidMatePreprocessor:
    ## 생성자
    def __init__(self, param:PARAMVAR):
        self.param = param
        self.data_dir = Path(self.param.rfp_data_dir)
        #self.normalize_filenames(self.param.rfp_data_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=param.chunk_size,
            chunk_overlap=param.chunk_overlap
        )
        self.embeddings = None
        if  param.is_openai:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model=param.embedding_model
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
            model_name=param.embedding_model)
            
        self.vector_store = None
        # 메타데이터 로드
        self.metadata_df = pd.read_csv(self.param.csv_path)
        OpLog(f"메타데이터 로드 완료: {len(self.metadata_df)}개 레코드", True)


    ## 파일 탐색기에서 한글 파일명이 깨져 보이는(자음/모음 분리) 현상은 주로 MacOS에서 압축된 파일을 Windows에서 풀었을 때 발생하는
    ## NFD(Normalization Form Decomposition) 인코딩 문제
    ## NFC(Normalization Form Composition)로 변환하여 해결
    def normalize_filenames(self, directory):
        target_dir = Path(directory)
        if not target_dir.exists():
            print(f"Directory not found: {target_dir}")
            return

        print(f"Scanning directory: {target_dir}")
        count = 0
        
        for file_path in target_dir.iterdir():
            if file_path.is_file():
                original_name = file_path.name
                # Normalize to NFC (NFC is standard for Windows/Linux, NFD is MacOS)
                normalized_name = unicodedata.normalize('NFC', original_name)
                
                if original_name != normalized_name:
                    new_path = target_dir / normalized_name
                    try:
                        # Rename the file
                        file_path.rename(new_path)
                        print(f"Renamed: {original_name} -> {normalized_name}")
                        count += 1
                    except Exception as e:
                        print(f"Error renaming {original_name}: {e}")

        print(f"Normalization complete. {count} files renamed.")


    ## HWP 텍스트 추출
    def _extract_hwp_text(self, file_path):
        #HWP 파일(v5)의 Record 구조를 파싱하여 텍스트(Tag ID 67)만 추출
        try:
            f = olefile.OleFileIO(file_path)
            dirs = f.listdir()
            text = ""
            for d in dirs:
                if "BodyText" in d:
                    section = f.openstream(d).read()
                    data = None
                    
                    # zlib 매직 넘버 체크 (0x789c or 0x78da, 0x7801, 0x785e 등)
                    if section[:2] in [b'\x78\x9c', b'\x78\xda', b'\x78\x01', b'\x78\x5e']:
                        # ✓ 압축된 데이터: zlib 압축 해제
                        decompress_errors = []
                        for wbits in [-15, 15, -zlib.MAX_WBITS, zlib.MAX_WBITS]:
                            try:
                                data = zlib.decompress(section, wbits)
                                OpLog(f"✓ HWP BodyText zlib 압축 해제 성공 [{os.path.basename(file_path)}]")
                                break
                            except zlib.error as e:
                                decompress_errors.append(f"wbits={wbits}: {str(e)[:30]}")
                                continue
                        
                        if data is None:
                            file_name = os.path.basename(file_path)
                            file_size = len(section)
                            OpLog(f"⚠️ HWP 압축 해제 실패 [{file_name}] (크기: {file_size} bytes) - 처리 스킵")
                            print(f"⚠️ HWP 압축 해제 실패, 건너뜀: {file_path}")
                            continue
                    else:
                        # ✓ 비압축 데이터: 직접 파싱 (HWP 5.0+에서 일부 스트림은 압축 안 됨)
                        file_name = os.path.basename(file_path)
                        magic = section[:2].hex()
                        OpLog(f"✓ HWP BodyText 비압축 형식 감지 [{file_name}] (매직: 0x{magic}) - 직접 파싱")
                        data = section
                    
                    pos = 0
                    size_mp = len(data)
                    
                    while pos < size_mp:
                        # 1. Record Header (4 bytes)
                        if pos + 4 > size_mp:
                            break
                        
                        header = struct.unpack('<I', data[pos:pos+4])[0]
                        pos += 4
                        
                        # 2. Tag ID & Length
                        # Tag ID: 하위 10비트
                        # Length: 상위 12비트
                        tag_id = header & 0x3FF
                        rec_len = (header >> 20) & 0xFFF
                        
                        # 길이가 0xFFF(4095)인 경우 추가 4바이트에 실제 길이 저장
                        if rec_len == 0xFFF:
                            if pos + 4 > size_mp:
                                break
                            rec_len = struct.unpack('<I', data[pos:pos+4])[0]
                            pos += 4
                            
                        if pos + rec_len > size_mp:
                            break
                            
                        # 3. 텍스트 추출 (Tag ID 67: HWPTAG_PARA_TEXT 추정)
                        # 디버깅 결과 67번 태그가 가변 길이의 텍스트 데이터를 담고 있음
                        if tag_id == 67: 
                            text_bytes = data[pos:pos+rec_len]
                            try:
                                # UTF-16LE 디코딩
                                decoded = text_bytes.decode('utf-16', errors='ignore')
                                
                                # 텍스트 내부의 제어문자 및 불필요한 기호 제거
                                # 한글(가-힣), 영문, 숫자, 기본 구두점만 허용
                                import re
                                clean = re.sub(r'[^가-힣a-zA-Z0-9\s.,()\-\[\]]', ' ', decoded)
                                clean = re.sub(r'\s+', ' ', clean).strip()
                                
                                if len(clean) > 0:
                                    text += clean + " "
                            except:
                                pass
                                
                        pos += rec_len
            
            # 전체 텍스트 공백 정리
            import re
            return re.sub(r'\s+', ' ', text).strip()
            
        except Exception as e:
            print(f"❌ HWP 추출 오류 ({file_path}): {e}")
            return ""
    ## PDF 텍스트 추출
    def _extract_pdf_text(self,file_path):
        try:
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            content = "\n".join([p.page_content for p in pages])
            return content
        except Exception as e:
            print(f"❌ PDF 추출 오류 ({file_path}): {e}")
            return ""
        
    def save_metadata(self):
        db = SQLiteDB() 
        self.metadata_df = pd.read_csv(self.param.csv_path)
        db.save_metadata(self.metadata_df)

    def get_all_docs(self):
        all_docs = []
        for _, row in self.metadata_df.iterrows():
            file_name = unicodedata.normalize('NFC', row['파일명'])
            file_path = os.path.join(self.param.rfp_data_dir, file_name)
            
            if not os.path.exists(file_path):
                print(f"⚠️ 파일 없음: {file_name}")
                continue

            # 1. 포맷별 텍스트 추출 및 Document 생성
            print(f"📄 처리 중: {file_name}")
            content = ""
            if file_name.endswith('.pdf'):
                content = self._extract_pdf_text(file_path)
            elif file_name.endswith('.hwp'):
                content = self._extract_hwp_text(file_path)

            # 2. 메타데이터 결합
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_name,
                    "Notice_no": row['공고 번호'],
                    "Notice_round": row['공고 차수'],
                    "project_name": row['사업명'],
                    "budget": row['사업 금액'],
                    "agency": row['발주 기관'],
                    ## 공개일자
                    "publish_date": row['공개 일자'],
                    ## 입찰 참여 여부
                    "participation_start_date": row['입찰 참여 시작일'],
                    ## 입찰 참여 마감일
                    "participation_end_date": row['입찰 참여 마감일'],
                    ## 사업요약
                    "project_summary": row['사업 요약'],
                    ## 파일형식
                    "file_type": row['파일형식'],
                    ## 파일명
                    "file_name": row['파일명'],
                    
                }
            )
            Lines(doc.metadata)
            
            # 3. 청킹 적용
            splits = self.text_splitter.split_documents([doc])
            all_docs.extend(splits)
        return all_docs



    ## FAISS 이름 생성
    def make_faiss_name(self):
        vector_name = f"{self.param.embedding_model.replace('/', '_')}_{self.param.llm_model.replace('/', '')}"
        faiss_name = f"faiss_store_{vector_name}_cs_{self.param.chunk_size}_co_{self.param.chunk_overlap}_{STORE_VER}"
        return faiss_name

    ## 기존 벡터 스토어 확인 및 로드
    def _check_vector_store_exists(self, newCreate):
        vector_store = None
        faiss_name = self.make_faiss_name()
        if not newCreate:
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
                except Exception as e:
                    print(f"⚠️ Vector DB 역직렬화 실패: {e}")
            else:
                print(f"⚠️ 기존 Vector DB 없음: {faiss_name}")
        return vector_store

    def get_hugging_vector_store(self,newCreate:bool = False):
        db = SQLiteDB() 
        vector_store = self._check_vector_store_exists(newCreate)
        if not vector_store is None:
            return vector_store
        faiss_name = self.make_faiss_name()
        if not newCreate:
            blob_bytes = db.load_blob(faiss_name)
            if blob_bytes:
                try:
                    self.vector_store = FAISS.deserialize_from_bytes(
                        blob_bytes,
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    print(f"✅ 기존 Vector DB 로드 완료: {faiss_name}")
                    return self.vector_store
                except Exception as e:
                    print(f"⚠️ Vector DB 역직렬화 실패: {e}")
            else:
                print(f"⚠️ 기존 Vector DB 없음: {faiss_name}")
        else :
            print(f"⚠️ 새로 생성: {faiss_name}")
                  
        all_docs = self.get_all_docs()
        embedding_dim = len(self.embeddings.embed_query("hello world"))
        # 코사인 유사도: 벡터 정규화 + IndexFlatIP
        index = faiss.IndexFlatIP(embedding_dim)
        texts = [doc.page_content for doc in all_docs]
        vectors = [self.embeddings.embed_query(text) for text in texts]
        vectors = np.array(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        # L2 정규화
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
         # 4. Vector DB 생성 및 저장 (Scenario B: OpenAI 기반)
        print(f"🚀 총 {len(all_docs)}개 청크를 벡터화하여 저장합니다...")
        self.vector_store = FAISS.from_documents(all_docs, self.embeddings)
        blob_bytes = self.vector_store.serialize_to_bytes()
        db = SQLiteDB() 
        db.save_blob(faiss_name, blob_bytes)
        return self.vector_store


  
    def get_openai_vector_store(self,newCreate):
        faiss_name = self.make_faiss_name()
        vector_store = self._check_vector_store_exists(newCreate)
        if not vector_store is None:
            return vector_store
        db = SQLiteDB() 
        all_docs = self.get_all_docs()
        
        # 4. Vector DB 생성 및 저장 (Scenario B: OpenAI 기반)
        print(f"🚀 총 {len(all_docs)}개 청크를 벡터화하여 저장합니다...")
        self.vector_store = FAISS.from_documents(all_docs, self.embeddings)
        self.vector_store.save_local(faiss_name)
        blob_bytes = self.vector_store.serialize_to_bytes()
        db = SQLiteDB() 
        db.save_blob(faiss_name, blob_bytes)
        return self.vector_store
   

    def get_vector_store(self,newCreatl):
        if self.param.is_openai:
            return self.get_openai_vector_store(newCreatl)
        else:
            return self.get_openai_vector_store(newCreatl)
            

# ════════════════════════════════════════
# ▣ 베이스 모델 클래스 
# ════════════════════════════════════════
class BaseModel():
    def __init__(self,param:PARAMVAR):
        self._my_name = f"embedding:{param.embedding_model}_llm:{param.llm_model}"
        self._param = param
        self._vector_store = None
        self._llm = None

    def rag_search(self, question):
        pass

    def query_answer(self,index,query):
        pass
    
    def clear_mem(self):
        import gc
        if self._vector_store is not None:
            del self._vector_store
            self._vector_store = None
        if self._faiss_db is not None:
            del self._faiss_db
            self._faiss_db = None
        torch.cuda.empty_cache()
        gc.collect()

      
# ════════════════════════════════════════
# ▣ OpenAI 및 HuggingFace 모델 클래스
# ════════════════════════════════════════
class OpenAIModel(BaseModel):
    def __init__(self,param:PARAMVAR):
        super().__init__(param)
        Lines(f"Make Model :: My_name:{self._my_name}, embedding_model:{self._param.embedding_model},llm_model:{self._param.llm_model},chunk_size:{self._param.chunk_size},chunk_overlap:{self._param.chunk_overlap},temperature:{self._param.temperature},repetition_penalty:{self._param.repetition_penalty}")
        processor = BidMatePreprocessor(self._param)
        self._vector_store = processor.get_vector_store(self._param.newCreate)

    def make_model(self):
        OpLog(f"OpenAI LLM 생성 시작: {self._param.llm_model}")
        self._llm = ChatOpenAI(model=self._param.llm_model)
        OpLog(f"OpenAI LLM 생성 완료")
        
        
    def rag_search(self, question):
        # 질문을 벡터로 변환하여 유사한 문서를 검색
        results = self._vector_store.similarity_search(question, k=self._param.k)
        # 검색된 문서의 내용을 텍스트로 결합
        context = "\n---\n".join([r.page_content for r in results])
        # LLM에게 질문과 관련된 문서 내용을 함께 전달하여 응답 생성
        prompt = f"""다음 메타데이터와 문서 내용을 참고하여 질문에 답변해주세요.

Context: 
{context}

Question: {question}

Answer:"""
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return response.content



class HugginFaceModel(BaseModel):
    def __init__(self,param:PARAMVAR):
        super().__init__(param)
        param.is_openai = False
        Lines(f"Make Model :: My_name:{self._my_name}, embedding_model:{self._param.embedding_model},llm_model:{self._param.llm_model},chunk_size:{self._param.chunk_size},chunk_overlap:{self._param.chunk_overlap},temperature:{self._param.temperature},repetition_penalty:{self._param.repetition_penalty}")
        processor = BidMatePreprocessor(param)
        self._vector_store = processor.get_vector_store(self._param.newCreate)
    
    def make_model(self):
        Lines(f"Make Model :: embedding_model_name:{self._param.embedding_model},llm_model:{self._param.llm_model},model_name:{self._my_name},temperature:{self._param.temperature},repetition_penalty:{self._param.repetition_penalty}")
        from transformers import AutoModelForCausalLM, AutoTokenizer        
        retriever = self._vector_store.as_retriever()
        if self._param.is_gpu:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        else:
            bnb_config = None
        device_map = "auto"
        if( self._param.is_gpu):
            device_map = "auto"
        else:
            device_map = None

        Lines("Make AutoModelForCausalLM")
        OpLog(f"AutoModelForCausalLM 로드 시작: {self._param.llm_model}")
        model = AutoModelForCausalLM.from_pretrained(
            self._param.llm_model,
            quantization_config= bnb_config if self._param.is_gpu else None,
            device_map=device_map,
            trust_remote_code=True,
        )
        OpLog(f"AutoModelForCausalLM 로드 완료")
        OpLog(f"Tokenizer 로드 시작: {self._param.llm_model}")
        tokenizer = AutoTokenizer.from_pretrained(self._param.llm_model)
        OpLog(f"Tokenizer 로드 완료")
        from transformers import pipeline
        Lines("Make LLM pipeline")
        OpLog(f"LLM Pipeline 생성 시작")
        llm_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=self._param.temperature,
            repetition_penalty=self._param.repetition_penalty,
            return_full_text=False,
            max_new_tokens=1000,
        )
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        chat_model = ChatHuggingFace(llm=llm)
        template = """다음 메타데이터와 문서 내용을 참고하여 질문에 답변해주세요.
        Context:
        {context}
        Question:
        {question}
        """
        prompt = PromptTemplate.from_template(template)

        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        def format_docs(docs):
            print(docs)
            return "\n\n".join(doc.page_content for doc in docs)

        self._retrieval_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        OpLog(f"Retrieval Chain 생성 완료")
    
    def rag_search(self, question):
        # LLM에서 최종 답변 가져오기
        answer = self._retrieval_chain.invoke(question)
        return answer

def Execute_Model(is_openai: bool, chunk_size: int, chunk_overlap: int, temperature: float, repetition_penalty: float, newCreate: bool, k: int, 
                  embedding_model: str , llm_model: str):
    param = PARAMVAR()
    param.is_openai = is_openai
    param.chunk_size =  chunk_size
    param.chunk_overlap = chunk_overlap
    param.temperature = temperature
    param.repetition_penalty = repetition_penalty
    param.newCreate = newCreate
    param.k = k
    model = None
    if( param.is_openai):
        param.embedding_model = embedding_model
        param.llm_model = llm_model
        model = OpenAIModel(param)
    else:
        param.embedding_model = embedding_model
        param.llm_model = llm_model
        model = HugginFaceModel(param)
    model.make_model()
    ## 테스트 질의를 입력 받는다. 
    ## ctrl+c 로 종료
    while True: 
        query = input("질문을 입력하세요 (종료하려면 Ctrl+C): ")
        answer = model.rag_search(query)
        print("답변:")
        Lines(answer)

def Execute_ModelEx(is_openai: bool, chunk_size: int, chunk_overlap: int, temperature: float, repetition_penalty: float, newCreate: bool, k: int, 
                  embedding_model: str , llm_model: str):
    param = PARAMVAR()
    param.is_openai = is_openai
    param.chunk_size =  chunk_size
    param.chunk_overlap = chunk_overlap
    param.temperature = temperature
    param.repetition_penalty = repetition_penalty
    param.newCreate = newCreate
    param.k = k
    model = None
    if( param.is_openai):
        param.embedding_model = embedding_model
        param.llm_model = llm_model
        model = OpenAIModel(param)
    else:
        param.embedding_model = embedding_model
        param.llm_model = llm_model
        model = HugginFaceModel(param)
    model.make_model()
    ## 테스트 질의를 입력 받는다. 
    ## ctrl+c 로 종료
    while True: 
        query = input("질문을 입력하세요 (종료하려면 Ctrl+C): ")
        answer = model.rag_search(query)
        print("답변:")
        Lines(answer)


#text-embedding-3-small" gpt-5-mini"
#nlpai-lab/KoE5 --llm_model nlpai-lab/KULLM3

if __name__ == "__main__":
    param = PARAMVAR()
    param.is_openai = False 
    param.chunk_size = 1000
    param.chunk_overlap = 100
    param.temperature = 0.7
    param.repetition_penalty = 1.2
    param.newCreate = False 
    param.k = 5
    model = None
    if( param.is_openai):
        param.embedding_model = "text-embedding-3-small"
        param.llm_model = "gpt-5-mini"
        model = OpenAIModel(param)
    else:
        param.embedding_model ="BAAI/bge-m3" # "nlpai-lab/KoE5"
        param.llm_model = "nlpai-lab/KULLM3"
        model = HugginFaceModel(param)
    model.make_model()
    ## 테스트 질의를 입력 받는다. 
    ## ctrl+c 로 종료
    while True: 
        query = input("질문을 입력하세요 (종료하려면 Ctrl+C): ")
        answer = model.rag_search(query)
        print("답변:")
        Lines(answer)
