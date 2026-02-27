# Standard Library
import sqlite3

# Third-party Libraries
import pandas as pd

# Project Modules
from .midprj_defines import SQLITEDB_PATH
from .midprj_func import Lines, OpLog, now_str


# ════════════════════════════════════════
# ▣ SQLite 데이터베이스 핸들러 클래스
# ════════════════════════════════════════
class SQLiteDB:
    ## SQLiteDB 초기화
    def __init__(self):
        self._db_path = SQLITEDB_PATH
        self.connection = None
        self.cursor = None

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

    def get_domain_keywords(self):
        """DB에서 모든 도메인 키워드 조회 (util 기능을 대체 혹은 유지)"""
        rows = self.select("SELECT category, keyword FROM domain_keywords")
        keywords_dict = {}
        for category, keyword in rows:
            if category not in keywords_dict:
                keywords_dict[category] = []
            keywords_dict[category].append(keyword)
        return keywords_dict

    ### 일반 SQL 쿼리 실행
    def execute(self, query):
        self._connect()
        OpLog(f"SQL 실행: {query}", level="INFO")
        self.cursor.execute(query)
        self.connection.commit()
        self._close()

    ## SELECT 쿼리 실행 결과 반환
    def select(self, sql):
        self._connect()
        OpLog(f"SELECT SQL 실행: {sql}", level="INFO")
        cursor = self.connection.execute(sql)
        rows = cursor.fetchall()
        self._close()
        return rows

    ## 파라미터화된 SELECT 쿼리 실행 결과 반환 (특수문자 안전)
    def select_with_params(self, sql, params):
        self._connect()
        OpLog(f"SELECT SQL 실행: {sql} with params {params}", level="INFO")
        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()
        self._close()
        return rows

    ## 데이터베이스 초기화 (테이블 삭제 후 재생성)
    def clear_db(self):
        OpLog("데이터베이스 초기화 시작", level="INFO")
        self.execute('DROP TABLE IF EXISTS blob_data')
        self.execute('DROP TABLE IF EXISTS result_data')
        self.execute('DROP TABLE IF EXISTS rfp_metadata')
        OpLog("데이터베이스 초기화 완료", level="INFO")

    def clear_adobe_summary_cache(self):
        """Adobe 요약 캐시 초기화"""
        try:
            self.execute('DELETE FROM adobe_summary_cache')
            OpLog("Adobe 요약 캐시 초기화 완료", level="INFO")
        except Exception as e:
            OpLog(f"Adobe 요약 캐시 초기화 실패: {e}", level="ERROR")

    ## BLOB 데이터 청크 단위 저장 
    def save_blob(self, blob_name: str, blob_content: bytes):
        OpLog(f"Blob 저장 시작: {blob_name} (크기: {len(blob_content) / (1024**3):.2f} GB)", level="INFO")

        # 2GB 단위로 분할 (2GB = 2 * 1024 * 1024 * 1024 바이트)
        CHUNK_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
        chunks = []
        for i in range(0, len(blob_content), CHUNK_SIZE):
            chunks.append(blob_content[i:i+CHUNK_SIZE])

        Lines(f"총 {len(chunks)}개의 청크로 분할됨")

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
            Lines(f"청크 {index}/{len(chunks)-1} 저장 완료 (크기: {len(chunk) / (1024**3):.2f} GB)")

        self.connection.commit()
        self._close()
        OpLog(f"Blob 저장 완료: {blob_name} ({len(chunks)}개 청크)", level="INFO")

    ## BLOB 데이터 청크 단위 로드 및 병합
    def load_blob(self, blob_name: str) -> bytes:
        OpLog(f"Blob 로드 시작: {blob_name}", level="INFO")
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
                Lines(f"청크 {blob_index} 로드 완료 (누적 크기: {len(combined_content) / (1024**3):.2f} GB)")

            OpLog(f"Blob 로드 완료: {blob_name} ({len(rows)}개 청크 병합)", level="INFO")
            return combined_content
        else:
            Lines(f"Blob 없음: {blob_name}")
            return None

    ## 결과 데이터 로드
    def load_results(self, execute_index: int) -> bool:
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


    def save_chat_history(self, session_id: str, role: str, content: str) -> None:
        self._connect()
        try:
            sql = '''
                INSERT INTO chat_history (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
            '''
            self.cursor.execute(sql, (session_id, role, content, now_str()))
            self.connection.commit()
        finally:
            self._close()

    def load_recent_chat_history(self, session_id: str, limit: int) -> list[dict]:
        self._connect()
        try:
            sql = '''
                SELECT role, content
                FROM chat_history
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            '''
            self.cursor.execute(sql, (session_id, limit))
            rows = self.cursor.fetchall()
            rows.reverse()
            return [{"role": row[0], "content": row[1]} for row in rows]
        finally:
            self._close()
        
# execute_index	실행 회차 ID (PK 연동용)
# k_value	검색 문서 개수 (k)
# chunk_size	문서 분할 크기
# chunk_overlap	문서 중첩 크기
# temperature	생성 다양성 지수
# repetition_penalty	반복 방지 패널티
# score	AI 심판의 평가 점수
# reason	평가 이유

    def load_results_for_evaluation(self, execute_index: int) -> list[dict]:
        """평가용 결과 데이터 조회"""
        sql = """
            SELECT id, query_index AS query_id, query, answer, context
            FROM result_data
            WHERE execute_index = ?
            ORDER BY id
        """
        params = (execute_index,)
        self._connect()
        try:
            self.cursor.execute(sql, params)
            rows = self.cursor.fetchall()
            return [
                {
                    "result_id": row[0],
                    "query_id": row[0],
                    "query": row[2],
                    "answer": row[3],
                    "context": row[4],
                }
                for row in rows
            ]
        finally:
            self._close()

    def save_evaluation_result(
        self,
        result_id: int,
        execute_index: int,
        query_id: int,
        faithfulness_score: int,
        faithfulness_reason: str,
        answer_relevance_score: int,
        answer_relevance_reason: str,
        context_relevance_score: int,
        context_relevance_reason: str,
        eval_date: str | None = None,
    ) -> None:
        """evaluation_results 테이블에 평가 결과 저장"""
        sql = '''
            INSERT OR REPLACE INTO evaluation_results (
                id, execute_index, query_id,
                faithfulness_score, faithfulness_reason,
                answer_relevance_score, answer_relevance_reason,
                context_relevance_score, context_relevance_reason,
                eval_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            result_id,
            execute_index,
            query_id,
            faithfulness_score,
            faithfulness_reason,
            answer_relevance_score,
            answer_relevance_reason,
            context_relevance_score,
            context_relevance_reason,
            eval_date or now_str(),
        )
        self._connect()
        try:
            self.cursor.execute(sql, params)
            self.connection.commit()
        finally:
            self._close()

    def delete_evaluation_results_by_execute_index(self, execute_index: int) -> int:
        """execute_index에 해당하는 evaluation_results 삭제, 삭제된 행 수 반환"""
        sql = "DELETE FROM evaluation_results WHERE execute_index=?"
        params = (execute_index,)
        try:
            self._connect()
            self.cursor.execute(sql, params)
            deleted = self.cursor.rowcount if self.cursor.rowcount is not None else 0
            self.connection.commit()
            OpLog(
                f"evaluation_results 삭제 완료: execute_index={execute_index}, rows={deleted}",
                level="INFO",
            )
            return deleted
        except Exception as e:
            OpLog(
                f"evaluation_results 삭제 실패: execute_index={execute_index}, error={e}",
                level="ERROR",
            )
            return 0
        finally:
            self._close()

    def delete_results_by_execute_index(self, execute_index: int) -> int:
        """execute_index에 해당하는 result_data 삭제, 삭제된 행 수 반환"""
        sql = f"DELETE FROM result_data WHERE execute_index={execute_index}"
        self.execute(sql)
        sql = f"DELETE FROM evaluation_results WHERE execute_index={execute_index}"
        self.execute(sql)
        return 0
    
      
    ## 결과 데이터 저장
    def save_results(
        self,
        execute_index: int,
        model_item: str,
        embedding_model: str,
        llm_model: str,
        retriever_llm_type: str,
        reranker_type: str,
        chunk_size: int,
        chunk_overlap: int,
        k: int,
        is_openai: int,
        is_gpu: int,
        store_ver: str,
        temperature: float,
        repetition_penalty: float,
        query_index: int,
        query: str,
        answer: str,
        context: str,
        start_time: str,
        end_time: str,
        do_retriever: str = None,
    ):
        sql = '''
            INSERT INTO result_data 
            (execute_index, model_item, embedding_model, llm_model, retriever_llm_type, reranker_type, chunk_size, chunk_overlap, k, is_openai, is_gpu, store_ver, temperature, repetition_penalty, query_index, query, answer, context, start_time, end_time, do_retriever)
            VALUES 
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            execute_index,
            model_item,
            embedding_model,
            llm_model,
            retriever_llm_type,
            reranker_type,
            chunk_size,
            chunk_overlap,
            k,
            is_openai,
            is_gpu,
            store_ver,
            temperature,
            repetition_penalty,
            query_index,
            query,
            answer,
            context,
            start_time,
            end_time,
            do_retriever,
        )
        self._connect()
        self.cursor.execute(sql, params)
        self.connection.commit()
        self._close()
        Lines(
            "Result 저장 완료: "
            f"execute_index={execute_index}, query_index={query_index}\n"
            f"model_item={model_item}, embedding_model={embedding_model}, llm_model={llm_model}\n"
            f"retriever_llm_type={retriever_llm_type}, reranker_type={reranker_type}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, k={k}\n"
            f"temperature={temperature}, repetition_penalty={repetition_penalty}\n"
            f"query={query}\nanswer={answer}\nstart_time={start_time}, end_time={end_time}\ndo_retriever={do_retriever}"
        )
        OpLog(
            f"Result 저장 완료: execute_index={execute_index}, query_index={query_index}",
            level="INFO",
        )

    ## 메타데이터 로드
    def load_metadata(self) -> pd.DataFrame:
        sql = '''
            SELECT * FROM rfp_metadata
        '''
        rows = self.select(sql)
        columns = ['Notice_no', 'Notice_round', 'project_name', 'budget', 'agency', 'publish_date', 'participation_start_date', 'participation_end_date', 'project_summary', 'file_type', 'file_name', 'text_content', 'domain', 'keywords', 'region']
        df = pd.DataFrame(rows, columns=columns)
        OpLog(f"메타데이터 로드 완료: {len(df)}개 레코드", level="INFO")
        return df

    def get_adobe_summary_cache(self, file_path, file_size, model_name, max_elements, chunk_size):
        sql = '''
            SELECT summary_text FROM adobe_summary_cache
            WHERE file_path = ?
              AND file_size = ?
              AND model_name = ?
              AND max_elements = ?
              AND chunk_size = ?
        '''
        params = (file_path, file_size, model_name, max_elements, chunk_size)
        rows = self.select_with_params(sql, params)
        if rows:
            return rows[0][0]
        return None

    def save_adobe_summary_cache(self, file_path, file_mtime, file_size, model_name, max_elements, chunk_size, summary_text):
        sql = '''
            INSERT OR REPLACE INTO adobe_summary_cache
            (file_path, file_mtime, file_size, model_name, max_elements, chunk_size, summary_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            file_path,
            file_mtime,
            file_size,
            model_name,
            max_elements,
            chunk_size,
            summary_text,
            now_str(),
        )
        try:
            self._connect()
            self.cursor.execute(sql, params)
            self.connection.commit()
        except Exception as e:
            OpLog(f"Adobe 요약 캐시 저장 실패: {e}", level="ERROR")
        finally:
            self._close()
