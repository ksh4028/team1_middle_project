-- SQLite result_data 테이블에 do_retriever 컬럼 추가 및 생성 쿼리 예시

-- 기존 테이블이 있다면 컬럼 추가
ALTER TABLE result_data ADD COLUMN do_retriever TEXT;

-- 테이블 생성 예시 (do_retriever 포함)
CREATE TABLE IF NOT EXISTS result_data (
    execute_index INTEGER,
    model_item TEXT,
    embedding_model TEXT,
    llm_model TEXT,
    retriever_type TEXT,
    reranker_type TEXT,
    chunk_size INTEGER,
    chunk_overlap INTEGER,
    k INTEGER,
    is_openai INTEGER,
    is_gpu INTEGER,
    store_ver TEXT,
    temperature REAL,
    repetition_penalty REAL,
    query_index INTEGER,
    query TEXT,
    answer TEXT,
    context TEXT,
    start_time TEXT,
    end_time TEXT,
    do_retriever TEXT,
    PRIMARY KEY (execute_index, model_item, embedding_model, llm_model, retriever_type, reranker_type, query_index)
);
