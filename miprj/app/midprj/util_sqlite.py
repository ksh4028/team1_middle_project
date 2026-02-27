from .midprj_sqlite import SQLiteDB
from .midprj_func import OpLog


def run_migrations() -> None:
	"""Run schema migrations for the sqlite database."""
	migrate_add_domain_columns()
	migrate_adobe_summary_cache_table()
	migrate_result_data_table()
	migrate_evaluation_results_table()


def create_tables() -> None:
	"""Create base tables if they do not exist."""
	db = SQLiteDB()
	try:
		db.execute(
			"""CREATE TABLE IF NOT EXISTS blob_data (
				blob_name TEXT,
				blob_index INTEGER,
				blob_content BLOB,
				PRIMARY KEY (blob_name, blob_index)
			)"""
		)
		db.execute(
			"""CREATE TABLE IF NOT EXISTS result_data (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				execute_index INTEGER,
				model_item TEXT,
				embedding_model TEXT,
				llm_model TEXT,
				retriever_llm_type TEXT,
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
				   do_retriever TEXT
			)"""
		)
		db.execute(
			"""CREATE TABLE IF NOT EXISTS rfp_metadata (
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
				domain TEXT,
				keywords TEXT,
				region TEXT,
				PRIMARY KEY (Notice_no)
			)"""
		)
		db.execute(
			"""CREATE TABLE IF NOT EXISTS domain_keywords (
				category TEXT,
				keyword TEXT,
				PRIMARY KEY (category, keyword)
			)"""
		)
		db.execute(
			"""CREATE TABLE IF NOT EXISTS adobe_summary_cache (
				file_path TEXT,
				file_mtime REAL,
				file_size INTEGER,
				model_name TEXT,
				max_elements INTEGER,
				chunk_size INTEGER,
				summary_text TEXT,
				created_at TEXT,
				PRIMARY KEY (file_path, file_mtime, file_size, model_name, max_elements, chunk_size)
			)"""
		)
	except Exception as exc:
		OpLog(f"create_tables failed: {exc}", level="ERROR")

def table_exists(db: SQLiteDB, table_name: str) -> bool:
	rows = db.select(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
	return len(rows) > 0

def migrate_add_domain_columns() -> None:
	"""Add rfp_metadata columns if they are missing."""
	db = SQLiteDB()
	try:
		columns = [row[1] for row in db.select("PRAGMA table_info(rfp_metadata)")]
		if not columns:
			return

		if "domain" not in columns:
			OpLog("rfp_metadata.domain missing - migrating", level="INFO")
			db.execute("ALTER TABLE rfp_metadata ADD COLUMN domain TEXT DEFAULT NULL")
		if "keywords" not in columns:
			OpLog("rfp_metadata.keywords missing - migrating", level="INFO")
			db.execute("ALTER TABLE rfp_metadata ADD COLUMN keywords TEXT DEFAULT NULL")
		if "region" not in columns:
			OpLog("rfp_metadata.region missing - migrating", level="INFO")
			db.execute("ALTER TABLE rfp_metadata ADD COLUMN region TEXT DEFAULT NULL")
	except Exception as exc:
		OpLog(f"rfp_metadata migration failed: {exc}", level="ERROR")

def migrate_adobe_summary_cache_table() -> None:
	"""Create adobe_summary_cache table if it is missing."""
	db = SQLiteDB()
	try:
		db.execute(
			"""CREATE TABLE IF NOT EXISTS adobe_summary_cache (
				file_path TEXT,
				file_mtime REAL,
				file_size INTEGER,
				model_name TEXT,
				max_elements INTEGER,
				chunk_size INTEGER,
				summary_text TEXT,
				created_at TEXT,
				PRIMARY KEY (file_path, file_mtime, file_size, model_name, max_elements, chunk_size)
			)"""
		)
	except Exception as exc:
		OpLog(f"adobe_summary_cache migration failed: {exc}", level="ERROR")


def migrate_result_data_table() -> None:
	"""Migrate result_data to include id PRIMARY KEY AUTOINCREMENT and do_retriever column."""
	expected_columns = [
		"id", "execute_index", "model_item", "embedding_model", "llm_model",
		"retriever_llm_type", "reranker_type", "chunk_size", "chunk_overlap", "k",
		"is_openai", "is_gpu", "store_ver", "temperature", "repetition_penalty",
		"query_index", "query", "answer", "context", "start_time", "end_time", "do_retriever"
	]
	db = SQLiteDB()
	try:
		# result_data_new → result_data 이름 변경 처리
		if table_exists(db, "result_data_new") and not table_exists(db, "result_data"):
			db.execute("ALTER TABLE result_data_new RENAME TO result_data")

		if not table_exists(db, "result_data"):
			return

		table_info = db.select("PRAGMA table_info(result_data)")
		if not table_info:
			return

		existing_columns = [row[1] for row in table_info]
		pk_columns = [row[1] for row in table_info if row[5] > 0]

		# do_retriever 컬럼만 없는 경우는 ALTER TABLE로 빠르게 추가
		if "do_retriever" not in existing_columns:
			try:
				db.execute("ALTER TABLE result_data ADD COLUMN do_retriever TEXT")
				OpLog("result_data.do_retriever 컬럼 추가(migration)", level="INFO")
				existing_columns.append("do_retriever")
				if set(expected_columns).issubset(set(existing_columns)) and pk_columns == ["id"]:
					return
			except Exception as e:
				OpLog(f"ALTER TABLE do_retriever 추가 실패: {e}", level="ERROR")

		needs_rebuild = False
		if "id" not in existing_columns or pk_columns != ["id"]:
			needs_rebuild = True
		if not set(expected_columns).issubset(set(existing_columns)):
			needs_rebuild = True

		if not needs_rebuild:
			return

		OpLog("result_data migration start", level="INFO")

		db.execute(
			"""CREATE TABLE IF NOT EXISTS result_data_tmp (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				execute_index INTEGER,
				model_item TEXT,
				embedding_model TEXT,
				llm_model TEXT,
				retriever_llm_type TEXT,
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
				do_retriever TEXT
			)"""
		)

		insert_columns = [col for col in expected_columns if col in existing_columns]
		if insert_columns:
			cols = ", ".join(insert_columns)
			db.execute(
				f"INSERT INTO result_data_tmp ({cols}) SELECT {cols} FROM result_data"
			)

		db.execute("DROP TABLE IF EXISTS result_data")
		db.execute("ALTER TABLE result_data_tmp RENAME TO result_data")
		OpLog("result_data migration complete", level="INFO")
	except Exception as exc:
		OpLog(f"result_data migration failed: {exc}", level="ERROR")
		if "id" not in existing_columns or pk_columns != ["id"]:
			needs_rebuild = True
		if not set(expected_columns).issubset(set(existing_columns)):
			needs_rebuild = True
		# unique index 체크 제거

		if not needs_rebuild:
			return

		OpLog("result_data migration start", level="INFO")

		db.execute(
			"""CREATE TABLE IF NOT EXISTS result_data_tmp (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				execute_index INTEGER,
				model_item TEXT,
				embedding_model TEXT,
				llm_model TEXT,
				retriever_llm_type TEXT,
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
				do_retriever TEXT
			)"""
		)

		insert_columns = [col for col in expected_columns if col in existing_columns]
		if insert_columns:
			cols = ", ".join(insert_columns)
			db.execute(
				f"INSERT INTO result_data_tmp ({cols}) SELECT {cols} FROM result_data"
			)

		db.execute("DROP TABLE IF EXISTS result_data")
		db.execute("ALTER TABLE result_data_tmp RENAME TO result_data")
		db.commit()
		OpLog("result_data migration complete", level="INFO")
	except Exception as exc:
		OpLog(f"result_data migration failed: {exc}", level="ERROR")
	finally:
		pass


def migrate_evaluation_results_table() -> None:
	"""Ensure evaluation_results table has expected columns."""
	expected_columns = [
		"id", "execute_index", "query_id", "faithfulness_score", "faithfulness_reason",
		"answer_relevance_score", "answer_relevance_reason", "context_relevance_score",
		"context_relevance_reason", "eval_date"
	]
	db = SQLiteDB()
	try:
		if not table_exists(db, "evaluation_results"):
			db.execute(
				"""CREATE TABLE IF NOT EXISTS evaluation_results (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					execute_index INTEGER,
					query_id INTEGER,
					faithfulness_score INTEGER,
					faithfulness_reason TEXT,
					answer_relevance_score INTEGER,
					answer_relevance_reason TEXT,
					context_relevance_score INTEGER,
					context_relevance_reason TEXT,
					eval_date TEXT
				)"""
			)
			return

		existing_columns = [row[1] for row in db.select("PRAGMA table_info(evaluation_results)")]
		if not existing_columns:
			return

		if not set(expected_columns).issubset(set(existing_columns)):
			OpLog("evaluation_results migration start", level="INFO")
			db.execute(
				"""CREATE TABLE IF NOT EXISTS evaluation_results_tmp (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					execute_index INTEGER,
					query_id INTEGER,
					faithfulness_score INTEGER,
					faithfulness_reason TEXT,
					answer_relevance_score INTEGER,
					answer_relevance_reason TEXT,
					context_relevance_score INTEGER,
					context_relevance_reason TEXT,
					eval_date TEXT
				)"""
			)
			insert_columns = [col for col in expected_columns if col in existing_columns]
			if insert_columns:
				cols = ", ".join(insert_columns)
				db.execute(
					f"INSERT OR IGNORE INTO evaluation_results_tmp ({cols}) "
					f"SELECT {cols} FROM evaluation_results"
				)
			db.execute("DROP TABLE IF EXISTS evaluation_results")
			db.execute("ALTER TABLE evaluation_results_tmp RENAME TO evaluation_results")
			OpLog("evaluation_results migration complete", level="INFO")
	except Exception as exc:
		OpLog(f"evaluation_results migration failed: {exc}", level="ERROR")



if __name__ == "__main__":
	print("Running SQLite DB create and migrations...")
	create_tables()
	run_migrations()
	print("Create and Migrations completed.")
