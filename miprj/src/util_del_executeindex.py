"""Delete result_data rows by execute_index from SQLite DB."""
# -*- coding: utf-8 -*-
import argparse

from midprj_func import Lines
from midprj_sqlite import SQLiteDB


def main():
	parser = argparse.ArgumentParser(description="execute_index 기반 평가 결과 삭제")
	parser.add_argument(
		"--execute_index",
		type=int,
		required=True,
		help="삭제할 execute_index",
	)
	args = parser.parse_args()

	db = SQLiteDB()
	deleted = db.delete_results_by_execute_index(args.execute_index)
	Lines(f"삭제 완료: execute_index={args.execute_index}, rows={deleted}")


if __name__ == "__main__":
	main()

# python midprj_del_executeindex.py --execute_index 1


