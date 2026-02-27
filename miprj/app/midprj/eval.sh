#!/usr/bin/env bash

# Cross-test runner (total 3*3*3*3*3 = 243, embeddings_llm loop is innermost)
# nohup bash eval.sh > /dev/null 2>&1 &
# ps -ef | grep eval.sh
# source /home/spai0641_codeit_sprint_kr/project/.venv/bin/activate
set -euo pipefail

llm_defines=(
	"openai|text-embedding-3-small|gpt-5-mini"
	"ollama|bge-m3|exaone3.5:7.8b"
)

temp_penalty=(
	"0.2|1.2"
	"0.3|1.0"
)

chunk_size_overlap=(
	"500|50"
	"1000|100"
)

k_values=(5 10)

execute_index=300
execute_indexEx=301

python3 util_sqlite.py

python3 util_del_executeindex.py --execute_index $execute_index
python3 util_del_executeindex.py --execute_index $execute_indexEx


for i in "${!llm_defines[@]}"; do
	triple="${llm_defines[$i]}"
	IFS='|' read -r whatmodelItem embedding_model llm_model <<< "$triple"
	llm_sel="$whatmodelItem"
	for tp in "${temp_penalty[@]}"; do
		IFS='|' read -r temperature repetition_penalty <<< "$tp"
		for co in "${chunk_size_overlap[@]}"; do
			IFS='|' read -r chunk_size chunk_overlap <<< "$co"
			for k in "${k_values[@]}"; do
				python3 call_midprj_eval.py \
					--execute_index "$execute_index" \
					--whatmodelItem "$whatmodelItem" \
					--embedding_model "$embedding_model" \
					--llm_model "$llm_model" \
					--temperature "$temperature" \
					--repetition_penalty "$repetition_penalty" \
					--chunk_size "$chunk_size" \
					--chunk_overlap "$chunk_overlap" \
					--k "$k" \
					--llm_selector "$llm_sel"
			done
		done
	done
done

python3 midprj_eval.py --execute_index "$execute_index"
python3 midprj_eval.py --execute_index "$execute_indexEx"



