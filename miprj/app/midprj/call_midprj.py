# -*- coding: utf-8 -*-
from . import midprj_main as prj
from call_midprj_args import build_argument_parser, build_param_from_args


def main():
    parser = build_argument_parser(description="MidPrj RAG 실행 스크립트")
    args = parser.parse_args()
    param = build_param_from_args(args)
    prj.Execute_Model(param)


if __name__ == "__main__":
    main()

"""
[HuggingFace 사용 방법]

1. 기본 실행 (nlpai-lab/KoE5, nlpai-lab/KULLM3):
   python call_midprj.py

2. HuggingFace + 파라미터 설정 (청크 크기, 온도, 검색 수):
   python call_midprj.py --chunk_size 500 --temperature 0.5 --k 3

3. HuggingFace 벡터 DB 생성:
   python call_midprj.py --newCreate True

4. HuggingFace 모델 변경:
   python call_midprj.py --embedding_model nlpai-lab/KoE5 --llm_model nlpai-lab/KULLM3 --chunk_size 1000 --chunk_overlap 200 --temperature 0.7 --repetition_penalty 1.5 --k 10

5. HuggingFace + Adobe 요약 설정:



[OpenAI 사용 방법]

1. OpenAI 기본 실행 (ini의 OPENAI_MODEL 사용):
   python call_midprj.py --whatmodelItem openai

2. OpenAI + 파라미터 설정 (청크 크기, 온도, 검색 수):
   python call_midprj.py --whatmodelItem openai --chunk_size 800 --temperature 0.3 --k 7

3. OpenAI 벡터 DB 생성:
   python call_midprj.py --whatmodelItem openai --newCreate True

4. OpenAI 모델 상세 설정 (직접 지정):
   python call_midprj.py --whatmodelItem openai --embedding_model <embedding> --llm_model <llm> --chunk_size 1200 --chunk_overlap 150 --temperature 0.2 --repetition_penalty 1.0 --k 10

5. OpenAI + 저장/경로/버전 설정:
   python call_midprj.py --whatmodelItem openai --csv_path "D:/project/CodeIt/data/rfp_files/data_list.csv" --rfp_data_dir "D:/project/CodeIt/data/rfp_files/files" --store_ver V16 --adobe_ver V02

[Ollama 사용 방법]

1. Ollama 기본 실행 (INI 설정 사용):
   python call_midprj.py --whatmodelItem ollama

[메타데이터 사전 저장]

1. CSV -> SQLite 메타데이터 저장:
   python call_midprj_preprocess.py --csv_path "D:/project/CodeIt/data/rfp_files/data_list.csv" --rfp_data_dir "D:/project/CodeIt/data/rfp_files/files"

========================================
"""
