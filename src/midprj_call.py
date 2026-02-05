# -*- coding: utf-8 -*-
import argparse
import midprj_main as prj

def main():
    parser = argparse.ArgumentParser(description="MidPrj RAG ??? ??? ?????")
    
    # ??? ?? ? LLM ??
    parser.add_argument("--embedding_model", type=str, default="nlpai-lab/KoE5", help="??? ?? ?? (???: nlpai-lab/KoE5)")
    parser.add_argument("--llm_model", type=str, default="nlpai-lab/KULLM3", help="LLM ?? ?? (???: nlpai-lab/KULLM3)")
    parser.add_argument("--chunk_size", type=int, default=1000, help="?? ?? (???: 1000)")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="?? ?? ?? (???: 100)")
    parser.add_argument("--temperature", type=float, default=0.7, help="?? ?? ? (???: 0.7)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="?? ??? ? (???: 1.2)")
    parser.add_argument("--k", type=int, default=5, help="?? ?? ? (???: 5)")
    parser.add_argument("--is_openai", type=str, default="False", help="OpenAI ?? ?? (???: False, True/False)")
    parser.add_argument("--newCreate", type=str, default="False", help="??? DB ?? ?? (???: False, True/False)")
    
    args = parser.parse_args()
    prj.Execute_Model(args.is_openai.lower() == "true", args.chunk_size, args.chunk_overlap, args.temperature, args.repetition_penalty, args.newCreate.lower() == "true", args.k, args.embedding_model, args.llm_model)

   
if __name__ == "__main__":
    main()
"""
[HuggingFace ?? ??]

1. ????? ?? (nlpai-lab/KoE5, nlpai-lab/KULLM3):
   python midprj_call.py

2. HuggingFace + ??? ?? (?? ??, ??, ?? ?):
   python midprj_call.py --chunk_size 500 --temperature 0.5 --k 3

3. HuggingFace ??? DB ??:
   python midprj_call.py --newCreate True

4. HuggingFace ?? ??:
   python midprj_call.py --embedding_model nlpai-lab/KoE5 --llm_model nlpai-lab/KULLM3 --chunk_size 1000 --chunk_overlap 200 --temperature 0.7 --repetition_penalty 1.5 --k 10


[OpenAI ?? ??]

1. OpenAI ??? ?? (text-embedding-3-small, gpt-5-mini):
   python midprj_call.py --embedding_model text-embedding-3-small --llm_model gpt-5-mini --is_openai True

2. OpenAI + ??? ?? (?? ??, ??, ?? ?):
   python midprj_call.py --embedding_model text-embedding-3-small --llm_model gpt-5-mini --is_openai True --chunk_size 800 --temperature 0.3 --k 7

3. OpenAI ??? DB ??:
   python midprj_call.py --embedding_model text-embedding-3-small --llm_model gpt-5-mini --is_openai True --newCreate True

4. OpenAI ?? ?? ?? (?? ??, ?? ??, ??, ?? ???, ?? ?):
   python midprj_call.py --embedding_model text-embedding-3-small --llm_model gpt-5-mini --is_openai True --chunk_size 1200 --chunk_overlap 150 --temperature 0.2 --repetition_penalty 1.0 --k 10

========================================
"""
