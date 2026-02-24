@echo off
REM eval.bat - Windows 11용 평가 스크립트

REM 변수 설정
setlocal enabledelayedexpansion

REM 배열 정의 (Windows에서는 배열 대신 구분자 사용)
REM 각 파라미터를 공백으로 구분
set "embeddings_llm=openai|text-embedding-3-small|gpt-5-mini ollama|bge-m3|exaone3.5:7.8b huggingface|nlpai-lab/KoE5|nlpai-lab/KULLM3"
set "llm_selector=gpt ollama"
set "temp_penalty=0.2|1.2 0.3|1.1 0.3|1.0"
set "chunk_size_overlap=500|50 700|70 1000|100"
set "k_values=5 7 10"
set "execute_index=100"

REM 초기화 스크립트 실행
python util_sqlite.py
python util_del_executeindex.py --execute_index %execute_index%




setlocal enabledelayedexpansion
for %%S in (%llm_selector%) do (
  for %%T in (%temp_penalty%) do (
    set "tp=%%T"
    for /f "tokens=1,2 delims=|" %%a in ("!tp!") do (
      set "temperature=%%a"
      set "repetition_penalty=%%b"
      for %%C in (%chunk_size_overlap%) do (
        set "co=%%C"
        for /f "tokens=1,2 delims=|" %%c in ("!co!") do (
          set "chunk_size=%%c"
          set "chunk_overlap=%%d"
          for %%K in (%k_values%) do (
            for %%E in (%embeddings_llm%) do (
              set "em=%%E"
              for /f "tokens=1,2,3 delims=|" %%e in ("!em!") do (
                set "whatmodelItem=%%e"
                set "embedding_model=%%f"
                set "llm_model=%%g"
                call python call_midprj_eval.py ^
                  --execute_index %execute_index% ^
                  --whatmodelItem !whatmodelItem! ^
                  --embedding_model !embedding_model! ^
                  --llm_model !llm_model! ^
                  --temperature !temperature! ^
                  --repetition_penalty !repetition_penalty! ^
                  --chunk_size !chunk_size! ^
                  --chunk_overlap !chunk_overlap! ^
                  --k %%K ^
                  --llm_selector %%S
              )
            )
          )
        )
      )
    )
  )
)
endlocal

endlocal
