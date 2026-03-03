import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_model_performance(db_path, target_index=202):
    # 1. 데이터베이스 연결
    conn = sqlite3.connect(db_path)
    
    # 2. SQL 쿼리: 결과와 평가 데이터 조인 및 점수 평균 계산
    # 점수 합계를 15점 만점 기준으로 백분율(%) 환산
    query = f"""
    SELECT 
        r.model_item,
        'T:' || r.temperature || '/P:' || r.repetition_penalty as temp_penalty,
        'C:' || r.chunk_size || '/O:' || r.chunk_overlap as chunk_config,
        'K=' || r.k as k_value,
        ((AVG(e.faithfulness_score) + AVG(e.answer_relevance_score) + AVG(e.context_relevance_score)) / 15.0) * 100 AS score_pct
    FROM result_data r
    JOIN evaluation_results e ON r.id = e.query_id
    WHERE r.execute_index = {target_index}
    GROUP BY r.model_item, r.temperature, r.repetition_penalty, r.chunk_size, r.chunk_overlap, r.k
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("데이터가 없습니다. execute_index를 확인하세요.")
        return

    # 3. 시각화 설정 (한글 깨짐 방지는 환경에 따라 설정 필요)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'RAG Performance Analysis (execute_index: {target_index})', fontsize=20)

    # 그래프 1: Temperature / Penalty 별 비교
    sns.barplot(x='temp_penalty', y='score_pct', hue='model_item', data=df, ax=axes[0])
    axes[0].set_title('Performance by Temp/Penalty')
    axes[0].set_ylabel('Score (%)')
    axes[0].set_ylim(0, 105)

    # 그래프 2: Chunk Size / Overlap 별 비교
    sns.barplot(x='chunk_config', y='score_pct', hue='model_item', data=df, ax=axes[1])
    axes[1].set_title('Performance by Chunk/Overlap')
    axes[1].set_ylabel('')
    axes[1].set_ylim(0, 105)

    # 그래프 3: K-Value 별 비교
    sns.barplot(x='k_value', y='score_pct', hue='model_item', data=df, ax=axes[2])
    axes[2].set_title('Performance by K-Value')
    axes[2].set_ylabel('')
    axes[2].set_ylim(0, 105)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_rag_performance(db_path, execute_indices):
    """
    execute_index별 OpenAI vs Ollama의 리트리버/리랭커 조합 성능 비교
    """
    conn = sqlite3.connect(db_path)
    
    # 인덱스 리스트를 SQL 쿼리용 문자열로 변환
    idx_str = ",".join(map(str, execute_indices))
    
    query = f"""
    SELECT 
        r.execute_index,
        r.model_item,
        r.do_retriever,
        r.reranker_type,
        -- 파이프라인 식별자 생성 (예: hybrid_llm + bge)
        r.do_retriever || ' / ' || r.reranker_type AS pipeline,
        AVG(e.faithfulness_score) AS faith,
        AVG(e.answer_relevance_score) AS rel,
        AVG(e.context_relevance_score) AS ctx,
        ((AVG(e.faithfulness_score) + AVG(e.answer_relevance_score) + AVG(e.context_relevance_score)) / 15.0) * 100 AS total_score_pct
    FROM result_data r
    JOIN evaluation_results e ON r.id = e.query_id
    WHERE r.execute_index IN ({idx_str})
    GROUP BY r.execute_index, r.model_item, r.do_retriever, r.reranker_type
    ORDER BY total_score_pct DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("조회된 데이터가 없습니다.")
        return

    # 시각화 설정
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # 막대 그래프 그리기 (Pipeline별 성과)
    g = sns.barplot(
        data=df, 
        x='pipeline', 
        y='total_score_pct', 
        hue='model_item',
        palette={'OpenAI': '#10a37f', 'Ollama': '#f15a24'} # OpenAI(녹색), Ollama(주황)
    )

    # 그래프 세부 설정
    plt.title(f'RAG Pipeline Performance Comparison (Index: {execute_indices})', fontsize=16, pad=20)
    plt.ylabel('Total Score (%)', fontsize=12)
    plt.xlabel('Retriever / Reranker Combination', fontsize=12)
    plt.ylim(0, 105)
    plt.xticks(rotation=15)
    plt.legend(title='LLM Model', loc='upper right')

    # 점수 텍스트 표시
    for p in g.patches:
        if p.get_height() > 0:
            g.annotate(f'{p.get_height():.1f}%', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontsize=10)

    plt.tight_layout()
    plt.show()

    # 상세 수치 표 출력
    print("\n[ 상세 성능 리포트 ]")
    print(df[['execute_index', 'model_item', 'pipeline', 'total_score_pct', 'faith', 'rel', 'ctx']].to_string(index=False))

# ════════════════════════════════════════
# ▣ 답변 시간 분석
# openai/ollama 별 답변 시간 분석 후 top3 비교(질문-응답 사간 평균 비교)
# ════════════════════════════════════════
def analyze_reply_time_top3(db_path, target_index):
    # 1. 데이터베이스 연결
    conn = sqlite3.connect(db_path)
    
    # 2. SQL 쿼리: model_item, pipeline 구성 요소 및 시작/종료 시간 조회
    # do_retriever가 NULL일 경우를 대비해 COALESCE 처리 (SQLite 문법)
    query = f"""
    SELECT 
        model_item,
        llm_model,
        IFNULL(do_retriever, 'Default') || ' / ' || IFNULL(reranker_type, 'None') AS pipeline,
        start_time,
        end_time
    FROM result_data
    WHERE execute_index = {target_index}
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("데이터가 없습니다. execute_index를 확인하세요.")
        return

    # 3. 시간 변환 및 수행 시간 계산 (초 단위)
    df['start_dt'] = pd.to_datetime(df['start_time'])
    df['end_dt'] = pd.to_datetime(df['end_time'])
    df['elapsed_sec'] = (df['end_dt'] - df['start_dt']).dt.total_seconds()

    # 4. 모델/파이프라인별 평균 응답 시간 계산
    df_avg = df.groupby(['model_item', 'pipeline', 'llm_model']).agg({
        'elapsed_sec': 'mean'
    }).reset_index()

    # 5. 모델별(OpenAI / Ollama) 상위 3개 추출
    # 각 모델별로 그룹지어 수행시간(elapsed_sec) 기준 오름차순(빠른 순) 상위 3개 선택
    top3_per_model = df_avg.sort_values('elapsed_sec').groupby('model_item').head(3).copy()

    # 6. 시각화
    plt.figure(figsize=(15, 8))
    sns.set_theme(style="whitegrid")
    
    # 색상 맵 설정 (OpenAI vs Ollama 구분)
    palette = {'OpenAI': '#10a37f', 'Ollama': '#f15a24'}
    
    # x축 라벨용 조합 이름 생성
    top3_per_model['config_label'] = top3_per_model['pipeline'] + "\n(" + top3_per_model['llm_model'] + ")"

    # 바 차트 생성
    barplot = sns.barplot(
        data=top3_per_model,
        x='config_label',
        y='elapsed_sec',
        hue='model_item',
        palette=palette
    )

    # 그래프 세부 설정
    plt.title(f'Reply Time Analysis: Top 3 Fastest by Model (Index: {target_index})', fontsize=16, pad=20)
    plt.ylabel('Average Reply Time (Seconds)', fontsize=12)
    plt.xlabel('Pipeline & LLM Configuration', fontsize=12)
    plt.legend(title='LLM Model Item', loc='upper right')
    
    # 바 위에 시간 표시
    for p in barplot.patches:
        height = p.get_height()
        if height > 0:
            barplot.annotate(f'{height:.2f}s', 
                             (p.get_x() + p.get_width() / 2., height), 
                             ha='center', va='center', 
                             xytext=(0, 9), 
                             textcoords='offset points',
                             fontsize=10, fontweight='bold')

    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.show()

    # 상세 수치 출력
    print("\n[ 답변 시간 분석 결과: 각 모델별 상위 3개 ]")
    print(top3_per_model[['model_item', 'llm_model', 'pipeline', 'elapsed_sec']].sort_values(['model_item', 'elapsed_sec']).to_string(index=False))

# 실행
if __name__ == "__main__":
    # 데이터베이스 파일명을 실제 파일명으로 수정하세요.
    db_path = r"D:\midprj.db"
    target_idx = 400
    
    # 1. 기존 성능 분석
   # analyze_model_performance(db_path, target_index=target_idx)
   # compare_rag_performance(db_path, execute_indices=[target_idx])
    
    # 2. 신규 답변 시간 분석
    analyze_reply_time_top3(db_path, target_index=target_idx)