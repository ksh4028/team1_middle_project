# midprj_eval.py
from datetime import datetime

# LangChain Imports
from .util_llmselector import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Project Imports
from .midprj_defines import OPENAI_SETTINGS, PARAMVAR
from .midprj_func import OpLog
from .midprj_sqlite import SQLiteDB

# ----------------------------------------------------------------
# 1. 평가 결과를 담을 Pydantic 모델 (JSON 파싱용)
# ----------------------------------------------------------------
class EvalScore(BaseModel):
    score: int = Field(description="1점부터 5점 사이의 점수")
    reason: str = Field(description="해당 점수를 부여한 이유")

# ----------------------------------------------------------------
# 2. 평가자(Judge) 클래스 정의
# ----------------------------------------------------------------
class RAGEvaluator:
    def __init__(self, param):
        ## LLM 및 파서 초기화
        #provider = "openai"
        if param and hasattr(param, 'llm_selector') and param.llm_selector:
            provider = param.llm_selector
        else:
            provider = "openai"
        provider = provider.lower()
        self.llm = get_llm(provider=provider)
        self.parser = JsonOutputParser(pydantic_object=EvalScore)

    def _get_score(self, prompt_template, **kwargs):
        """LLM에게 평가를 요청하고 점수를 파싱하는 공통 함수"""
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | self.parser
            result = chain.invoke(kwargs)
            return result
        except Exception as e:
            OpLog(f"평가 실패: {e}", level="ERROR")
            return {"score": 0, "reason": "Error during evaluation"}

    def eval_faithfulness(self, question, answer, context):
        """[신뢰성 평가] 답변이 문맥(Context)에 기반했는가?"""
        template = """
        당신은 공정한 평가자입니다. 아래 제공된 [문맥]만을 사용하여 [답변]이 생성되었는지 평가하세요.
        [질문]이나 외부 지식은 고려하지 말고, 오직 [문맥]과 [답변]의 논리적 연결성만 봅니다.
        
        [문맥]: {context}
        [답변]: {answer}
        
        1점: 답변이 문맥과 전혀 상관없거나 거짓 정보를 포함함.
        3점: 답변이 문맥을 일부 반영하지만, 문맥에 없는 내용도 섞여 있음.
        5점: 답변이 문맥의 내용만을 기반으로 충실하게 작성됨.
        
        결과를 JSON 형식으로 출력하세요: {{ "score": 점수, "reason": "이유" }}
        """
        return self._get_score(template, context=context, answer=answer)

    def eval_answer_relevance(self, question, answer):
        """[답변 관련성 평가] 답변이 질문의 의도에 맞는가?"""
        template = """
        당신은 공정한 평가자입니다. [답변]이 [질문]에 대해 얼마나 유용하고 적절한지 평가하세요.
        문맥의 사실 여부는 따지지 말고, 동문서답인지 아닌지를 판단하세요.
        
        [질문]: {question}
        [답변]: {answer}
        
        1점: 질문과 전혀 상관없는 답변 (동문서답).
        3점: 질문에 대답은 했으나 핵심을 놓치거나 모호함.
        5점: 질문의 의도를 정확히 파악하고 명확하게 답변함.
        
        결과를 JSON 형식으로 출력하세요: {{ "score": 점수, "reason": "이유" }}
        """
        return self._get_score(template, question=question, answer=answer)

    def eval_context_relevance(self, question, context):
        """[문맥 관련성 평가] 검색된 문서가 질문과 관련이 있는가?"""
        template = """
        당신은 검색 시스템 평가자입니다. [문맥]이 [질문]에 답하는 데 필요한 정보를 포함하고 있는지 평가하세요.
        
        [질문]: {question}
        [문맥]: {context}
        
        1점: 질문과 전혀 관련 없는 문서만 검색됨.
        3점: 일부 관련 정보가 있으나, 핵심 정보가 부족함.
        5점: 질문에 답하기 위한 충분하고 정확한 정보가 포함됨.
        
        결과를 JSON 형식으로 출력하세요: {{ "score": 점수, "reason": "이유" }}
        """
        return self._get_score(template, question=question, context=context)

# ----------------------------------------------------------------
# 3. 메인 실행 함수
# ----------------------------------------------------------------
def run_evaluation(execute_index: int, param: PARAMVAR = None):
    """
    특정 execute_index에 해당하는 데이터를 DB에서 읽어와 평가 후 저장
    """
    OpLog(f"=== 평가 시작 (Index: {execute_index}) ===", bLines=True)
    
    db = SQLiteDB()

    # 2. 데이터 조회 (result_data 테이블 구조에 맞춰 수정 필요)
    # 가정: id, execute_index, query, answer, context 컬럼이 존재한다고 가정
    # contexts가 JSON 문자열이나 리스트로 저장되어 있다고 가정
    try:
        rows = db.load_results_for_evaluation(execute_index)
    except Exception as e:
        OpLog(f"DB 조회 실패: {e}", level="ERROR")
        return

    if not rows:
        OpLog("평가할 데이터가 없습니다.", level="WARNING")
        return

    # 3. 평가 수행
    evaluator = RAGEvaluator(param=param) # 비용 절감을 위해 mini 사용 추천
    # 루프에 진입하기 전에 evaluatoin_result를 execute_index로 삭제
    db.delete_evaluation_results_by_execute_index(execute_index)
    
    for index, row in enumerate(rows):
        q_id = row["query_id"]
        result_id = row["result_id"]
        question = row["query"]
        answer = row["answer"]
        context = row["context"] if row["context"] is not None else ""

        print(f"\n[{index+1}/{len(rows)}] 평가 중: {question[:30]}...")

        # (1) 신뢰성 평가
        f_res = evaluator.eval_faithfulness(question, answer, context)
        # (2) 답변 관련성 평가
        a_res = evaluator.eval_answer_relevance(question, answer)
        # (3) 문맥 관련성 평가
        c_res = evaluator.eval_context_relevance(question, context)

        # 4. 결과 DB 저장
        db.save_evaluation_result(
            result_id=result_id,
            execute_index=execute_index,
            query_id=q_id,
            faithfulness_score=f_res['score'],
            faithfulness_reason=f_res['reason'],
            answer_relevance_score=a_res['score'],
            answer_relevance_reason=a_res['reason'],
            context_relevance_score=c_res['score'],
            context_relevance_reason=c_res['reason'],
            eval_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    OpLog(f"=== 평가 완료 (Index: {execute_index}) ===", bLines=True)

if __name__ == "__main__":
    from .midprj_main import init_Env
    init_Env()
    # 테스트용 (직접 실행 시)
    # python midprj_eval.py --index 1 형식으로 받을 수 있게 argparse 추가 가능
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute_index", type=int, default=1)
    args = parser.parse_args()
    param = PARAMVAR()  # 필요에 따라 PARAMVAR 초기화
    param.llm_selector = "openai"  # 평가 시 사용할 LLM 지정 (예: "gpt", "mini" 등)
    execute_index = args.execute_index
    run_evaluation(execute_index, param)

