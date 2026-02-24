from __future__ import annotations

from typing import Iterable, Optional



from midprj_func import OpLog
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from util_llmselector import get_llm
import numpy as np



class Reranker:
    def __init__(self, param):
        self._param = param
        # BGE-Reranker 모델 로컬 로드 (GPU 사용 가능 시 use_fp16=True 권장)
        try:
            self.bge_model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=self._param.is_gpu)
            OpLog("BGE-Reranker v2-m3 로컬 모델 로드 완료", level="INFO")
        except Exception as e:
            OpLog(f"BGE 모델 로드 실패: {e}", level="WARN")
            self.bge_model = None

    def rerank_and_get_context(self, question, docs, embedder=None):
        if not docs:
            return [], "관련된 문서를 찾을 수 없습니다."
        reranker_type = self._param.reranker_type
        k = self._param.k

        # 1. BGE-Reranker 적용 (최신 추천 방식)
        if reranker_type == "bge":
            top_docs = self._bge_rerank(question, docs, k)
        
        # 2. Dense 리랭킹 (단순 벡터 유사도 기반)
        elif reranker_type == "dense":
            top_docs = self._dense_rerank_embedding(question, docs, k, embedder)
        
        # 3. LLM 리랭킹 (OpenAI/Ollama API 활용)
        elif reranker_type in ("openai", "ollama"):
            top_docs = self._llm_rerank(question, reranker_type, docs, k)
        
        # 4. Hybrid 리랭킹 (BGE + Dense 또는 BGE + LLM 조합 가능)
        elif reranker_type == "hybrid":
            top_docs = self._hybrid_rerank_complex(question, docs, k, embedder)
        
        else:
            raise NotImplementedError("지원하지 않는 리랭커 타입입니다: " + str(reranker_type))
        
        context = "\n---\n".join([doc.page_content for doc in top_docs])
        return top_docs, context

    def _bge_rerank(self, question, docs, k):
        """BGE Cross-Encoder를 이용한 로컬 리랭킹"""
        if not self.bge_model:
            return docs[:k]
        
        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.bge_model.compute_score(pairs)
        
        # 점수와 문서를 묶어서 정렬
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k]]

    def _dense_rerank_embedding(self, question, docs, k, embedder):
        """기존 벡터 유사도 기반 재정렬"""
        if not embedder:
            return docs[:k]
        q_vec = np.array(embedder.embed_query(question))
        doc_vecs = np.array(embedder.embed_documents([d.page_content for d in docs]))
        
        # 코사인 유사도 계산
        norm_q = np.linalg.norm(q_vec)
        norm_docs = np.linalg.norm(doc_vecs, axis=1)
        scores = np.dot(doc_vecs, q_vec) / (norm_docs * norm_q + 1e-8)
        
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k]]
    
    def _hybrid_rerank_complex(self, question, docs, k, embedder):
        """BGE(의미) + Dense(유사도) 결합 하이브리드 리랭킹"""
        # BGE 점수 확보 (0~1 사이로 정규화 권장)
        pairs = [[question, doc.page_content] for doc in docs]
        bge_scores = np.array(self.bge_model.compute_score(pairs))
        
        # Dense 점수 확보
        q_vec = np.array(embedder.embed_query(question))
        doc_vecs = np.array(embedder.embed_documents([d.page_content for d in docs]))
        dense_scores = np.dot(doc_vecs, q_vec) / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-8)
        
        # 가중치 결합 (BGE에 더 높은 가중치 부여 권장)
        combined_scores = 0.7 * bge_scores + 0.3 * dense_scores
        scored_docs = sorted(zip(combined_scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k]]


    def _dense_rerank(self, question, docs, k=10):
        # 간단한 dense 임베딩 유사도 기반 리랭킹 (예시)
        # 실제로는 임베딩 모델을 불러와서 question/doc 임베딩 후 유사도 계산 필요
        # 여기서는 임시로 doc.metadata['score'] 사용 (실제 구현시 교체)
        scored_docs = []
        for doc in docs:
            score = doc.metadata.get('dense_score', 0) if hasattr(doc, 'metadata') else 0
            scored_docs.append((score, doc))
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]

    def _llm_rerank(self, question, provider, docs, k=10):
        # provider는 param.llm_selector에서 받아야 함
        llm = get_llm(provider=provider)
        class RerankScore(BaseModel):
            score: int = Field(description="0에서 5까지의 관련도 점수")
            reason: str = Field(description="간단한 이유")
        parser = PydanticOutputParser(pydantic_object=RerankScore)
        prompt = ChatPromptTemplate.from_template(
            """
            당신은 검색 결과 리랭커입니다. 질문과 문서의 관련도를 0~5점으로 평가하세요.
            질문: {question}
            문서: {document}
            {format_instructions}
            """
        ).partial(format_instructions=parser.get_format_instructions())
        scored_docs = []
        for doc in docs:
            try:
                doc_text = doc.page_content[:1200]
                result = (prompt | llm | parser).invoke({"question": question, "document": doc_text})
                scored_docs.append((result.score, doc))
            except Exception as exc:
                OpLog(f"LLM 리랭킹 실패: {exc}", level="WARN")
                scored_docs.append((0, doc))
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]

    def _hybrid_rerank(self, question, docs, k=10):
        # dense 점수와 LLM 점수를 조합 (가중 평균)
        dense_scores = []
        for doc in docs:
            score = doc.metadata.get('dense_score', 0) if hasattr(doc, 'metadata') else 0
            dense_scores.append(score)
        dense_scores = np.array(dense_scores)
        dense_scores = (dense_scores - dense_scores.min()) / (np.ptp(dense_scores) + 1e-8) if len(dense_scores) > 0 else dense_scores

        llm = get_llm(provider=self._param.llm_selector)
        class RerankScore(BaseModel):
            score: int = Field(description="0에서 5까지의 관련도 점수")
            reason: str = Field(description="간단한 이유")
        parser = PydanticOutputParser(pydantic_object=RerankScore)
        prompt = ChatPromptTemplate.from_template(
            """
            당신은 검색 결과 리랭커입니다. 질문과 문서의 관련도를 0~5점으로 평가하세요.
            질문: {question}
            문서: {document}
            {format_instructions}
            """
        ).partial(format_instructions=parser.get_format_instructions())
        llm_scores = []
        for doc in docs:
            try:
                doc_text = doc.page_content[:1200]
                result = (prompt | llm | parser).invoke({"question": question, "document": doc_text})
                llm_scores.append(result.score)
            except Exception as exc:
                OpLog(f"LLM 리랭킹 실패: {exc}", level="WARN")
                llm_scores.append(0)
        llm_scores = np.array(llm_scores)
        llm_scores = (llm_scores - llm_scores.min()) / (np.ptp(llm_scores) + 1e-8) if len(llm_scores) > 0 else llm_scores

        # 가중치 조정 (예: dense 0.5, llm 0.5)
        hybrid_scores = 0.5 * dense_scores + 0.5 * llm_scores
        scored_docs = list(zip(hybrid_scores, docs))
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]

# BGE-Reranker 적용 예시 (개념 코드)
from FlagEmbedding import FlagReranker

class BGEReranker(Reranker):
    def __init__(self, param):
        super().__init__(param)
        # 가벼운 v2-m3 모델 추천
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) 

    def rerank(self, question, docs, k=5):
        # 질문과 각 문서의 페어를 생성
        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.reranker.compute_score(pairs)
        
        # 점수 기준으로 정렬 후 상위 k개 반환
        doc_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [ds[0] for ds in doc_scores[:k]]