

from abc import ABC, abstractmethod
from langchain_community.retrievers import BM25Retriever

from langchain_core.retrievers import BaseRetriever



from typing import Any



class HybridRetriever(BaseRetriever):
# BM25 (키워드) + FAISS (의미) 검색을 결합하는 하이브리드 검색기 (BaseRetriever 상속)
    bm25_retriever: Any = None
    faiss_retriever: Any = None
    weights: list = [0.5, 0.5]

    def __init__(self, bm25_retriever, faiss_retriever, weights=[0.5, 0.5]):
        super().__init__()
        object.__setattr__(self, 'bm25_retriever', bm25_retriever)
        object.__setattr__(self, 'faiss_retriever', faiss_retriever)
        object.__setattr__(self, 'weights', weights)

    def _get_relevant_documents(self, query: str, *, filter: dict = None, **kwargs):
        # BaseRetriever requires _get_relevant_documents to be implemented
        return self.get_relevant_documents(query, filter=filter, **kwargs)

    def get_relevant_documents(self, query: str, filter: dict = None, **kwargs):
        # 기존 invoke 로직을 get_relevant_documents로 이동
        # 필터가 있을 경우 충분한 후보군 확보를 위해 k를 크게 설정
        base_k = self.faiss_retriever.search_kwargs.get("k", 5)
        search_k = max(base_k * 2, 25) if filter else base_k
        
        OpLog(
            f"HybridRetriever invoke: k={search_k}, filter={'on' if filter else 'off'}",
            level="INFO",
        )
        
        # 1. BM25 검색 수행
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # 2. FAISS 검색 수행 (임시로 k를 늘려서 검색 후 수동 필터링)
        # BUG FIX: IndexFlat.search() 에러 방지를 위해 similarity_search(filter=filter) 대신 invoke 후 수동 필터링 사용
        if filter:
            # retriever의 search_kwargs를 잠시 변경하여 더 많은 문서를 가져옴
            old_k = self.faiss_retriever.search_kwargs.get("k", 5)
            self.faiss_retriever.search_kwargs["k"] = search_k
            faiss_docs = self.faiss_retriever.invoke(query)
            self.faiss_retriever.search_kwargs["k"] = old_k # 복구
        else:
            faiss_docs = self.faiss_retriever.invoke(query)

        # 3. 수동 필터링 적용
        if filter:
            # BM25 필터링
            filtered_bm25 = []
            for doc in bm25_docs:
                if filter(doc.metadata):
                    filtered_bm25.append(doc)
            bm25_docs = filtered_bm25
            
            # FAISS 필터링
            filtered_faiss = []
            for doc in faiss_docs:
                if filter(doc.metadata):
                    filtered_faiss.append(doc)
            faiss_docs = filtered_faiss

        # 4. 상위 k개 유지 (전체 k로 다시 제한)
        final_k = self.faiss_retriever.search_kwargs.get("k", 5)
        bm25_docs = bm25_docs[:final_k]
        faiss_docs = faiss_docs[:final_k]

        # 5. 가중치 기반 결합 (RRF 방식)
        bm25_weight, faiss_weight = self.weights
        combined = {}
        scores = {}
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            combined.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + (bm25_weight / (rank + 1))
        for rank, doc in enumerate(faiss_docs):
            key = doc.page_content
            combined.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + (faiss_weight / (rank + 1))
        def safe_score(item):
            score = scores.get(item[0], 0.0)
            # NaN 방지: score가 NaN이면 0.0으로 대체
            if isinstance(score, float) and (score != score or score is None):
                return 0.0
            return score
        ordered = sorted(combined.items(), key=safe_score, reverse=True)
        return [doc for _, doc in ordered[:final_k]]

from .util_llmselector import get_llm
from .midprj_func import OpLog


def get_multiquery_retriever_class():
    # langchain_classic.retrievers에서 MultiQueryRetriever 클래스를 동적으로 import하여 반환
    try:
        from langchain_classic.retrievers import MultiQueryRetriever
        return MultiQueryRetriever
    except ImportError:
        raise ImportError(
            "MultiQueryRetriever is unavailable. Install/upgrade langchain_classic to use multiquery retrievers."
        )

class BaseRetrieverWrapper(ABC):
# 모든 검색기의 인터페이스를 정의하는 추상 클래스
    def __init__(self, vector_store, all_docs, param):
        self.vector_store = vector_store
        self.all_docs = all_docs
        self.param = param

    @abstractmethod
    def get_retriever(self):
        # 실제 LangChain 혹은 커스텀 retriever 객체를 반환
        pass



class BM25RetrieverModule(BaseRetrieverWrapper):
# 키워드 기반 BM25 검색기
    def get_retriever(self):
        retriever = BM25Retriever.from_documents(self.all_docs)
        retriever.k = self.param.k
        OpLog(f"BM25 retriever ready: k={self.param.k}", level="INFO")
        return retriever

class VectorRetrieverModule(BaseRetrieverWrapper):
# FAISS 전용 의미 기반 검색기
    def get_retriever(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.param.k})
        OpLog(f"Vector retriever ready: k={self.param.k}", level="INFO")
        return retriever

class HybridRetrieverModule(BaseRetrieverWrapper):
# BM25 + FAISS 하이브리드 검색기 (기존 HybridRetriever 활용)
    def get_retriever(self):
        bm25 = BM25Retriever.from_documents(self.all_docs)
        bm25.k = self.param.k
        faiss = self.vector_store.as_retriever(search_kwargs={"k": self.param.k})
        alpha = getattr(self.param, "hybrid_alpha", 0.5)
        alpha = max(0.0, min(1.0, float(alpha)))
        # 기존에 정의하신 HybridRetriever 클래스 사용
        OpLog(f"Hybrid retriever ready: k={self.param.k}, alpha={alpha}", level="INFO")
        return HybridRetriever(bm25, faiss, weights=[alpha, 1.0 - alpha])


class MultiQueryRetrieverModule(BaseRetrieverWrapper):
# 질문을 다각도로 재해석하여 검색하는 리트리버
    def __init__(self, vector_store, all_docs, param, llm):
        super().__init__(vector_store, all_docs, param)
        self.llm = llm  # 질문 생성을 위한 LLM 필요

    def get_retriever(self):
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.param.k})
        MultiQueryRetriever = get_multiquery_retriever_class()
        OpLog(f"MultiQuery retriever ready: k={self.param.k}", level="INFO")
        return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.llm)


DEFAULT_RETRIVER = "hybrid_llm" # 

class MultiQueryHybridRetrieverModule:
# Hybrid 검색(BM25+Vector)에 Multi-Query(LLM)를 결합한 모듈
    def __init__(self, vector_store, all_docs, param, llm_selector):
        self.vector_store = vector_store
        self.all_docs = all_docs
        self.param = param
        self.llm = get_llm(provider=llm_selector) 
        self.MultiQueryRetriever = get_multiquery_retriever_class()


    def get_retriever(self, do_retriever=DEFAULT_RETRIVER):
        # 1. 문자열 값에 따른 메서드 매핑 테이블 정의
        retriever_map = {
            "bm25": self.get_bm25_retriever,
            "vector": self.get_vector_retriever,
            "hybrid": self.get_hybrid_retriever,
            "llm": self.get_llm_retriever,
            "bm25_llm": self.get_bm25_llm_retriever,
            "vector_llm": self.get_vector_llm_retriever,
            "hybrid_llm": self.get_hybrid_llm_retriever,
        }

        # 2. 매핑 테이블에서 함수 호출
        retriever_func = retriever_map.get(do_retriever)
        
        if retriever_func:
            OpLog(f"Selected retriever type: {do_retriever}", level="INFO")
            return retriever_func()
        else:
            # 지원하지 않는 타입일 경우 예외 처리
            raise ValueError(
                f"Unsupported retriever type: {do_retriever}. "
                f"Supported types: {list(retriever_map.keys())}"
            )
        
    def get_bm25_retriever(self):
        # 키워드(BM25) 기반 리트리버 반환
        bm25 = BM25Retriever.from_documents(self.all_docs)
        bm25.k = getattr(self.param, "k", 4)
        return bm25

    def get_vector_retriever(self):
        # 임베딩(FAISS) 기반 리트리버 반환
        return self.vector_store.as_retriever(search_kwargs={"k": getattr(self.param, "k", 4)})

    def get_hybrid_retriever(self):
        # BM25+임베딩 하이브리드 리트리버 반환
        bm25 = BM25Retriever.from_documents(self.all_docs)
        bm25.k = getattr(self.param, "k", 4)
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": getattr(self.param, "k", 4)})
        alpha = getattr(self.param, "hybrid_alpha", 0.5)
        alpha = max(0.0, min(1.0, float(alpha)))
        return HybridRetriever(bm25, vector_retriever, weights=[alpha, 1.0 - alpha])

    def get_bm25_llm_retriever(self):
        bm25 = BM25Retriever.from_documents(self.all_docs)
        bm25.k = getattr(self.param, "k", 4)
        return self.MultiQueryRetriever.from_llm(retriever=bm25, llm=self.llm)
    
    def get_vector_llm_retriever(self):
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": getattr(self.param, "k", 4)}
        )
        return self.MultiQueryRetriever.from_llm(retriever=vector_retriever, llm=self.llm)

    def get_llm_retriever(self):
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.param.k})
        return self.MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.llm)

    def get_hybrid_llm_retriever(self):
        # 1. 기본 BM25 리트리버 설정
        bm25 = BM25Retriever.from_documents(self.all_docs)
        bm25.k = getattr(self.param, "k", 4)

        # 2. 기본 Vector 리트리버 설정
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": getattr(self.param, "k", 4)}
        )

        # 3. Hybrid 리트리버 생성 (BM25 + Vector)
        alpha = getattr(self.param, "alpha", 0.5)
        hybrid_retriever = HybridRetriever(
            bm25_retriever=bm25,
            faiss_retriever=vector_retriever,
            weights=[alpha, 1.0 - alpha]
        )
        # 4. Multi-Query 적용 (LLM이 질문을 확장하여 Hybrid 검색 수행)
        # 여기서 llm은 OpenAI 혹은 Ollama 객체입니다.
        mq_hybrid_retriever = self.MultiQueryRetriever.from_llm(
            retriever=hybrid_retriever, 
            llm=self.llm
        )
        print(f"INFO: Multi-Query Hybrid Retriever 구성 완료 (LLM: {self.llm.model_name if hasattr(self.llm, 'model_name') else 'Ollama'})")
        return mq_hybrid_retriever


def build_retriever_from_param(vector_store, all_docs, param, llm_selector,
                               do_retriever =DEFAULT_RETRIVER):
    multi = MultiQueryHybridRetrieverModule(vector_store, all_docs, param, llm_selector)
    retriever = multi.get_retriever(do_retriever=do_retriever)
    return retriever

if __name__ == "__main__":
    get_multiquery_retriever_class()
    print("MultiQueryRetriever class resolved successfully.")


    