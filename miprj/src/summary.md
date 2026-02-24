# midprj_main.py 커맨드라인 질의응답 파이프라인 (병렬적/데이터 흐름 강조)

아래는 주요 컴포넌트 간 데이터 흐름과 병렬적 처리를 강조한 Mermaid 그래프입니다.

```mermaid
flowchart TD
    subgraph 사용자
        A[질의 입력 CLI]
    end

    subgraph 데이터베이스
        B1[SQLite: 메타데이터 조회]
        B2[SQLite: 문서 본문 조회]
    end

    subgraph 검색
        C1["Retriever: BM25/Vector/Hybrid/LLM"]
        C2[캐시/중복확인]
    end

    subgraph 후처리
        D1["Reranker: BGE/Dense/LLM/Hybrid"]
        D2[상위 문서 컨텍스트 추출]
    end

    subgraph 생성
        E1[LLM/프롬프트 기반 답변 생성]
    end

    subgraph 결과
        F1[최종 답변 출력 CLI]
        F2[검색/요약 결과 DB 저장]
    end

    %% 데이터 흐름 (병렬 및 주고받음 강조)
    A -->|질의| B1
    A -->|질의| B2
    B1 -->|메타데이터| C1
    B2 -->|문서 본문| C1
    C1 -->|후보 문서 집합| D1
    C1 -->|캐시 질의| C2
    C2 -->|캐시 hit/miss| C1
    D1 -->|재정렬 결과| D2
    D2 -->|컨텍스트| E1
    E1 -->|답변| F1
    E1 -->|요약/검색 결과| F2
    F2 -->|DB 저장 완료 알림| C2

    %% 병렬적 의미 강조
    B1 -..->|동시에| B2
    D1 -..->|동시에| D2
```

- 메타데이터/본문 조회, 캐시 확인 등은 병렬적으로 처리될 수 있음을 점선으로 표현
- 각 컴포넌트 간 데이터가 주고받는 흐름을 명확히 화살표로 표시
- 검색 결과와 요약 결과는 DB에 저장되어 재활용 및 캐싱에 활용됨
