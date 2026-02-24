# SQL 검색 및 의도 감지 관련 분리 모듈
from midprj_func import Lines, OpLog
from midprj_sqlite import SQLiteDB
import os
import configparser

class SQLSearchHelper:
    @staticmethod
    def detect_search_intent(question, filters):
        """질문의 동의어를 ini에서 읽어와 처리하고 리스트 검색(SQL) 의도가 있는지 판단 (키워드 DB 기반)"""
        config = configparser.ConfigParser()
        ini_path = os.path.join(os.path.dirname(__file__), 'midprj.ini')
        config.read(ini_path, encoding='utf-8')
        synonym_map = {}
        if config.has_section('synonym_map'):
            for k, v in config.items('synonym_map'):
                synonym_map[k.strip()] = v.strip()
        else:
            synonym_map = {
                "파일이름": "파일명",
                "금액": "예산",
                "사업비": "예산",
                "마감일": "입찰 참여 마감일",
                "공고번호": "공고 번호",
                "공고차수": "공고 차수",
                "발주기관": "발주 기관",
                "사업요약": "사업 요약",
                "파일형식": "파일 형식",
                "파일명": "파일명",
                "공개일자": "공개 일자",
                "입찰참여시작일": "입찰 참여 시작일",
                "입찰참여마감일": "입찰 참여 마감일",
            }
        processed_question = question
        for old, new in synonym_map.items():
            processed_question = processed_question.replace(old, new)

        db = SQLiteDB()
        try:
            intent_rows = db.select("SELECT keyword FROM intent_keywords")
            intent_keywords = [row[0] for row in intent_rows]
        except Exception:
            intent_keywords = [
                '공고 번호', '공고 차수', '사업명', '발주 기관', '예산', '공개 일자',
                '입찰 참여 시작일', '입찰 참여 마감일', '사업 요약', '파일형식', '파일명',
                '목록', '리스트', '찾아줘', '보여줘', '전체', '있나요', 'RFP', 'RFQ'
            ]
        is_list_query = any(word in processed_question for word in intent_keywords)

        try:
            domain_rows = db.select("SELECT DISTINCT category FROM domain_keywords")
            domain_keywords = [row[0] for row in domain_rows]
        except Exception:
            domain_keywords = [
                '분야', '영역', '산업', '부문', '섹터', '카테고리', '종류',
                '과학', 'IT', '정보', '기술', '건설', '의료', '교육', '환경', '문화', '복지'
            ]
        is_domain_query = any(word in processed_question for word in domain_keywords)

        has_any_filter = any([
            getattr(filters, 'notice_no', None), getattr(filters, 'notice_round', None), getattr(filters, 'project_name', None), getattr(filters, 'agency', None),
            getattr(filters, 'budget_min', None), getattr(filters, 'budget_max', None), getattr(filters, 'publish_date', None), getattr(filters, 'start_date', None),
            getattr(filters, 'end_date', None), getattr(filters, 'project_summary', None), getattr(filters, 'file_type', None), getattr(filters, 'file_name', None),
            getattr(filters, 'domain', None), getattr(filters, 'keywords', None)
        ])
        return is_list_query or has_any_filter or is_domain_query

    @staticmethod
    def format_budget(budget_value) -> str:
        if budget_value is None:
            return "정보 없음"
        try:
            if isinstance(budget_value, str):
                normalized = budget_value.replace(",", "").replace("원", "")
                budget_value = float(normalized)
            else:
                budget_value = float(budget_value)
            return f"{budget_value:,.0f}원"
        except (ValueError, TypeError):
            return str(budget_value)

    @staticmethod
    def sql_metadata_search(filters):
        """SQLite에서 필터 조건에 맞는 공고 목록을 직접 조회"""
        db = SQLiteDB()
        OpLog("SQL 메타데이터 검색 시작", level="INFO")
        query_parts = []
        params = []

        if getattr(filters, 'budget_max', None):
            query_parts.append("budget <= ?")
            params.append(filters.budget_max)
        
        if getattr(filters, 'budget_min', None):
            query_parts.append("budget >= ?")
            params.append(filters.budget_min)
        
        if getattr(filters, 'agency', None):
            query_parts.append("agency LIKE ?")
            params.append(f"%{filters.agency}%")

        if getattr(filters, 'notice_no', None):
            query_parts.append("Notice_no LIKE ?")
            params.append(f"%{filters.notice_no}%")
            
        if getattr(filters, 'project_name', None):
            query_parts.append("project_name LIKE ?")
            params.append(f"%{filters.project_name}%")
            
        if getattr(filters, 'publish_date', None):
            query_parts.append("publish_date LIKE ?")
            params.append(f"%{filters.publish_date}%")
            
        if getattr(filters, 'end_date', None):
            query_parts.append("participation_end_date LIKE ?")
            params.append(f"%{filters.end_date}%")

        if getattr(filters, 'project_summary', None):
            query_parts.append("project_summary LIKE ?")
            params.append(f"%{filters.project_summary}%")
        
        if getattr(filters, 'domain', None):
            domains = [d.strip() for d in str(filters.domain).split(',') if d.strip()]
            if domains:
                domain_clauses = []
                for domain in domains:
                    domain_clauses.append(
                        "(domain = ? OR domain LIKE ? OR domain LIKE ? OR domain LIKE ?)"
                    )
                    params.extend([
                        domain,
                        f"{domain},%",
                        f"%,{domain},%",
                        f"%,{domain}",
                    ])
                query_parts.append("(" + " OR ".join(domain_clauses) + ")")

        if getattr(filters, 'keywords', None):
            query_parts.append("keywords LIKE ?")
            params.append(f"%{filters.keywords}%")

        if getattr(filters, 'region', None):
            query_parts.append("region LIKE ?")
            params.append(f"%{filters.region}%")

        if not query_parts:
            # 필터가 없으면 전체 중 상위 10개만
            sql = "SELECT * FROM rfp_metadata LIMIT 10"
            rows = db.select(sql)
        else:
            where_clause = " AND ".join(query_parts)
            sql = f"SELECT * FROM rfp_metadata WHERE {where_clause} LIMIT 15"
            rows = db.select_with_params(sql, params)

        if not rows:
            OpLog("SQL 메타데이터 검색 결과 없음", level="INFO")
            return []

        # 결과 포맷팅
        formatted_list = []
        for row in rows:
            formatted_list.append({
                "Notice_no": row[0],
                "Notice_round": row[1],
                "project_name": row[2],
                "budget": row[3],
                "agency": row[4],
                "publish_date": row[5],
                "participation_start_date": row[6],
                "participation_end_date": row[7],
                "project_summary": row[8],
                "file_name": row[10],
            })
        
        return formatted_list

    @staticmethod
    def format_metadata_items(items: list[dict]) -> str:
        if not items:
            return ""
        formatted_list = []
        for item in items:
            formatted_list.append(
                "\n".join([
                    f"- 사업명: {item.get('project_name', '')}",
                    f"  공고번호: {item.get('Notice_no', '')} | 공고차수: {item.get('Notice_round', '')}",
                    f"  발주기관: {item.get('agency', '')} | 예산: {SQLSearchHelper.format_budget(item.get('budget'))}",
                    f"  공개일자: {item.get('publish_date', '')} | 입찰기간: {item.get('participation_start_date', '')} ~ {item.get('participation_end_date', '')}",
                    f"  파일명: {item.get('file_name', '')}",
                    f"  사업요약: {item.get('project_summary', '')}",
                ])
            )
        return "\n\n".join(formatted_list)

    @staticmethod
    def format_project_names(items: list[dict]) -> str:
        if not items:
            return ""
        names = []
        seen = set()
        for item in items:
            name = item.get("project_name", "")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        if not names:
            return ""
        return "\n".join([f"- {name}" for name in names])

    @staticmethod
    def collect_metadata_from_docs(docs) -> list[dict]:
        items = []
        seen = set()
        for doc in docs or []:
            meta = doc.metadata or {}
            key = (
                meta.get("Notice_no"),
                meta.get("Notice_round"),
                meta.get("project_name"),
                meta.get("file_name"),
            )
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "Notice_no": meta.get("Notice_no"),
                "Notice_round": meta.get("Notice_round"),
                "project_name": meta.get("project_name"),
                "budget": meta.get("budget"),
                "agency": meta.get("agency"),
                "publish_date": meta.get("publish_date"),
                "participation_start_date": meta.get("participation_start_date"),
                "participation_end_date": meta.get("participation_end_date"),
                "project_summary": meta.get("project_summary"),
                "file_name": meta.get("file_name"),
            })
        return items
