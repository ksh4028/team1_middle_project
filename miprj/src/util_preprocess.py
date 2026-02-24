# ════════════════════════════════════════
# ▣ Domain Extraction Utility for RFP Metadata
# ════════════════════════════════════════
# Purpose: One-time utility to extract and save domain/category information
# Usage: python midprj_util.py [--force] [--sample N] [--dry-run]

import os
import sys
import argparse
import pandas as pd
from typing import Optional
from tqdm import tqdm

# Import from main module
from midprj_defines import OPENAI_SETTINGS
from midprj_main import SQLiteDB, init_Env, PARAMVAR
from midprj_func import OpLog, Lines, filter_clean_for_rfp
from util_llmselector import get_llm
from langchain_core.messages import HumanMessage
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
plt.style.use('fivethirtyeight')

# Initialize environment
init_Env()

# Domain classification prompt template
DOMAIN_CLASSIFICATION_PROMPT = """다음 사업 요약과 문서 내용을 읽고, 가장 적합한 분야를 선택하세요.

분야 목록:
- 과학: 과학기술, 연구개발, R&D, 나노, 반도체, 소재
- IT: 정보기술, 소프트웨어, 클라우드, AI, 인공지능, 데이터, 시스템 구축
- 건설: 건설, 공사, 토목, 건축, 시설, 인프라
- 의료: 의료, 보건, 병원, 의약품, 제약, 헬스케어
- 교육: 교육, 학교, 대학, 연수, 훈련
- 환경: 환경, 에너지, 친환경, 신재생, 그린
- 문화: 문화, 예술, 관광, 체육, 스포츠
- 복지: 복지, 사회복지, 복지시설, 요양
- 국방: 국방, 방위, 군사, 안보
- 통신: 통신, 네트워크, 5G, 6G, 이동통신
- 행정: 행정, 정부, 공공, 민원, 시스템 운영
- 기타: 위 분야에 해당하지 않음

사업 요약: {project_summary}

문서 내용 (처음 500자): {text_preview}

**중요**: 답변은 분야명만 출력하세요. 여러 분야면 콤마(,)로 구분하세요 (예: IT, 교육)
다른 설명 없이 분야명만 출력하세요."""

# Keyword-based fallback classification


def classify_domain_with_llm(project_summary: str, text_content: str) -> Optional[str]:
    """LLM을 사용하여 도메인 분류"""
    try:
        # provider를 외부에서 인자로 받도록 변경 (self 제거)
        provider = None
        # param이 있으면 provider를 param.llm_selector에서 가져옴
        import inspect
        frame = inspect.currentframe().f_back
        param = frame.f_locals.get('param', None)
        if param and hasattr(param, 'llm_selector'):
            provider = param.llm_selector
        llm = get_llm(provider=provider)
        text_preview = text_content[:500] if text_content else "(내용 없음)"
        prompt = DOMAIN_CLASSIFICATION_PROMPT.format(
            project_summary=project_summary or "(요약 없음)",
            text_preview=text_preview
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        domain = response.content.strip()
        valid_domains = ['과학', 'IT', '건설', '의료', '교육', '환경', '문화', '복지', '국방', '통신', '행정', '기타']
        normalized = domain.replace("/", ",").replace("|", ",").replace(";", ",")
        candidates = [item.strip() for item in normalized.split(",") if item.strip()]
        domains = []
        for item in candidates:
            if item in valid_domains and item not in domains:
                domains.append(item)
        if domains:
            return ",".join(domains)
        Lines(f"LLM returned invalid domain: {domain}, using fallback")
        return None
    except Exception as e:
        OpLog(f"LLM classification failed: {e}", level="WARN")
        return None


def classify_domain_with_keywords(project_summary: str, text_content: str) -> str:
    """키워드 기반 도메인 분류 (fallback)"""
    """키워드 기반 도메인 분류 (DB 기반)"""
    from midprj_main import SQLiteDB
    db = SQLiteDB()
    rules = db.get_domain_keywords()
    combined_text = f"{project_summary} {text_content[:1000]}"
    domain_scores = {}
    for domain, keywords in rules.items():
        score = sum(1 for keyword in keywords if keyword in combined_text)
        if score > 0:
            domain_scores[domain] = score
    if domain_scores:
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain
    else:
        return '기타'


def extract_domain(row):
    """사업명과 발주기관을 기반으로 도메인 태깅 (중복 허용)"""
    # SQLiteDB의 get_domain_keywords를 사용하여 동적으로 규칙 생성
    db = SQLiteDB()
    rules = db.get_domain_keywords()
    
    # 만약 DB에 키워드가 없으면 기본값 사용 (fallback)
    if not rules:
        rules = {
            '교육': ['교육', '대학', '학교', '학습', '강의', 'LMS', '인재', '연수'],
            '과학/IT': ['시스템', '정보', '개발', '데이터', 'SW', 'AI', '클라우드', '나노', '연구', '기술', 'R&D', '플랫폼'],
            '예술/문화': ['문화', '예술', '전시', '관광', '콘텐츠', '디자인', '행사'],
            '행정/기타': ['행정', '운영', '관리', '시설', '청사'],
            '건설/토목': ['건설', '공사', '도로', '건축']
        }

    text = str(row['사업명']) + " " + str(row['발주 기관']) + " " + str(row['사업 요약'])
    domains = []
    
    for domain, keywords in rules.items():
        for keyword in keywords:
            if keyword.lower() in text.lower():
                domains.append(domain)
                break
    
    if not domains:
        return '기타'
    
    return ','.join(domains)

def extract_keywords(text):
    """사업명에서 핵심 단어 추출"""
    import re
    if pd.isna(text): return ""
    words = text.split()
    keywords = [re.sub(r'[^\w]', '', w) for w in words if len(w) > 1]
    return ','.join(keywords[:5])

def extract_region(file_name, agency_name):
    """파일명에서 지역을 우선 추출하고 없으면 기관명에서 추출"""
    regions = [
        '서울', '경기', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
        '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주',
    ]

    def find_region(value):
        if pd.isna(value):
            return None
        text = str(value)
        for r in regions:
            if r in text:
                return r
        return None

    file_region = find_region(file_name)
    if file_region:
        return file_region

    agency_region = find_region(agency_name)
    if agency_region:
        return agency_region

    if pd.isna(file_name) and pd.isna(agency_name):
        return '미상'
    return '전국/기타'

def classify_domain(project_summary: str, text_content: str, use_llm: bool = True) -> str:
    """도메인 분류 (LLM + keyword fallback)"""
    if use_llm:
        # Try LLM first
        domain = classify_domain_with_llm(project_summary, text_content)
        if domain:
            return domain
    
    # Fallback to keyword matching
    return classify_domain_with_keywords(project_summary, text_content)


def extract_all_domains(force: bool = False, sample: Optional[int] = None, dry_run: bool = False, use_llm: bool = True):
    """모든 RFP 메타데이터의 도메인 추출 및 저장"""
    Lines("도메인 추출 시작")
    OpLog(f"도메인 추출 시작 - force={force}, sample={sample}, dry_run={dry_run}, use_llm={use_llm}", level="INFO")
    
    db = SQLiteDB()
    
    # Load metadata
    sql = "SELECT Notice_no, project_summary, text_content, domain FROM rfp_metadata"
    rows = db.select(sql)
    
    if not rows:
        print("❌ 메타데이터가 없습니다.")
        return
    
    print(f"총 {len(rows)}개의 레코드 발견")
    
    # Filter records that need processing
    records_to_process = []
    for row in rows:
        notice_no, project_summary, text_content, existing_domain = row
        if force or existing_domain is None:
            records_to_process.append((notice_no, project_summary, text_content))
    
    if sample:
        records_to_process = records_to_process[:sample]
    
    print(f"처리할 레코드: {len(records_to_process)}개")
    
    if not records_to_process:
        print("✅ 이미 모든 레코드에 도메인이 설정되어 있습니다.")
        return
    
    # Process each record
    update_params = []
    for notice_no, project_summary, text_content in tqdm(records_to_process, desc="도메인 추출 중"):
        try:
            domain = classify_domain(project_summary, text_content, use_llm=use_llm)
            update_params.append((domain, notice_no))
            Lines(f"Notice {notice_no}: domain={domain}")
        except Exception as e:
            OpLog(f"Notice {notice_no} 처리 실패: {e}", level="ERROR")
            update_params.append(('기타', notice_no))
    
    # Save to database
    if not dry_run:
        db._connect()
        db.cursor.executemany(
            "UPDATE rfp_metadata SET domain = ? WHERE Notice_no = ?",
            update_params
        )
        db.connection.commit()
        db._close()
        print(f"✅ {len(update_params)}개 레코드 업데이트 완료")
        OpLog(f"도메인 추출 완료: {len(update_params)}개 레코드 업데이트", level="INFO")
    else:
        print(f"🔍 Dry-run mode: {len(update_params)}개 레코드가 업데이트될 예정")
        # Show sample
        for i, (domain, notice_no) in enumerate(update_params[:5]):
            print(f"  {i+1}. Notice {notice_no} -> {domain}")
        if len(update_params) > 5:
            print(f"  ... and {len(update_params) - 5} more")


def initialize_domain_keywords_to_db(db: SQLiteDB):
    """초기 도메인 키워드를 DB에 저장 (midprj_main에서 이동됨)"""
    rows = db.select("SELECT count(*) FROM domain_keywords")
    if rows[0][0] == 0:
        OpLog("도메인 키워드 초기화 시작", level="INFO")
        domain_keywords = {
            '과학': ['과학', '과학기술', '연구개발', 'R&D'],
            'IT': ['IT', '정보기술', '소프트웨어', 'SW', '하드웨어', 'HW', '클라우드', 'AI', '인공지능', '데이터'],
            '건설': ['건설', '공사', '토목', '건축'],
            '의료': ['의료', '보건', '병원', '의약', '제약'],
            '교육': ['교육', '학교', '대학', '연수'],
            '환경': ['환경', '에너지', '친환경', '신재생'],
            '문화': ['문화', '예술', '관광', '체육'],
            '복지': ['복지', '사회복지', '복지시설'],
            '국방': ['국방', '방위', '군사'],
            '통신': ['통신', '네트워크', '5G', '6G'],
        }
        params = []
        for category, keywords in domain_keywords.items():
            for kw in keywords:
                params.append((category, kw))
        
        db._connect()
        db.cursor.executemany("INSERT INTO domain_keywords (category, keyword) VALUES (?, ?)", params)
        db.connection.commit()
        db._close()
        OpLog("도메인 키워드 초기화 완료", level="INFO")
    else:
        Lines("도메인 키워드가 이미 존재합니다.")

def save_metadata_from_csv(param: PARAMVAR):
    """CSV 파일을 읽어 SQLite DB에 메타데이터 저장 (멀티 도메인, 키워드, 지역 추출 포함)"""
    import re
    db = SQLiteDB()
    OpLog(f"CSV 메타데이터 저장 시작: {param.csv_path}", level="INFO")
    
    try:
        df = pd.read_csv(param.csv_path)
        
        print("데이터 분석 및 메타데이터 생성 중...")
        # 1. 도메인, 키워드, 지역 추출 적용
        df['domain'] = df.apply(extract_domain, axis=1)
        df['keywords'] = df['사업명'].apply(extract_keywords)
        df['region'] = df.apply(lambda row: extract_region(row['파일명'], row['발주 기관']), axis=1)

        # 2. 예산 데이터 정제
        if df['사업 금액'].dtype == 'object':
            df['사업 금액'] = df['사업 금액'].str.replace(',', '').astype(float)
        else:
            df['사업 금액'] = df['사업 금액'].fillna(0).astype(float)

        # 3. DB 저장
        db.execute("DELETE FROM rfp_metadata")
        db._connect()
        sql = '''
            INSERT OR REPLACE INTO rfp_metadata (Notice_no, Notice_round, project_name, budget, agency, publish_date, participation_start_date, participation_end_date, project_summary, file_type, file_name, text_content, domain, keywords, region)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        params_list = []
        for _, row in df.iterrows():
            params_list.append((
                row['공고 번호'],
                row['공고 차수'],
                row['사업명'],
                row['사업 금액'],
                row['발주 기관'],
                row['공개 일자'],
                row['입찰 참여 시작일'],
                row['입찰 참여 마감일'],
                row['사업 요약'],
                row['파일형식'],
                row['파일명'],
                filter_clean_for_rfp(row['텍스트']),
                row['domain'],
                row['keywords'],
                row['region']
            ))
            
        db.cursor.executemany(sql, params_list)
        db.connection.commit()
        db._close()
        OpLog(f"메타데이터 저장 완료: {len(df)}개 레코드", level="INFO")
        return True
    except Exception as e:
        OpLog(f"CSV 저장 실패: {e}", level="ERROR")
        import traceback
        OpLog(traceback.format_exc(), level="ERROR")
        return False

def view_domain_keywords_statistics():
    
    # 도메인/키워드/도메인-키워드 조합별 통계 시각화
    db = SQLiteDB()
    sql = "SELECT category, keyword FROM domain_keywords"
    rows = db.select(sql)

    # 1. 도메인별 키워드 수
    category_counts = {}
    # 2. 키워드별 도메인 수
    keyword_counts = {}
    # 3. 도메인-키워드 조합 빈도 (여기선 모두 1이지만, 확장성 위해)
    domain_keyword_pairs = []

    for category, keyword in rows:
        if category not in category_counts:
            category_counts[category] = set()
        category_counts[category].add(keyword)
        if keyword not in keyword_counts:
            keyword_counts[keyword] = set()
        keyword_counts[keyword].add(category)
        domain_keyword_pairs.append((category, keyword))

    print("\n" + "="*50)
    print("도메인 키워드 통계")
    print("="*50)
    for category, keywords in category_counts.items():
        print(f"{category:10s}: {len(keywords):3d}개 키워드 - {', '.join(list(keywords)[:5])}...")
    print("="*50 + "\n")

    try:
        import matplotlib.pyplot as plt
        # 1. 도메인별 키워드 수
        categories = list(category_counts.keys())
        n_keywords = [len(category_counts[c]) for c in categories]
        plt.figure(figsize=(8, 5))
        plt.bar(categories, n_keywords, color='orange')
        plt.xlabel('도메인')
        plt.ylabel('키워드 수')
        plt.title('도메인별 키워드 개수')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # 2. 키워드별 도메인 수 (상위 20개만)
        keywords = list(keyword_counts.keys())
        n_domains = [len(keyword_counts[k]) for k in keywords]
        # 상위 20개만 표시
        top_k = 20
        sorted_kw = sorted(zip(keywords, n_domains), key=lambda x: x[1], reverse=True)[:top_k]
        top_keywords, top_n_domains = zip(*sorted_kw) if sorted_kw else ([],[])
        plt.figure(figsize=(10, 5))
        plt.bar(top_keywords, top_n_domains, color='skyblue')
        plt.xlabel('키워드')
        plt.ylabel('도메인 수')
        plt.title('키워드별 도메인 개수 (상위 20)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # 3. 도메인-키워드 조합 히트맵 (상위 10개 도메인, 20개 키워드)
        import numpy as np
        from collections import Counter
        top_domains = sorted(category_counts.keys(), key=lambda c: len(category_counts[c]), reverse=True)[:10]
        top_keywords2 = [k for k, _ in sorted(keyword_counts.items(), key=lambda x: len(x[1]), reverse=True)[:20]]
        # 매트릭스 생성
        matrix = np.zeros((len(top_domains), len(top_keywords2)), dtype=int)
        pair_set = set(domain_keyword_pairs)
        for i, d in enumerate(top_domains):
            for j, k in enumerate(top_keywords2):
                if (d, k) in pair_set:
                    matrix[i, j] = 1
        plt.figure(figsize=(12, 6))
        plt.imshow(matrix, cmap='Blues', aspect='auto')
        plt.colorbar(label='존재 여부')
        plt.xticks(np.arange(len(top_keywords2)), top_keywords2, rotation=45, ha='right')
        plt.yticks(np.arange(len(top_domains)), top_domains)
        plt.xlabel('키워드')
        plt.ylabel('도메인')
        plt.title('도메인-키워드 조합 히트맵 (상위 도메인/키워드)')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[그래프 출력 오류] {e}")

def setup_all_data(param: PARAMVAR, force_domain: bool = False):
    """전체 데이터 초기화 프로세스 (키워드 초기화 -> 메타데이터 저장)"""
    Lines("전체 데이터 초기화 프로세스 시작")
    
    # 1. 도메인 키워드 초기화 (추출 로직이 DB 키워드를 사용하므로 먼저 수행)
    db = SQLiteDB()
    initialize_domain_keywords_to_db(db)
    
    # 2. CSV -> DB 저장 (내부에서 도메인, 키워드, 지역 추출 수행)
    save_metadata_from_csv(param)
    
    # 3. 추가적인 도메인 추출 (필요한 경우)
    # extract_all_domains(force=force_domain)
    show_domain_statistics()
    
    print("✅ 모든 데이터 준비가 완료되었습니다.")

def show_domain_statistics():
    """도메인별 통계 출력"""
    db = SQLiteDB()
    sql = "SELECT domain FROM rfp_metadata"
    rows = db.select(sql)
    
    print("\n" + "="*50)
    print("도메인별 통계")
    print("="*50)

    domain_counts = {}
    total_records = 0
    for (domain_value,) in rows:
        total_records += 1
        domain_text = domain_value or "(미분류)"
        parts = [item.strip() for item in str(domain_text).split(",") if item.strip()]
        if not parts:
            parts = ["(미분류)"]
        for item in parts:
            domain_counts[item] = domain_counts.get(item, 0) + 1

    total_tags = 0
    for domain_name, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{domain_name:10s}: {count:4d}개")
        total_tags += count
    
    print("="*50)
    print(f"{'총계(레코드)':10s}: {total_records:4d}개")
    print(f"{'총계(태그)':10s}: {total_tags:4d}개")
    print("="*50 + "\n")
    


def main():
    parser = argparse.ArgumentParser(description="RFP 메타데이터 도메인 추출 유틸리티")
    parser.add_argument("--force", action="store_true", help="이미 도메인이 있는 레코드도 재추출")
    parser.add_argument("--sample", type=int, help="처리할 레코드 수 제한 (테스트용)")
    parser.add_argument("--dry-run", action="store_true", help="실제 저장 없이 미리보기만")
    parser.add_argument("--no-llm", action="store_true", help="LLM 사용 안 함 (키워드만 사용)")
    parser.add_argument("--stats", action="store_true", help="도메인 통계만 출력")
    
    args = parser.parse_args()
    
    if args.stats:
        show_domain_statistics()
    else:
        extract_all_domains(
            force=args.force,
            sample=args.sample,
            dry_run=args.dry_run,
            use_llm=not args.no_llm
        )
        show_domain_statistics()

def Execute_domain_extraction():
    """도메인 추출 실행 (이제 setup_all_data를 사용)"""
    param = PARAMVAR()
    setup_all_data(param, force_domain=True)

def change_hwp_to_pdf_dir():
    hwp_search_dir = r"D:\project\TodoPrj_Anti\data\rfp_files\files"
    pdf_search_dir = r"D:\project\TodoPrj_Anti\data\rfp_files"
    for root, dirs, files in os.walk(hwp_search_dir):
        for file in files:
            if file.endswith(".hwp"):
                hwp_path = os.path.join(root, file)
                pdf_path = os.path.join(pdf_search_dir, file.replace(".hwp", ".pdf"))
                change_hwp_to_pdf(hwp_path, pdf_path)
                

def change_hwp_to_pdf(hwp_path: str, pdf_path: str) -> bool:
    """한글 파일을 PDF로 변환 (midprj_main에서 이동됨)"""
    try:
        # from hwp_converter import HWPConverter  # 오류 방지: 임시 주석 처리
        converter = HWPConverter()
        converter.convert_to_pdf(hwp_path, pdf_path)
        Lines(f"HWP to PDF 변환 성공: {hwp_path} -> {pdf_path}")
        return True
    except Exception as e:
        Lines(f"HWP to PDF 변환 실패: {e}")
        return False

if __name__ == "__main__":
    view_domain_keywords_statistics()
    #Execute_domain_extraction()
    #change_hwp_to_pdf_dir()

