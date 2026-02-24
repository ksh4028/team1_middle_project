
import re

def filter_clean_for_rfp(text):
    """RFP 문서용 텍스트 전처리 (HTML 제거, 반복 축소, 중요 기호 보존)"""
    if not text:
        return ""
    
    # 1. HTML 태그 및 URL 제거 (Garbage removal)
    text = re.sub(r'<[^>]+>', ' ', str(text))
    text = re.sub(r'(http|https)://\S+', ' ', text)

    # 2. 반복 문자 정규화 (3회 이상 반복은 2회로 축소)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 3. 특수문자 필터링 (RFP용 허용 범위 확대 - 기술/문서 기호 보존)
    # 보존: . , ? ! ~ ; : - _ ( ) [ ] { } < > / % & = + " ' ` * #
    # 제거: 이모지, 제어 문자 등
    text = re.sub(r'[^\w\s.,?!~;:()\[\]{}<>/%&=+"\'`*#-]', ' ', text)

    # 4. 다중 공백 -> 단일 공백
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    test_cases = [
        "Budget: 100,000,000 (100%) - Section 1-1 [Note]",
        "Spec: i7-12700 / 3.5GHz / 16GB RAM",
        "Email: test@example.com",
        "Hello <br> world! http://link.com",
        "Repeat: ㅋㅋㅋㅋㅋㅋ okay",
        "Symbols: #Hash & Amp * Star + Plus = Equal"
    ]
    
    print("-" * 50)
    for text in test_cases:
        cleaned = filter_clean_for_rfp(text)
        print(f"Origin: {text}")
        print(f"Clean : {cleaned}")
        print("-" * 50)
