
import pandas as pd
import os

# Define paths
BASE_DIR = r"D:\project\TodoPrj_Anti"
CSV_PATH = os.path.join(BASE_DIR, "data", "rfp_files", "data_list.csv")
OUTPUT_MD_PATH = os.path.join(BASE_DIR, "data", "rfp_list_under_100m.md")

def generate_rfp_list():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    try:
        # Load CSV
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} records.")

        # Ensure '사업 금액' is numeric
        df['사업 금액'] = pd.to_numeric(df['사업 금액'], errors='coerce').fillna(0)

        # Filter: Amount <= 100,000,000
        filtered_df = df[df['사업 금액'] <= 100000000].copy()
        
        # Filter out 0 or very small values if needed (optional, keeping for now)
        filtered_df = filtered_df[filtered_df['사업 금액'] > 0]

        print(f"Filtered {len(filtered_df)} records <= 100,000,000 won.")

        # Sort by Agency, then Amount (desc)
        filtered_df.sort_values(by=['발주 기관', '사업 금액'], ascending=[True, False], inplace=True)

        # Create Markdown Content
        md_content = "# 발주기관별 1억 원 이하 RFP 목록\n\n"
        md_content += f"총 {len(filtered_df)}건 검색됨 (기준: 100,000,000원 이하)\n\n"

        current_agency = None
        
        for index, row in filtered_df.iterrows():
            agency = str(row['발주 기관']).strip()
            title = str(row['사업명']).strip()
            amount = int(row['사업 금액'])
            date = str(row['공개 일자']).strip()
            bid_no = str(row['공고 번호']).strip()
            
            # G2B Direct Link Generation (Try to infer standard G2B link)
            # URL: https://www.g2b.go.kr:8101/ep/invitation/publish/bidInfoDtl.do?bidno=20241001798&bidseq=00
            # Assuming '공고 차수' exists or default to 00.
            # In the CSV sample, '공고 차수' is float 0.0 -> convert to string '00'
            try:
                bid_seq = str(int(float(row['공고 차수']))).zfill(2)
            except:
                bid_seq = "00"
            
            link = f"https://www.g2b.go.kr:8101/ep/invitation/publish/bidInfoDtl.do?bidno={bid_no}&bidseq={bid_seq}"

            if agency != current_agency:
                md_content += f"\n## {agency}\n\n"
                md_content += "| 공고번호 | 사업명 | 사업금액 | 공고일 | 링크 |\n"
                md_content += "|---|---|---|---|---|\n"
                current_agency = agency
            
            md_content += f"| {bid_no} | {title} | {amount:,}원 | {date} | [바로가기]({link}) |\n"

        # Write to file
        with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        print(f"Successfully generated report at: {OUTPUT_MD_PATH}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_rfp_list()
