# -*- coding: utf-8 -*-
import argparse
from . import midprj_main as prj
from .util_preprocess import setup_all_data


def main():
    parser = argparse.ArgumentParser(description="MidPrj 메타데이터 사전 저장")
    parser.add_argument("--csv_path", type=str, default=prj.CSV_PATH, help="CSV 경로")
    parser.add_argument("--rfp_data_dir", type=str, default=prj.RFP_DATA_DIR, help="RFP 데이터 디렉토리")
    parser.add_argument("--store_ver", type=str, default="V15", help="스토어 버전")
    parser.add_argument("--adobe_ver", type=str, default="V01", help="Adobe 요약 버전")
    parser.add_argument("--force_domain", action="store_true", help="도메인 재추출 강제")

    args = parser.parse_args()

    param = prj.PARAMVAR()
    param.csv_path = args.csv_path
    param.rfp_data_dir = args.rfp_data_dir
    param.store_ver = args.store_ver
    param.adobe_ver = args.adobe_ver

    setup_all_data(param, force_domain=args.force_domain)


if __name__ == "__main__":
    main()
