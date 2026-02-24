import os
import win32com.client
import pythoncom

class HWPConverter:
    def __init__(self):
        self.hwp = None

    def _get_hwp_object(self):
        """Hancom Office HWP 객체 생성"""
        try:
            # 기존에 열려 있는 한글이 있으면 가져오고, 없으면 새로 생성
            #gencache.EnsureDispatch를 사용하여 초기 속도 및 호환성 확보
            hwp = win32com.client.gencache.EnsureDispatch("HWPFrame.HwpObject")
            return hwp
        except Exception as e:
            print(f"HWP 객체 생성 실패: {e}")
            return None

    def convert_to_pdf(self, hwp_path: str, pdf_path: str) -> bool:
        """HWP 파일을 PDF로 변환"""
        hwp_path = os.path.abspath(hwp_path)
        pdf_path = os.path.abspath(pdf_path)

        if not os.path.exists(hwp_path):
            print(f"파일을 찾을 수 없습니다: {hwp_path}")
            return False

        # COM 초기화 (멀티스레딩 환경 고려)
        pythoncom.CoInitialize()
        print(f"한글파일({hwp_path})을 PDF({pdf_path})로 변환합니다.")
        
        hwp = self._get_hwp_object()
        if not hwp:
            print("한글 프로그램을 실행할 수 없습니다. 한글(Hancom Office)이 설치되어 있는지 확인하세요.")
            return False

        try:
            # 보안 승인 모듈 등록 (보안 팝업 차단)
            res = hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModuleExample")
            if res:
                print("보안 승인 모듈 등록 성공")
            else:
                print("보안 승인 모듈 등록 실패 (이미 설정되어 있거나 경로 문제일 수 있습니다.)")
            
            # 메시지 박스 차단 (0: 모두 차단)
            hwp.SetMessageBoxMode(0)
            
            print(f"파일 여는 중: {hwp_path}")
            # 파일 열기
            if not hwp.Open(hwp_path):
                print("파일 열기 실패")
                return False
            
            print("PDF 변환 설정 중...")
            # PDF로 저장
            hwp.HAction.GetDefault("FileSaveAs_Pdf", hwp.HParameterSet.HFileOpenSave.HSet)
            hwp.HParameterSet.HFileOpenSave.filename = pdf_path
            hwp.HParameterSet.HFileOpenSave.Format = "PDF"
            
            print("PDF 변환 실행 중...")
            # hwp.SaveAs(Path, Format, arg)
            # Format "PDF"는 HWP 2010 이후부터 지원됨
            if hwp.SaveAs(pdf_path, "PDF"):
                print(f"변환 완료: {pdf_path}")
                return True
            else:
                print("PDF 변환 실행 실패 (SaveAs가 False 반환)")
                return False
        except Exception as e:
            print(f"변환 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if hwp:
                hwp.Quit()
            pythoncom.CoUninitialize()


def change_hwp_to_pdf_dir():
    hwp_search_dir = r"D:\project\TodoPrj_Anti\data\rfp_files\files"
    pdf_search_dir = r"D:\project\TodoPrj_Anti\data\rfp_files\hwp2pdf"
    for root, dirs, files in os.walk(hwp_search_dir):
        for file in files:
            if file.endswith(".hwp"):
                hwp_path = os.path.join(root, file)
                pdf_path = os.path.join(pdf_search_dir, file.replace(".hwp", ".pdf"))
                change_hwp_to_pdf(hwp_path, pdf_path)
                

def change_hwp_to_pdf(hwp_path: str, pdf_path: str) -> bool:
    """한글 파일을 PDF로 변환 (midprj_main에서 이동됨)"""
    try:
        converter = HWPConverter()
        converter.convert_to_pdf(hwp_path, pdf_path)
        print(f"HWP to PDF 변환 성공: {hwp_path} -> {pdf_path}")
        return True
    except Exception as e:
        print(f"HWP to PDF 변환 실패: {e}")
        return False

if __name__ == "__main__":
    # 간단한 테스트 코드
    change_hwp_to_pdf_dir()
    
    # converter = HWPConverter()
    # # 테스트용 파일 경로가 있다면 지정
    # converter.convert_to_pdf(r"D:\project\TodoPrj_Anti\data\rfp_files\files\(사)벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업 .hwp", 
    # r"D:\project\TodoPrj_Anti\data\rfp_files\hwp2pdf\test.pdf")
