import reflex as rx
import os
import base64

def page():
    # assets 폴더 내의 상대 경로 (Reflex는 assets 폴더를 / 경로로 마운트함)
    pdf_filename = "midprj_report.pdf"
    
    # 서버 측 파일 존재 여부 확인을 위한 절대 경로 계산
    # d:\project\MidPrj\app\pages\summary.py -> d:\project\MidPrj\assets\midprj_report.pdf
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    asset_pdf_path = os.path.join(project_root, "assets", pdf_filename)
    
    error_msg = ""
    file_exists = os.path.exists(asset_pdf_path)
    
    if not file_exists:
        error_msg = f"PDF 파일을 찾을 수 없습니다. (확인된 경로: {asset_pdf_path})"
        pdf_url = ""
    else:
        pdf_url = f"/{pdf_filename}"

    return rx.vstack(
        rx.center(
            rx.heading("📊 프로젝트 리포트 요약", size="8", margin_y="4"),
            width="100%",
        ),
        rx.divider(),
        rx.cond(
            pdf_url != "",
            rx.box(
                rx.html(
                    f'<iframe src="{pdf_url}" style="width: 100%; height: 85vh; min-height: 800px; border: none;"></iframe>'
                ),
                width="100%",
                height="85vh",
                min_height="800px",
                border="1px solid #e5e7eb",
                border_radius="xl",
                box_shadow="lg",
                overflow="hidden",
                bg="white",
            ),
            rx.center(
                rx.vstack(
                    rx.text("❌ PDF 파일을 불러올 수 없습니다.", size="5", color="red", font_weight="bold"),
                    rx.text(error_msg, size="3", color="gray"),
                    spacing="4",
                    padding="8",
                    align="center",
                    border="1px dashed red",
                    border_radius="md",
                ),
                width="100%",
                height="60vh",
            )
        ),
        spacing="4",
        padding_x="8",
        padding_y="4",
        width="100%",
        height="100%",
        min_height="90vh",
        background_color="#f9fafb",
    )
