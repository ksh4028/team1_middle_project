import reflex as rx
import os
import base64

def page():
    # 구글 슬라이드 링크
    doc_link = "https://docs.google.com/presentation/d/1GGCKnv8VLKtxa-S2wWff059i_fJz1wWP1mgmSjeRjEo/edit?usp=sharing"
    
    # 임베드용 URL로 변환 (/edit 부분을 /embed로 변경)
    if "/edit" in doc_link:
        embed_url = doc_link.split("/edit")[0] + "/embed"
    else:
        embed_url = doc_link

    return rx.vstack(
        rx.center(
            rx.heading("📊 프로젝트 리포트 요약", size="8", margin_y="4"),
            width="100%",
        ),
        rx.divider(),
        rx.box(
            rx.html(
                f'<iframe src="{embed_url}" style="width: 100%; height: 85vh; min-height: 800px; border: none;" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>'
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
        spacing="4",
        padding_x="8",
        padding_y="4",
        width="100%",
        height="100%",
        min_height="90vh",
        background_color="#f9fafb",
    )
