import reflex as rx

def page():
	return rx.center(
		rx.text("이곳은 Rerank 페이지입니다.", size="6"),
		style={"height": "60vh"}
	)
