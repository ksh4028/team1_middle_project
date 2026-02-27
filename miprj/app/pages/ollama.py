import reflex as rx
from app.midprj.midprj_main import Make_Model
import time

class OllamaState(rx.State):
    query: str = ""
    answer: str = ""
    is_loading: bool = False
    use_sqlite: bool = True # SQLite 사용 여부 기본값 True
    search_time: float = 0.0

    def handle_submit(self):
        if not self.query:
            return
        self.answer = ""
        self.is_loading = True
        self.search_time = 0.0
        yield
        start_time = time.time()
        try:
            # UI 상태를 모델 파라미터에 반영
            if model:
                model._param.use_sqlite = self.use_sqlite
            
            # Using the global model instance initialized in this module
            self.answer = model.rag_search(self.query)
        except Exception as e:
            self.answer = f"오류 발생: {str(e)}"
        finally:
            self.is_loading = False
            self.search_time = round(time.time() - start_time, 2)

def page():
    return rx.vstack(
        rx.hstack(
            rx.heading("Ollama RAG 검색", size="6"),
            rx.hstack(
                rx.checkbox(
                    checked=OllamaState.use_sqlite,
                    on_change=OllamaState.set_use_sqlite,
                ),
                rx.text("SQLite 검색 사용", size="2"),
                align="center",
                spacing="2",
            ),
            rx.button(
                "검색",
                on_click=OllamaState.handle_submit,
                loading=OllamaState.is_loading,
            ),
            rx.cond(
                OllamaState.search_time > 0,
                rx.text(f"(검색 시간 : {OllamaState.search_time}초)", size="2", color="gray"),
            ),
            spacing="4",
            align="center",
            width="100%",
        ),
        rx.input(
            placeholder="질문을 입력하세요...",
            on_change=OllamaState.set_query,
            value=OllamaState.query,
            width="100%",
        ),
        rx.divider(),
        rx.cond(
            OllamaState.answer,
            rx.box(
                rx.markdown(OllamaState.answer),
                padding="4",
                border="1px solid #ccc",
                border_radius="md",
                width="100%",
                flex="1",
                overflow="auto",
                bg="white",
            ),
        ),
        spacing="4",
        padding_x="4",
        padding_bottom="4",
        padding_top="2em", # 상단 여백 추가로 한 줄 내림 효과
        width="100%",
        height="95vh",
    )

# Initialize model once at module level
try:
    model = Make_Model("ollama")
except Exception as e:
    print(f"Model initialization failed: {e}")
    model = None

