import reflex as rx
from app.midprj.midprj_main import Make_Model
import time

class OllamaState(rx.State):
    query: str = ""
    answer: str = ""
    is_loading: bool = False
    use_sqlite: bool = True # SQLite 사용 여부 기본값 True
    search_time: float = 0.0
    is_listening: bool = False

    def toggle_listening(self):
        """버튼 클릭 시 듣는 중/대기 상태 토글"""
        self.is_listening = not self.is_listening

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

# 브라우저에서 실행될 자바스크립트 코드
STT_JS = """
    if (!window.voice2text_recognition) {
        window.voice2text_recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        window.voice2text_recognition.lang = 'ko-KR';
        window.voice2text_recognition.interimResults = false;
        window.voice2text_recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            const input = document.getElementById('query_input');
            if (input) {
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                nativeInputValueSetter.call(input, transcript);
                input.dispatchEvent(new Event('input', { bubbles: true }));
            }
            const btn = document.getElementById('stt_btn');
            if (btn) btn.click();
        };
        window.voice2text_recognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
        };
    }
    if (!window.voice2text_listening) {
        window.voice2text_recognition.start();
        window.voice2text_listening = true;
    } else {
        window.voice2text_recognition.stop();
        window.voice2text_listening = false;
    }
"""

def page():
    return rx.vstack(
        rx.hstack(
            rx.heading("Cloud(GCP) RAG 검색", size="6"),
            rx.hstack(
                rx.checkbox(
                    checked=OllamaState.use_sqlite,
                    on_change=OllamaState.set_use_sqlite,
                ),
                rx.text("SQLite 검색 사용", size="2"),
                align="center",
                spacing="2",
            ),
            rx.cond(
                OllamaState.search_time > 0,
                rx.text(f"(검색 시간 : {OllamaState.search_time}초)", size="2", color="gray"),
            ),
            spacing="4",
            align="center",
            width="100%",
        ),
        rx.hstack(
            rx.button(
                rx.cond(OllamaState.is_listening, rx.icon("mic-off"), rx.icon("mic")),
                on_click=[OllamaState.toggle_listening, rx.call_script(STT_JS)],
                color_scheme=rx.cond(OllamaState.is_listening, "red", "blue"),
                id="stt_btn",
            ),
            rx.button(
                "검색",
                on_click=OllamaState.handle_submit,
                loading=OllamaState.is_loading,
            ),
            rx.input(
                placeholder="질문을 입력하세요...",
                on_change=OllamaState.set_query,
                value=OllamaState.query,
                width="100%",
                id="query_input",
            ),
            width="100%",
        ),
        rx.divider(),
        rx.dialog.root(
            rx.dialog.content(
                rx.vstack(
                    rx.spinner(size="3"),
                    rx.text("답변을 생성하고 있습니다. 잠시만 기다려 주세요...", size="3", font_weight="bold"),
                    align="center",
                    spacing="3",
                ),
                style={"max_width": "400px", "padding": "2em"},
            ),
            open=OllamaState.is_loading,
        ),
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

