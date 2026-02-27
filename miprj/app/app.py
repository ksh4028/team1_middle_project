import reflex as rx


from .pages import summary, preprocess, retriever, rerank, openai, ollama


def index() -> rx.Component:
    return rx.vstack(
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("Summary", value="summary"),
                rx.tabs.trigger("OpenAI", value="openai"),
                rx.tabs.trigger("Ollama", value="ollama"),
                rx.tabs.trigger("Version", value="version"),
                width="100%",
            ),
            rx.tabs.content(summary.page(), value="summary"),
            rx.tabs.content(openai.page(), value="openai"),
            rx.tabs.content(ollama.page(), value="ollama"),
            rx.tabs.content(
                rx.center(
                    rx.text("version : 1.0.02", size="4"),
                    padding="4",
                ),
                value="version",
            ),
            default_value="summary",
            width="100%",
            padding="4",
        ),
        min_height="100vh",
        spacing="0",
        width="100%",
    )


app = rx.App()
app.add_page(index)
