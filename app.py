import gradio as gr
from transformers import pipeline


def analytics_emo(x):
    data = classifier(x)[0]
    return data


if __name__ == "__main__":
    classifier = pipeline(
        "text-classification", model="Shunian/yelp_review_rating_reberta_base", top_k=1
    )

    with gr.Blocks() as demo:
        gr.Markdown(
            "<center><h1>Comment rating</h1> A Simple Comment Rating Tool</center>"
        )
        with gr.Tab("Review Rating"):
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    analytics_input = gr.Textbox(
                        label="Text Content",
                        lines=4,
                        max_lines=100,
                        placeholder="Analyzing Rating...",
                    )
                    analytics_button = gr.Button("Analyze")
                text_output = gr.Textbox(
                    label="Result", lines=10, max_lines=100, placeholder="Rates..."
                )
        analytics_button.click(
            analytics_emo,
            api_name="analytics",
            inputs=analytics_input,
            outputs=text_output,
        )
    demo.launch(debug=True, server_name="0.0.0.0", share=True, server_port=5000)
