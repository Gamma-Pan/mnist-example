import gradio as gr


demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()


