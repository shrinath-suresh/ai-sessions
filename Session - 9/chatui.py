import gradio as gr
import random
import time
import json
import requests


def random_response(message, history):

    url = "http://localhost:5000/answer"

    headers = {
    'Content-Type': 'application/json'
    }

    payload = {"query" : message}
    response = requests.post(url, json=payload).json()
    print("UI Response: ", response)
    return response


gr.ChatInterface(
    random_response,
    chatbot=gr.Chatbot(height=450),
    textbox=gr.Textbox(placeholder="Ask me a question", container=False, scale=7),
    title="Movie Chatbot",
    description="Ask me a question",
    theme="soft",
    examples=["List the last 3 Vijay movies?",
               "Who directed the second movie?",
                ],
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
    css="#textbox_id textbox {color: red}"
).launch()
