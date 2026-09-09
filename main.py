import torch
import requests

from ollama import chat 
while True:
    x=input()
    response = chat(
    model='qwen2.5:1.5b',
    messages=[{'role': 'user', 'content': f'{x}'}],
    )
    print(response.message.content)
