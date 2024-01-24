import requests

url = "http://localhost:8000/chatbot?prompt=What kind of AI are you?"

with requests.get(url, stream=True) as r:
    for chunk in r.iter_content(1024, decode_unicode=True):
        print(chunk, end="", flush=True)
