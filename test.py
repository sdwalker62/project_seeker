import requests
import fire


def main(prompt: str):
    url = f"http://localhost:8000/chatbot?prompt={prompt}"
    with requests.get(url, stream=True) as r:
        for chunk in r.iter_content(1024, decode_unicode=True):
            print(chunk, end="", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
