from langchain_mixtral import Mixtral8x7b


if __name__ == "__main__":
    model = Mixtral8x7b()
    for chunk in model.stream("Hey there!"):
        print(chunk.content, end="", flush=True)
