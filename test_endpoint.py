import openai
from foundry_local_sdk import Configuration, FoundryLocalManager

def main():
    config = Configuration(app_name="local-rag-assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model_alias = "qwen2.5-0.5b"
    model_info = manager.catalog.get_model(model_alias)
    print("Loading model...")
    model_info.load()

    print("Starting web service...")
    manager.start_web_service()
    print("Service URLs:", manager.urls)

    # 1. Test base_url = url + "/v1"
    print("\n--- Testing with /v1 ---")
    client = openai.OpenAI(
        base_url=f"{manager.urls[0]}/v1",
        api_key="local"
    )
    try:
        response = client.chat.completions.create(
            model=model_info.id,
            messages=[{"role": "user", "content": "Hello, write a 3 word greeting."}],
            max_tokens=20
        )
        print("Success! Response:", response.choices[0].message.content)
    except Exception as e:
        print("Failed with /v1:", e)

    # 2. Test base_url = url directly
    print("\n--- Testing directly ---")
    client_direct = openai.OpenAI(
        base_url=manager.urls[0],
        api_key="local"
    )
    try:
        response = client_direct.chat.completions.create(
            model=model_info.id,
            messages=[{"role": "user", "content": "Hello, write a 3 word greeting."}],
            max_tokens=20
        )
        print("Success! Response:", response.choices[0].message.content)
    except Exception as e:
        print("Failed direct:", e)

    manager.stop_web_service()
    print("Web service stopped.")

if __name__ == "__main__":
    main()
