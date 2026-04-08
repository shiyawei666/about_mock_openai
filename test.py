# test_client.py
import requests
import json

def test_non_stream():
    """测试非流式请求"""
    url = "http://localhost:8889/v1/chat/completions"
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": "你好，请介绍一下自己"}
        ],
        "stream": False,
        "temperature": 0.7
    }
    
    response = requests.post(url, json=payload)
    print("非流式响应:")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

def test_stream():
    """测试流式请求"""
    url = "http://localhost:8889/v1/chat/completions"
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "给我讲个笑话"}
        ],
        "stream": True,
        "temperature": 0.7
    }
    
    response = requests.post(url, json=payload, stream=True)
    print("流式响应:")
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                data = line[6:]  # 移除 "data: " 前缀
                if data != "[DONE]":
                    chunk = json.loads(data)
                    if chunk['choices'][0]['delta'].get('content'):
                        print(chunk['choices'][0]['delta']['content'], end='', flush=True)
    print("\n")

def test_models():
    """测试模型列表接口"""
    url = "http://localhost:8889/v1/models"
    response = requests.get(url)
    print("可用模型:")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # 先测试健康检查
    health = requests.get("http://localhost:8889/health")
    print(f"健康状态: {health.json()}\n")
    
    # 测试各个接口
    test_models()
    print("\n" + "="*50 + "\n")
    test_non_stream()
    print("\n" + "="*50 + "\n")
    test_stream()
