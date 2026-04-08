# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import time
import json
import asyncio
from enum import Enum

app = FastAPI(title="Mock OpenAI API", version="1.0.0")

# ============ 数据模型定义 ============
class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"

class Message(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

# ============ 辅助函数 ============
def generate_response_id() -> str:
    """生成响应ID"""
    return f"chatcmpl-{int(time.time())}{hash(time.time())}"

def count_tokens(text: str) -> int:
    """简单模拟token计数（实际应用应该使用tiktoken）"""
    return len(text) // 4  # 粗略估计

def generate_mock_response(messages: List[Message], model: str) -> str:
    """生成模拟响应内容"""
    last_message = messages[-1].content if messages else ""
    
    # 简单的响应生成逻辑
    if "你好" in last_message or "hello" in last_message.lower():
        return "你好！我是AI助手，很高兴为你服务。请问有什么可以帮助你的吗？"
    elif "天气" in last_message:
        return "抱歉，我无法获取实时天气信息。建议你查看天气应用或网站获取最新天气情况。"
    elif "笑话" in last_message:
        return "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 = Dec 25！"
    elif "翻译" in last_message:
        return f"收到翻译请求：'{last_message}' 的相关内容。作为演示API，我会返回模拟的翻译结果。"
    else:
        return f"这是对你消息 '{last_message}' 的模拟响应。我是Mock OpenAI API，支持流式和非流式输出。"

# ============ 流式响应生成器 ============
async def generate_stream_response(request: ChatCompletionRequest, response_id: str, created_time: int):
    """生成流式响应"""
    mock_content = generate_mock_response(request.messages, request.model)
    
    # 模拟逐词输出
    words = mock_content.split()
    for i, word in enumerate(words):
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": word + (" " if i < len(words) - 1 else "")
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.1)  # 模拟延迟
    
    # 发送结束标志
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

# ============ API端点 ============
@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1686935002,
                "owned_by": "openai"
            },
            {
                "id": "gpt-4",
                "object": "model",
                "created": 1687882411,
                "owned_by": "openai"
            },
            {
                "id": "mock-model",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """聊天补全接口，支持流式和非流式"""
    response_id = generate_response_id()
    created_time = int(time.time())
    
    # 处理流式请求
    if request.stream:
        return StreamingResponse(
            generate_stream_response(request, response_id, created_time),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    # 处理非流式请求
    mock_content = generate_mock_response(request.messages, request.model)
    
    # 计算token使用量（粗略估计）
    prompt_tokens = sum(count_tokens(msg.content) for msg in request.messages)
    completion_tokens = count_tokens(mock_content)
    
    response = ChatCompletionResponse(
        id=response_id,
        created=created_time,
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(role=Role.assistant, content=mock_content),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )
    
    return response

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "timestamp": int(time.time())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)
