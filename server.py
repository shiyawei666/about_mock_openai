# server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import time
import json
import asyncio
from enum import Enum

app = FastAPI(title="Optimized Mock OpenAI API", version="2.0.0")


# ============ 数据模型定义（保持不变） ============
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
    return f"chatcmpl-{int(time.time() * 1000)}-{id(time.time())}"


def count_tokens(text: str) -> int:
    """简单模拟token计数"""
    return max(len(text) // 4, 10)  # 至少10个token


# 预定义的响应模板（避免每次都重新生成）
RESPONSE_TEMPLATES = {
    "你好": "你好！我是AI助手，很高兴为你服务。请问有什么可以帮助你的吗？",
    "hello": "Hello! I'm an AI assistant, happy to help you. How can I assist you today?",
    "天气": "抱歉，我无法获取实时天气信息。建议你查看天气应用或网站获取最新天气情况。",
    "笑话": "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 = Dec 25！",
    "翻译": "这是对翻译请求的模拟响应。作为高性能Mock API，我能够快速返回结果。",
}

DEFAULT_RESPONSE = "这是高性能模拟响应。Mock OpenAI API支持高并发流式输出。"


def generate_mock_response_fast(messages: List[Message], model: str) -> str:
    """快速生成模拟响应（微秒级）"""
    if not messages:
        return DEFAULT_RESPONSE

    last_message = messages[-1].content.lower()

    # 快速匹配
    for key, response in RESPONSE_TEMPLATES.items():
        if key in last_message:
            return response

    return DEFAULT_RESPONSE


# ============ 优化后的流式响应生成器 ============
async def generate_stream_response_optimized(request: ChatCompletionRequest,
                                             response_id: str,
                                             created_time: int):
    """优化后的流式响应 - 零延迟首Token"""

    # 🚀 优化1：立即发送第一个chunk（首Token延迟接近0）
    first_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": ""},  # 空content也可以，但客户端会立即收到响应
                "finish_reason": None
            }
        ]
    }
    yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

    # 🚀 优化2：异步生成内容（不阻塞）
    mock_content = generate_mock_response_fast(request.messages, request.model)

    # 🚀 优化3：可选择是否模拟流式输出
    temperature = request.temperature or 0.7  # 默认为0.7
    if temperature > 0.5:  # 用temperature控制是否模拟延迟
        # 模拟流式输出（但速度更快）
        # 将内容分成更小的块（单个字符而不是词）
        chunk_size = max(1, len(mock_content) // 10)  # 分成10块
        for i in range(0, len(mock_content), chunk_size):
            chunk_text = mock_content[i:i + chunk_size]
            chunk_data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            # 🚀 优化4：减少延迟（0.01秒而不是0.1秒）
            await asyncio.sleep(0.01)  # 10ms per chunk，而非100ms
    else:
        # 🚀 优化5：一次性返回所有内容（最快）
        chunk_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": mock_content},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

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

    # 发送 usage 信息（OpenAI API 标准行为）
    prompt_tokens = sum(count_tokens(msg.content) for msg in request.messages)
    completion_tokens = count_tokens(mock_content)
    usage_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": request.model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


# ============ API端点 ============
@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-3-5-turbo",
                "object": "model",
                "created": 1686935002,
                "owned_by": "mock"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """聊天补全接口 - 高性能版本"""
    response_id = generate_response_id()
    created_time = int(time.time())

    # 处理流式请求
    if request.stream:
        return StreamingResponse(
            generate_stream_response_optimized(request, response_id,
                                               created_time),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # 禁用nginx缓冲
            }
        )

    # 处理非流式请求（极快）
    mock_content = generate_mock_response_fast(request.messages, request.model)

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


# ============ 启动配置 ============
if __name__ == "__main__":
    import uvicorn

    # 🚀 关键修改：使用 "main_optimized:app" 字符串格式
    uvicorn.run(
        "server:app",  # 注意：这里是字符串，不是 app 对象
        host="0.0.0.0",
        port=8889,
        workers=4,  # 现在可以正常工作了
        loop="uvloop",
        http="httptools",
        limit_concurrency=1000,
        backlog=2048
    )