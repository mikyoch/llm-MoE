"""FastAPI server exposing OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

from unified_llm.unified import UnifiedRouterLLM

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("fastapi and pydantic are required to use unified_llm.server") from exc


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 256
    stream: bool = False
    mode: Optional[str] = None
    verbose: bool = False


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 256
    stream: bool = False
    mode: Optional[str] = None
    verbose: bool = False


def _messages_to_prompt(messages: List[Message]) -> str:
    lines = []
    for msg in messages:
        lines.append(f"{msg.role}: {msg.content}")
    lines.append("assistant:")
    return "\n".join(lines)


def _usage(prompt: str, completion: str) -> Dict[str, int]:
    # Approximate token usage with whitespace tokenization.
    p = len(prompt.split())
    c = len(completion.split())
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _sse_chat_chunks(text_iter: Iterator[str], completion_id: str) -> Iterator[str]:
    for chunk in text_iter:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"
    done = {"id": completion_id, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    yield f"data: {json.dumps(done, ensure_ascii=True)}\n\n"
    yield "data: [DONE]\n\n"


def _sse_completion_chunks(text_iter: Iterator[str], completion_id: str) -> Iterator[str]:
    for chunk in text_iter:
        payload = {
            "id": completion_id,
            "object": "text_completion",
            "choices": [{"index": 0, "text": chunk, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"
    done = {"id": completion_id, "object": "text_completion", "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]}
    yield f"data: {json.dumps(done, ensure_ascii=True)}\n\n"
    yield "data: [DONE]\n\n"


def create_app(config_path: str) -> FastAPI:
    app = FastAPI(title="UnifiedRouterLLM", version="0.1.0")
    model = UnifiedRouterLLM.from_config(config_path)

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> Any:
        try:
            prompt = _messages_to_prompt(req.messages)
            gen_kwargs = {
                "temperature": req.temperature,
                "top_p": req.top_p,
                "max_new_tokens": req.max_tokens,
            }
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            if req.stream:
                iterator = model.stream(prompt=prompt, mode=req.mode, **gen_kwargs)
                return StreamingResponse(
                    _sse_chat_chunks(iterator, completion_id),
                    media_type="text/event-stream",
                )

            started = time.time()
            result = model.generate(
                prompt=prompt,
                return_metadata=req.verbose,
                mode=req.mode,
                **gen_kwargs,
            )
            if req.verbose:
                text = result["text"]
                metadata = result["metadata"]
            else:
                text = result
                metadata = None

            body = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model or "unified-router-llm",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": _usage(prompt, text),
                "latency_ms": round((time.time() - started) * 1000.0, 2),
            }
            if metadata is not None:
                body["metadata"] = metadata
            return JSONResponse(body)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/completions")
    def completions(req: CompletionRequest) -> Any:
        try:
            prompt = req.prompt[0] if isinstance(req.prompt, list) else req.prompt
            gen_kwargs = {
                "temperature": req.temperature,
                "top_p": req.top_p,
                "max_new_tokens": req.max_tokens,
            }
            completion_id = f"cmpl-{uuid.uuid4().hex}"
            if req.stream:
                iterator = model.stream(prompt=prompt, mode=req.mode, **gen_kwargs)
                return StreamingResponse(
                    _sse_completion_chunks(iterator, completion_id),
                    media_type="text/event-stream",
                )

            started = time.time()
            result = model.generate(
                prompt=prompt,
                return_metadata=req.verbose,
                mode=req.mode,
                **gen_kwargs,
            )
            if req.verbose:
                text = result["text"]
                metadata = result["metadata"]
            else:
                text = result
                metadata = None

            body = {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model or "unified-router-llm",
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": _usage(prompt, text),
                "latency_ms": round((time.time() - started) * 1000.0, 2),
            }
            if metadata is not None:
                body["metadata"] = metadata
            return JSONResponse(body)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app
