"""
llm_backend.py
==============
analyze 노드가 쓸 챗 모델을 '환경변수로' 골라 반환하는 팩토리.
그래프 코드는 이 함수만 호출하므로, 백엔드를 바꿔도 graph는 손대지 않는다.

선택: 환경변수 LLM_BACKEND
  - "ollama"        (기본) 로컬 Ollama. 키/비용 없음. 예: OLLAMA_MODEL=qwen2.5:7b
  - "openai_compat" OpenAI 호환 엔드포인트(vLLM/LM Studio/HyperCLOVA X 등)
  - "google"        Google Gemini

각 백엔드는 LangChain BaseChatModel을 반환하므로 .invoke([...])가 동일하게 동작.
"""

from __future__ import annotations

import os


def get_chat_model(temperature: float = 0.2, max_tokens: int = 2000):
    backend = os.getenv("LLM_BACKEND", "ollama").lower()

    if backend == "ollama":
        # pip install langchain-ollama ; ollama pull qwen3:8b
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            num_predict=max_tokens,   # Ollama의 max_tokens 대응 파라미터
            # 핵심: 기본 num_ctx가 작아 긴 '사업의 내용'이 조용히 잘린다.
            # 16GB+8B 기준 16K가 안전. 본문이 더 길면 32768까지(메모리 여유 확인).
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")),
        )

    if backend == "openai_compat":
        # vLLM / LM Studio / llama.cpp server / 네이버 HyperCLOVA X 등
        # pip install langchain-openai
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "local-model"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "not-needed"),  # 로컬 서버는 더미키 허용
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if backend == "google":
        # pip install langchain-google-genai ; GOOGLE_API_KEY 필요
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    raise ValueError(f"알 수 없는 LLM_BACKEND: {backend}")
