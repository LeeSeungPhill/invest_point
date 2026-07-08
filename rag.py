"""
rag.py
======
사업보고서 원문 RAG 엔진 (전부 로컬, API 없음).

  chunk_text()       : 사업의 내용 본문을 출처 메타데이터(chunk_id/소제목/오프셋)와
                       함께 청크로 분할.
  OllamaEmbedder     : 맥 Ollama로 임베딩 (기본 bge-m3, 한국어 강함). API 키 불필요.
  LocalVectorIndex   : numpy 코사인 유사도 기반 경량 벡터 인덱스 + 디스크 캐시.
  Retriever          : 임베더+인덱스 묶음. build(chunks) 후 query(text, k).

인용 추적의 핵심: 각 청크는 안정적인 chunk_id(C001..)와 소제목/오프셋을 보존한다.
LLM이 [C012] 형태로 인용하면, citation.py가 그 id가 실재하는지 검증할 수 있다.
"""
from __future__ import annotations

import os
import re
import json
import hashlib
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import requests

logger = logging.getLogger("rag")

CACHE_DIR = Path(os.getenv("RAG_CACHE_DIR", Path.home() / ".cache" / "dart_mvp" / "rag"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 소제목처럼 보이는 줄: '가.' '나.' '1.' '(1)' 'II.' '제1절' 등
_HEADING = re.compile(r"^\s*(?:[가-힣]\.|[0-9]{1,2}\.|\([0-9]{1,2}\)|[IVX]{1,4}\.|제?\s*[0-9]+\s*[절항])\s*\S")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    heading: str       # 가장 가까운 직전 소제목(추정)
    start: int         # 원문 내 시작 오프셋
    end: int


# ---------------------------------------------------------------------- #
# 1) 청킹
# ---------------------------------------------------------------------- #
def chunk_text(text: str, *, max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    """문단 경계 기준으로 누적 청킹. 한국어 임베딩에 적당한 ~800자/청크."""
    if not text:
        return []

    # 문단 분리(원문 오프셋 추적을 위해 위치 보존)
    paras = []
    pos = 0
    for part in re.split(r"\n{2,}", text):
        start = text.find(part, pos)
        if start < 0:
            start = pos
        paras.append((start, part.strip()))
        pos = start + len(part)

    chunks: list[Chunk] = []
    cur, cur_start, heading = "", None, ""
    chunk_heading = ""   # 현재 만들고 있는 청크가 '시작된 시점'의 소제목
    idx = 1

    def flush(end_pos: int):
        nonlocal cur, cur_start, idx
        if cur.strip():
            chunks.append(Chunk(
                chunk_id=f"C{idx:03d}",
                text=cur.strip(),
                heading=chunk_heading,
                start=cur_start if cur_start is not None else 0,
                end=end_pos,
            ))
            idx += 1
        cur, cur_start = "", None

    for start, p in paras:
        if not p:
            continue
        # 짧고 소제목 패턴이면 heading 갱신
        first_line = p.splitlines()[0] if p else ""
        if len(first_line) <= 40 and _HEADING.match(first_line):
            heading = first_line.strip()
        if cur_start is None:
            cur_start = start
            chunk_heading = heading   # 청크 시작 시점의 소제목 고정
        if len(cur) + len(p) + 1 > max_chars and cur:
            flush(start)
            # overlap: 직전 청크 꼬리를 새 청크 앞에 덧댐
            tail = chunks[-1].text[-overlap:] if overlap and chunks else ""
            cur, cur_start = (tail + "\n" + p), start
            chunk_heading = heading
        else:
            cur = (cur + "\n" + p) if cur else p
    flush(len(text))
    return chunks


# ---------------------------------------------------------------------- #
# 2) 임베딩 (Ollama, 로컬)
# ---------------------------------------------------------------------- #
class OllamaEmbedder:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: int = 120):
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
        self.base = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        r = requests.post(f"{self.base}{path}", json=payload, timeout=self.timeout)
        if r.status_code == 404:
            raise RuntimeError(
                f"Ollama 404 ({path}): 임베딩 모델 '{self.model}'가 없는 듯합니다. "
                f"맥에서 `ollama pull {self.model}` 를 실행하세요. (응답: {r.text[:160]})")
        r.raise_for_status()
        return r.json()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        # 신형 배치 엔드포인트 우선
        try:
            data = self._post("/api/embed", {"model": self.model, "input": texts})
            embs = data.get("embeddings")
            if embs:
                return np.array(embs, dtype=np.float32)
        except RuntimeError:
            raise  # 모델 없음(404): 명확한 안내를 그대로 올림
        except requests.RequestException as e:
            logger.warning("/api/embed 실패, 단건 폴백: %s", e)

        # 구형 단건 엔드포인트 폴백
        out = []
        for t in texts:
            data = self._post("/api/embeddings", {"model": self.model, "prompt": t})
            out.append(data["embedding"])
        return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------- #
# 3) 로컬 벡터 인덱스 (numpy 코사인)
# ---------------------------------------------------------------------- #
def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class LocalVectorIndex:
    def __init__(self):
        self.vectors: Optional[np.ndarray] = None
        self.chunks: list[dict] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = [asdict(c) for c in chunks]
        self.vectors = _normalize(vectors.astype(np.float32))

    def search(self, qvec: np.ndarray, k: int = 4) -> list[dict]:
        if self.vectors is None or not len(self.chunks):
            return []
        q = _normalize(qvec.reshape(1, -1))[0]
        sims = self.vectors @ q
        order = np.argsort(-sims)[:k]
        res = []
        for i in order:
            c = dict(self.chunks[int(i)])
            c["score"] = float(sims[int(i)])
            res.append(c)
        return res

    # 디스크 캐시 (종목+보고서 단위 재사용 → 재실행 속도/안정성)
    def save(self, key: str):
        p = CACHE_DIR / hashlib.sha1(key.encode()).hexdigest()
        np.savez_compressed(p.with_suffix(".npz"), vectors=self.vectors)
        p.with_suffix(".json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")

    def load(self, key: str) -> bool:
        p = CACHE_DIR / hashlib.sha1(key.encode()).hexdigest()
        if not (p.with_suffix(".npz").exists() and p.with_suffix(".json").exists()):
            return False
        self.vectors = np.load(p.with_suffix(".npz"))["vectors"]
        self.chunks = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        return True


# ---------------------------------------------------------------------- #
# 4) Retriever (임베더 + 인덱스)
# ---------------------------------------------------------------------- #
class Retriever:
    def __init__(self, embedder=None):
        self.embedder = embedder or OllamaEmbedder()
        self.index = LocalVectorIndex()

    def build(self, chunks: list[Chunk], cache_key: Optional[str] = None,
              use_cache: bool = True) -> int:
        if cache_key and use_cache and self.index.load(cache_key):
            logger.info("RAG 인덱스 캐시 로드: %s", cache_key)
            return len(self.index.chunks)
        vectors = self.embedder.embed([c.text for c in chunks])
        self.index.add(chunks, vectors)
        if cache_key:
            self.index.save(cache_key)
        return len(chunks)

    def query(self, text: str, k: int = 4) -> list[dict]:
        qv = self.embedder.embed([text])
        if qv.shape[0] == 0:
            return []
        return self.index.search(qv[0], k=k)

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        return next((c for c in self.index.chunks if c["chunk_id"] == chunk_id), None)
