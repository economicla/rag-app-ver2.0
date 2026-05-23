"""
Open WebUI Pipe — Kapsamlı RAG + Native Citation UI

Kullanım:
  1. Open WebUI → Workspace → Functions → bu dosyayı yapıştır
  2. Valves: RAG_API_URL ve SCOPED_FILENAMES ayarla
  3. SCOPED_FILENAMES = virgülle ayrılmış dosya adları (ingest ile birebir aynı isimler)

Özellikler:
  - Backend RAG API'ye scoped sorgu gönderir
  - Cevaptaki KAYNAKLAR metnini siler, yerine Open WebUI native citation UI kullanır
  - Popup'ta sayfa, benzerlik skoru ve içerik önizlemesi gösterilir

Backend:
  POST {RAG_API_URL}/api/v2/query/scoped
"""

from typing import Optional, Any, Union, Generator, Iterator, Callable, Awaitable, List
from pydantic import BaseModel, Field
import aiohttp
import asyncio
import re
import logging


class Pipe:
    class Valves(BaseModel):
        RAG_API_URL: str = Field(
            default="http://10.144.100.204:8005",
            description="RAG API kök URL (örn. http://10.x.x.x:8005)",
        )
        SCOPED_FILENAMES: str = Field(
            default="",
            description="Virgülle ayrılmış dosya adları (opsiyonel — unit/collection yeterliyse boş bırak)",
        )
        UNIT: str = Field(
            default="",
            description="Birim/departman adı (ör. 'krediler', 'egitim') — bu pipe sadece bu birimde arar",
        )
        COLLECTION: str = Field(
            default="",
            description="Birim içindeki koleksiyon/firma/doküman grubu (ör. 'karesi', 'prosedur')",
        )
        SYSTEM_PROMPT: str = Field(
            default="",
            description="Doküman tipine/RAG botuna özel ek talimatlar. Backend genel promptunu ezmez; en alta eklenir.",
        )
        TEMPERATURE: float = Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="LLM yanıt yaratıcılığı (0=kesin, 1=yaratıcı). Ek talimat varsa geçerlidir.",
        )
        TASK_MODEL: str = Field(
            default="",
            description="Başlık/tag üretimi için model ID (boşsa pipe kendisi cevaplar)",
        )
        STREAM_DELAY_SECONDS: float = Field(
            default=0.01,
            ge=0.0,
            le=0.2,
            description="Cevap parçaları arasındaki bekleme süresi. 0 daha hızlı, 0.01 typewriter etkisi verir.",
        )
        STREAM_CHUNK_WORDS: int = Field(
            default=6,
            ge=1,
            le=30,
            description="Open WebUI'ya tek seferde gönderilecek kelime sayısı.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Scoped RAG (fixed files)"
        self.log = logging.getLogger("scoped_rag_pipe")

    def _strip_kaynaklar(self, text: str) -> str:
        """LLM cevabındaki ayrı kaynak bölümünü temizler, metin içindeki 'kaynaklarda' gibi kelimelere dokunmaz."""
        pattern = r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(KAYNAKLAR|SOURCES|المصادر)\s*:?\s*(?:\*\*)?\s*$"
        match = re.search(pattern, text)
        if not match:
            return text.rstrip()
        return text[: match.start()].rstrip()

    def _build_citation(self, source: dict) -> dict:
        """Backend source nesnesinden Open WebUI citation event'i oluşturur."""
        filename = source.get("filename", "")
        unit = source.get("unit")
        content = source.get("content", "")
        score = source.get("similarity_score", 0)
        pages = source.get("source_pages")

        meta = {"source": filename}
        if unit:
            meta["unit"] = unit
        if pages:
            meta["page"] = pages

        label = filename
        if pages:
            label += f" (sayfa {pages})"

        return {
            "type": "citation",
            "data": {
                "document": [content],
                "metadata": [meta],
                "source": {"name": label},
                "distances": [round(1 - score, 4)],
            },
        }

    def _answer_chunks(self, text: str):
        """Cevabı Open WebUI'ya daha stabil yazdırmak için küçük parçalara böler."""
        words = re.findall(r"\S+\s*", text)
        if not words:
            return

        chunk_size = max(1, int(self.valves.STREAM_CHUNK_WORDS))
        for idx in range(0, len(words), chunk_size):
            yield "".join(words[idx : idx + chunk_size])

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __task__: Optional[str] = None,
        **kwargs,
    ) -> Union[str, Generator, Iterator]:

        if __task__ is not None:
            return

        messages = body.get("messages", [])
        if not messages:
            yield "Mesaj yok."
            return
        user_input = (messages[-1].get("content") or "").strip()
        if not user_input:
            yield "Lütfen soru yazın."
            return

        raw = self.valves.SCOPED_FILENAMES.strip()
        filenames = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
        unit = self.valves.UNIT.strip() or None
        collection = self.valves.COLLECTION.strip() or None

        if not filenames and not collection and not unit:
            yield "❌ Valves'te UNIT, COLLECTION veya SCOPED_FILENAMES ayarlanmalı."
            return

        url = f"{self.valves.RAG_API_URL.rstrip('/')}/api/v2/query/scoped"
        custom_prompt = self.valves.SYSTEM_PROMPT.strip()
        payload = {
            "query": user_input,
            "top_k": 10,
            "temperature": self.valves.TEMPERATURE if custom_prompt else 0.0,
        }
        if filenames:
            payload["filenames"] = filenames
        if unit:
            payload["unit"] = unit
        if collection:
            payload["collection"] = collection
        if custom_prompt:
            payload["system_prompt"] = custom_prompt

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        try:
                            err = await resp.json()
                            detail = err.get("detail", str(err))
                        except Exception:
                            detail = await resp.text()
                        yield f"❌ API {resp.status}: {detail}"
                        return

                    data = await resp.json()
        except Exception as e:
            yield f"❌ Bağlantı hatası: {e}"
            return

        sources = data.get("sources", [])
        if __event_emitter__:
            for src in sources:
                await __event_emitter__(self._build_citation(src))

        answer = data.get("answer", "")
        clean_answer = self._strip_kaynaklar(answer)

        delay = max(0.0, float(self.valves.STREAM_DELAY_SECONDS))
        for chunk in self._answer_chunks(clean_answer):
            yield chunk
            if delay:
                await asyncio.sleep(delay)
