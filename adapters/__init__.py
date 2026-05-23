
"""

Adapters - Framework-specific implementations

"""

from app_refactored.adapters.jina_embedding_adapter import JinaEmbeddingAdapter
from app_refactored.adapters.openai_embedding_adapter import OpenAIEmbeddingAdapter

from app_refactored.adapters.postgres_document_adapter import PostgresDocumentAdapter

from app_refactored.adapters.vllm_adapter import VLLMAdapter
 
__all__ = [

    "JinaEmbeddingAdapter",
    "OpenAIEmbeddingAdapter",

    "PostgresDocumentAdapter",

    "VLLMAdapter"

]
