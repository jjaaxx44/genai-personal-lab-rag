from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Uploads
    max_upload_mb: int = 5
    max_pdf_pages: int = 50

    # LLM providers, tried in this order: Gemini -> Groq -> OpenAI -> local Ollama.
    # A provider joins the chain only when both its key and its model are set.
    gemini_api_key: str = ""
    gemini_llm_model: str = ""
    gemini_llm_fast_model: str = ""

    groq_api_key: str = ""
    groq_llm_model: str = ""
    groq_llm_fast_model: str = ""

    openai_api_key: str = ""
    openai_llm_model: str = ""
    openai_llm_fast_model: str = ""

    # Ollama runs on the host, not in Docker, and serves one model -- no fast variant.
    local_ollama_base_url: str = "http://host.docker.internal:11434"
    local_ollama_llm_model: str = ""

    # Local models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # MongoDB Atlas Local
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "genai_lab"

    # Neo4j AuraDB
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""

    # Langfuse (optional)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Demo-specific caps
    contextual_max_chunks: int = 60
    contextual_chunk_size: int = 800
    contextual_chunk_overlap: int = 100
    contextual_chunks_per_call: int = 8
    # Every enrichment call carries the whole document; beyond this it's truncated.
    contextual_doc_max_chars: int = 60_000
    naive_chunk_size: int = 800
    naive_chunk_overlap: int = 100
    chunking_chunk_size: int = 800
    chunking_chunk_overlap: int = 100
    chunking_sliding_overlap: int = 400
    chunking_semantic_breakpoint_percentile: float = 80.0
    hybrid_chunk_size: int = 800
    hybrid_chunk_overlap: int = 100
    hybrid_rrf_k: int = 60
    rerank_chunk_size: int = 800
    rerank_chunk_overlap: int = 100
    rerank_candidate_k: int = 20
    rerank_top_k: int = 5
    eval_test_set_size: int = 10
    agentic_chunk_size: int = 800
    agentic_chunk_overlap: int = 100
    agentic_top_k: int = 5
    # get_page returns a whole page verbatim; capped so one call can't blow the
    # context window on a page-dense PDF.
    agentic_page_char_cap: int = 4000
    agentic_recursion_limit: int = 12
    crag_chunk_size: int = 800
    crag_chunk_overlap: int = 100
    crag_top_k: int = 5
    crag_web_results: int = 3
    self_rag_chunk_size: int = 800
    self_rag_chunk_overlap: int = 100
    self_rag_top_k: int = 5
    self_rag_max_retries: int = 2
    adaptive_chunk_size: int = 800
    adaptive_chunk_overlap: int = 100
    adaptive_top_k: int = 5
    adaptive_max_subquestions: int = 3
    multi_hop_chunk_size: int = 800
    multi_hop_chunk_overlap: int = 100
    multi_hop_top_k: int = 5
    multi_hop_max_hops: int = 2
    # Pages grouped per node when a PDF has no table of contents to build a tree from.
    vectorless_page_group_size: int = 5
    # Safety cap on how many descend-into-a-child steps navigation can take.
    vectorless_max_depth: int = 6
    # A leaf section's page text is truncated to this many characters before the
    # final answer call, so one huge section can't blow the context window.
    vectorless_max_read_chars: int = 8000
    sql_rag_row_limit: int = 50
    # Total attempts are this plus one: the first try, then this many fix-and-retry passes.
    sql_rag_max_retries: int = 1
    graph_rag_chunk_size: int = 1200
    graph_rag_chunk_overlap: int = 100
    # Extraction costs one LLM call per chunk (LLMGraphTransformer), capped like
    # contextual_max_chunks -- a full document's worth would be far too slow/expensive.
    graph_rag_max_chunks: int = 8
    graph_rag_extract_concurrency: int = 3
    graph_rag_top_k: int = 5
    kag_chunk_size: int = 1200
    kag_chunk_overlap: int = 100
    kag_max_chunks: int = 8
    kag_extract_concurrency: int = 3
    kag_max_steps: int = 4
    kag_row_limit: int = 25
    # Total attempts per step are this plus one: the first try, then this many fix-and-retry passes.
    kag_max_retries: int = 1
    multimodal_chunk_size: int = 800
    multimodal_chunk_overlap: int = 100
    multimodal_top_k: int = 5
    # Embedded images smaller than this on either side are skipped (icons, bullets, rules).
    multimodal_min_image_px: int = 80
    # Captioning costs one LLM call per image; caps a large PDF's ingest like graph_rag_max_chunks does.
    multimodal_max_images: int = 20
    multimodal_caption_concurrency: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
