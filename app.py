import os
import json
import time
import traceback
from functools import wraps

import gradio as gr
from openai import OpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# CONFIG & SECRETS
# -------------------------------------------------
APP_USER = os.getenv("APP_USER")
APP_PASS = os.getenv("APP_PASS")
COLLECTION = "second_brain_local"
EMBED_DIM = 384
GROK_MODEL = os.getenv("GROK_MODEL", "grok-beta")

# -------------------------------------------------
# CLIENTS
# -------------------------------------------------
print("App starting - imports loaded.")

client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.getenv("GROK_API_KEY"))
print("Grok client initialized.")

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
print("Qdrant client initialized.")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedder loaded.")

# -------------------------------------------------
# COLLECTION SETUP
# -------------------------------------------------
try:
    if not qdrant.collection_exists(COLLECTION):  # Fixed: collection_exists() for newer clients
        qdrant.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION}'.")
    else:
        info = qdrant.get_collection(COLLECTION)
        if info.config.params.vectors.size != EMBED_DIM:
            raise ValueError(f"Dim mismatch: expected {EMBED_DIM}, got {info.config.params.vectors.size}")
        print(f"Collection '{COLLECTION}' ready.")
except Exception as e:
    print(f"ERROR: Collection setup failed – {e}\n{traceback.format_exc()}")

print("DEBUG: Startup complete.")
print(f"DEBUG: Using collection: {COLLECTION}")

# -------------------------------------------------
# RETRY DECORATOR (max 3 attempts)
# -------------------------------------------------
def retry(max_attempts=3, delay=1):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except UnexpectedResponse as e:
                    last_exc = e
                    if attempt == max_attempts:
                        break
                    print(f"Retry {attempt}/{max_attempts} after Qdrant error: {e}")
                    time.sleep(delay)
            raise last_exc or RuntimeError("Retry failed")
        return wrapper
    return decorator

# -------------------------------------------------
# TEXT EXTRACTION (all ChatGPT formats)
# -------------------------------------------------
def extract_text_chunks(obj):
    chunks = []

    # Newer: {"messages": [{"role":..., "content": {"parts": [...]}}]}
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            c = m.get("content")
            if isinstance(c, dict) and "parts" in c:
                chunks.extend([p for p in c["parts"] if isinstance(p, str)])
            elif isinstance(c, str):
                chunks.append(c)
        return [t.strip() for t in chunks if t.strip()]

    # Older: {"mapping": {id: {"message": {"content": {"parts": [...]}}}}}
    mapping = obj.get("mapping")
    if isinstance(mapping, dict):
        for node in mapping.values():
            msg = (node or {}).get("message", {})
            content = msg.get("content", {})
            parts = content.get("parts") or []
            chunks.extend([p for p in parts if isinstance(p, str)])
        return [t.strip() for t in chunks if t.strip()]

    # Fallback: list of dicts with "content"
    if isinstance(obj, list):
        for m in obj:
            c = m.get("content")
            if isinstance(c, str):
                chunks.append(c)
    return [t.strip() for t in chunks if t.strip()]

# -------------------------------------------------
# INDEX FUNCTION
# -------------------------------------------------
@retry()
def index_json(file, project="All", tag=""):
    if file is None:
        return "No file uploaded."

    print(f"DEBUG: index_json – project: {project}, tag: {tag}, file: {file.name}")

    # Load JSON
    try:
        data = json.load(file)
        print("DEBUG: JSON loaded.")
    except Exception as e:
        print(f"ERROR: JSON load failed – {e}")
        return f"Error: Invalid JSON – {str(e)}"

    # Extract text
    try:
        chunks = extract_text_chunks(data)
        print(f"DEBUG: Extracted {len(chunks)} chunk(s).")
        if not chunks:
            return "Warning: No text found in JSON."
    except Exception as e:
        print(f"ERROR: Extraction failed – {e}")
        return f"Error: Extraction failed – {str(e)}"

    # Embed
    try:
        embeddings = embedder.encode(chunks, convert_to_tensor=False)
        print(f"DEBUG: Generated {len(embeddings)} embedding(s).")
    except Exception as e:
        print(f"ERROR: Embedding failed – {e}")
        return f"Error: Embedding failed – {str(e)}"

    # Batch upsert
    try:
        points = [
            models.PointStruct(
                id=f"{project}_{tag}_{os.path.basename(file.name)}_{i}",
                vector=emb.tolist(),
                payload={"text": chunk, "project": project, "tag": tag},
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        qdrant.upsert(collection_name=COLLECTION, points=points)
        print(f"DEBUG: Upserted {len(points)} point(s) to Qdrant.")
        return f"Success: Indexed {len(points)} chunk(s)."
    except Exception as e:
        print(f"ERROR: Upsert failed – {e}")
        return f"Error: Upsert failed – {str(e)}"

# -------------------------------------------------
# ASK FUNCTION
# -------------------------------------------------
@retry()
def ask(q, proj="All", tag=""):
    if not q.strip():
        return "No query entered."

    try:
        query_vector = embedder.encode(q).tolist()

        must_filters = []
        if proj != "All":
            must_filters.append(models.FieldCondition(key="project", match=models.MatchValue(value=proj)))
        if tag:
            must_filters.append(models.FieldCondition(key="tag", match=models.MatchValue(value=tag)))

        results = qdrant.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            query_filter=models.Filter(must=must_filters) if must_filters else None,
            limit=5,
        )

        context = "\n".join([hit.payload.get("text", "") for hit in results if hit.payload])
        if not context.strip():
            return "No relevant context found."

        response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "system", "content": f"Answer using only this context:\n{context}"},
                {"role": "user", "content": q},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"ERROR: Ask failed – {e}")
        return f"Error: Ask failed – {str(e)}"

# -------------------------------------------------
# GRADIO UI
# -------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Second Brain MVP")
    with gr.Row():
        with gr.Column():
            upload = gr.File(label="Upload ChatGPT JSON", file_types=[".json"])
            proj = gr.Dropdown(
                ["All", "BYLD", "SUNRUN", "Church", "Pond", "Fish Farm", "D&D"],
                label="Project"
            )
            tag = gr.Textbox(label="Tag")
            index_btn = gr.Button("Index")
            status = gr.Textbox(label="Status", interactive=False)
        with gr.Column():
            q = gr.Textbox(label="Ask")
            ask_btn = gr.Button("Ask")
            output = gr.Textbox(label="Answer", interactive=False)

    index_btn.click(index_json, inputs=[upload, proj, tag], outputs=status)
    ask_btn.click(ask, inputs=[q, proj, tag], outputs=output)

# -------------------------------------------------
# LAUNCH (simple, no FastAPI mount to avoid error)
# -------------------------------------------------
if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", "7860"))
        print(f"Launching Gradio on 0.0.0.0:{port}")

        # Optional basic auth
        auth_fn = None
        if APP_USER and APP_PASS:
            def _auth(username, password):
                return username == APP_USER and password == APP_PASS
            auth_fn = _auth

        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            debug=True,
            auth=auth_fn
        )
    except Exception as e:
        print(f"ERROR launching app: {e}\n{traceback.format_exc()}")
