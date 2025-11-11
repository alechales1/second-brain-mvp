import os, json, time, traceback, uuid
from functools import wraps

import gradio as gr
from openai import OpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

# ── Config
APP_USER   = os.getenv("APP_USER")
APP_PASS   = os.getenv("APP_PASS")
COLLECTION = os.getenv("QDRANT_COLLECTION", "second_brain_local")
EMBED_DIM  = 384
GROK_MODEL = os.getenv("GROK_MODEL", "grok-beta")

print("App starting - imports loaded.")

# ── Clients
client  = OpenAI(base_url="https://api.x.ai/v1", api_key=os.getenv("GROK_API_KEY"))
print("Grok client initialized.")
qdrant  = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
print("Qdrant client initialized.")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedder loaded.")

# ── Collection setup
try:
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION}'.")
    else:
        # Verify collection vector size matches
        info = qdrant.get_collection(COLLECTION)
        actual_size = info.config.params.vectors.size
        if actual_size != EMBED_DIM:
            raise ValueError(f"Collection '{COLLECTION}' has vector size {actual_size}, expected {EMBED_DIM}. Delete collection or use different name.")
        print(f"Collection '{COLLECTION}' ready (vector size {actual_size}).")
except Exception as e:
    print(f"FATAL: Collection setup failed — {e}\n{traceback.format_exc()}")
    raise

print("DEBUG: Startup complete.")
print(f"DEBUG: Using collection: {COLLECTION}")

# ── Retry decorator (for transient Qdrant errors)
def retry(max_attempts=3, delay=1.0):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except UnexpectedResponse as e:
                    last = e
                    if attempt == max_attempts:
                        break
                    print(f"Retry {attempt}/{max_attempts} after Qdrant error: {e}")
                    time.sleep(delay)
            raise last or RuntimeError("Retry failed")
        return wrapper
    return deco

# ── ChatGPT export extractors
def extract_text_chunks(obj):
    chunks = []
    
    # Format 1: messages with content.parts
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        print(f"DEBUG: Found {len(msgs)} messages in 'messages' key")
        for i, m in enumerate(msgs):
            c = m.get("content")
            if isinstance(c, dict) and "parts" in c:
                parts = [p for p in c["parts"] if isinstance(p, str)]
                print(f"DEBUG: Message {i} has {len(parts)} text parts")
                chunks.extend(parts)
            elif isinstance(c, str):
                chunks.append(c)
        result = [t.strip() for t in chunks if t and t.strip()]
        print(f"DEBUG: Format 1 extracted {len(result)} chunks")
        return result
    
    # Format 2: mapping structure (ChatGPT conversations.json)
    mapping = obj.get("mapping")
    if isinstance(mapping, dict):
        print(f"DEBUG: Found 'mapping' key with {len(mapping)} nodes")
        for node in mapping.values():
            msg = (node or {}).get("message", {})
            content = msg.get("content", {})
            parts = content.get("parts") or []
            chunks.extend([p for p in parts if isinstance(p, str)])
        result = [t.strip() for t in chunks if t and t.strip()]
        print(f"DEBUG: Format 2 extracted {len(result)} chunks")
        return result
    
    # Format 3: simple list of messages
    if isinstance(obj, list):
        print(f"DEBUG: Found list with {len(obj)} items")
        for m in obj:
            c = m.get("content")
            if isinstance(c, str):
                chunks.append(c)
    
    result = [t.strip() for t in chunks if t and t.strip()]
    print(f"DEBUG: Final extraction: {len(result)} chunks")
    return result

# ── Index
@retry()
def index_json(file_path, project="All", tag=""):
    if not file_path:
        return "❌ No file uploaded."
    
    print(f"DEBUG: index_json — project={project}, tag={tag}, file={file_path}")
    
    # 1. Load JSON with BOM safety
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
        
        # Strip BOM if present
        if raw.startswith(b'\xef\xbb\xbf'):
            raw = raw[3:]
            print("DEBUG: Stripped UTF-8 BOM from file")
        
        text = raw.decode("utf-8")
        data = json.loads(text)
        print(f"DEBUG: JSON loaded successfully ({len(text)} chars)")
    except FileNotFoundError:
        return f"❌ File not found: {file_path}"
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parse failed at line {e.lineno}, col {e.colno}")
        return f"❌ Invalid JSON: {e.msg} (line {e.lineno})"
    except Exception as e:
        print(f"ERROR: File read failed — {e}")
        return f"❌ File read error: {str(e)}"
    
    # 2. Extract chunks
    try:
        chunks = extract_text_chunks(data)
        print(f"DEBUG: Extracted {len(chunks)} chunk(s)")
        if not chunks:
            return "⚠️ No text found in JSON. Check file format."
    except Exception as e:
        print(f"ERROR: Extraction failed — {e}\n{traceback.format_exc()}")
        return f"❌ Extraction failed: {str(e)}"
    
    # 3. Generate embeddings
    try:
        embeddings = embedder.encode(chunks, convert_to_tensor=False, show_progress_bar=False)
        embeddings = [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]
        print(f"DEBUG: Generated {len(embeddings)} embeddings (dim {len(embeddings[0])})")
    except Exception as e:
        print(f"ERROR: Embedding failed — {e}\n{traceback.format_exc()}")
        return f"❌ Embedding failed: {str(e)}"
    
    # 4. Upsert to Qdrant
    try:
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "text": chunk[:1000],  # Truncate long chunks to save space
                    "project": project,
                    "tag": tag or "untagged",
                    "file": os.path.basename(file_path),
                    "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        
        qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)
        print(f"DEBUG: Upserted {len(points)} points to Qdrant")
        
        return f"✅ Success: Indexed {len(points)} chunk(s) to project '{project}'"
    except Exception as e:
        print(f"ERROR: Upsert failed — {e}\n{traceback.format_exc()}")
        return f"❌ Upsert failed: {str(e)}"

# ── Ask
@retry()
def ask(q, proj="All", tag=""):
    if not q or not q.strip():
        return "No query entered."

    try:
        query_vector = embedder.encode(q).tolist()

        must = []
        if proj != "All":
            must.append(models.FieldCondition(key="project", match=models.MatchValue(value=proj)))
        if tag:
            must.append(models.FieldCondition(key="tag", match=models.MatchValue(value=tag)))

        results = qdrant.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            query_filter=models.Filter(must=must) if must else None,
            limit=5,
        )

        context = "\n".join([hit.payload.get("text", "") for hit in results if hit.payload])
        if not context.strip():
            return "No relevant context found."

        resp = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "system", "content": f"Answer using only this context:\n{context}"},
                {"role": "user", "content": q},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"ERROR: Ask failed — {e}")
        return f"Error: Ask failed — {str(e)}"

# ── UI
with gr.Blocks() as demo:
    gr.Markdown("## Second Brain MVP")
    with gr.Row():
        with gr.Column():
            upload = gr.File(label="Upload ChatGPT JSON", file_types=[".json"], type="filepath")
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

# ── Launch
if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", "7860"))
        print(f"Launching Gradio on 0.0.0.0:{port}")

        auth_fn = None
        if APP_USER and APP_PASS:
            def _auth(u, p): return (u == APP_USER and p == APP_PASS)
            auth_fn = _auth

        demo.launch(server_name="0.0.0.0", server_port=port, debug=True, auth=auth_fn)
    except Exception as e:
        print(f"ERROR launching app: {e}\n{traceback.format_exc()}")
