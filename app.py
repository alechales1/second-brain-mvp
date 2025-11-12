import os
import json
import time
import traceback
import uuid
from functools import wraps
import numpy as np

import gradio as gr
from openai import OpenAI
from httpx import Client as HttpxClient
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

# CONFIG & SECRETS
APP_USER = os.getenv("APP_USER")
APP_PASS = os.getenv("APP_PASS")
COLLECTION = "second_brain_local"
EMBED_DIM = 384
GROK_MODEL = os.getenv("GROK_MODEL", "grok-beta")

# CLIENTS
print("App starting - imports loaded.")

client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=os.getenv("GROK_API_KEY"),
    http_client=HttpxClient(trust_env=False)  # Ignores env proxies reliably
)
print("Grok client initialized (env proxies ignored).")

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
print("Qdrant client initialized.")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedder loaded.")

# COLLECTION SETUP
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
    raise  # Stop app launch if collection broken

# RETRY DECORATOR
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

# TEXT EXTRACTION
def extract_text_chunks(obj):
    chunks = []
    
    # Format 1: messages with content.parts (YOUR TEST FILE USES THIS)
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

# INDEXING
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
        
        # Filter out NaN values
        clean_embeddings = []
        for i, emb in enumerate(embeddings):
            if any(np.isnan(emb)):
                print(f"WARNING: Chunk {i} has NaN embedding, replacing with zeros")
                clean_embeddings.append([0.0] * EMBED_DIM)
            else:
                clean_embeddings.append(emb)
        
        embeddings = clean_embeddings
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

# QUERYING
@retry()
def ask_grok(query, project="All", tag=""):
    if not query:
        return "❌ No query provided."
    
    print(f"DEBUG: ask_grok — query='{query}', project={project}, tag={tag}")
    
    try:
        query_emb = embedder.encode(query).tolist()
        
        filter_conditions = []
        if project != "All":
            filter_conditions.append(models.FieldCondition(key="project", match=models.MatchValue(value=project)))
        if tag:
            filter_conditions.append(models.FieldCondition(key="tag", match=models.MatchValue(value=tag)))
        
        results = qdrant.search(
            collection_name=COLLECTION,
            query_vector=query_emb,
            query_filter=models.Filter(must=filter_conditions) if filter_conditions else None,
            limit=5,
        )
        
        context = "\n\n".join([r.payload.get("text", "") for r in results])
        print(f"DEBUG: Retrieved {len(results)} chunks for context")
        
        if not context:
            return "⚠️ No relevant context found."
        
        response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use ONLY the provided context to answer the query accurately and concisely."},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"ERROR: Query failed — {e}\n{traceback.format_exc()}")
        return f"❌ Query failed: {str(e)}"

# GRADIO INTERFACE
with gr.Blocks(title="Second Brain MVP") as demo:
    gr.Markdown("# Second Brain MVP\nPrivate knowledge base with Grok-powered queries.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Index JSON")
            file_input = gr.File(label="Upload ChatGPT JSON Export", type="filepath")
            project_input = gr.Dropdown(choices=["All", "Pond", "BYLD", "Church", "Other"], value="All", label="Project")
            tag_input = gr.Textbox(label="Tag (optional)")
            index_button = gr.Button("Index File")
            status_output = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column(scale=2):
            gr.Markdown("### Query Knowledge Base")
            query_input = gr.Textbox(label="Your Question")
            project_filter = gr.Dropdown(choices=["All", "Pond", "BYLD", "Church", "Other"], value="All", label="Filter by Project")
            tag_filter = gr.Textbox(label="Filter by Tag (optional)")
            ask_button = gr.Button("Ask Grok")
            answer_output = gr.Textbox(label="Answer", interactive=False)
    
    index_button.click(index_json, inputs=[file_input, project_input, tag_input], outputs=status_output)
    ask_button.click(ask_grok, inputs=[query_input, project_filter, tag_filter], outputs=answer_output)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, debug=True)
