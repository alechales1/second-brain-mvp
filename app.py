import gradio as gr
from openai import OpenAI
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import json
from datetime import datetime
import hashlib

# Setup
client = QdrantClient(
    url=os.getenv("QDRANT_URL", "https://free.qdrant.io"),
    api_key=os.getenv("QDRANT_API_KEY")
)
openai_client = OpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.x.ai/v1"
)
COLLECTION = "second_brain"

# Create collection if not exists
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

def embed(text):
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def add_doc(doc: dict, source: str = "manual"):
    doc.setdefault("project", "general")
    doc.setdefault("tags", [])
    doc.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
    doc["source"] = source
    
    text = json.dumps(doc, indent=2)
    doc_id = hashlib.md5(text.encode()).hexdigest()
    
    client.upsert(
        collection_name=COLLECTION,
        points=[{
            "id": doc_id,
            "vector": embed(text),
            "payload": {"text": text, **doc}
        }]
    )
    return doc_id

def search(query: str, project: str = None, tag: str = None, k=5):
    vector = embed(query)
    filters = []
    if project and project != "All":
        filters.append({"key": "project", "match": {"value": project}})
    if tag:
        filters.append({"key": "tags", "match": {"value": tag}})
    
    results = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=k,
        query_filter={"must": filters} if filters else None
    )
    return [r.payload for r in results]

def ask(q, proj, tag):
    results = search(q, proj, tag)
    context = "\n\n".join([f"From {r['source']}:\n{r['text'][:800]}" for r in results])
    prompt = f"Context:\n{context}\n\nQ: {q}\nA: Clear, concise."
    
    resp = openai_client.chat.completions.create(
        model="grok-beta",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

def upload_file(file):
    if file is None:
        return "No file uploaded."
    if file.name.endswith('.json'):
        doc = json.load(open(file.name))
        add_doc(doc, file.name)
        return f"Indexed {file.name}!"
    else:
        return "Only JSON files supported for now."

with gr.Blocks() as demo:
    gr.Markdown("# Second Brain MVP")
    with gr.Row():
        with gr.Column():
            file_in = gr.File(label="Upload JSON")
            file_out = gr.Textbox(label="Status")
            file_in.change(upload_file, file_in, file_out)
        with gr.Column():
            q = gr.Textbox(label="Ask")
            proj = gr.Dropdown(["All", "general", "Work"], label="Project")
            tag = gr.Textbox(label="Tag")
            ask_btn = gr.Button("Ask")
    output = gr.Textbox(label="Answer")
    ask_btn.click(ask, [q, proj, tag], output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
