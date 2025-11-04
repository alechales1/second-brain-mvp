import gradio as gr
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os
import json

# Grok for chat queries
client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.getenv("GROK_API_KEY"))

# Qdrant for vector DB
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Local embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')

collection_name = "second_brain_local"  # New name to reset

if not qdrant.has_collection(collection_name):
    qdrant.create_collection(
        collection_name,
        vectors_config=models.VectorParams(
            size=384,  # Matches embedder
            distance=models.Distance.COSINE,
        ),
    )

def embed(text):
    return embedder.encode(text).tolist()

def index_json(file, project="All", tag=""):
    if file is None:
        return "No file uploaded."
    try:
        data = json.load(file)
        # Extract content from typical ChatGPT JSON structure
        chunks = []
        for item in data:
            if isinstance(item, dict) and 'mapping' in item:
                for msg_id, msg in item['mapping'].items():
                    content = msg.get('message', {}).get('content', {}).get('parts', [None])[0]
                    if content:
                        chunks.append(content)
            elif isinstance(item, dict) and 'message' in item:
                content = item['message'].get('content', {}).get('parts', [None])[0]
                if content:
                    chunks.append(content)
        for i, chunk in enumerate(chunks):
            vector = embed(chunk)
            qdrant.upsert(
                collection_name,
                points=[models.PointStruct(
                    id=f"{os.path.basename(file.name)}_{i}",
                    vector=vector,
                    payload={"text": chunk, "project": project, "tag": tag}
                )]
            )
        return f"Indexed {file.name}!"
    except Exception as e:
        return f"Error: {str(e)}"

def ask(q, proj="All", tag=""):
    if not q:
        return "No query."
    try:
        query_vector = embed(q)
        filter_conditions = [cond for cond in [
            models.FieldCondition(key="project", match=models.MatchValue(value=proj)) if proj != "All" else None,
            models.FieldCondition(key="tag", match=models.MatchValue(value=tag)) if tag else None
        ] if cond]
        results = qdrant.search(
            collection_name,
            query_vector=query_vector,
            query_filter=models.Filter(must=filter_conditions) if filter_conditions else None,
            limit=5,
        )
        context = "\n".join([hit.payload['text'] for hit in results])
        if not context:
            return "No relevant context found."
        response = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": f"Answer based on this context: {context}"},
                {"role": "user", "content": q}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("Second Brain MVP")
    with gr.Row():
        with gr.Column():
            upload = gr.File(label="Upload ChatGPT JSON", file_types=[".json"])
            proj = gr.Dropdown(["All"], label="Project")  # Add options as needed
            tag = gr.Textbox(label="Tag")
            index_btn = gr.Button("Index")
            status = gr.Textbox(label="Status")
        with gr.Column():
            q = gr.Textbox(label="Ask")
            ask_btn = gr.Button("Ask")
    output = gr.Textbox(label="Answer")
    index_btn.click(index_json, [upload, proj, tag], status)
    ask_btn.click(ask, [q, proj, tag], output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
