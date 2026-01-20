import os
import torch
import clip
import lancedb
import numpy as np
import pyarrow as pa
import gradio as gr
import imageio
import pandas as pd

from PIL import Image
from gtts import gTTS
from diffusers import StableDiffusionPipeline

# ==========================================
# ⚙ SETUP
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

clip_model, preprocess = clip.load("ViT-B/32", device=device)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to(device)

# ==========================================
# 🧱 LANCEDB SETUP
# ==========================================
embedding_dim = clip_model.visual.output_dim

schema = pa.schema([
    pa.field("prompt", pa.string()),
    pa.field("image_path", pa.string()),
    pa.field("audio_path", pa.string()),
    pa.field("video_path", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), list_size=embedding_dim))
])

os.makedirs("generated_images", exist_ok=True)
os.makedirs("generated_audio", exist_ok=True)
os.makedirs("generated_videos", exist_ok=True)

db = lancedb.connect("./lancedb")

if "multimodal" in db.table_names():
    table = db.open_table("multimodal")
else:
    table = db.create_table("multimodal", schema=schema, mode="create")

# ==========================================
# 🔧 HELPER FUNCTIONS
# ==========================================
def compute_embedding(prompt):
    with torch.no_grad():
        tokens = clip.tokenize([prompt]).to(device)
        features = clip_model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32).flatten().tolist()

def generate_image(prompt):
    img = pipe(prompt).images[0]
    path = f"generated_images/{prompt.replace(' ', '_')}.png"
    img.save(path)
    return path

def generate_audio(prompt):
    path = f"generated_audio/{prompt.replace(' ', '_')}.mp3"
    gTTS(prompt).save(path)
    return path

def generate_video(prompt, num_frames=6):
    frames = []
    for i in range(num_frames):
        frame = pipe(f"{prompt}, frame {i}").images[0]
        temp_path = f"generated_videos/tmp_{i}.png"
        frame.save(temp_path)
        frames.append(imageio.imread(temp_path))

    video_path = f"generated_videos/{prompt.replace(' ', '_')}.mp4"
    imageio.mimsave(video_path, frames, fps=2)
    return video_path

def store_in_db(prompt, img, aud, vid):
    emb = compute_embedding(prompt)
    table.add([{
        "prompt": prompt,
        "image_path": img or "",
        "audio_path": aud or "",
        "video_path": vid or "",
        "embedding": emb
    }])

# ==========================================
# 🔍 SEARCH HANDLER
# ==========================================
def search_handler(query, mode):

    df = table.to_pandas()
    if df.empty:
        return [], None, None, "❌ No data found in database"

    query_emb = compute_embedding(query)

    results = (
        table.search(query_emb, vector_column_name="embedding")
             .limit(2)
             .to_pandas()
    )

    if results.empty:
        return [], None, None, "❌ No similar match found"

    if mode == "Image":
        results = results[results["image_path"] != ""]
    elif mode == "Audio":
        results = results[results["audio_path"] != ""]
    elif mode == "Video":
        results = results[results["video_path"] != ""]

    if results.empty:
        return [], None, None, "❌ No matching media type found"

    if mode == "Image":
        images = [(r["image_path"], r["prompt"]) for _, r in results.iterrows()]
        return images, None, None, "✅ Images Found"

    elif mode == "Audio":
        audio = results.iloc[0]["audio_path"]
        return [], audio, None, "✅ Audio Found"

    elif mode == "Video":
        video = results.iloc[0]["video_path"]
        return [], None, video, "✅ Video Found"

    else:  # All
        images = [(r["image_path"], r["prompt"]) for _, r in results.iterrows() if r["image_path"]]
        audio = results.iloc[0]["audio_path"] if results.iloc[0]["audio_path"] else None
        video = results.iloc[0]["video_path"] if results.iloc[0]["video_path"] else None
        return images, audio, video, "✅ All Results Found"

# ==========================================
# 📂 VIEW DB
# ==========================================
def view_lancedb():
    df = table.to_pandas()
    if df.empty:
        return pd.DataFrame({"message": ["No records found"]})
    return df[["prompt", "image_path", "audio_path", "video_path"]]

# ==========================================
# 🎨 CUSTOM CSS FOR BEAUTIFUL UI
# ==========================================
custom_css = """
body {
    background: linear-gradient(135deg, #6a82fb 0%, #a683ff 100%);
    background-attachment: fixed;
}

.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    border-radius: 20px;
}

.section-card {
    padding: 25px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 25px rgba(0,0,0,0.12);
    margin-bottom: 25px;
}

#title-text {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    padding: 10px;
    color: #FFF;
}

#subtitle-text {
    text-align: center;
    font-size: 18px;
    opacity: 0.8;
    margin-bottom: 20px;
    color: #FFF;
}

.gr-button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    font-weight: bold;
    height: 48px;
    border-radius: 14px !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.1);
}

.gr-radio label {
    font-size: 16px;
}

.gr-textbox input {
    border-radius: 10px !important;
}
"""


# ==========================================
# 🎛 GRADIO UI
# ==========================================
with gr.Blocks(css=custom_css, title="Multimodal Search") as demo:

    gr.Markdown("<h1 id='title-text'>✨ Multimodal Search Using OpenAI CLIP and LanceDB ✨</h1>")
    gr.Markdown("<p id='subtitle-text'>Image • Audio • Video Generation & Search</p>")

    # ---------------- GENERATE TAB ----------------
    with gr.Tab("🎨 Generate Content"):
        with gr.Column(elem_classes="section-card"):
            gr.Markdown("### 🎯 Generate • Store • Explore")

            prompt = gr.Textbox(label="💬 Enter a prompt", placeholder="Example: A futuristic city at sunset")
            choice = gr.Radio(["Image", "Audio", "Video", "All"], label="Select type", value="Image")
            btn = gr.Button("🚀 Generate")

            out_img = gr.Image(label="📸 Generated Image", height=300)
            out_aud = gr.Audio(label="🎧 Generated Audio")
            out_vid = gr.Video(label="🎬 Generated Video")

            def generate_handler(prompt, choice):
                img = aud = vid = None
                if choice in ["Image", "All"]:
                    img = generate_image(prompt)
                if choice in ["Audio", "All"]:
                    aud = generate_audio(prompt)
                if choice in ["Video", "All"]:
                    vid = generate_video(prompt)

                store_in_db(prompt, img, aud, vid)
                return img, aud, vid

            btn.click(generate_handler, [prompt, choice], [out_img, out_aud, out_vid])

    # ---------------- SEARCH TAB ----------------
    with gr.Tab("🔍 Search"):
        with gr.Column(elem_classes="section-card"):
            gr.Markdown("### 🔎 Search Stored Content")

            q = gr.Textbox(label="🔎 Search query", placeholder="Example: sunset city")
            mode = gr.Radio(["All", "Image", "Audio", "Video"], label="Search in", value="All")

            btn2 = gr.Button("🔍 Search Now")

            gallery = gr.Gallery(label="🖼 Matched Images", columns=2, height=300)
            audio_out = gr.Audio(label="🎧 Audio Result")
            video_out = gr.Video(label="🎬 Video Result")
            status = gr.Textbox(label="⚡ Status")

            btn2.click(search_handler, [q, mode], [gallery, audio_out, video_out, status])

    # ---------------- DB TAB ----------------
    with gr.Tab("📁 View Database"):
        with gr.Column(elem_classes="section-card"):
            gr.Markdown("### 📂 Stored Records in LanceDB")
            show_btn = gr.Button("📄 Show Records")
            db_output = gr.Dataframe(label="LanceDB Table")

            show_btn.click(view_lancedb, [], db_output)

demo.launch()
