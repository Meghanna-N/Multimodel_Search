# Multimodal Search

This project implements a **multimodal AI system** where a **text query** can be used to search and generate **images, videos, and audio** based on semantic meaning rather than keyword matching.

The system uses **OpenAI CLIP** to convert text and visual data into a shared embedding space, **LanceDB** to store and retrieve embeddings efficiently, **Stable Diffusion** to generate images from text prompts, **imageio** for image and video processing, and **gTTS** to convert text responses into audio output.

---

## Features
- Text → Image semantic search  
- Text → Video retrieval  
- Text → Audio generation  
- Meaning-based (semantic) search  
- Fast vector similarity search using LanceDB  
- Interactive user interface using Gradio  

---

## Installation

```bash
pip install torch torchvision torchaudio --quiet
pip install ftfy regex tqdm --quiet
pip install git+https://github.com/openai/CLIP.git --quiet
pip install lancedb pyarrow --quiet
pip install gradio imageio gTTS diffusers transformers accelerate safetensors --quiet
