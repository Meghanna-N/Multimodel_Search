Multimodal Search

This project is a multimodal AI system where a text query is used to search and generate images, videos, and audio. It uses OpenAI CLIP for semantic understanding, LanceDB for vector search, Stable Diffusion for image generation, imageio for image/video processing, and gTTS for text-to-audio output.

Installation
pip install torch torchvision torchaudio --quiet
pip install ftfy regex tqdm --quiet
pip install git+https://github.com/openai/CLIP.git --quiet
pip install lancedb pyarrow --quiet
pip install gradio imageio gTTS diffusers transformers accelerate safetensors --quiet

Tech Used

Python

OpenAI CLIP

LanceDB

Stable Diffusion

imageio

gTTS

Gradio
