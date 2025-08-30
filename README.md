# ComfyUI LLM SDXL Adapter　フォーク版

これはフォーク版です。

T5Gemmaについてのみ修正を加えています。

１　入力されたカンマをスペースに変換

２　attention_maskをモデルに渡し、パディング部分を無視させるようにした

３　llm_to_sdxl_adapter.pyで、MultiheadAttentionのロジックを正しく実装し直し、温度を適用するクラスclass SharpenedMultiheadAttention(nn.MultiheadAttention)にした

４　オプションでデバッグを出力することが出来るようにした

３についての意義はなんとも言えないところですが追加してみたというものです。temperature=0.60 としていますので、ここを変えて調整してください

# ComfyUI LLM SDXL Adapter

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange.svg)

A comprehensive set of ComfyUI nodes for using Large Language Models (LLM) as text encoders for SDXL image generation through a trained adapter.

<img width="1803" height="904" alt="image" src="https://github.com/user-attachments/assets/e8e5f047-37e7-4f8b-9bbd-78d70e2a7d80" />

[Image with workflow](https://files.catbox.moe/om6tc4.png)


## 🎯 Available Adapters

### RouWei-Gemma Adapter 
Trained adapter for using Gemma-3-1b as text encoder for [Rouwei v0.8](https://civitai.com/models/950531) (vpred or epsilon or [base](https://huggingface.co/Minthy/RouWei-0.8/blob/main/rouwei_080_base_fp16.safetensors)).

**Download Links:**
- [CivitAI Model](https://civitai.com/models/1782437)
- [HuggingFace Repository](https://huggingface.co/Minthy/RouWei-Gemma)

## 📦 Installation
### Install Nodes
1. Clone the repository to `ComfyUI/custom_nodes/`:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter.git
```

2. Restart ComfyUI

### Setup RouWei-Gemma Adapter

1. **Download the adapter:**
   - Download from [CivitAI](https://civitai.com/models/1782437) or [HuggingFace](https://huggingface.co/Minthy/RouWei-Gemma)
   - Place the adapter file in `ComfyUI/models/llm_adapters/`

2. **Download Gemma-3-1b-it model:**
   - Download [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) ([non-gated mirror](https://huggingface.co/unsloth/gemma-3-1b-it))
   - Place in `ComfyUI/models/llm/gemma-3-1b-it/`
   - **Note:** You need ALL files from the original model for proper functionality (not just .safetensors)

3. **Download Rouwei checkpoint:**
   - Get [Rouwei v0.8](https://civitai.com/models/950531) (vpred, epsilon, or [base](https://huggingface.co/Minthy/RouWei-0.8/blob/main/rouwei_080_base_fp16.safetensors)) if you don't have it
   - Place in your regular ComfyUI checkpoints folder

## 📁 File Structure Example

```
ComfyUI/models/
├── llm/gemma-3-1b-it/
│   ├── added_tokens.json
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer.model
│   └── tokenizer_config.json
├── llm_adapters/
│   └── rouweiGemma_g31b27k.safetensors
└── checkpoints/
    └── rouwei_v0.8_vpred.safetensors
```

## 🔍 Debugging

To enable detailed logging, edit `__init__.py`:
```python
# Change from:
logger.setLevel(logging.WARN)
# To:
logger.setLevel(logging.INFO)
```
