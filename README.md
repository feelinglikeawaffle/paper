# SHREDCO

A browser-based incremental / physics-sandbox prototype set inside OmniCorp's Redaction Division.

## Play
Open \`index.html\` locally, or enable GitHub Pages for this repository (Deploy from branch: \`main\`, folder: \`/ (root)\`).

## SHREDCO Image Studio
A separate local AI image generator and photo editor powered by **FLUX.2 Klein 4B**.

- Open \`studio.html\` through the local Python server (instructions below).
- Generate images from text prompts.
- Upload a PNG, JPG, or WEBP photo and edit it with natural-language instructions.
- Download the generated PNG.
- The model runs locally; there is no per-image API charge. Your computer supplies the compute.

### Run the image studio locally

1. Install Python 3.10+ and a compatible PyTorch build. For NVIDIA CUDA, use the selector at https://pytorch.org/get-started/locally/.
2. In a terminal, enter this repository folder.
3. Install the web dependencies and Diffusers:

   \`\`\`bash
   pip install -r studio_requirements.txt
   pip install git+https://github.com/huggingface/diffusers.git
   \`\`\`

4. Start the local engine:

   \`\`\`bash
   python server.py
   \`\`\`

5. Open http://127.0.0.1:7860

The first generation downloads the model weights and can take a while. A GPU with around 13 GB VRAM is recommended. CPU mode is available as a fallback but may be extremely slow and memory-intensive. Keep the server running while using the studio.

**Important:** GitHub Pages only hosts static files; it cannot run FLUX. The image studio must be opened through the local Python server above. Do not expose this development server directly to the public internet.

Model: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Apache-2.0). Review the model card and license before redistribution or deployment.

## Prototype controls
- Click **Issue Document** to dispense a sheet from the overhead printer.
- Drag documents into the shredder slot along the bottom of the desk.
- Earn Clean-Up Credits and Confetti XP from shredded documents.
- Purchase blade, motor, hopper, and auto-spooler upgrades.

## Art direction
Hybrid pixel-art / 2.5D: pixel-inspired document and machine details, layered desk surfaces, dimensional shadows, and mechanical feedback.

## Current build
Build 0.1.0 — first playable desk loop. Additional printer tiers, narrative evidence, skill trees, prestige, and richer physical effects are part of the evolving game.
