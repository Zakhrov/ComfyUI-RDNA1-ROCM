<div align="center">

# ComfyUI

**The most powerful and modular AI engine for content creation.**

[![Website](https://img.shields.io/badge/comfy.org-4285F4?style=flat)](https://www.comfy.org/)
[![Discord](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total)](https://discord.com/invite/comfyorg)
[![Twitter](https://img.shields.io/twitter/follow/ComfyUI)](https://x.com/ComfyUI)
<br>
[![Release](https://img.shields.io/github/v/release/Comfy-Org/ComfyUI?style=flat&sort=semver)](https://docs.comfy.org/changelog)
[![Downloads](https://img.shields.io/github/downloads/Comfy-Org/ComfyUI/total?style=flat)](https://github.com/Comfy-Org/ComfyUI/releases)

<img width="1024" height="720" alt="ComfyDefaultWorkflow" src="https://github.com/user-attachments/assets/59232d56-277e-4144-9a77-9fc972925874" />

</div>

ComfyUI is the AI creation engine for visual professionals who demand control over every model, every parameter, and every output. Its powerful and modular node graph interface empowers creatives to generate images, videos, 3D models, audio, and more...

- ComfyUI natively supports the latest open-source state of the art models.
- It is by far the most optimized inference engine for diffusion models in the world with a focus on local consumer hardware.
- Can run even the biggest open source models on as low as 4GB vram + 8GB ram relatively quickly (saturating your GPU compute) using our state of the art asynchronous weight streaming technology.
- [Partner nodes](https://docs.comfy.org/tutorials/partner-nodes/overview#partner-nodes) provide access to the best closed source models such as Nano Banana, Seedance, Hunyuan3D, etc.
- It is available on Windows, Linux, and macOS, locally with our [desktop application](https://www.comfy.org/download), our [portable install](docs/installation.md#windows-portable) or on our [cloud](https://www.comfy.org/cloud).
- The most sophisticated workflows can be exposed through a simple UI thanks to App Mode.
- It integrates seamlessly into production pipelines with our API endpoints.

## This fork: RDNA1 (gfx1010)

This is a hard fork of [ComfyUI](https://github.com/Comfy-Org/ComfyUI) for the RX 5600M
class of GPU (RDNA1, `gfx1010`, 6 GB), optionally beside a Vega APU (`gfx90c`). It carries
the [comfy-kitchen fork](https://github.com/Zakhrov/comfy-kitchen) as the `comfy-kitchen`
submodule (branch `rdna1-backend`), built from source for its `rdna1` backend (see
[the backend's README](comfy-kitchen/comfy_kitchen/backends/rdna1/README.md)), and pairs
with the ComfyUI-MultiGPU fork.

### Install

The comfy-kitchen PyPI wheel has no `rdna1` backend, so `requirements.txt` installs the
submodule as an editable package and compiles the backend into the source tree. That
needs a ROCm compiler, found through `ROCM_HOME`, `/opt/rocm` or `amdclang++`/`hipcc` on
`PATH`, and takes a few minutes. Without one the install still succeeds but has no
`rdna1` backend: check for `Found comfy_kitchen backend rdna1` in the startup log.

```bash
git clone https://github.com/Zakhrov/ComfyUI-RDNA1-ROCM.git
cd ComfyUI-RDNA1-ROCM
git submodule update --init

uv venv
uv pip install --extra-index-url https://nightly.repo.amd.com/rocm/whl-next/ "rocm[device-gfx90c,device-gfx1010]" "torch[device-gfx90c,device-gfx1010]==2.12" torchaudio==2.11 "torchvision[device-gfx90c,device-gfx1010]==0.27"
uv pip install --extra-index-url https://nightly.repo.amd.com/rocm/whl-next/ -r requirements.txt
uv run main.py --disable-dynamic-vram --use-ck-attention
```

- Drop the `device-gfx90c` extras on a machine without the Vega APU.
- Do not pass `--recursive` to the submodule update: comfy-kitchen's own submodules
  (CUTLASS, flash-attention) are only read by its CUDA backend.
- The build runs in an isolated environment that cannot see the GPU. uv takes the target
  from `[tool.uv.extra-build-variables]` in `pyproject.toml`; with pip, set it yourself:
  `COMFY_HIP_ARCHS=gfx1010 pip install -r requirements.txt`.
- After pulling, run `git submodule update` to move comfy-kitchen to the recorded commit.
  If its native sources (`.hip`, `.cpp`, `.h`) changed, rebuild incrementally from
  `comfy-kitchen/`: `COMFY_HIP_ARCHS=gfx1010 ../.venv/bin/python setup.py build_ext --inplace`.
  Python changes need no rebuild.

### Changes from upstream

Every path below is taken only where comfy-kitchen reports its packed fp16 kernels as
accelerated for the tensor's device, and never while training; elsewhere the code runs
upstream's torch ops.

- **Linear layers** (`comfy/ops.py`): plain fp32, fp16 and bf16 linears run on kitchen's
  packed fp16 GEMM in place of rocBLAS, above the size where it is faster. The weight is
  cast straight to fp16 rather than through the input's dtype.
- **Convolutions** (`comfy/ops.py`): every float Conv1d, Conv2d, Conv3d and transposed
  conv runs on kitchen's kernels, so MIOpen never runs. MIOpen searches for a kernel on the
  first call with each new shape (gfx1010 ships no find-db) and its fp32 GEMM solvers take
  an im2col workspace that has run tiled VAE decodes out of memory; a Flux VAE decode
  takes 2.3 s on its first call here against 23 s with an empty MIOpen cache. Per conv:
  - fp32 2D convs at stride 1 or 2 run MIOpen's own fp32 Winograd kernels, built into
    kitchen (0.7-1.05x MIOpen's warm time, no search);
  - 1x1 convs, heads of at most 8 output channels and stems with few input channels run
    kitchen's direct conv;
  - grouped convs with one filter per channel run the depthwise kernel, other groups
    run one group at a time;
  - everything else runs the packed fp16 implicit GEMM, whose 3x3 stride-1 convs take a
    fused Winograd F(2x2, 3x3) kernel.

  `padding_mode` other than zeros and string padding are applied to the input first, as
  torch does. Only dilated transposed convs still run on torch. A Conv3d whose kernel
  covers its whole input runs as a linear. Once warm, MIOpen stays ahead only on 1x1 convs
  past 256 channels (1.05-1.5x), tiny heads (1.2-1.4x) and small fp16 transposed convs (1.4x).
- **BatchNorm** (`comfy/ops.py`): eval-mode batch norm runs as one fused multiply-add in
  torch instead of MIOpen's batch norm.
- **Model code convs**: `comfy.ops.conv_nd` takes the same route for model code that calls
  a conv on its own tensors; the LTXV vocoder's STFT uses it. Other raw `F.conv*` calls in
  model files still run on torch.
- **Depthwise 1D resampling**: the anti-aliasing filters in the LTXV vocoder, the MMAudio
  VAE and the MiniMax audio VAE use kitchen's depthwise conv kernels.
- **VAE attention** (`comfy/ldm/modules/diffusionmodules/model.py`): the sliced attention
  fallback runs its two matrix products on the packed GEMM.
- **MiniMax H3 VAE**: fp16 decode uses kitchen's fused group norm, SiLU and pad on ROCm,
  and the tiled decode canvas is kept on the intermediate device so finished tiles leave
  VRAM.
- **Text-encoder LM heads** (`comfy/text_encoders/llama.py`, `qwen35.py`, `ace15.py`):
  logits read the vocabulary table as fp16 through the packed GEMV instead of casting
  the whole table to fp32 (about 1.8 GB on the ACE-Step 1.5 2B LM). ACE-Step audio-code
  sampling computes only the EOS and audio-code rows.
- **LoRA** (`comfy/weight_adapter/lora.py`, `comfy/controlnet.py`): LoRA weight merges
  and control LoRA layers use the same linear and conv paths.
- **Uploads** (`folder_paths.py`): `text/xsl` and `text/x-xslt` are added to the blocked
  content types, as some distributions map `.xsl` to them.

### Results

How the pieces fit, against comfy-kitchen's plain eager backend (stock PyTorch ops,
no working INT8 matmul and no INT8 attention on gfx1010):

- **comfy-kitchen `rdna1`** supplies kernels built for a GPU with no matrix cores, dot
  instructions or bf16: a register-blocked GEMM, packed fp16 linear and conv, and INT8
  attention. On upstream ComfyUI it only changes attention: SD1.5 runs 16% faster and in
  ~1.5 GB less VRAM.
- **This fork** routes ComfyUI's fp16 linears and convs to those kernels as well, which
  upstream does not: SD1.5 sampling goes from 1.20 it/s (eager) to 1.86 it/s.
- **ComfyUI-MultiGPU** streams models that do not fit in 6 GB through the same kernels;
  its streamed execution requires the rdna1 path and ck-attention. Z-Image Turbo runs at
  normal VRAM, and SDXL in 1.9 GB.

The Vega iGPU only stores weights in these workflows; nothing computes on it.

512x512 text to image on a Dell G5 15 SE (RX 5600M + Ryzen Vega APU), PyTorch 2.12 ROCm
nightly, `--disable-cuda-malloc --disable-dynamic-vram`, 2026-10-09. Cold includes model
loading; warm is a second prompt with a new seed.

| | Upstream, kitchen eager, sub-quadratic | Upstream, rdna1, `--use-ck-attention` | This fork, rdna1, `--use-ck-attention` | This fork + MultiGPU, both GPUs |
| --- | --- | --- | --- | --- |
| SD1.5, 20 steps: cold / warm, peak VRAM | 20.6 / 18.8 s, 4.0-4.4 GB | 17.8 / 15.8 s, 2.7-2.8 GB | 14.9 / 12.7 s, 2.6-2.7 GB | 16.8 / 13.0 s, 2.4 GB |
| SDXL, 20 steps: cold / warm, peak VRAM | 37.9 / 34.4 s, 5.4-5.6 GB | 36.4 / 32.9 s, 5.4-5.5 GB | 35.3 / 30.8 s, 5.3-5.5 GB | 45.3 / 35.4 s, 1.9 GB |
| Z-Image Turbo, 9 steps | out of memory | out of memory | out of memory | 74.6 / 29.6 s, 2.6 GB warm |
| Z-Image Turbo with `--lowvram` | 147.5 / 65.5 s | 138.7 / 65.4 s | 111.8 / 33.2 s | not needed |

- Upstream ComfyUI does not run on this GPU at its defaults: dynamic VRAM fails on the
  first model load (`VBAR allocation failed`), so every column uses `--disable-dynamic-vram`.
- Z-Image Turbo runs out of memory in its Q8_0 GGUF text encoder at normal VRAM. The
  MultiGPU column splits that encoder across both GPUs and RAM and streams the UNet
  (`z_image_turbo_int8_convrot.safetensors`; the other columns use the Q8_0 GGUF).
- MultiGPU streams SDXL's UNet from the iGPU and RAM: 15% slower sampling, 3.5 GB less VRAM.
- With the Q8_0 GGUF UNet instead, the MultiGPU Z-Image run took 56.4 / 30.3 s: within 5%
  of the int8 safetensors once loaded.
- Full setup, per-step rates and caveats are in the rdna1 backend README.

## Features

- A visual node graph for building and reusing image, video, audio, 3D, and text workflows without code.
- Broad native model support. This is a representative list; browse the [workflow library](https://comfy.org/workflows/) for maintained, ready-to-run templates.
  - [Image generation](https://comfy.org/workflows/tag/text-to-image/): Stable Diffusion 1.5, SDXL, SD3.5, Flux.1, Flux.2, Qwen Image, Z-Image, Hunyuan Image 2.1, HiDream, Lumina Image 2.0, Chroma, Anima, LongCat Image, Ideogram 4, Krea 2, MageFlow, Microsoft Lens, PixelDiT, Kandinsky 5, Ernie Image and Moss Image.
  - [Image editing](https://comfy.org/workflows/tag/image-edit/): Flux Kontext, Flux.2 Klein, Qwen Image Edit, HiDream E1.1 and O1, OmniGen2, Boogu, JoyImage Edit, MageFlow Edit, LongCat Image Edit, Fibo 2.0, Flux 3 image klein.
  - [Video generation](https://comfy.org/workflows/tag/video-generation/): Wan 2.1 and 2.2, LTX-Video 2 and 2.3, HunyuanVideo 1.5, Kandinsky 5 Video, CogVideoX, Cosmos Predict2, Bernini-R, SCAIL 2, and Mochi.
  - [Audio and video generation](https://comfy.org/workflows/): MiniMax H3 and LTX-AV.
  - [Audio generation](https://comfy.org/workflows/tag/text-to-audio/): ACE-Step 1.5, Stable Audio 3, MiniMax Music 3 and Yue 2.
  - [3D and vision](https://comfy.org/workflows/): Hunyuan3D 2.1, TripoSplat, SeedVR2, SUPIR, Depth Anything 3, MoGe, SAM 3 and 3.1, RT-DETRv4, and BiRefNet.
  - [Text generation](https://comfy.org/workflows/tag/text-generation/): Gemma 3 and 4, Qwen3, Qwen3.5, and Qwen3-VL, including multimodal inputs.
- Efficient local execution with asynchronous queueing, partial graph re-execution, smart VRAM and RAM management, model offloading, and support for quantized models.
- Reusable subgraphs, workflow templates, [App Mode](https://docs.comfy.org/interface/app-mode), and a [local API](https://docs.comfy.org/development/comfyui-server/api-examples) for integrating workflows into applications.
- Load complete checkpoints or separate diffusion models, VAEs, text encoders, LoRAs, ControlNets, adapters, and upscalers from supported model formats.
- Built-in tools for inpainting, outpainting, reference conditioning, masks and compositing, model merging, upscaling, frame interpolation, segmentation, depth estimation, and media processing.
- Save and load workflows as JSON. You can also drag any image, video, audio, or 3D file made with ComfyUI onto the canvas to get back the full workflow that made it, seeds included.
- Runs fully offline: core does not download anything unless you request it. Use `--offline` to disable the optional paid [Comfy API nodes](https://docs.comfy.org/tutorials/api-nodes/overview) and force all built-in functionality to stay offline, or `--disable-partner-nodes` to only disable the API nodes.
- Extend ComfyUI with custom nodes, installed and updated through [ComfyUI-Manager](docs/installation.md#comfyui-manager).
- Configure additional model locations with [`extra_model_paths.yaml`](extra_model_paths.yaml.example).
- Support for saving and loading high bit depth images and videos: 16 bit PNG images, 32 bit EXR, 10 bit AVIF are supported and more, including HDR.

## Products

| Product | Who it's for |
| --- | --- |
| [Comfy Desktop](https://www.comfy.org/download) | Anyone who wants ComfyUI running locally from a normal installer, on Windows or macOS. The easiest way to start. |
| [Developer Platform](https://www.comfy.org/platform) | Developers who want to take Comfy to production. Deploy a workflow as an API endpoint that scales, or call thousands of models through one API. |
| [Comfy Cloud](https://www.comfy.org/cloud) | Running Comfy without local hardware. Good for experimentation and lightweight workflows. |
| [Comfy Agent](https://www.comfy.org/agent) | Describing what you want inside ComfyUI and having it plan, build and run the workflow, editing the canvas alongside you. |
| [Comfy MCP](https://www.comfy.org/mcp) | Driving ComfyUI from an MCP client, local or cloud: generate media, search models, nodes and templates, and run real workflows. |
| [Comfy CLI](https://www.comfy.org/cli) | Driving the whole engine from a terminal. Built for CI jobs, scripting, and piping outputs somewhere else. |
| [Comfy Enterprise](https://www.comfy.org/enterprise) | Organizations standardizing ComfyUI across teams: one approved reproducible build everywhere, dedicated GPU capacity and SLAs, commercial licensing and security review. |

## Installing

### [Comfy Desktop](https://www.comfy.org/download) is the recommended way to install ComfyUI

It is the easiest and best way to get started, on Windows and macOS.

Other paths:

- **[Windows Portable](docs/installation.md#windows-portable)** for a self-contained build you can unzip and run.
- **[Comfy Cloud](https://www.comfy.org/cloud)** if you would rather not run it locally at all.
- **[comfy-cli](https://www.comfy.org/cli)** if you prefer a CLI: `pip install comfy-cli && comfy install`.
- **[Developer Platform](https://www.comfy.org/platform)** if you are deploying a workflow as a production API endpoint rather than running the app yourself.
- **Manual install**, below, when you need control over your Python and Torch versions.

### Manual install

First install PyTorch for your GPU, then:

```bash
git clone https://github.com/Comfy-Org/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
python main.py
```

Python 3.13 is very well supported and 3.14 works, though some custom nodes may have
issues. torch 2.7 is minimally supported and cu130 or above is required on Nvidia 20
series and above; if your pytorch is more than 6 months old, please update it.

Put your checkpoints in `models/checkpoints` and your VAE in `models/vae`.

**[docs/installation.md](docs/installation.md) has the full detail:** PyTorch commands
for NVIDIA, AMD ROCm on Linux and Windows, Intel Arc, Apple Silicon, Huawei Ascend NPU,
Cambricon MLU and Iluvatar Corex, plus ComfyUI-Manager setup, portable builds, previews
and TLS.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to ask questions, report issues, open pull
requests, and how releases are cut across core, Desktop and the frontend.

## Community

Ask questions in [#help or #feedback on Discord](https://comfy.org/discord).

_psst, we're hiring:_ [comfy.org/careers](https://www.comfy.org/careers)

## References

| Resource | Link |
| --- | --- |
| Documentation | <https://docs.comfy.org> |
| Installing (this repo) | [docs/installation.md](docs/installation.md) |
| Keyboard shortcuts | <https://docs.comfy.org/interface/shortcuts> |
| Launch / startup flags | <https://docs.comfy.org/development/comfyui-server/startup-flags> |
| Changelog | <https://docs.comfy.org/changelog> |
| Workflow templates | <https://comfy.org/workflows> |
| Older example workflows | <https://comfyanonymous.github.io/ComfyUI_examples/> |
| Built-in node reference | <https://docs.comfy.org/built-in-nodes/overview> |
| Writing custom nodes | <https://docs.comfy.org/custom-nodes/overview> |
| Prompt syntax: weighting, `{wildcards}`, `embedding:`, `//` comments | <https://docs.comfy.org/built-in-nodes/ClipTextEncode> |
| Recovering a workflow from generated media | <https://docs.comfy.org/development/api-development/workflow-metadata> |
| Troubleshooting | <https://docs.comfy.org/troubleshooting/overview> |
| Which GPU should I buy? | <https://github.com/Comfy-Org/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI> |
| Blog | <https://blog.comfy.org> |
| Discord | <https://comfy.org/discord> |
| Frontend repository | <https://github.com/Comfy-Org/ComfyUI_frontend> |
