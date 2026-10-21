# Patrick: run your half of Qwen3-8B (Windows, RTX 4060 8 GB)

You serve decoder layers 20-36 (16 of 36) of Qwen3-8B in NF4 (about 1.6 GiB of GPU memory).
Miles runs layers 0-20 and the rest. Everything below is PowerShell. Items marked
[UNVERIFIED] were read from docs/wheel metadata on a Mac and have never run on Windows.
The NF4 code path (`commonnet/backends/nf4_block.py`) has never run on any CUDA GPU:
expect to send Miles the error text if it fails.

## 1. Install (once, about 15 min plus downloads)

1. NVIDIA driver: run `nvidia-smi`. It must print your RTX 4060 and "CUDA Version: 12.x or 13.x".
   If not, install the latest Game Ready/Studio driver from nvidia.com and reboot.
   You do NOT need the CUDA Toolkit; the pip wheels bring their own CUDA runtime.
2. Python 3.12 (64-bit) from python.org, tick "Add python.exe to PATH". [3.10-3.13 should all work;
   bitsandbytes 0.50.2 needs >= 3.10. 3.12 is the safe pick. UNVERIFIED on your PC.]
3. Git: `winget install --id Git.Git -e`  (close and reopen PowerShell afterwards)
4. cloudflared: `winget install --id Cloudflare.cloudflared`  (close and reopen PowerShell; check `cloudflared --version`)
5. Code and Python packages:

```powershell
cd $HOME
git clone -b feat/pedal-integration https://github.com/TheCommonAI/common-network.git
cd common-network
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "transformers>=5" safetensors numpy bitsandbytes huggingface_hub
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import bitsandbytes as b;print(b.__version__)"
```

If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.
The torch line must print `True` and `NVIDIA GeForce RTX 4060`. If it prints `False`, you got the CPU wheel:
rerun the torch install with `--force-reinstall`. [cu126 wheel index and `torch<3` requirement of bitsandbytes
checked on paper only; if cu126 fails use cu128 instead.]
If the git clone asks for a login, the repo is private: tell Miles to add you.

## 2. Download the model (6.1 GB, one file)

```powershell
huggingface-cli download unsloth/Qwen3-8B-bnb-4bit --local-dir $HOME\models\Qwen3-8B-bnb-4bit
```
(If `huggingface-cli` is not found, use `hf download unsloth/Qwen3-8B-bnb-4bit --local-dir $HOME\models\Qwen3-8B-bnb-4bit`.)
You need the whole thing: the weights are a single `model.safetensors` file, so there is no way to fetch only your layers.
Your member reads only layers 20-35 from it and ignores the embedding and head.
If the download stalls, press Ctrl+C and rerun the same command; it resumes.

## 3. Get the token from Miles

Miles sends you a small file `member.token` through a private channel (not a group chat).
Save it as `$HOME\member.token`. Do not paste it anywhere, and do not put it on a command line.
(Windows has no 0600 permissions; just keep the file in your user folder.)

## 4. Start your member

```powershell
cd $HOME\common-network
.\.venv\Scripts\Activate.ps1
python -m commonnet.member --layers 20 36 --backend nf4 --checkpoint $HOME\models\Qwen3-8B-bnb-4bit --token-file $HOME\member.token --out $HOME\member.json
```
Model load takes 1-2 minutes. When it works you see `block server on http://127.0.0.1:...`, then
`tunnel URL: https://....trycloudflare.com`, then `published to ...member.json`.

**Send Miles the tunnel URL (the https://...trycloudflare.com line) and nothing else.** Not the token.
It changes every time you restart.

Leave the window open. To stop: press Ctrl+C once (it stops the server and the tunnel). Closing the window also stops it.

## If something fails

Send Miles the last 20 lines of the window and of `$HOME\block-server.log` (next to `member.json`).
If it ends with "the tunnel did not answer /health", cloudflared started but your DNS is slow to learn the new
name: just run the same command again (the tools now fall back to 1.1.1.1 automatically when your DNS does not know the name; setting your DNS to 1.1.1.1 is a backup).
Known risks: [UNVERIFIED] NF4 loading on CUDA; Windows Defender Firewall prompt (click Cancel/deny, only loopback is used);
GPU out-of-memory (close games/browsers; check with `nvidia-smi`).
