# Nulla v0.0.6

<p align="center">
  <img src="assets/Nulla.png" alt="Nulla banner" width="960">
</p>

*A local Windows bootstrapper for a talkative AI companion — sets up Whisper (ASR), XTTS v2 (TTS), llama.cpp with OpenHermes GGUF, and ships sample mini-games.*

---

### Status
**Alpha — functional CLI.** `nulla setup` creates isolated venvs, fetches llama.cpp Windows binaries and the OpenHermes-2.5-Mistral-7B GGUF from **official upstreams**, and wires in Whisper (CPU) + XTTS v2 (CUDA/CPU).  
This package **does not redistribute** third-party models/binaries; they are downloaded during setup under their respective licenses.

### Tested Environment
- **GPU:** RTX 5070 Ti 16 GB  
- **CPU:** Ryzen 5 5600X  
- **RAM:** 32 GB  
- **Storage:** ~20 GB free recommended  
- **OS:** Windows 11  
- **Python:** **3.11.6 (required)**

### Requirements
- Windows 11
- **Python 3.11.6 exactly**
- NVIDIA CUDA (optional; CPU fallbacks exist but are slower)

**Python requirement:** This project targets **Python 3.11.6 exactly**.

**What this package does *not* include:**  
- No Whisper code/weights, no llama.cpp binaries, no XTTS-v2 models, no GGUF models.  

## Quick Start (Windows, PowerShell)

```powershell
# =============================================================================
# Nulla — Quick Start (Windows, PowerShell)
# =============================================================================
# Requirements:
#   • Windows 11
#   • Python 3.11.6 installed (required). In PowerShell, `py -3.11 --version` should work.
#   • Disk space: ~20–30 GB free (models + binaries + caches)
#   • GPU optional (XTTS v2 can use CUDA if present; CPU fallback works but is slower)
#
# What this does:
#   • Creates a project folder with its own Python venv
#   • Installs the LATEST Nulla from PyPI
#   • Runs `nulla setup` to fetch official upstream components:
#       - llama.cpp Windows binaries (from ggml-org releases)
#       - OpenHermes-2.5-Mistral-7B GGUF (from TheBloke on Hugging Face)
#       - Whisper (CPU) & XTTS v2 (TTS) inside their isolated venvs
#   • Nothing is redistributed by Nulla; you’ll accept each upstream license on first use.
#
# If `py -3.11` doesn’t exist on your machine, install Python 3.11, or replace `py -3.11`
# with the full path to your Python 3.11 interpreter (e.g., "C:\Python311\python.exe").
# =============================================================================

# 1) Choose where to install (change this path to any folder you prefer)
$Base  = "C:\Users\Public\Nulla"
$Venv  = "$Base\.venv"
$Root  = "$Base\nulla_source"   # Nulla will download all runtime pieces here

# 2) Create the folder + a fresh virtual environment
New-Item -ItemType Directory -Force $Base | Out-Null
py -3.11 -m venv $Venv

# 3) Install the latest Nulla from PyPI into the venv
& "$Venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Venv\Scripts\pip.exe" install --no-cache-dir nulla

# 4) Sanity checks (help + version)
& "$Venv\Scripts\nulla.exe" --help
& "$Venv\Scripts\python.exe" -m nulla --version

# 5) One-command setup (downloads official upstream components into $Root)
#    Add -y to auto-accept prompts.
& "$Venv\Scripts\nulla.exe" setup --root $Root -y

# 6) (Optional) If llama.cpp EXEs didn’t appear, fetch official binaries explicitly:
#    Adjust CUDA version if needed (supported tags depend on the upstream release).
# & "$Venv\Scripts\nulla.exe" setup-llama-bins --root $Root --cuda 12.4

# ---------------------------------------------------------------------------
# Re-running setup later (no reinstall needed):
# & "$Venv\Scripts\nulla.exe" setup --root $Root -y
#
# If `nulla.exe` isn’t found for any reason, you can always call the module:
# & "$Venv\Scripts\python.exe" -m nulla --help
#
# Uninstall / remove:
#   • Close any running Nulla/llama.cpp processes
#   • Delete the $Base folder to remove the venv and all downloaded assets
#
# Troubleshooting tips:
#   • Ensure you’re using Python 3.11.6 exactly for best compatibility.
#   • Corporate proxies/firewalls can block downloads—try a different network.
#   • If Whisper/XTTS download caches are slow or interrupted, re-run step 5.
#   • GPU users: keep your NVIDIA drivers up to date for CUDA builds.
# =============================================================================
