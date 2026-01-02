. "$PSScriptRoot\.venv\Scripts\Activate.ps1"
# keep everything local to this folder
$env:XDG_CACHE_HOME      = "$PSScriptRoot\.cache"
$env:HF_HOME             = "$PSScriptRoot\.cache\huggingface"
$env:TRANSFORMERS_CACHE  = "$PSScriptRoot\.cache\huggingface\transformers"
$env:PIP_CACHE_DIR       = "$PSScriptRoot\.cache\pip"
$env:TORCH_HOME          = "$PSScriptRoot\.cache\torch"
# if you place a portable ffmpeg at .\ffmpeg\bin\ffmpeg.exe, add it to PATH for this venv only
if (Test-Path "$PSScriptRoot\ffmpeg\bin\ffmpeg.exe") {
  $env:PATH = "$PSScriptRoot\ffmpeg\bin;$env:PATH"
}
