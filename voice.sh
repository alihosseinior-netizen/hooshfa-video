#!/bin/bash
# Persian TTS with Piper (open-source, runs on CPU). Usage: voice.sh <voice> <text> <out.wav>
set -euo pipefail
V=${1:-amir}; mkdir -p voices
M=voices/fa_IR-$V-medium.onnx
python3 -c "import piper" 2>/dev/null || pip install -q piper-tts >/dev/null
if [ ! -f "$M" ]; then
  B=https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/$V/medium/fa_IR-$V-medium.onnx
  curl -sfL "$B" -o "$M"; curl -sfL "$B.json" -o "$M.json"
fi
python3 -m piper -m "$M" -f "$3" -- "$2"
