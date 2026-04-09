#!/bin/bash
# mere.run — Demo Showcase
# Generates one phenomenal output per modality, tuned for maximum quality.
# Outputs saved to ~/Desktop/mererun-demo/
set -uo pipefail

OUT="$HOME/Desktop/mererun-demo"
mkdir -p "$OUT"

ts() { date "+%H:%M:%S"; }
log() { echo "[$(ts)] $*"; }
step() { echo ""; log "━━━ $* ━━━"; }

ACESTEP_ROOT="$HOME/Library/Application Support/MereRun/models/music-acestep"

# ═══════════════════════════════════════════════════════════════════════════════
step "1/9  TEXT — Creative writing"
# ═══════════════════════════════════════════════════════════════════════════════

mere.run model pull text-chat-gemma4 2>/dev/null
mere.run text chat \
  -p "Write a short, vivid paragraph describing what it feels like to stand inside a thunderstorm on an alien planet with three moons. Make it poetic but grounded in sensory detail." \
  -s "You are a literary fiction author known for precise, evocative prose. Never use clichés." \
  --temperature 0.85 \
  --max-tokens 512 \
  | tee "$OUT/text_creative.txt"
log "Saved: $OUT/text_creative.txt"

# ═══════════════════════════════════════════════════════════════════════════════
step "2/9  CODE — Elegant solution"
# ═══════════════════════════════════════════════════════════════════════════════

# Try dedicated code model first; fall back to chat model
if mere.run model pull text-code-qwen3 2>/dev/null; then
  mere.run text code \
    -p "Write a beautiful, well-documented Swift function that generates the Mandelbrot set as a 2D array of iteration counts. Include a companion function that renders it to a Unicode art string using a gradient of characters. Make it elegant and idiomatic Swift." \
    --stream \
    | tee "$OUT/code_mandelbrot.txt"
else
  log "Code model unavailable — falling back to chat model for code gen"
  mere.run text chat \
    -p "Write a beautiful, well-documented Swift function that generates the Mandelbrot set as a 2D array of iteration counts. Include a companion function that renders it to a Unicode art string using a gradient of characters. Make it elegant and idiomatic Swift." \
    -s "You are an expert Swift developer. Output only code with comments, no prose." \
    --max-tokens 2048 \
    | tee "$OUT/code_mandelbrot.txt"
fi
log "Saved: $OUT/code_mandelbrot.txt"

# ═══════════════════════════════════════════════════════════════════════════════
step "3/9  IMAGE — Hero artwork"
# ═══════════════════════════════════════════════════════════════════════════════

mere.run model pull image-klein-nano 2>/dev/null
mere.run image generate \
  -m image-klein-nano \
  -p "A solitary astronaut standing on the edge of a vast crystalline canyon on an alien world, bioluminescent flora glowing in deep turquoise and violet along the cliff walls, three moons visible in a twilight sky streaked with aurora-like ribbons, cinematic composition, hyper-detailed, volumetric lighting, concept art quality" \
  -n "blurry, low quality, distorted, text, watermark" \
  -W 1024 -H 1024 \
  -s 8 \
  --seed 7777 \
  -o "$OUT/image_hero.png"
log "Saved: $OUT/image_hero.png"

# ═══════════════════════════════════════════════════════════════════════════════
step "4/9  VISION — Detailed analysis"
# ═══════════════════════════════════════════════════════════════════════════════

if [[ -f "$OUT/image_hero.png" ]]; then
  log "Inspecting the hero image..."
  mere.run vision inspect "$OUT/image_hero.png" \
    "Analyze this image in detail: describe the composition, lighting, color palette, mood, and any notable artistic techniques. What story does it tell?" \
    --max-tokens 1024 \
    | tee "$OUT/vision_analysis.txt"
  log "Saved: $OUT/vision_analysis.txt"

  # Segment the astronaut
  if mere.run model pull vision-segment-sam31 2>/dev/null; then
    mere.run vision segment "$OUT/image_hero.png" \
      --prompt "astronaut" "canyon" "moons" \
      -o "$OUT/vision_segmented.png" \
      --show-boxes
    log "Saved: $OUT/vision_segmented.png"
  else
    log "⚠ Skipping segmentation — model pull failed"
  fi
else
  log "⚠ Skipping vision — no hero image generated"
fi

# ═══════════════════════════════════════════════════════════════════════════════
step "5/9  SPEECH — Expressive narration"
# ═══════════════════════════════════════════════════════════════════════════════

mere.run model pull speech-tts-qwen3-nano 2>/dev/null

# Read back the creative text as narration
NARRATION=$(cat "$OUT/text_creative.txt" 2>/dev/null || echo "The three moons hung low over the crystalline canyon, their light refracting through sheets of alien rain. Thunder rolled across frequencies no human ear was built to parse.")

mere.run speech synthesize "$NARRATION" \
  --voice "A deep, resonant male voice with gravitas, like a nature documentary narrator" \
  --temperature 0.5 \
  -o "$OUT/speech_narration.wav"
log "Saved: $OUT/speech_narration.wav"

# ═══════════════════════════════════════════════════════════════════════════════
step "6/9  SPEECH — Round-trip (TTS → ASR)"
# ═══════════════════════════════════════════════════════════════════════════════

mere.run model pull speech-asr-parakeet 2>/dev/null
log "Transcribing narration back..."
mere.run speech transcribe "$OUT/speech_narration.wav" \
  --timestamps \
  | tee "$OUT/speech_roundtrip.txt"
log "Saved: $OUT/speech_roundtrip.txt"

# ═══════════════════════════════════════════════════════════════════════════════
step "7/9  MUSIC — Full production track"
# ═══════════════════════════════════════════════════════════════════════════════

cat > "$OUT/demo_lyrics.txt" << 'LYRICS'
[intro]

[verse]
Standing on the edge of worlds unknown
Crystal canyons carved from ancient stone
Three moons rising through the violet haze
Lost in wonder, lost in endless days

[chorus]
We are stardust, we are light
Burning through the cosmic night
Every signal, every spark
Echoes dancing in the dark

[verse]
Bioluminescent rivers flow
Painting shadows with an emerald glow
Thunder speaks in frequencies unheard
Every silence holds an alien word

[chorus]
We are stardust, we are light
Burning through the cosmic night
Every signal, every spark
Echoes dancing in the dark

[outro]
LYRICS

mere.run music generate \
  "epic cinematic synthwave with soaring analog synth leads, deep pulsing bass, atmospheric pads, gated reverb snare, arpeggiated sequences, and a building orchestral string section, 1980s retro-futuristic sci-fi film soundtrack quality" \
  --lyrics-file "$OUT/demo_lyrics.txt" \
  --use-lm \
  --bpm 126 \
  --key "E minor" \
  --timesig 4 \
  --duration 60 \
  --steps 8 \
  --checkpoints-root "$ACESTEP_ROOT" \
  -o "$OUT/music_epic.wav"
log "Saved: $OUT/music_epic.wav"

# ═══════════════════════════════════════════════════════════════════════════════
step "8/9  VIDEO — Cinematic scene"
# ═══════════════════════════════════════════════════════════════════════════════

if mere.run model pull video-ltx-av 2>/dev/null; then
  mere.run video generate \
    "cinematic slow dolly forward through a bioluminescent alien canyon at twilight, crystalline walls glowing turquoise and violet, particles floating in the air, volumetric fog, three moons visible in the sky, smooth camera movement, film grain" \
    --width 768 --height 512 \
    --num-frames 65 \
    --fps 24 \
    --seed 7777 \
    -o "$OUT/video_cinematic.mp4"
  log "Saved: $OUT/video_cinematic.mp4"

  # img2vid from hero image
  if [[ -f "$OUT/image_hero.png" ]]; then
    mere.run video generate \
      "the scene slowly comes to life, bioluminescent plants gently pulsing, aurora ribbons shifting in the sky, the astronaut turns their head slightly" \
      --image "$OUT/image_hero.png" \
      --image-strength 0.9 \
      --width 768 --height 512 \
      --num-frames 33 \
      --fps 24 \
      -o "$OUT/video_from_image.mp4"
    log "Saved: $OUT/video_from_image.mp4"
  else
    log "⚠ Skipping img2vid — no hero image"
  fi
else
  log "⚠ Skipping video — model pull failed"
fi

# ═══════════════════════════════════════════════════════════════════════════════
step "9/9  EMBEDDINGS — Semantic similarity"
# ═══════════════════════════════════════════════════════════════════════════════

mere.run model pull text-embed-qwen3-0.6b 2>/dev/null
mere.run text embed \
  "astronaut exploring alien canyon" \
  "bioluminescent crystal caves on another planet" \
  "a cat sitting on a windowsill" \
  -o "$OUT/embeddings_demo.json" --pretty
log "Saved: $OUT/embeddings_demo.json (compare cosine similarity — first two should be close)"

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  DEMO COMPLETE"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
log "  All outputs in: $OUT/"
echo ""
find "$OUT" -type f | sort | while read -r f; do
  log "  $(du -h "$f" | cut -f1)	$f"
done
echo ""
log "  Open the folder:  open $OUT"
