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
  -p "photograph of a lone astronaut in a weathered spacesuit standing at the edge of a massive crystalline canyon on an alien planet, bioluminescent moss and lichen glowing turquoise and violet on the rock faces, three pale moons low in a dusky indigo sky with faint aurora streaks, shot on 35mm film, shallow depth of field, golden hour backlight catching dust particles, photojournalistic composition, ultra realistic, 8k" \
  -n "painting, illustration, cartoon, anime, 3d render, cgi, blurry, low quality, distorted, text, watermark, oversaturated" \
  -W 1280 -H 720 \
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
    "EXT. ALIEN CANYON – TWILIGHT – CINEMATIC ESTABLISHING SHOT. The camera begins on a wide shot of a vast crystalline canyon stretching to the horizon, walls covered in bioluminescent moss glowing deep turquoise and violet. Three pale moons hang low in an indigo sky streaked with shimmering aurora ribbons. The camera pushes forward in a slow, steady dolly move, gliding just above the canyon floor. Tiny luminous particles drift through the air like embers, catching the faint moonlight. Volumetric fog rolls through the lower passages, partially obscuring jagged crystal formations that jut from the walls. The ambient sound is a deep, resonant hum — the canyon itself vibrating at some low alien frequency — layered with the soft chime of crystal faces shifting in a gentle wind. As the camera advances, the bioluminescent glow intensifies, casting rippling turquoise reflections across the fog. The shot holds a shallow depth of field, the foreground crystals softly blurred as the vast canyon beyond sharpens into focus. Film grain texture, anamorphic lens flares from the brightest crystal clusters." \
    --variant unified-av \
    --width 768 --height 512 \
    --num-frames 65 \
    --fps 24 \
    --seed 7777 \
    -o "$OUT/video_cinematic.mp4"
  log "Saved: $OUT/video_cinematic.mp4"

  # img2vid from hero image
  if [[ -f "$OUT/image_hero.png" ]]; then
    mere.run video generate \
      "The still scene slowly comes to life. A faint wind picks up, causing the bioluminescent moss on the cliff walls to pulse gently in waves of turquoise light. The astronaut, silhouetted against the canyon, shifts their weight and turns their head slightly to the right, gazing upward at the three moons. Aurora ribbons in the sky begin to ripple and shift, their colors bleeding from violet into emerald green. Small luminous spores lift off the canyon floor and drift upward past the astronaut like slow-motion fireflies. The camera holds steady for a beat, then begins a barely perceptible push-in toward the astronaut's visor, where the aurora light reflects across the curved glass. The ambient sound is a low crystalline resonance with the soft whisper of alien wind." \
      --variant unified-av \
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
step "BONUS  TOOL USE — Agentic code compile & run"
# ═══════════════════════════════════════════════════════════════════════════════

if [[ -f "$OUT/code_mandelbrot.txt" ]]; then
  MANDELBROT_CODE=$(cat "$OUT/code_mandelbrot.txt")
  mere.run text chat \
    -p "Here is Swift code for a Mandelbrot set renderer. Extract ONLY the Swift code (no markdown fences), save it to mandelbrot.swift using write_file, then compile it with 'swiftc mandelbrot.swift -o mandelbrot' using shell_exec, and finally run './mandelbrot' using shell_exec. Show me the output.

$MANDELBROT_CODE" \
    -s "You have write_file and shell_exec tools. Use them step by step. Be concise." \
    --tools write_file,shell_exec \
    --tool-loop \
    --sandbox-dir "$OUT/tool_sandbox" \
    --max-tokens 4096 \
    --temperature 0.3 \
    | tee "$OUT/tool_use_demo.txt"
  log "Saved: $OUT/tool_use_demo.txt"
  log "Sandbox: $OUT/tool_sandbox/"
else
  log "⚠ Skipping tool use demo — no code output from step 2"
fi

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
