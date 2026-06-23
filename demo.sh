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
  -p "Write a short, vivid paragraph describing what it feels like to step into a 1950s American roadside diner just before midnight during a heavy rainstorm. Make it poetic but grounded in sensory detail." \
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
  -p "photograph of a 1950s American roadside diner just before midnight, heavy rain streaking the large plate-glass windows, warm amber tungsten light spilling onto a wet asphalt parking lot, long polished chrome counter and red vinyl stools, a waitress in a mint-green uniform with a white apron pouring coffee from a glass pot for a single customer hunched over the counter in a wool overcoat, glowing red neon EAT sign reflected in the puddles outside, black-and-white checkerboard tile floor, a Wurlitzer jukebox glowing in the corner, faint cigarette smoke curling through the light, shot on Kodak Portra 400 film, anamorphic lens, shallow depth of field, cinematic Edward Hopper composition, ultra realistic, 8k" \
  -n "painting, illustration, cartoon, anime, 3d render, cgi, blurry, low quality, distorted, text, watermark, oversaturated, modern smartphones, modern cars, modern clothing, anachronistic" \
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

  # Ground objects with Falcon Perception
  if mere.run model pull vision-ground-falcon-perception 2>/dev/null; then
    mere.run vision ground "$OUT/image_hero.png" \
      --query "waitress" --query "jukebox" --query "neon sign" \
      -o "$OUT/vision_grounded.png" \
      --json-output "$OUT/vision_grounded.json" \
      --mask-output-dir "$OUT/vision_masks"
    log "Saved: $OUT/vision_grounded.png"
    log "Saved: $OUT/vision_grounded.json"
  else
    log "⚠ Skipping grounding — model pull failed"
  fi
else
  log "⚠ Skipping vision — no hero image generated"
fi

# ═══════════════════════════════════════════════════════════════════════════════
step "5/9  SPEECH — Expressive narration"
# ═══════════════════════════════════════════════════════════════════════════════

mere.run model pull speech-tts-qwen3-nano 2>/dev/null

# Read back the creative text as narration
NARRATION=$(cat "$OUT/text_creative.txt" 2>/dev/null || echo "Rain drummed against the diner windows like a slow brushed snare. Inside, the jukebox glowed amber, the chrome counter caught the neon, and the coffee pot hissed quietly against the late hour.")

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
Rain on the window, neon on the sign
Coffee getting cold by a quarter past nine
Jukebox humming an old slow tune
Waitress smiling like a harvest moon

[chorus]
Honey stay one more song with me
Underneath the chrome and the canopy
Red vinyl shining in the smoky light
Save me from the lonely night

[verse]
Headlights drifting down the wet asphalt road
Ceiling fan turning slow and low
Pie under glass and a cigarette glow
One more dance before you go

[chorus]
Honey stay one more song with me
Underneath the chrome and the canopy
Red vinyl shining in the smoky light
Save me from the lonely night

[outro]
LYRICS

mere.run music generate \
  "warm vintage rockabilly ballad with brushed snare, walking upright bass, twangy reverb-soaked Telecaster, dreamy doo-wop backing vocals, a soft tenor saxophone solo, late-night 1950s American diner jukebox sound, analog tape warmth, mono recording" \
  --lyrics-file "$OUT/demo_lyrics.txt" \
  --use-lm \
  --bpm 88 \
  --key "G major" \
  --timesig 4 \
  --duration 60 \
  --steps 8 \
  --checkpoints-root "$ACESTEP_ROOT" \
  -o "$OUT/music_epic.wav"
log "Saved: $OUT/music_epic.wav"

# ═══════════════════════════════════════════════════════════════════════════════
step "8/9  VIDEO — Cinematic scene"
# ═══════════════════════════════════════════════════════════════════════════════

if mere.run model pull video-ltx23-av-mlx 2>/dev/null; then
  mere.run video generate \
    "EXT./INT. 1950s AMERICAN ROADSIDE DINER – NIGHT – CINEMATIC ESTABLISHING SHOT. The camera begins outside on a rain-slicked asphalt street, glowing red neon EAT and OPEN signs reflecting in deep puddles. A slow, steady dolly push moves the camera toward the large plate-glass window of the diner, rain beading and streaking down the glass. Through the window, warm amber tungsten light reveals a long polished chrome counter, red vinyl stools, and a single customer in a wool overcoat hunched over a coffee cup. A waitress in a mint-green uniform with a white apron pours coffee from a glass pot, steam curling into the light. A Wurlitzer jukebox glows softly in the corner with shifting amber and ruby tones. Cigarette smoke drifts in slow lazy curls through the warm light. The camera continues its push, transitioning through the glass into the diner interior, depth of field shifting as the foreground rain blur dissolves and the chrome counter and red booths sharpen into focus. The shot holds a shallow depth of field, anamorphic lens flares pulling soft horizontal streaks from the neon and polished chrome. 35mm Kodak Portra film grain, slightly desaturated palette except for the warm tungsten interior and the saturated reds and ambers of the neon." \
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
      "The still scene slowly comes to life. Rain continues to streak the windows, headlights of a passing car briefly washing across the wet street outside. The waitress turns slightly, lifts the glass coffee pot, and tops off the cup, steam curling upward through the warm tungsten light. The customer in the wool overcoat shifts on the stool and lifts the cup to his lips. The jukebox in the corner pulses gently as a record changes, its amber glow brightening for a moment. Cigarette smoke from an ashtray on the counter drifts in slow lazy spirals. The camera holds steady for a beat, then begins a barely perceptible push-in toward the customer's profile, the neon EAT sign blurred but readable in the reflection of the window behind him. The ambient sound is the soft hum of refrigeration, the faint clink of a spoon against ceramic, and the muffled patter of rain on the roof." \
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
  "rainy 1950s American diner late at night" \
  "neon-lit chrome counter and red vinyl booths" \
  "a cat sitting on a windowsill" \
  -o "$OUT/embeddings_demo.json" --pretty
log "Saved: $OUT/embeddings_demo.json (compare cosine similarity — first two should be close)"

# ═══════════════════════════════════════════════════════════════════════════════
step "BONUS  TOOL USE — Agentic code compile & run"
# ═══════════════════════════════════════════════════════════════════════════════

mere.run text chat \
  -p "Write a Swift program that prints the Mandelbrot set as ASCII art (60x30 characters, using characters ' .:-=+*#%@' for increasing iteration depth). Save it to mandelbrot.swift using write_file, then compile and run it in one shell_exec call: 'swiftc mandelbrot.swift -o mandelbrot && ./mandelbrot'." \
  -s "You have write_file and shell_exec tools. Use them to complete the task. Be concise." \
  --tools write_file,shell_exec \
  --tool-loop \
  --allow-shell-exec \
  --sandbox-dir "$OUT/tool_sandbox" \
  --max-tokens 2048 \
  --temperature 0.3 \
  | tee "$OUT/tool_use_demo.txt"
log "Saved: $OUT/tool_use_demo.txt"
log "Sandbox: $OUT/tool_sandbox/"

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
