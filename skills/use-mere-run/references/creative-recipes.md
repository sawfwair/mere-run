# Creative Recipes

Use this reference when the user wants a good creative output, not just the syntactically correct `mere.run` command. The CLI exposes mechanics; the agent should provide taste, structure, and iteration.

## General Pattern

Turn vague requests into a concrete brief:

1. Medium: song, beat, image, video, speech, caption, code, or API run.
2. Subject: what should be present.
3. Style: genre, era, format, reference qualities, production texture, or visual language.
4. Constraints: duration, size, output path, model, seed, tempo, key, or language.
5. Iteration: run one small result first, then vary one or two controls.

Prefer one strong command and one next-step variation over a long menu of possibilities.

## Music

The music CLI can make better tracks when the prompt contains musical structure rather than a generic adjective. Build captions from:

- Genre/subgenre: synthwave, jazz trio, indie folk, lo-fi hip hop, ambient, cinematic strings.
- Instrumentation: brushed drums, walking bass, Rhodes, arpeggiated synths, acoustic guitar, string quartet.
- Groove/tempo: relaxed 82 BPM, driving 128 BPM, slow ballad, half-time drums.
- Key/time: `--key "A minor"`, `--timesig 4`; use common keys unless the user asks for something unusual.
- Mood/context: late-night, warm, triumphant, fragile, dreamlike, rainy city.
- Arrangement: intro, verse, chorus, bridge, drop, outro. Put actual vocals in `--lyrics` or `--lyrics-file`.
- Production: tape warmth, dry intimate vocal, wide stereo pads, clean modern pop, gritty live room.

Use `--use-lm` when the user wants a more structured piece, lyrics, or metadata-guided music. It requires a model layout with the 5Hz LM subdirectory; if it fails, pull `music-acestep`, use the managed default layout, or remove `--use-lm` for a simpler direct diffusion run.

Start with `music-acestep`:

```bash
mere.run model pull music-acestep
```

Instrumental beat:

```bash
mere.run music generate \
  "warm lo-fi hip hop beat, dusty drums, mellow Rhodes chords, round sub bass, vinyl texture, relaxed late-night mood" \
  --use-lm \
  --bpm 82 \
  --key "F minor" \
  --timesig 4 \
  --duration 20 \
  --output ./lofi-night.wav
```

Vocal song with lyrics:

```bash
mere.run music generate \
  "indie folk ballad with fingerpicked acoustic guitar, soft male vocal, brushed percussion, warm room ambience, intimate and hopeful" \
  --lyrics-file ./lyrics.txt \
  --use-lm \
  --bpm 94 \
  --key "G major" \
  --timesig 4 \
  --duration 30 \
  --output ./folk-demo.wav
```

Cinematic cue:

```bash
mere.run music generate \
  "cinematic orchestral cue, low strings pulse, distant brass swells, shimmering high strings, slow heroic build, trailer underscore" \
  --use-lm \
  --bpm 72 \
  --key "D minor" \
  --timesig 4 \
  --duration 24 \
  --output ./cinematic-cue.wav
```

Iteration tips:

- If the result is too chaotic, shorten the duration, simplify the caption, lower arrangement complexity, or use a seed and iterate from it.
- If the result lacks structure, add `--use-lm`, BPM, key, and time signature.
- If lyrics are garbled or misplaced, make the caption explicitly vocal-led and put lyrics in simple `[verse]` and `[chorus]` sections.
- If the user wants repeatable A/B tests, add `--seed 12345` and change only the caption or one metadata flag at a time.
- If the command reports missing checkpoints, run `mere.run model info music-acestep` and verify the model store, or pass `--checkpoints-root`.

## Image And Video

Image and video prompts benefit from visible subject, medium, composition, lighting, and constraints.

Image prompt pattern:

```text
<subject>, <medium/style>, <composition>, <lighting>, <materials/details>, <mood>
```

Example:

```bash
mere.run image generate \
  --prompt "ceramic coffee mug on a walnut desk, product photograph, three-quarter view, soft morning window light, visible steam, matte glaze texture" \
  --output ./mug.png
```

Video prompt pattern:

```text
<subject action>, <camera movement>, <scene>, <lighting>, <motion quality>
```

Example:

```bash
mere.run video generate \
  "slow dolly shot through a rainy neon alley, reflections on pavement, cinematic night lighting, gentle handheld motion" \
  --variant unified-av \
  --output ./neon-alley.mp4
```

Use `--seed` for reproducible variants, dimensions/frame controls only when the user needs a specific format, and image-to-video or image-to-image inputs when preserving an existing visual direction matters.

## Speech

For speech, write text that sounds natural when spoken. Prefer shorter sentences, explicit voice style, and an output path:

```bash
mere.run speech synthesize \
  "Welcome back. Your local model is ready, and the next run should be much faster." \
  --voice "warm, calm narrator with clear studio diction" \
  --output ./welcome.wav
```

If the user wants a reusable voice, use the profile commands rather than repeating long voice descriptions.
