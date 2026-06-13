# Open WebUI

## Purpose

Connect Open WebUI to a local mere.run API server while keeping mere.run as the
runtime provider. Open WebUI is an optional companion UI; do not vendor,
rename, or rebrand it unless a deliberate packaging and license review says so.

## Required Models

- Chat: any installed API-capable `text-chat-*` model, such as
  `text-chat-gemma4-12b`, `text-chat-gemma4`, or `text-chat-q36-nano`.
- Vision chat: `vision-chat-gemma4-12b`.
- RAG embeddings: `text-embed-qwen3-0.6b`.
- Image generation: `image-zimage-nano`.
- Text-to-speech: `speech-tts-qwen3-nano`.
- Speech-to-text: `speech-asr-parakeet`.

## Quickstart Command

Use the quickstart command when you want the same path mere.run uses for live
Open WebUI smoke testing: start the local API, run the official Open WebUI
Docker image, configure the OpenAI-compatible provider, filter the chat picker
to text and vision chat models, and keep image/TTS/STT/RAG sidecars in their
own Open WebUI settings.

```bash
mere.run open-webui quickstart --pull
```

The default chat pair is `text-chat-gemma4-12b` and
`vision-chat-gemma4-12b`. The command binds the mere.run API to `0.0.0.0:8080`
with a bearer key so Docker can reach it at
`http://host.docker.internal:8080/v1`, publishes Open WebUI at
`http://127.0.0.1:3000`, disables image editing for the live smoke path, and
sets Open WebUI RAG to mere.run's `/v1/embeddings`.

Preview the exact commands without starting Docker:

```bash
mere.run open-webui quickstart --dry-run
```

Use an already-running mere.run API and Open WebUI instance when you only want
to apply the admin API configuration:

```bash
mere.run open-webui quickstart \
  --skip-server \
  --skip-docker \
  --host 127.0.0.1 \
  --port 8080 \
  --webui-port 3000
```

If an old Open WebUI test container or volume exists, recreate it:

```bash
mere.run open-webui quickstart --reset
```

## Install And Check

Pull the models you want Open WebUI to see:

```bash
mere.run model pull text-chat-gemma4-12b
mere.run model pull vision-chat-gemma4-12b
mere.run model pull text-embed-qwen3-0.6b
mere.run model pull image-zimage-nano
mere.run model pull speech-tts-qwen3-nano
mere.run model pull speech-asr-parakeet
mere.run model list
```

Start mere.run as the local OpenAI-compatible provider. For Docker bridge
networking, bind the API to the host interface and protect it with a bearer key:

```bash
export MERERUN_API_KEY=change-me
mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-12b \
  --host 0.0.0.0 \
  --port 8080 \
  --api-key "$MERERUN_API_KEY"
```

For a same-host Python/pip Open WebUI install, loopback is enough:

```bash
export MERERUN_API_KEY=change-me
mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-12b \
  --host 127.0.0.1 \
  --port 8080 \
  --api-key "$MERERUN_API_KEY"
```

## Docker Path

The repeatable smoke path uses a dedicated Open WebUI container and volume:

```bash
MERERUN_API_KEY=change-me \
MERERUN_OPENWEBUI_TEXT_MODEL=text-chat-gemma4-12b \
MERERUN_OPENWEBUI_RESET=1 \
scripts/smoke-open-webui.sh docker-run

MERERUN_API_KEY=change-me \
MERERUN_OPENWEBUI_TEXT_MODEL=text-chat-gemma4-12b \
scripts/smoke-open-webui.sh live-smoke
```

`live-smoke` waits for Open WebUI, signs in to the disposable no-auth smoke
admin user, filters the OpenAI chat connection to the configured text and vision
chat models, imports per-model metadata wrappers, sends text and vision chat
through Open WebUI's own proxy, then exercises the direct mere.run API surface
and the Open WebUI model list. Use `configure`, `proxy-smoke`, `api-smoke`, and
`ui-smoke` separately when debugging one stage.

For the fully expanded Docker command, run the official Open WebUI container and
point it at the host's mere.run API:

```bash
DEFAULT_MODEL_METADATA='{"capabilities":{"file_context":true,"vision":false,"file_upload":true,"web_search":false,"image_generation":true,"code_interpreter":false,"terminal":false,"citations":true,"status_updates":true,"builtin_tools":true}}'
docker run -d \
  --name open-webui \
  --restart unless-stopped \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e OPENAI_API_BASE_URLS=http://host.docker.internal:8080/v1 \
  -e OPENAI_API_KEY="$MERERUN_API_KEY" \
  -e OPENAI_API_KEYS="$MERERUN_API_KEY" \
  -e DEFAULT_MODELS=text-chat-gemma4-12b \
  -e DEFAULT_MODEL_PARAMS='{"function_calling":"native"}' \
  -e DEFAULT_MODEL_METADATA="$DEFAULT_MODEL_METADATA" \
  -e RAG_EMBEDDING_ENGINE=openai \
  -e RAG_OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e RAG_OPENAI_API_KEY="$MERERUN_API_KEY" \
  -e RAG_EMBEDDING_MODEL=text-embed-qwen3-0.6b \
  -e ENABLE_IMAGE_GENERATION=True \
  -e ENABLE_IMAGE_EDIT=False \
  -e IMAGE_GENERATION_ENGINE=openai \
  -e IMAGE_GENERATION_MODEL=image-zimage-nano \
  -e IMAGE_SIZE=1024x1024 \
  -e IMAGE_EDIT_ENGINE=openai \
  -e IMAGE_EDIT_MODEL=qwen-image-edit \
  -e IMAGE_EDIT_SIZE=1024x1024 \
  -e IMAGES_OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e IMAGES_OPENAI_API_KEY="$MERERUN_API_KEY" \
  -e IMAGES_EDIT_OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e IMAGES_EDIT_OPENAI_API_KEY="$MERERUN_API_KEY" \
  -e AUDIO_TTS_ENGINE=openai \
  -e AUDIO_TTS_MODEL=speech-tts-qwen3-nano \
  -e AUDIO_TTS_VOICE=nova \
  -e AUDIO_TTS_OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e AUDIO_TTS_OPENAI_API_KEY="$MERERUN_API_KEY" \
  -e AUDIO_TTS_OPENAI_PARAMS='{"response_format":"wav"}' \
  -e AUDIO_STT_ENGINE=openai \
  -e AUDIO_STT_MODEL=speech-asr-parakeet \
  -e AUDIO_STT_OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e AUDIO_STT_OPENAI_API_KEY="$MERERUN_API_KEY" \
  -e ENABLE_PERSISTENT_CONFIG=False \
  -e WEBUI_AUTH=False \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

Open `http://localhost:3000`. The environment variables above preconfigure the
OpenAI-compatible chat provider and RAG embedding provider on a fresh Open WebUI
data volume. Run `scripts/smoke-open-webui.sh configure` after the container is
healthy to set Open WebUI's OpenAI connection `model_ids` filter and import
per-model capability wrappers. That keeps image, embedding, TTS, and STT
sidecars out of the chat selector while still using them in their own settings.
You can also set the same values in the admin UI:

- Base URL: `http://host.docker.internal:8080/v1`
- API key: the value of `MERERUN_API_KEY`
- Chat model: select an installed `text-chat-*` model from `/v1/models`
- Vision model: select `vision-chat-gemma4-12b`
- Embedding model for RAG/knowledge: `text-embed-qwen3-0.6b`
- Image generation engine/model: `openai` / `image-zimage-nano`
- Image editing: disabled for the live smoke path with `ENABLE_IMAGE_EDIT=False`
- Image edit engine/model when enabled: `openai` / `qwen-image-edit`
- TTS engine/model: `openai` / `speech-tts-qwen3-nano`
- STT engine/model: `openai` / `speech-asr-parakeet`
- Function calling: set model params to native mode with
  `{"function_calling":"native"}`

If Docker cannot resolve `host.docker.internal` on your Linux install, keep the
same Open WebUI settings and add the `--add-host=host.docker.internal:host-gateway`
flag shown above. If your Docker runtime still cannot reach the host, use the
host's LAN IP in the Base URL and keep the mere.run API key enabled.
If an existing Open WebUI volume ignores changed environment variables, update
the values in the admin UI, set `ENABLE_PERSISTENT_CONFIG=False` for smoke, or
start with a fresh data volume.

## Docker Compose / DGX Spark Path

Use Compose for a longer-lived Open WebUI companion on Linux, a LAN workstation,
or DGX Spark. Keep mere.run itself on the host so it owns the local model store
and GPU/runtime setup, then point the Open WebUI container at the host's
mere.run `/v1` API.

Start the mere.run provider first:

```bash
export MERERUN_API_KEY=change-me
mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-12b \
  --host 0.0.0.0 \
  --port 8080 \
  --api-key "$MERERUN_API_KEY"
```

Create a small Compose directory:

```bash
mkdir -p open-webui-mere-run
cd open-webui-mere-run
openssl rand -hex 32
```

Write `.env` and replace the secret plus URL for your machine:

```dotenv
MERERUN_API_KEY=change-me
WEBUI_SECRET_KEY=replace-with-openssl-rand-hex-32
OPEN_WEBUI_IMAGE=ghcr.io/open-webui/open-webui:main
OPEN_WEBUI_BIND=0.0.0.0
OPEN_WEBUI_URL=http://spark.local:3000
MERERUN_OPENWEBUI_API_URL=http://host.docker.internal:8080/v1
MERERUN_OPENWEBUI_TEXT_MODEL=text-chat-gemma4-12b
MERERUN_OPENWEBUI_VISION_MODEL=vision-chat-gemma4-12b
```

For Open WebUI running on the same Linux/Spark host as mere.run, keep
`host.docker.internal` and the `extra_hosts` entry below. If Open WebUI runs on
a different machine, set `MERERUN_OPENWEBUI_API_URL` to the mere.run host's LAN
URL, for example `http://192.168.1.50:8080/v1`, and keep the API key enabled.

Write `compose.yaml`:

```yaml
services:
  open-webui:
    image: ${OPEN_WEBUI_IMAGE:-ghcr.io/open-webui/open-webui:main}
    container_name: open-webui
    restart: unless-stopped
    ports:
      - "${OPEN_WEBUI_BIND:-0.0.0.0}:3000:8080"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      WEBUI_URL: ${OPEN_WEBUI_URL:-http://localhost:3000}
      WEBUI_AUTH: "True"
      WEBUI_SECRET_KEY: ${WEBUI_SECRET_KEY}
      ENABLE_PERSISTENT_CONFIG: "True"
      OPENAI_API_BASE_URL: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      OPENAI_API_BASE_URLS: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      OPENAI_API_KEY: ${MERERUN_API_KEY}
      OPENAI_API_KEYS: ${MERERUN_API_KEY}
      DEFAULT_MODELS: ${MERERUN_OPENWEBUI_TEXT_MODEL:-text-chat-gemma4-12b}
      DEFAULT_MODEL_PARAMS: '{"function_calling":"native"}'
      DEFAULT_MODEL_METADATA: '{"capabilities":{"file_context":true,"vision":false,"file_upload":true,"web_search":false,"image_generation":true,"code_interpreter":false,"terminal":false,"citations":true,"status_updates":true,"builtin_tools":true}}'
      RAG_EMBEDDING_ENGINE: openai
      RAG_OPENAI_API_BASE_URL: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      RAG_OPENAI_API_KEY: ${MERERUN_API_KEY}
      RAG_EMBEDDING_MODEL: text-embed-qwen3-0.6b
      ENABLE_IMAGE_GENERATION: "True"
      ENABLE_IMAGE_EDIT: "False"
      IMAGE_GENERATION_ENGINE: openai
      IMAGE_GENERATION_MODEL: image-zimage-nano
      IMAGE_SIZE: 1024x1024
      IMAGE_EDIT_ENGINE: openai
      IMAGE_EDIT_MODEL: qwen-image-edit
      IMAGE_EDIT_SIZE: 1024x1024
      IMAGES_OPENAI_API_BASE_URL: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      IMAGES_OPENAI_API_KEY: ${MERERUN_API_KEY}
      IMAGES_EDIT_OPENAI_API_BASE_URL: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      IMAGES_EDIT_OPENAI_API_KEY: ${MERERUN_API_KEY}
      AUDIO_TTS_ENGINE: openai
      AUDIO_TTS_MODEL: speech-tts-qwen3-nano
      AUDIO_TTS_VOICE: nova
      AUDIO_TTS_OPENAI_API_BASE_URL: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      AUDIO_TTS_OPENAI_API_KEY: ${MERERUN_API_KEY}
      AUDIO_TTS_OPENAI_PARAMS: '{"response_format":"wav"}'
      AUDIO_STT_ENGINE: openai
      AUDIO_STT_MODEL: speech-asr-parakeet
      AUDIO_STT_OPENAI_API_BASE_URL: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      AUDIO_STT_OPENAI_API_KEY: ${MERERUN_API_KEY}
    volumes:
      - open-webui:/app/backend/data

volumes:
  open-webui:
```

Start it:

```bash
docker compose up -d
docker compose logs -f open-webui
```

Open the `OPEN_WEBUI_URL` value from `.env`, create the first admin account,
then run the configure step if you want the same-host per-model wrappers
as the quickstart path:

```bash
mere.run open-webui quickstart \
  --skip-server \
  --skip-docker \
  --host 127.0.0.1 \
  --port 8080 \
  --webui-port 3000 \
  --api-key "$MERERUN_API_KEY" \
  --admin-email you@example.com \
  --admin-password your-open-webui-password
```

For Open WebUI on a different machine, use the admin UI to keep Base URL set to
that machine's `MERERUN_OPENWEBUI_API_URL` value.

For team or production-like deployments, pin `OPEN_WEBUI_IMAGE` to a release tag
after testing it, back up the `open-webui` volume before upgrades, and avoid
`WEBUI_AUTH=False` on a LAN.

## Pip Path

Run Open WebUI directly on the same host:

```bash
python3.11 -m pip install --upgrade open-webui
DEFAULT_MODEL_METADATA='{"capabilities":{"file_context":true,"vision":false,"file_upload":true,"web_search":false,"image_generation":true,"code_interpreter":false,"terminal":false,"citations":true,"status_updates":true,"builtin_tools":true}}'
OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_API_BASE_URLS=http://127.0.0.1:8080/v1 \
OPENAI_API_KEY="$MERERUN_API_KEY" \
OPENAI_API_KEYS="$MERERUN_API_KEY" \
DEFAULT_MODELS=text-chat-gemma4-12b \
DEFAULT_MODEL_PARAMS='{"function_calling":"native"}' \
DEFAULT_MODEL_METADATA="$DEFAULT_MODEL_METADATA" \
RAG_EMBEDDING_ENGINE=openai \
RAG_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
RAG_OPENAI_API_KEY="$MERERUN_API_KEY" \
RAG_EMBEDDING_MODEL=text-embed-qwen3-0.6b \
ENABLE_IMAGE_GENERATION=True \
ENABLE_IMAGE_EDIT=False \
IMAGE_GENERATION_ENGINE=openai \
IMAGE_GENERATION_MODEL=image-zimage-nano \
IMAGE_SIZE=1024x1024 \
IMAGE_EDIT_ENGINE=openai \
IMAGE_EDIT_MODEL=qwen-image-edit \
IMAGE_EDIT_SIZE=1024x1024 \
IMAGES_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
IMAGES_OPENAI_API_KEY="$MERERUN_API_KEY" \
IMAGES_EDIT_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
IMAGES_EDIT_OPENAI_API_KEY="$MERERUN_API_KEY" \
AUDIO_TTS_ENGINE=openai \
AUDIO_TTS_MODEL=speech-tts-qwen3-nano \
AUDIO_TTS_VOICE=nova \
AUDIO_TTS_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
AUDIO_TTS_OPENAI_API_KEY="$MERERUN_API_KEY" \
AUDIO_TTS_OPENAI_PARAMS='{"response_format":"wav"}' \
AUDIO_STT_ENGINE=openai \
AUDIO_STT_MODEL=speech-asr-parakeet \
AUDIO_STT_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
AUDIO_STT_OPENAI_API_KEY="$MERERUN_API_KEY" \
ENABLE_PERSISTENT_CONFIG=False \
WEBUI_AUTH=False \
open-webui serve --host 127.0.0.1 --port 3000
```

Open `http://127.0.0.1:3000`. The environment variables above preconfigure the
same connection values on a fresh Open WebUI data directory. You can also set
them in the admin UI:

- Base URL: `http://127.0.0.1:8080/v1`
- API key: the value of `MERERUN_API_KEY`
- Chat model: select an installed `text-chat-*` model from `/v1/models`
- Vision model: select `vision-chat-gemma4-12b`
- Embedding model for RAG/knowledge: `text-embed-qwen3-0.6b`
- Image generation engine/model: `openai` / `image-zimage-nano`
- Image editing: disabled for the live smoke path with `ENABLE_IMAGE_EDIT=False`
- Image edit engine/model when enabled: `openai` / `qwen-image-edit`
- TTS engine/model: `openai` / `speech-tts-qwen3-nano`
- STT engine/model: `openai` / `speech-asr-parakeet`
- Function calling: set model params to native mode with
  `{"function_calling":"native"}`

For pip-based smoke configuration, use the same-host provider URL:

```bash
OPEN_WEBUI_MERERUN_API_URL=http://127.0.0.1:8080/v1 \
scripts/smoke-open-webui.sh configure
```

## uv Path

Use `uvx` when you want a disposable same-host Open WebUI process without
managing a virtualenv by hand. This is the same connection shape as the pip
path: mere.run stays on loopback and Open WebUI talks to `127.0.0.1`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export DATA_DIR="$HOME/.open-webui-mere-run"
export MERERUN_API_KEY=change-me

mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-12b \
  --host 127.0.0.1 \
  --port 8080 \
  --api-key "$MERERUN_API_KEY"
```

In another terminal:

```bash
DEFAULT_MODEL_METADATA='{"capabilities":{"file_context":true,"vision":false,"file_upload":true,"web_search":false,"image_generation":true,"code_interpreter":false,"terminal":false,"citations":true,"status_updates":true,"builtin_tools":true}}'
DATA_DIR="$HOME/.open-webui-mere-run" \
OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_API_BASE_URLS=http://127.0.0.1:8080/v1 \
OPENAI_API_KEY="$MERERUN_API_KEY" \
OPENAI_API_KEYS="$MERERUN_API_KEY" \
DEFAULT_MODELS=text-chat-gemma4-12b \
DEFAULT_MODEL_PARAMS='{"function_calling":"native"}' \
DEFAULT_MODEL_METADATA="$DEFAULT_MODEL_METADATA" \
RAG_EMBEDDING_ENGINE=openai \
RAG_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
RAG_OPENAI_API_KEY="$MERERUN_API_KEY" \
RAG_EMBEDDING_MODEL=text-embed-qwen3-0.6b \
ENABLE_IMAGE_GENERATION=True \
ENABLE_IMAGE_EDIT=False \
IMAGE_GENERATION_ENGINE=openai \
IMAGE_GENERATION_MODEL=image-zimage-nano \
IMAGE_SIZE=1024x1024 \
IMAGES_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
IMAGES_OPENAI_API_KEY="$MERERUN_API_KEY" \
AUDIO_TTS_ENGINE=openai \
AUDIO_TTS_MODEL=speech-tts-qwen3-nano \
AUDIO_TTS_VOICE=nova \
AUDIO_TTS_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
AUDIO_TTS_OPENAI_API_KEY="$MERERUN_API_KEY" \
AUDIO_TTS_OPENAI_PARAMS='{"response_format":"wav"}' \
AUDIO_STT_ENGINE=openai \
AUDIO_STT_MODEL=speech-asr-parakeet \
AUDIO_STT_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
AUDIO_STT_OPENAI_API_KEY="$MERERUN_API_KEY" \
ENABLE_PERSISTENT_CONFIG=False \
WEBUI_AUTH=False \
uvx --python 3.11 open-webui@latest serve --host 127.0.0.1 --port 3000
```

Open `http://127.0.0.1:3000`. If you omit `--port 3000`, current Open WebUI
`uvx` examples default to port `8080`, which conflicts with the mere.run API
recipe above.

## Model Metadata

Open WebUI defaults many capabilities to enabled. For mere.run, start with
conservative defaults so text models do not advertise web search, code
interpreter, terminal access, or vision by accident:

```json
{
  "capabilities": {
    "file_context": true,
    "vision": false,
    "file_upload": true,
    "web_search": false,
    "image_generation": true,
    "code_interpreter": false,
    "terminal": false,
    "citations": true,
    "status_updates": true,
    "builtin_tools": true
  }
}
```

For the specific `vision-chat-gemma4-12b` model, set the per-model capability
override `vision=true` in Workspace/Admin Models before testing image uploads.
Keep `function_calling=native` in the model params for Open WebUI tools,
knowledge tools, and agentic RAG.

The smoke helper applies those wrappers for you:

```bash
scripts/smoke-open-webui.sh configure
```

Set `OPEN_WEBUI_CHAT_MODELS` to a semicolon-separated list when you want a
different chat selector allowlist, for example
`OPEN_WEBUI_CHAT_MODELS='text-chat-gemma4-12b;vision-chat-gemma4-12b'`.

## Smoke Tests

Check the provider before opening the UI:

```bash
curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer $MERERUN_API_KEY"
```

Text chat:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "text-chat-gemma4",
    "messages": [{"role": "user", "content": "Reply with only: mere.run online"}],
    "max_tokens": 16
  }'
```

Embeddings for RAG:

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "text-embed-qwen3-0.6b",
    "input": ["mere.run native embeddings", "Open WebUI RAG"]
}'
```

Image generation:

```bash
curl http://127.0.0.1:8080/v1/images/generations \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "image-zimage-nano",
    "prompt": "a compact local AI workstation in morning light",
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

Text-to-speech:

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -H "Content-Type: application/json" \
  --output speech.wav \
  --data '{
    "model": "speech-tts-qwen3-nano",
    "input": "Open WebUI is speaking through mere.run.",
    "voice": "nova",
    "response_format": "wav"
  }'
```

Speech-to-text:

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -F model=speech-asr-parakeet \
  -F response_format=json \
  -F file=@speech.wav
```

Vision chat, using a local image encoded as a data URL:

```bash
IMAGE_DATA_URL="$(python3 - <<'PY'
import base64
from pathlib import Path

print("data:image/png;base64," + base64.b64encode(Path("smoke.png").read_bytes()).decode())
PY
)"
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -H "Content-Type: application/json" \
  --data "{
    \"model\": \"vision-chat-gemma4-12b\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Describe this image in one sentence.\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"$IMAGE_DATA_URL\"}}
      ]
    }],
    \"max_tokens\": 64
  }"
```

Then smoke Open WebUI itself:

1. Start a new chat and pick each installed `text-chat-*` model you want to
   support.
2. Send a one-line prompt and confirm the response comes from the local mere.run
   server logs.
3. Pick `vision-chat-gemma4-12b`, upload a local image, and confirm the model
   answers about the image. If the upload control is hidden, set that model's
   per-model `vision` capability to `true`.
4. Create or update a knowledge/RAG collection with the embedding model set to
   `text-embed-qwen3-0.6b`, then ask a question that retrieves from it.
5. Generate an image and confirm Open WebUI receives a rendered PNG.
6. Keep image editing disabled during the first live smoke; enable it later if
   you want to exercise `/v1/images/edits` with `qwen-image-edit`.
7. Use voice playback and audio upload/transcription with the OpenAI-compatible
   audio engines set to the mere.run base URL.
8. Attach a small knowledge base and enable native function calling to exercise
   Open WebUI's knowledge/tool path against mere.run `tool_calls`.

## Troubleshooting

- Open WebUI says no models: check `curl /v1/models` with the same Base URL and
  API key configured in Open WebUI.
- Docker cannot reach mere.run: use `--host 0.0.0.0` on the mere.run side and
  `http://host.docker.internal:8080/v1` on the Open WebUI side.
- RAG asks for another embedding provider: set the OpenAI-compatible embedding
  endpoint to the same mere.run Base URL and use `text-embed-qwen3-0.6b`.
- Image generation fails: set `ENABLE_IMAGE_GENERATION=True`,
  `IMAGE_GENERATION_ENGINE=openai`, and point `IMAGES_OPENAI_API_BASE_URL` at
  the same mere.run `/v1` base URL.
- Image editing controls appear during smoke: set `ENABLE_IMAGE_EDIT=False`.
- Image editing fails after enabling it: set `IMAGE_EDIT_ENGINE=openai`,
  `IMAGE_EDIT_MODEL=qwen-image-edit`, and point `IMAGES_EDIT_OPENAI_API_BASE_URL`
  at the same mere.run `/v1` base URL.
- Tools or knowledge do not run: set `DEFAULT_MODEL_PARAMS` or the per-model
  params to `{"function_calling":"native"}`.
- TTS fails because `ffmpeg` is unavailable: keep
  `AUDIO_TTS_OPENAI_PARAMS='{"response_format":"wav"}'` so Open WebUI asks for
  native Qwen3-TTS WAV output. For MP3/Opus/AAC/FLAC, install `ffmpeg`.
- STT upload fails: set `AUDIO_STT_ENGINE=openai`,
  `AUDIO_STT_MODEL=speech-asr-parakeet`, and upload an audio file as multipart
  form data.
- Vision uploads fail: use `vision-chat-gemma4-12b`; the local runtime accepts
  one image content part per message as a file path, `file://` URL, or base64
  data URL.
- Non-loopback bind rejected: set `MERERUN_API_KEY` or pass `--api-key`.

## Sources

- https://github.com/open-webui/open-webui
- https://docs.openwebui.com/getting-started/quick-start/
- https://docs.openwebui.com/openai-compatible/
- https://docs.openwebui.com/reference/env-configuration/
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/APIServeCommand.swift
