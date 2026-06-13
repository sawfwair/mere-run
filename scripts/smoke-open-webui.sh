#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MERERUN_API_URL="${MERERUN_API_URL:-http://127.0.0.1:8080/v1}"
MERERUN_DOCKER_API_URL="${MERERUN_DOCKER_API_URL:-http://host.docker.internal:8080/v1}"
MERERUN_API_KEY="${MERERUN_API_KEY:-change-me}"

OPEN_WEBUI_CONTAINER="${OPEN_WEBUI_CONTAINER:-open-webui-mere-run-smoke}"
OPEN_WEBUI_VOLUME="${OPEN_WEBUI_VOLUME:-open-webui-mere-run-smoke}"
OPEN_WEBUI_IMAGE="${OPEN_WEBUI_IMAGE:-ghcr.io/open-webui/open-webui:main}"
OPEN_WEBUI_PORT="${OPEN_WEBUI_PORT:-3000}"
OPEN_WEBUI_URL="${OPEN_WEBUI_URL:-http://127.0.0.1:${OPEN_WEBUI_PORT}}"
OPEN_WEBUI_AUTH_EMAIL="${OPEN_WEBUI_AUTH_EMAIL:-admin@localhost}"
OPEN_WEBUI_AUTH_PASSWORD="${OPEN_WEBUI_AUTH_PASSWORD:-admin}"
OPEN_WEBUI_TOKEN="${OPEN_WEBUI_TOKEN:-}"
OPEN_WEBUI_WAIT_SECONDS="${OPEN_WEBUI_WAIT_SECONDS:-120}"
OPEN_WEBUI_MERERUN_API_URL="${OPEN_WEBUI_MERERUN_API_URL:-${MERERUN_DOCKER_API_URL}}"

MERERUN_OPENWEBUI_TEXT_MODEL="${MERERUN_OPENWEBUI_TEXT_MODEL:-text-chat-gemma4-12b}"
MERERUN_OPENWEBUI_VISION_MODEL="${MERERUN_OPENWEBUI_VISION_MODEL:-vision-chat-gemma4-12b}"
MERERUN_OPENWEBUI_EMBED_MODEL="${MERERUN_OPENWEBUI_EMBED_MODEL:-text-embed-qwen3-0.6b}"
MERERUN_OPENWEBUI_IMAGE_MODEL="${MERERUN_OPENWEBUI_IMAGE_MODEL:-image-zimage-nano}"
MERERUN_OPENWEBUI_IMAGE_EDIT_MODEL="${MERERUN_OPENWEBUI_IMAGE_EDIT_MODEL:-qwen-image-edit}"
MERERUN_OPENWEBUI_TTS_MODEL="${MERERUN_OPENWEBUI_TTS_MODEL:-speech-tts-qwen3-nano}"
MERERUN_OPENWEBUI_STT_MODEL="${MERERUN_OPENWEBUI_STT_MODEL:-speech-asr-parakeet}"
MERERUN_OPENWEBUI_TTS_FORMAT="${MERERUN_OPENWEBUI_TTS_FORMAT:-wav}"
OPEN_WEBUI_CHAT_MODELS="${OPEN_WEBUI_CHAT_MODELS:-${MERERUN_OPENWEBUI_TEXT_MODEL};${MERERUN_OPENWEBUI_VISION_MODEL}}"

DEFAULT_MODEL_METADATA_JSON='{"capabilities":{"file_context":true,"vision":false,"file_upload":true,"web_search":false,"image_generation":true,"code_interpreter":false,"terminal":false,"citations":true,"status_updates":true,"builtin_tools":true}}'
DEFAULT_MODEL_PARAMS_JSON='{"function_calling":"native"}'
OPEN_WEBUI_MODEL_METADATA="${OPEN_WEBUI_MODEL_METADATA:-$DEFAULT_MODEL_METADATA_JSON}"
OPEN_WEBUI_MODEL_PARAMS="${OPEN_WEBUI_MODEL_PARAMS:-$DEFAULT_MODEL_PARAMS_JSON}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/smoke-open-webui.sh print-env
  scripts/smoke-open-webui.sh docker-run
  scripts/smoke-open-webui.sh configure
  scripts/smoke-open-webui.sh proxy-smoke
  scripts/smoke-open-webui.sh api-smoke
  scripts/smoke-open-webui.sh ui-smoke
  scripts/smoke-open-webui.sh live-smoke
  scripts/smoke-open-webui.sh stop

Environment:
  MERERUN_API_URL=http://127.0.0.1:8080/v1
  MERERUN_DOCKER_API_URL=http://host.docker.internal:8080/v1
  MERERUN_API_KEY=change-me
  OPEN_WEBUI_URL=http://127.0.0.1:3000
  OPEN_WEBUI_AUTH_EMAIL=admin@localhost
  OPEN_WEBUI_AUTH_PASSWORD=admin
  OPEN_WEBUI_TOKEN=                    # optional existing admin JWT/API token
  OPEN_WEBUI_MERERUN_API_URL=http://host.docker.internal:8080/v1
  MERERUN_OPENWEBUI_TEXT_MODEL=text-chat-gemma4-12b
  MERERUN_OPENWEBUI_VISION_MODEL=vision-chat-gemma4-12b
  MERERUN_OPENWEBUI_EMBED_MODEL=text-embed-qwen3-0.6b
  MERERUN_OPENWEBUI_IMAGE_MODEL=image-zimage-nano
  MERERUN_OPENWEBUI_IMAGE_EDIT_MODEL=qwen-image-edit
  MERERUN_OPENWEBUI_TTS_MODEL=speech-tts-qwen3-nano
  MERERUN_OPENWEBUI_STT_MODEL=speech-asr-parakeet
  MERERUN_OPENWEBUI_TTS_FORMAT=wav
  OPEN_WEBUI_CHAT_MODELS='text-chat-gemma4;vision-chat-gemma4-12b'
  MERERUN_OPENWEBUI_RESET=1        # remove the smoke container and volume first

Start mere.run first, for example:
  MERERUN_API_KEY=change-me mere.run api serve --engine text-chat-gemma4 --model text-chat-gemma4-12b --host 0.0.0.0 --port 8080 --api-key "$MERERUN_API_KEY"
USAGE
}

curl_json() {
  local method="$1"
  local path="$2"
  local data="${3:-}"
  local url="${MERERUN_API_URL%/}${path}"
  if [[ -n "${data}" ]]; then
    curl -fsS -X "${method}" "${url}" \
      -H "Authorization: Bearer ${MERERUN_API_KEY}" \
      -H "Content-Type: application/json" \
      --data "${data}"
  else
    curl -fsS -X "${method}" "${url}" \
      -H "Authorization: Bearer ${MERERUN_API_KEY}"
  fi
}

openwebui_json() {
  local method="$1"
  local path="$2"
  local token="$3"
  local data="${4:-}"
  local url="${OPEN_WEBUI_URL%/}${path}"
  if [[ -n "${data}" ]]; then
    curl -fsS -X "${method}" "${url}" \
      -H "Authorization: Bearer ${token}" \
      -H "Content-Type: application/json" \
      --data "${data}"
  else
    curl -fsS -X "${method}" "${url}" \
      -H "Authorization: Bearer ${token}" \
      -H "Content-Type: application/json"
  fi
}

wait_openwebui() {
  local health_url="${OPEN_WEBUI_URL%/}/health"
  local deadline=$((SECONDS + OPEN_WEBUI_WAIT_SECONDS))
  echo "Waiting for Open WebUI at ${health_url}"
  until curl -fsS "${health_url}" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for Open WebUI at ${health_url}" >&2
      exit 1
    fi
    sleep 2
  done
}

openwebui_token() {
  if [[ -n "${OPEN_WEBUI_TOKEN}" ]]; then
    printf '%s\n' "${OPEN_WEBUI_TOKEN}"
    return
  fi

  local signin_payload
  signin_payload="$(python3 - "${OPEN_WEBUI_AUTH_EMAIL}" "${OPEN_WEBUI_AUTH_PASSWORD}" <<'PY'
import json
import sys

print(json.dumps({"email": sys.argv[1], "password": sys.argv[2]}))
PY
)"

  local signin_response
  signin_response="$(curl -fsS -X POST "${OPEN_WEBUI_URL%/}/api/v1/auths/signin" \
    -H "Content-Type: application/json" \
    --data "${signin_payload}")"

  python3 -c 'import json, sys
payload = json.load(sys.stdin)
token = payload.get("token")
if not token:
    raise SystemExit("Open WebUI signin did not return a token")
print(token)' <<<"${signin_response}"
}

chat_models_json() {
  local models_file="$1"
  python3 - "${models_file}" "${OPEN_WEBUI_CHAT_MODELS}" <<'PY'
import json
import re
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)

available = {item.get("id") for item in payload.get("data", []) if item.get("id")}
requested = [item for item in re.split(r"[;,]", sys.argv[2]) if item]
selected = [item for item in requested if item in available]
if not selected:
    selected = requested
print(json.dumps(selected))
PY
}

configure_openwebui_payloads() {
  local output_dir="$1"
  local chat_models="$2"
  python3 - "${output_dir}" \
    "${OPEN_WEBUI_MERERUN_API_URL}" \
    "${MERERUN_API_KEY}" \
    "${MERERUN_OPENWEBUI_TEXT_MODEL}" \
    "${MERERUN_OPENWEBUI_VISION_MODEL}" \
    "${OPEN_WEBUI_MODEL_METADATA}" \
    "${OPEN_WEBUI_MODEL_PARAMS}" \
    "${chat_models}" <<'PY'
import copy
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
provider_api_url = sys.argv[2]
api_key = sys.argv[3]
text_model = sys.argv[4]
vision_model = sys.argv[5]
default_metadata = json.loads(sys.argv[6] or "{}")
default_params = json.loads(sys.argv[7] or "{}")
chat_models = json.loads(sys.argv[8])

openai_config = {
    "ENABLE_OPENAI_API": True,
    "OPENAI_API_BASE_URLS": [provider_api_url],
    "OPENAI_API_KEYS": [api_key],
    "OPENAI_API_CONFIGS": {
        "0": {
            "enable": True,
            "model_ids": chat_models,
        }
    },
}

models_config = {
    "DEFAULT_MODELS": text_model,
    "DEFAULT_PINNED_MODELS": text_model,
    "MODEL_ORDER_LIST": chat_models,
    "DEFAULT_MODEL_METADATA": default_metadata,
    "DEFAULT_MODEL_PARAMS": default_params,
}

def wrapper(model_id: str, *, vision: bool, name: str) -> dict:
    metadata = copy.deepcopy(default_metadata)
    capabilities = dict(metadata.get("capabilities") or {})
    capabilities["vision"] = vision
    capabilities["file_upload"] = True
    metadata["capabilities"] = capabilities
    metadata["description"] = f"{name} served locally by mere.run."
    return {
        "id": model_id,
        "base_model_id": model_id,
        "name": name,
        "meta": metadata,
        "params": default_params,
        "is_active": True,
    }

models = []
if text_model in chat_models:
    models.append(wrapper(text_model, vision=False, name=f"mere.run {text_model}"))
if vision_model in chat_models:
    models.append(wrapper(vision_model, vision=True, name=f"mere.run {vision_model}"))

(output_dir / "openai-config.json").write_text(json.dumps(openai_config), encoding="utf-8")
(output_dir / "models-config.json").write_text(json.dumps(models_config), encoding="utf-8")
(output_dir / "models-import.json").write_text(json.dumps({"models": models}), encoding="utf-8")
PY
}

configure_openwebui() {
  wait_openwebui
  local work_dir
  work_dir="$(mktemp -d "${TMPDIR:-/tmp}/mere-openwebui-config.XXXXXX")"

  local models_file="${work_dir}/mere-models.json"
  echo "Checking mere.run provider models for Open WebUI chat filtering"
  curl_json GET "/models" >"${models_file}"

  local chat_models
  chat_models="$(chat_models_json "${models_file}")"
  echo "Configuring Open WebUI chat models: ${chat_models}"

  local token
  token="$(openwebui_token)"

  configure_openwebui_payloads "${work_dir}" "${chat_models}"

  openwebui_json POST "/openai/config/update" "${token}" "$(cat "${work_dir}/openai-config.json")" >/dev/null
  openwebui_json POST "/api/v1/configs/models" "${token}" "$(cat "${work_dir}/models-config.json")" >/dev/null
  openwebui_json POST "/api/v1/models/import" "${token}" "$(cat "${work_dir}/models-import.json")" >/dev/null

  openwebui_json GET "/api/models" "${token}" >"${work_dir}/openwebui-models.json"
  python3 - "${work_dir}/openwebui-models.json" "${chat_models}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)
expected = set(json.loads(sys.argv[2]))
ids = {item.get("id") for item in payload.get("data", [])}
missing = sorted(expected - ids)
if missing:
    raise SystemExit(f"Open WebUI model list is missing configured ids: {', '.join(missing)}")
print("Open WebUI models configured:", ", ".join(sorted(expected)))
PY
  rm -rf "${work_dir}"
}

model_present() {
  local models_file="$1"
  local model_id="$2"
  python3 - "$models_file" "$model_id" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)
ids = {item.get("id") for item in payload.get("data", [])}
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
}

write_smoke_png() {
  local output="$1"
  python3 - "$output" <<'PY'
import base64
import sys

png = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
with open(sys.argv[1], "wb") as fh:
    fh.write(base64.b64decode(png))
PY
}

print_env() {
  cat <<ENV
OPENAI_API_BASE_URL=${MERERUN_API_URL}
OPENAI_API_BASE_URLS=${MERERUN_API_URL}
OPENAI_API_KEY=${MERERUN_API_KEY}
OPENAI_API_KEYS=${MERERUN_API_KEY}
DEFAULT_MODELS=${MERERUN_OPENWEBUI_TEXT_MODEL}
DEFAULT_MODEL_PARAMS=${OPEN_WEBUI_MODEL_PARAMS}
DEFAULT_MODEL_METADATA=${OPEN_WEBUI_MODEL_METADATA}
RAG_EMBEDDING_ENGINE=openai
RAG_OPENAI_API_BASE_URL=${MERERUN_API_URL}
RAG_OPENAI_API_KEY=${MERERUN_API_KEY}
RAG_EMBEDDING_MODEL=${MERERUN_OPENWEBUI_EMBED_MODEL}
ENABLE_IMAGE_GENERATION=True
ENABLE_IMAGE_EDIT=False
IMAGE_GENERATION_ENGINE=openai
IMAGE_GENERATION_MODEL=${MERERUN_OPENWEBUI_IMAGE_MODEL}
IMAGE_SIZE=1024x1024
IMAGE_EDIT_ENGINE=openai
IMAGE_EDIT_MODEL=${MERERUN_OPENWEBUI_IMAGE_EDIT_MODEL}
IMAGE_EDIT_SIZE=1024x1024
IMAGES_OPENAI_API_BASE_URL=${MERERUN_API_URL}
IMAGES_OPENAI_API_KEY=${MERERUN_API_KEY}
IMAGES_EDIT_OPENAI_API_BASE_URL=${MERERUN_API_URL}
IMAGES_EDIT_OPENAI_API_KEY=${MERERUN_API_KEY}
AUDIO_TTS_ENGINE=openai
AUDIO_TTS_MODEL=${MERERUN_OPENWEBUI_TTS_MODEL}
AUDIO_TTS_VOICE=nova
AUDIO_TTS_OPENAI_API_BASE_URL=${MERERUN_API_URL}
AUDIO_TTS_OPENAI_API_KEY=${MERERUN_API_KEY}
AUDIO_TTS_OPENAI_PARAMS={"response_format":"${MERERUN_OPENWEBUI_TTS_FORMAT}"}
AUDIO_STT_ENGINE=openai
AUDIO_STT_MODEL=${MERERUN_OPENWEBUI_STT_MODEL}
AUDIO_STT_OPENAI_API_BASE_URL=${MERERUN_API_URL}
AUDIO_STT_OPENAI_API_KEY=${MERERUN_API_KEY}
ENABLE_PERSISTENT_CONFIG=False
WEBUI_AUTH=False
ENV
}

docker_run() {
  command -v docker >/dev/null 2>&1 || {
    echo "docker is required for docker-run" >&2
    exit 127
  }

  if [[ "${MERERUN_OPENWEBUI_RESET:-0}" == "1" ]]; then
    docker rm -f "${OPEN_WEBUI_CONTAINER}" >/dev/null 2>&1 || true
    docker volume rm "${OPEN_WEBUI_VOLUME}" >/dev/null 2>&1 || true
  elif docker ps -a --format '{{.Names}}' | grep -Fxq "${OPEN_WEBUI_CONTAINER}"; then
    echo "Container ${OPEN_WEBUI_CONTAINER} already exists. Set MERERUN_OPENWEBUI_RESET=1 to recreate it." >&2
    exit 1
  fi

  docker run -d \
    --name "${OPEN_WEBUI_CONTAINER}" \
    --restart unless-stopped \
    -p "${OPEN_WEBUI_PORT}:8080" \
    --add-host=host.docker.internal:host-gateway \
    -e OPENAI_API_BASE_URL="${MERERUN_DOCKER_API_URL}" \
    -e OPENAI_API_BASE_URLS="${MERERUN_DOCKER_API_URL}" \
    -e OPENAI_API_KEY="${MERERUN_API_KEY}" \
    -e OPENAI_API_KEYS="${MERERUN_API_KEY}" \
    -e DEFAULT_MODELS="${MERERUN_OPENWEBUI_TEXT_MODEL}" \
    -e DEFAULT_MODEL_PARAMS="${OPEN_WEBUI_MODEL_PARAMS}" \
    -e DEFAULT_MODEL_METADATA="${OPEN_WEBUI_MODEL_METADATA}" \
    -e RAG_EMBEDDING_ENGINE=openai \
    -e RAG_OPENAI_API_BASE_URL="${MERERUN_DOCKER_API_URL}" \
    -e RAG_OPENAI_API_KEY="${MERERUN_API_KEY}" \
    -e RAG_EMBEDDING_MODEL="${MERERUN_OPENWEBUI_EMBED_MODEL}" \
    -e ENABLE_IMAGE_GENERATION=True \
    -e ENABLE_IMAGE_EDIT=False \
    -e IMAGE_GENERATION_ENGINE=openai \
    -e IMAGE_GENERATION_MODEL="${MERERUN_OPENWEBUI_IMAGE_MODEL}" \
    -e IMAGE_SIZE=1024x1024 \
    -e IMAGE_EDIT_ENGINE=openai \
    -e IMAGE_EDIT_MODEL="${MERERUN_OPENWEBUI_IMAGE_EDIT_MODEL}" \
    -e IMAGE_EDIT_SIZE=1024x1024 \
    -e IMAGES_OPENAI_API_BASE_URL="${MERERUN_DOCKER_API_URL}" \
    -e IMAGES_OPENAI_API_KEY="${MERERUN_API_KEY}" \
    -e IMAGES_EDIT_OPENAI_API_BASE_URL="${MERERUN_DOCKER_API_URL}" \
    -e IMAGES_EDIT_OPENAI_API_KEY="${MERERUN_API_KEY}" \
    -e AUDIO_TTS_ENGINE=openai \
    -e AUDIO_TTS_MODEL="${MERERUN_OPENWEBUI_TTS_MODEL}" \
    -e AUDIO_TTS_VOICE=nova \
    -e AUDIO_TTS_OPENAI_API_BASE_URL="${MERERUN_DOCKER_API_URL}" \
    -e AUDIO_TTS_OPENAI_API_KEY="${MERERUN_API_KEY}" \
    -e AUDIO_TTS_OPENAI_PARAMS="{\"response_format\":\"${MERERUN_OPENWEBUI_TTS_FORMAT}\"}" \
    -e AUDIO_STT_ENGINE=openai \
    -e AUDIO_STT_MODEL="${MERERUN_OPENWEBUI_STT_MODEL}" \
    -e AUDIO_STT_OPENAI_API_BASE_URL="${MERERUN_DOCKER_API_URL}" \
    -e AUDIO_STT_OPENAI_API_KEY="${MERERUN_API_KEY}" \
    -e ENABLE_PERSISTENT_CONFIG=False \
    -e WEBUI_AUTH=False \
    -v "${OPEN_WEBUI_VOLUME}:/app/backend/data" \
    "${OPEN_WEBUI_IMAGE}"

  echo "Open WebUI smoke container starting at http://127.0.0.1:${OPEN_WEBUI_PORT}"
}

api_smoke() {
  local work_dir
  work_dir="$(mktemp -d "${TMPDIR:-/tmp}/mere-openwebui-smoke.XXXXXX")"

  local models_file="${work_dir}/models.json"
  echo "Checking ${MERERUN_API_URL%/}/models"
  curl_json GET "/models" >"${models_file}"

  python3 - "$models_file" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)
ids = [item.get("id") for item in payload.get("data", [])]
print("Models:", ", ".join(ids))
PY

  if model_present "${models_file}" "${MERERUN_OPENWEBUI_TEXT_MODEL}"; then
    echo "Smoking text chat: ${MERERUN_OPENWEBUI_TEXT_MODEL}"
    curl_json POST "/chat/completions" "{
      \"model\":\"${MERERUN_OPENWEBUI_TEXT_MODEL}\",
      \"messages\":[{\"role\":\"user\",\"content\":\"Reply with only: mere.run online\"}],
      \"max_tokens\":16
    }" >/dev/null
  else
    echo "Skipping text chat; ${MERERUN_OPENWEBUI_TEXT_MODEL} is not listed."
  fi

  if model_present "${models_file}" "${MERERUN_OPENWEBUI_VISION_MODEL}"; then
    echo "Smoking vision chat: ${MERERUN_OPENWEBUI_VISION_MODEL}"
    local image_path="${work_dir}/smoke.png"
    write_smoke_png "${image_path}"
    local image_data_url
    image_data_url="$(python3 - "$image_path" <<'PY'
import base64
import sys

with open(sys.argv[1], "rb") as fh:
    print("data:image/png;base64," + base64.b64encode(fh.read()).decode())
PY
)"
    curl_json POST "/chat/completions" "{
      \"model\":\"${MERERUN_OPENWEBUI_VISION_MODEL}\",
      \"messages\":[{
        \"role\":\"user\",
        \"content\":[
          {\"type\":\"text\",\"text\":\"Describe this image in one short sentence.\"},
          {\"type\":\"image_url\",\"image_url\":{\"url\":\"${image_data_url}\"}}
        ]
      }],
      \"max_tokens\":64
    }" >/dev/null
  else
    echo "Skipping vision chat; ${MERERUN_OPENWEBUI_VISION_MODEL} is not listed."
  fi

  if model_present "${models_file}" "${MERERUN_OPENWEBUI_EMBED_MODEL}"; then
    echo "Smoking embeddings: ${MERERUN_OPENWEBUI_EMBED_MODEL}"
    curl_json POST "/embeddings" "{
      \"model\":\"${MERERUN_OPENWEBUI_EMBED_MODEL}\",
      \"input\":[\"mere.run native embeddings\",\"Open WebUI RAG\"]
    }" >/dev/null
  else
    echo "Skipping embeddings; ${MERERUN_OPENWEBUI_EMBED_MODEL} is not listed."
  fi

  if model_present "${models_file}" "${MERERUN_OPENWEBUI_IMAGE_MODEL}"; then
    echo "Smoking image generation: ${MERERUN_OPENWEBUI_IMAGE_MODEL}"
    curl_json POST "/images/generations" "{
      \"model\":\"${MERERUN_OPENWEBUI_IMAGE_MODEL}\",
      \"prompt\":\"a compact local AI workstation in morning light\",
      \"size\":\"1024x1024\",
      \"response_format\":\"b64_json\"
    }" >/dev/null
  else
    echo "Skipping image generation; ${MERERUN_OPENWEBUI_IMAGE_MODEL} is not listed."
  fi

  local speech_file="${work_dir}/speech.${MERERUN_OPENWEBUI_TTS_FORMAT}"
  if model_present "${models_file}" "${MERERUN_OPENWEBUI_TTS_MODEL}"; then
    echo "Smoking TTS: ${MERERUN_OPENWEBUI_TTS_MODEL} (${MERERUN_OPENWEBUI_TTS_FORMAT})"
    curl -fsS -X POST "${MERERUN_API_URL%/}/audio/speech" \
      -H "Authorization: Bearer ${MERERUN_API_KEY}" \
      -H "Content-Type: application/json" \
      --output "${speech_file}" \
      --data "{
        \"model\":\"${MERERUN_OPENWEBUI_TTS_MODEL}\",
        \"input\":\"Open WebUI is speaking through mere.run.\",
        \"voice\":\"nova\",
        \"response_format\":\"${MERERUN_OPENWEBUI_TTS_FORMAT}\"
      }"
    test -s "${speech_file}"
  else
    echo "Skipping TTS; ${MERERUN_OPENWEBUI_TTS_MODEL} is not listed."
  fi

  if [[ -s "${speech_file}" ]] && model_present "${models_file}" "${MERERUN_OPENWEBUI_STT_MODEL}"; then
    echo "Smoking STT: ${MERERUN_OPENWEBUI_STT_MODEL}"
    curl -fsS -X POST "${MERERUN_API_URL%/}/audio/transcriptions" \
      -H "Authorization: Bearer ${MERERUN_API_KEY}" \
      -F "model=${MERERUN_OPENWEBUI_STT_MODEL}" \
      -F "response_format=json" \
      -F "file=@${speech_file}" >/dev/null
  else
    echo "Skipping STT; missing speech file or ${MERERUN_OPENWEBUI_STT_MODEL} is not listed."
  fi
  rm -rf "${work_dir}"
}

proxy_smoke() {
  wait_openwebui
  local work_dir
  work_dir="$(mktemp -d "${TMPDIR:-/tmp}/mere-openwebui-proxy.XXXXXX")"

  local token
  token="$(openwebui_token)"

  local models_file="${work_dir}/openwebui-models.json"
  openwebui_json GET "/api/models" "${token}" >"${models_file}"

  if model_present "${models_file}" "${MERERUN_OPENWEBUI_TEXT_MODEL}"; then
    echo "Smoking Open WebUI text proxy: ${MERERUN_OPENWEBUI_TEXT_MODEL}"
    openwebui_json POST "/api/chat/completions" "${token}" "{
      \"model\":\"${MERERUN_OPENWEBUI_TEXT_MODEL}\",
      \"messages\":[{\"role\":\"user\",\"content\":\"Reply with only: mere.run via Open WebUI\"}],
      \"stream\":false,
      \"max_tokens\":24
    }" >/dev/null
  else
    echo "Skipping Open WebUI text proxy; ${MERERUN_OPENWEBUI_TEXT_MODEL} is not listed."
  fi

  if model_present "${models_file}" "${MERERUN_OPENWEBUI_VISION_MODEL}"; then
    echo "Smoking Open WebUI vision proxy: ${MERERUN_OPENWEBUI_VISION_MODEL}"
    local image_path="${work_dir}/smoke.png"
    write_smoke_png "${image_path}"
    local image_data_url
    image_data_url="$(python3 - "$image_path" <<'PY'
import base64
import sys

with open(sys.argv[1], "rb") as fh:
    print("data:image/png;base64," + base64.b64encode(fh.read()).decode())
PY
)"
    openwebui_json POST "/api/chat/completions" "${token}" "{
      \"model\":\"${MERERUN_OPENWEBUI_VISION_MODEL}\",
      \"messages\":[{
        \"role\":\"user\",
        \"content\":[
          {\"type\":\"text\",\"text\":\"Describe this image in one short sentence.\"},
          {\"type\":\"image_url\",\"image_url\":{\"url\":\"${image_data_url}\"}}
        ]
      }],
      \"stream\":false,
      \"max_tokens\":64
    }" >/dev/null
  else
    echo "Skipping Open WebUI vision proxy; ${MERERUN_OPENWEBUI_VISION_MODEL} is not listed."
  fi

  rm -rf "${work_dir}"
}

ui_smoke() {
  wait_openwebui
  local work_dir
  work_dir="$(mktemp -d "${TMPDIR:-/tmp}/mere-openwebui-ui.XXXXXX")"

  curl -fsS "${OPEN_WEBUI_URL%/}/" >"${work_dir}/index.html"
  if ! grep -Eiq '(<html|Open WebUI)' "${work_dir}/index.html"; then
    echo "Open WebUI root did not look like an HTML UI." >&2
    exit 1
  fi

  local token
  token="$(openwebui_token)"

  openwebui_json GET "/api/config" "${token}" >"${work_dir}/config.json"
  openwebui_json GET "/api/models" "${token}" >"${work_dir}/models.json"

  python3 - "${work_dir}/models.json" "${OPEN_WEBUI_CHAT_MODELS}" <<'PY'
import json
import re
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)
expected = {item for item in re.split(r"[;,]", sys.argv[2]) if item}
ids = {item.get("id") for item in payload.get("data", []) if item.get("id")}
missing = sorted(expected - ids)
if missing:
    raise SystemExit(f"Open WebUI UI model list is missing: {', '.join(missing)}")
print("Open WebUI UI smoke models:", ", ".join(sorted(expected)))
PY
  rm -rf "${work_dir}"
}

live_smoke() {
  configure_openwebui
  proxy_smoke
  api_smoke
  ui_smoke
}

stop_openwebui() {
  command -v docker >/dev/null 2>&1 || return 0
  docker rm -f "${OPEN_WEBUI_CONTAINER}" >/dev/null 2>&1 || true
}

command_name="${1:-help}"
case "${command_name}" in
  print-env)
    print_env
    ;;
  docker-run)
    docker_run
    ;;
  configure)
    configure_openwebui
    ;;
  proxy-smoke)
    proxy_smoke
    ;;
  api-smoke)
    api_smoke
    ;;
  ui-smoke)
    ui_smoke
    ;;
  live-smoke)
    live_smoke
    ;;
  stop)
    stop_openwebui
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
