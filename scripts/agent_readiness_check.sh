#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

failures=0

record_failure() {
  printf 'agent-readiness: %s\n' "$1" >&2
  failures=$((failures + 1))
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    record_failure "missing required file: $path"
  fi
}

for required_file in \
  CODEBASE.md \
  DECISIONS.md \
  docs/development-workflow.md \
  docs/repository-tour.md \
  docs/testing.md \
  README.md
do
  require_file "$required_file"
done

if [[ -f CODEBASE.md ]]; then
  codebase_words="$(wc -w < CODEBASE.md | tr -d ' ')"
  if (( codebase_words > 500 )); then
    record_failure "CODEBASE.md must stay under 500 words for agent orientation; found $codebase_words"
  fi
fi

required_module_readmes=(
  Sources/AudioCore/README.md
  Sources/AudioCodecs/README.md
  Sources/AudioSTT/Parakeet/README.md
  Sources/AudioSTT/Qwen3ASR/Model/README.md
  Sources/AudioSTT/Qwen3ASR/README.md
  Sources/AudioTTS/Qwen3TTS/README.md
  Sources/MereRunApp/README.md
  Sources/MereRunCLI/Commands/README.md
  Sources/MereRunCLI/Support/README.md
  Sources/MereRunCore/ACEStep/Model/README.md
  Sources/MereRunCore/ACEStep/README.md
  Sources/MereRunCore/ACEStep/VAE/README.md
  Sources/MereRunCore/CodeGen/README.md
  Sources/MereRunCore/FalconPerception/README.md
  Sources/MereRunCore/Flux2Klein/Model/Transformer/README.md
  Sources/MereRunCore/Flux2Klein/README.md
  Sources/MereRunCore/Gemma4/README.md
  Sources/MereRunCore/LightOnOCR/README.md
  Sources/MereRunCore/LoRA/README.md
  Sources/MereRunCore/LTX/README.md
  Sources/MereRunCore/PrivacyFilter/README.md
  Sources/MereRunCore/Psi/README.md
  Sources/MereRunCore/Q35/README.md
  Sources/MereRunCore/QwenImageEdit/Model/Transformer/README.md
  Sources/MereRunCore/QwenImageEdit/Model/VAE/README.md
  Sources/MereRunCore/QwenImageEdit/README.md
  Sources/MereRunCore/SAM3/README.md
  Sources/MereRunCore/Support/README.md
  Sources/MereRunCore/VLM/README.md
  Sources/MereRunCore/ZImageI2L/Model/README.md
  Sources/MereRunCore/ZImageI2L/README.md
  Sources/MereRunCore/ZImageTurbo/Model/TextEncoder/README.md
  Sources/MereRunCore/ZImageTurbo/Model/TextEncoder/Vision/README.md
  Sources/MereRunCore/ZImageTurbo/Model/Transformer/README.md
  Sources/MereRunCore/ZImageTurbo/Model/VAE/README.md
  Sources/MereRunCore/ZImageTurbo/README.md
  Sources/MereRunCore/ZImageTurbo/Util/README.md
  Sources/MereRunCore/README.md
)

for readme in "${required_module_readmes[@]}"; do
  require_file "$readme"
done

while IFS= read -r module_dir; do
  direct_lines="$(
    find "$module_dir" -maxdepth 1 -name '*.swift' -print0 |
      xargs -0 wc -l 2>/dev/null |
      tail -1 |
      awk '{print $1}'
  )"
  if [[ -n "$direct_lines" && "$direct_lines" =~ ^[0-9]+$ && "$direct_lines" -ge 500 && ! -f "$module_dir/README.md" ]]; then
    record_failure "module over 500 direct Swift LOC needs README.md: $module_dir ($direct_lines lines)"
  fi
done < <(find Sources -type d | sort)

dynamic_boundary_files=(
  "Sources/AudioSTT/Qwen3ASR/Qwen3ASRTokenizer.swift"
  # Studio artifact explorers deliberately accept polymorphic manifests/results from saved
  # runs. Voice Studio is included because AVAudioRecorder's settings API requires [String: Any].
  "Sources/MereRunApp/Studio3DCreationView.swift"
  "Sources/MereRunApp/StudioMusicToolsView.swift"
  "Sources/MereRunApp/StudioTrainingView.swift"
  "Sources/MereRunApp/StudioUtilityLabView.swift"
  "Sources/MereRunApp/StudioVoiceView.swift"
  "Sources/MereRunCLI/Support/ResumeLoRABootstrap.swift"
  "Sources/MereRunCore/Asset3D/MeshGLBWriter.swift"
  "Sources/MereRunCore/FalconPerception/FalconPerceptionTokenizer.swift"
  "Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+Chat.swift"
  "Sources/MereRunCore/Geometry/MultiViewGeometryExporter.swift"
  "Sources/MereRunCore/Geometry/PointCloudGLBWriter.swift"
  "Sources/MereRunCore/LTX/LTXVideoMP4Writer.swift"
  "Sources/MereRunCore/LoRA/LoRACheckpointResolver.swift"
  "Sources/MereRunCore/LoRA/LoRAWeightLoader.swift"
  "Sources/MereRunCore/LoRA/QwenTextLoRAWeightLoader.swift"
  "Sources/MereRunCore/NativeMediaAssembler.swift"
  "Sources/MereRunCore/Psi/GLM47Tokenizer.swift"
  "Sources/MereRunCore/QwenImageEdit/Model/VAE/AutoencoderKL3D.swift"
  "Sources/MereRunCore/QwenImageEdit/Tokenizer/Qwen25VLTokenizer.swift"
  "Sources/MereRunCore/SAM3/SAM31Tokenizer.swift"
  "Sources/MereRunCore/SAM3/SAM31VideoIO.swift"
  "Sources/MereRunCore/Trellis2/Trellis2TexturedGLBWriter.swift"
  "Sources/MereRunCore/VLM/Qwen3VLAutoCaptioner.swift"
  "Sources/MereRunCore/ZImageTurbo/Model/TextEncoder/LLMGeneration/QwenGeneration.swift"
  "Sources/MereRunCore/ZImageTurbo/Tokenizer/QwenTokenizer.swift"
  "Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator.swift"
)

is_dynamic_boundary_file() {
  local candidate="$1"
  local allowed
  for allowed in "${dynamic_boundary_files[@]}"; do
    if [[ "$candidate" == "$allowed" ]]; then
      return 0
    fi
  done
  return 1
}

dynamic_pattern='JSONSerialization\.jsonObject|JSONSerialization\.data\(withJSONObject|\[String\s*:\s*Any\]|\[\[String\s*:\s*Any\]\]|\[Any\]|Any\?'
dynamic_matches=()
while IFS= read -r match; do
  dynamic_matches+=("$match")
done < <(rg -l "$dynamic_pattern" Sources | sort || true)
for match in "${dynamic_matches[@]}"; do
  if ! is_dynamic_boundary_file "$match"; then
    record_failure "raw dynamic JSON appeared outside the typed-boundary inventory: $match"
  fi
done

if (( failures > 0 )); then
  exit 1
fi

printf 'agent-readiness: ok\n'
