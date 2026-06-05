#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/vendor/magentart.xcframework"
MAGENTART_URL="${MAGENTART_URL:-https://github.com/magenta/magenta-realtime.git}"
MAGENTART_COMMIT="${MAGENTART_COMMIT:-51836beddf5fbb33636830efd919884f40ef56c5}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "[magentart-xcframework] error: Magenta RT2 native runtime must be built on Apple Silicon macOS." >&2
  exit 1
fi

for tool in git cmake xcodebuild install_name_tool; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "[magentart-xcframework] error: missing required tool: $tool" >&2
    exit 1
  fi
done

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/magentart-xcframework.XXXXXX")"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

src_dir="$tmpdir/magenta-realtime"
shim_dir="$tmpdir/magentart-shim"
build_dir="$tmpdir/build"
framework_root="$tmpdir/framework"
framework_dir="$framework_root/magentart.framework"

echo "[magentart-xcframework] cloning ${MAGENTART_URL}"
git clone --filter=blob:none --recurse-submodules "$MAGENTART_URL" "$src_dir"
git -C "$src_dir" checkout "$MAGENTART_COMMIT"
git -C "$src_dir" submodule update --init --recursive

python3 - "$src_dir" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
header = src / "core/include/magentart/mlx_engine.h"
runner_header = src / "core/include/magentart/realtime_runner.h"
impl = src / "core/src/mlx_engine.cpp"

header_text = header.read_text()
header_text = header_text.replace(
    "  void set_unmask_width(int w);\n  int get_unmask_width() const;\n",
    "  void set_unmask_width(int w);\n  int get_unmask_width() const;\n"
    "  void set_musiccoca_token_count(int count);\n"
    "  int get_musiccoca_token_count() const;\n",
)
header.write_text(header_text)

runner_text = runner_header.read_text()
runner_text = runner_text.replace(
    "    void set_unmask_width(int w) { engine_.set_unmask_width(w); }\n"
    "    int get_unmask_width() const { return engine_.get_unmask_width(); }\n",
    "    void set_unmask_width(int w) { engine_.set_unmask_width(w); }\n"
    "    int get_unmask_width() const { return engine_.get_unmask_width(); }\n"
    "    void set_musiccoca_token_count(int count) { engine_.set_musiccoca_token_count(count); }\n"
    "    int get_musiccoca_token_count() const { return engine_.get_musiccoca_token_count(); }\n",
)
runner_header.write_text(runner_text)

impl_text = impl.read_text()
impl_text = impl_text.replace(
    "  std::atomic<int> unmask_width_{0};\n",
    "  std::atomic<int> unmask_width_{0};\n"
    "  std::atomic<int> musiccoca_token_count_{static_cast<int>(kMusicCoCaRVQLevels - kMusicCoCaMaskedTailLevels)};\n",
)
impl_text = impl_text.replace(
    "    const int kept_mc_levels =\n"
    "        static_cast<int>(kMusicCoCaRVQLevels - kMusicCoCaMaskedTailLevels);\n",
    "    const int kept_mc_levels = std::max(0, std::min(\n"
    "        static_cast<int>(kMusicCoCaRVQLevels),\n"
    "        musiccoca_token_count_.load(std::memory_order_relaxed)));\n",
)
impl_text = impl_text.replace(
    "  const int kept_mc_levels =\n"
    "      static_cast<int>(kMusicCoCaRVQLevels - kMusicCoCaMaskedTailLevels);\n",
    "  const int kept_mc_levels = std::max(0, std::min(\n"
    "      static_cast<int>(kMusicCoCaRVQLevels),\n"
    "      musiccoca_token_count_.load(std::memory_order_relaxed)));\n",
)
impl_text = impl_text.replace(
    "int MLXEngine::get_unmask_width() const {\n"
    "  return impl_->unmask_width_.load(std::memory_order_relaxed);\n"
    "}\n",
    "int MLXEngine::get_unmask_width() const {\n"
    "  return impl_->unmask_width_.load(std::memory_order_relaxed);\n"
    "}\n"
    "void MLXEngine::set_musiccoca_token_count(int count) {\n"
    "  const int clamped = std::max(0, std::min(static_cast<int>(kMusicCoCaRVQLevels), count));\n"
    "  impl_->musiccoca_token_count_.store(clamped, std::memory_order_relaxed);\n"
    "}\n"
    "int MLXEngine::get_musiccoca_token_count() const {\n"
    "  return impl_->musiccoca_token_count_.load(std::memory_order_relaxed);\n"
    "}\n",
)
impl.write_text(impl_text)
PY

mkdir -p "$shim_dir/include/magentart" "$shim_dir/src"
cat >"$shim_dir/include/magentart/mrt2_c_api.h" <<'HEADER'
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define MRT2_EXPORT __attribute__((visibility("default")))
#else
#define MRT2_EXPORT
#endif

typedef struct mrt2_engine mrt2_engine_t;
typedef struct mrt2_runner mrt2_runner_t;

MRT2_EXPORT mrt2_engine_t *mrt2_engine_create(void);
MRT2_EXPORT void mrt2_engine_destroy(mrt2_engine_t *engine);
MRT2_EXPORT bool mrt2_engine_init_assets(mrt2_engine_t *engine, const char *resource_dir, const char *model_subfolder);
MRT2_EXPORT bool mrt2_engine_load_model(mrt2_engine_t *engine, const char *mlxfn_path);
MRT2_EXPORT bool mrt2_engine_prefill_silence(mrt2_engine_t *engine, int32_t duration_frames);
MRT2_EXPORT void mrt2_engine_reset_state(mrt2_engine_t *engine);
MRT2_EXPORT void mrt2_engine_set_text_prompt(mrt2_engine_t *engine, const char *prompt);
MRT2_EXPORT int32_t mrt2_engine_get_text_encoder_status(mrt2_engine_t *engine);
MRT2_EXPORT int32_t mrt2_engine_get_quantizer_status(mrt2_engine_t *engine);
MRT2_EXPORT bool mrt2_engine_generate_frame(mrt2_engine_t *engine, float *left, float *right, int32_t sample_count);
MRT2_EXPORT void mrt2_engine_set_temperature(mrt2_engine_t *engine, float value);
MRT2_EXPORT void mrt2_engine_set_top_k(mrt2_engine_t *engine, int32_t value);
MRT2_EXPORT void mrt2_engine_set_cfg_musiccoca(mrt2_engine_t *engine, float value);
MRT2_EXPORT void mrt2_engine_set_cfg_notes(mrt2_engine_t *engine, float value);
MRT2_EXPORT void mrt2_engine_set_cfg_drums(mrt2_engine_t *engine, float value);
MRT2_EXPORT void mrt2_engine_set_drumless(mrt2_engine_t *engine, bool value);
MRT2_EXPORT void mrt2_engine_set_unmask_width(mrt2_engine_t *engine, int32_t value);
MRT2_EXPORT void mrt2_engine_set_musiccoca_token_count(mrt2_engine_t *engine, int32_t value);
MRT2_EXPORT void mrt2_engine_set_seed_rotation(mrt2_engine_t *engine, int32_t value);
MRT2_EXPORT void mrt2_engine_set_note_on(mrt2_engine_t *engine, int32_t note);
MRT2_EXPORT void mrt2_engine_set_note_off(mrt2_engine_t *engine, int32_t note);
MRT2_EXPORT void mrt2_engine_set_onset_mode(mrt2_engine_t *engine, int32_t mode);

MRT2_EXPORT mrt2_runner_t *mrt2_runner_create(void);
MRT2_EXPORT void mrt2_runner_destroy(mrt2_runner_t *runner);
MRT2_EXPORT bool mrt2_runner_init_assets(mrt2_runner_t *runner, const char *resource_dir);
MRT2_EXPORT bool mrt2_runner_load_model(mrt2_runner_t *runner, const char *mlxfn_path);
MRT2_EXPORT bool mrt2_runner_prefill_silence(mrt2_runner_t *runner, int32_t duration_frames);
MRT2_EXPORT void mrt2_runner_start(mrt2_runner_t *runner);
MRT2_EXPORT void mrt2_runner_stop(mrt2_runner_t *runner);
MRT2_EXPORT void mrt2_runner_set_text_prompt(mrt2_runner_t *runner, const char *prompt);
MRT2_EXPORT int32_t mrt2_runner_get_text_encoder_status(mrt2_runner_t *runner);
MRT2_EXPORT int32_t mrt2_runner_get_quantizer_status(mrt2_runner_t *runner);
MRT2_EXPORT bool mrt2_runner_read_audio_stereo(mrt2_runner_t *runner, float *left, float *right, int32_t sample_count, bool blocking);
MRT2_EXPORT void mrt2_runner_set_temperature(mrt2_runner_t *runner, float value);
MRT2_EXPORT void mrt2_runner_set_top_k(mrt2_runner_t *runner, int32_t value);
MRT2_EXPORT void mrt2_runner_set_cfg_musiccoca(mrt2_runner_t *runner, float value);
MRT2_EXPORT void mrt2_runner_set_cfg_notes(mrt2_runner_t *runner, float value);
MRT2_EXPORT void mrt2_runner_set_cfg_drums(mrt2_runner_t *runner, float value);
MRT2_EXPORT void mrt2_runner_set_drumless(mrt2_runner_t *runner, bool value);
MRT2_EXPORT void mrt2_runner_set_unmask_width(mrt2_runner_t *runner, int32_t value);
MRT2_EXPORT void mrt2_runner_set_musiccoca_token_count(mrt2_runner_t *runner, int32_t value);
MRT2_EXPORT void mrt2_runner_set_seed_rotation(mrt2_runner_t *runner, int32_t value);

#undef MRT2_EXPORT

#ifdef __cplusplus
}
#endif
HEADER

cat >"$shim_dir/src/mrt2_c_api.cpp" <<'SOURCE'
#include <magentart/mrt2_c_api.h>

#include <magentart/mlx_engine.h>
#include <magentart/realtime_runner.h>

#include <exception>

struct mrt2_engine {
  magentart::core::MLXEngine impl;
};

struct mrt2_runner {
  magentart::core::RealtimeRunner impl;
};

extern "C" {

mrt2_engine_t *mrt2_engine_create(void) {
  try {
    return new mrt2_engine();
  } catch (...) {
    return nullptr;
  }
}
void mrt2_engine_destroy(mrt2_engine_t *engine) {
  try {
    delete engine;
  } catch (...) {}
}

bool mrt2_engine_init_assets(mrt2_engine_t *engine, const char *resource_dir, const char *model_subfolder) {
  try {
    return engine && resource_dir && engine->impl.init_assets(resource_dir, model_subfolder ? model_subfolder : "musiccoca");
  } catch (...) {
    return false;
  }
}

bool mrt2_engine_load_model(mrt2_engine_t *engine, const char *mlxfn_path) {
  try {
    return engine && mlxfn_path && engine->impl.load_model(mlxfn_path);
  } catch (...) {
    return false;
  }
}

bool mrt2_engine_prefill_silence(mrt2_engine_t *engine, int32_t duration_frames) {
  try {
    return engine && engine->impl.prefill_silence(duration_frames);
  } catch (...) {
    return false;
  }
}

void mrt2_engine_reset_state(mrt2_engine_t *engine) {
  try {
    if (engine) engine->impl.reset_state();
  } catch (...) {}
}

void mrt2_engine_set_text_prompt(mrt2_engine_t *engine, const char *prompt) {
  try {
    if (engine) engine->impl.set_text_prompt(prompt ? prompt : "");
  } catch (...) {}
}

int32_t mrt2_engine_get_text_encoder_status(mrt2_engine_t *engine) {
  try {
    return engine ? engine->impl.get_text_encoder_status() : 3;
  } catch (...) {
    return 3;
  }
}

int32_t mrt2_engine_get_quantizer_status(mrt2_engine_t *engine) {
  try {
    return engine ? engine->impl.get_quantizer_status() : 3;
  } catch (...) {
    return 3;
  }
}

bool mrt2_engine_generate_frame(mrt2_engine_t *engine, float *left, float *right, int32_t sample_count) {
  try {
    return engine && left && right && sample_count >= static_cast<int32_t>(magentart::core::kFrameSamples)
        && engine->impl.generate_frame(left, right);
  } catch (...) {
    return false;
  }
}

void mrt2_engine_set_temperature(mrt2_engine_t *engine, float value) { try { if (engine) engine->impl.set_temperature(value); } catch (...) {} }
void mrt2_engine_set_top_k(mrt2_engine_t *engine, int32_t value) { try { if (engine) engine->impl.set_top_k(value); } catch (...) {} }
void mrt2_engine_set_cfg_musiccoca(mrt2_engine_t *engine, float value) { try { if (engine) engine->impl.set_cfg_musiccoca(value); } catch (...) {} }
void mrt2_engine_set_cfg_notes(mrt2_engine_t *engine, float value) { try { if (engine) engine->impl.set_cfg_notes(value); } catch (...) {} }
void mrt2_engine_set_cfg_drums(mrt2_engine_t *engine, float value) { try { if (engine) engine->impl.set_cfg_drums(value); } catch (...) {} }
void mrt2_engine_set_drumless(mrt2_engine_t *engine, bool value) { try { if (engine) engine->impl.set_drumless(value); } catch (...) {} }
void mrt2_engine_set_unmask_width(mrt2_engine_t *engine, int32_t value) { try { if (engine) engine->impl.set_unmask_width(value); } catch (...) {} }
void mrt2_engine_set_musiccoca_token_count(mrt2_engine_t *engine, int32_t value) { try { if (engine) engine->impl.set_musiccoca_token_count(value); } catch (...) {} }
void mrt2_engine_set_seed_rotation(mrt2_engine_t *engine, int32_t value) { try { if (engine) engine->impl.set_seed_rotation(value); } catch (...) {} }
void mrt2_engine_set_note_on(mrt2_engine_t *engine, int32_t note) { try { if (engine) engine->impl.set_note_on(note); } catch (...) {} }
void mrt2_engine_set_note_off(mrt2_engine_t *engine, int32_t note) { try { if (engine) engine->impl.set_note_off(note); } catch (...) {} }
void mrt2_engine_set_onset_mode(mrt2_engine_t *engine, int32_t mode) { try { if (engine) engine->impl.set_onset_mode(mode); } catch (...) {} }

mrt2_runner_t *mrt2_runner_create(void) {
  try {
    return new mrt2_runner();
  } catch (...) {
    return nullptr;
  }
}
void mrt2_runner_destroy(mrt2_runner_t *runner) {
  try {
    delete runner;
  } catch (...) {}
}

bool mrt2_runner_init_assets(mrt2_runner_t *runner, const char *resource_dir) {
  try {
    return runner && resource_dir && runner->impl.init_assets(resource_dir);
  } catch (...) {
    return false;
  }
}

bool mrt2_runner_load_model(mrt2_runner_t *runner, const char *mlxfn_path) {
  try {
    return runner && mlxfn_path && runner->impl.load_model(mlxfn_path);
  } catch (...) {
    return false;
  }
}

bool mrt2_runner_prefill_silence(mrt2_runner_t *runner, int32_t duration_frames) {
  try {
    return runner && runner->impl.prefill_silence(duration_frames);
  } catch (...) {
    return false;
  }
}

void mrt2_runner_start(mrt2_runner_t *runner) { try { if (runner) runner->impl.start(); } catch (...) {} }
void mrt2_runner_stop(mrt2_runner_t *runner) { try { if (runner) runner->impl.stop(); } catch (...) {} }
void mrt2_runner_set_text_prompt(mrt2_runner_t *runner, const char *prompt) { try { if (runner) runner->impl.set_text_prompt(prompt ? prompt : ""); } catch (...) {} }
int32_t mrt2_runner_get_text_encoder_status(mrt2_runner_t *runner) { try { return runner ? runner->impl.get_text_encoder_status() : 3; } catch (...) { return 3; } }
int32_t mrt2_runner_get_quantizer_status(mrt2_runner_t *runner) { try { return runner ? runner->impl.get_quantizer_status() : 3; } catch (...) { return 3; } }

bool mrt2_runner_read_audio_stereo(mrt2_runner_t *runner, float *left, float *right, int32_t sample_count, bool blocking) {
  try {
    return runner && left && right && sample_count > 0
        && runner->impl.read_audio_stereo(left, right, static_cast<std::size_t>(sample_count), blocking);
  } catch (...) {
    return false;
  }
}

void mrt2_runner_set_temperature(mrt2_runner_t *runner, float value) { try { if (runner) runner->impl.set_temperature(value); } catch (...) {} }
void mrt2_runner_set_top_k(mrt2_runner_t *runner, int32_t value) { try { if (runner) runner->impl.set_top_k(value); } catch (...) {} }
void mrt2_runner_set_cfg_musiccoca(mrt2_runner_t *runner, float value) { try { if (runner) runner->impl.set_cfg_musiccoca(value); } catch (...) {} }
void mrt2_runner_set_cfg_notes(mrt2_runner_t *runner, float value) { try { if (runner) runner->impl.set_cfg_notes(value); } catch (...) {} }
void mrt2_runner_set_cfg_drums(mrt2_runner_t *runner, float value) { try { if (runner) runner->impl.set_cfg_drums(value); } catch (...) {} }
void mrt2_runner_set_drumless(mrt2_runner_t *runner, bool value) { try { if (runner) runner->impl.set_drumless(value); } catch (...) {} }
void mrt2_runner_set_unmask_width(mrt2_runner_t *runner, int32_t value) { try { if (runner) runner->impl.set_unmask_width(value); } catch (...) {} }
void mrt2_runner_set_musiccoca_token_count(mrt2_runner_t *runner, int32_t value) { try { if (runner) runner->impl.set_musiccoca_token_count(value); } catch (...) {} }
void mrt2_runner_set_seed_rotation(mrt2_runner_t *runner, int32_t value) { try { if (runner) runner->impl.set_seed_rotation(value); } catch (...) {} }

}
SOURCE

cat >"$shim_dir/CMakeLists.txt" <<CMAKE
cmake_minimum_required(VERSION 3.27)
project(MereRunMagentaRT2 LANGUAGES C CXX OBJCXX)

set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS deployment version" FORCE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory("${src_dir}" magenta-realtime)

add_library(magentart SHARED src/mrt2_c_api.cpp)
target_include_directories(magentart PUBLIC "\${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(magentart PRIVATE magentart::core)
target_compile_options(magentart PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
set_target_properties(magentart PROPERTIES
  OUTPUT_NAME magentart
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)
CMAKE

echo "[magentart-xcframework] configuring"
cmake -S "$shim_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_OSX_ARCHITECTURES=arm64

echo "[magentart-xcframework] building"
cmake --build "$build_dir" --config "$CMAKE_BUILD_TYPE" --target magentart --parallel "$(sysctl -n hw.ncpu)"

mkdir -p "$framework_dir/Headers" "$framework_dir/Modules" "$framework_dir/Resources"
cp "$build_dir/libmagentart.dylib" "$framework_dir/magentart"
install_name_tool -id "@rpath/magentart.framework/magentart" "$framework_dir/magentart"
cp "$shim_dir/include/magentart/mrt2_c_api.h" "$framework_dir/Headers/mrt2_c_api.h"
mlx_metallib="$(find "$build_dir" -name 'mlx.metallib' -type f -print -quit)"
if [[ -z "$mlx_metallib" ]]; then
  echo "[magentart-xcframework] error: mlx.metallib was not produced by the MLX build." >&2
  exit 1
fi
cp "$mlx_metallib" "$framework_dir/Resources/mlx.metallib"
cp "$mlx_metallib" "$framework_dir/Resources/default.metallib"
cat >"$framework_dir/Modules/module.modulemap" <<'MODULEMAP'
framework module magentart {
  umbrella header "mrt2_c_api.h"
  export *
  module * { export * }
}
MODULEMAP
cat >"$framework_dir/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>magentart</string>
  <key>CFBundleIdentifier</key>
  <string>run.mere.magentart</string>
  <key>CFBundleName</key>
  <string>magentart</string>
  <key>CFBundlePackageType</key>
  <string>FMWK</string>
  <key>CFBundleShortVersionString</key>
  <string>0.0.1</string>
  <key>CFBundleVersion</key>
  <string>0.0.1</string>
  <key>MinimumOSVersion</key>
  <string>15.0</string>
</dict>
</plist>
PLIST

rm -rf "$VENDOR_DIR"
echo "[magentart-xcframework] creating ${VENDOR_DIR}"
xcodebuild -create-xcframework -framework "$framework_dir" -output "$VENDOR_DIR"
echo "[magentart-xcframework] done"
