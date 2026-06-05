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
