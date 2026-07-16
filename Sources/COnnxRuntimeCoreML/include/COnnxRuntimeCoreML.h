#pragma once

#include <CONNXRuntime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

OrtStatus *MereRunOrtAppendCoreML(OrtSessionOptions *options, uint32_t flags);

#ifdef __cplusplus
}
#endif
