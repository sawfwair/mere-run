#include "COnnxRuntimeCoreML.h"
#include <coreml_provider_factory.h>

OrtStatus *MereRunOrtAppendCoreML(OrtSessionOptions *options, uint32_t flags) {
    return OrtSessionOptionsAppendExecutionProvider_CoreML(options, flags);
}
