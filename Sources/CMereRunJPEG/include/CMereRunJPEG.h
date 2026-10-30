#ifndef MERE_RUN_JPEG_H
#define MERE_RUN_JPEG_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MRJPEGSuccess = 0,
    MRJPEGInvalidData = 1,
    MRJPEGUnsupportedPrecision = 2,
    MRJPEGAllocationFailed = 3
} MRJPEGStatus;

typedef struct {
    uint8_t *pixels;
    size_t count;
    uint32_t width;
    uint32_t height;
} MRJPEGImage;

MRJPEGStatus mererun_jpeg_decode(const uint8_t *data, size_t count, MRJPEGImage *image);
void mererun_jpeg_release(MRJPEGImage *image);
const char *mererun_jpeg_version(void);
#ifdef __cplusplus
}
#endif
#endif
