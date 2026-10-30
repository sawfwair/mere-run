#include "CMereRunJPEG.h"
#include <limits.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jpeglib.h"
#include "jerror.h"

typedef struct {
    struct jpeg_error_mgr base;
    jmp_buf jump;
} MRJPEGError;

typedef struct {
    struct jpeg_decompress_struct info;
    MRJPEGError error;
    uint8_t *pixels;
    MRJPEGStatus status;
} MRJPEGContext;

static void fail(j_common_ptr info) {
    MRJPEGError *error = (MRJPEGError *)info->err;
    longjmp(error->jump, 1);
}

static void discard_message(j_common_ptr info) { (void)info; }

static boolean end_of_input(j_decompress_ptr info) {
    // Match Pillow's normal strict loading, rather than jpeg_mem_src's
    // synthesized EOI marker for truncated inputs.
    ERREXIT(info, JERR_INPUT_EOF);
    return FALSE;
}

MRJPEGStatus mererun_jpeg_decode(const uint8_t *data, size_t count, MRJPEGImage *image) {
    memset(image, 0, sizeof(*image));
    if (count > ULONG_MAX || count < 2) return MRJPEGInvalidData;
    MRJPEGContext *context = calloc(1, sizeof(*context));
    if (!context) return MRJPEGAllocationFailed;
    context->status = MRJPEGInvalidData;
    context->info.err = jpeg_std_error(&context->error.base);
    context->error.base.error_exit = fail;
    context->error.base.output_message = discard_message;
    if (setjmp(context->error.jump)) goto cleanup;
    jpeg_create_decompress(&context->info);
    jpeg_mem_src(&context->info, data, (unsigned long)count);
    context->info.src->fill_input_buffer = end_of_input;
    jpeg_read_header(&context->info, TRUE);
    if (context->info.data_precision != 8) {
        context->status = MRJPEGUnsupportedPrecision;
        goto cleanup;
    }
    const int cmyk = context->info.jpeg_color_space == JCS_CMYK || context->info.jpeg_color_space == JCS_YCCK;
    context->info.out_color_space = cmyk ? JCS_CMYK : JCS_EXT_RGBA;
    context->info.dct_method = JDCT_ISLOW;
    context->info.do_fancy_upsampling = TRUE;
    jpeg_start_decompress(&context->info);
    const size_t width = context->info.output_width;
    const size_t height = context->info.output_height;
    if (!width || !height || width > SIZE_MAX / height / 4) goto cleanup;
    const size_t bytes = width * height * 4;
    context->pixels = malloc(bytes);
    if (!context->pixels) {
        context->status = MRJPEGAllocationFailed;
        goto cleanup;
    }
    while (context->info.output_scanline < context->info.output_height) {
        JSAMPROW row = context->pixels + context->info.output_scanline * width * 4;
        jpeg_read_scanlines(&context->info, &row, 1);
        if (cmyk) {
            // Pillow loads JPEG CMYK as inverted samples before convert("RGB").
            // Its integer conversion is nearest-rounded inverted C/M/Y * K.
            for (size_t x = 0; x < width; ++x) {
                unsigned int black = row[x * 4 + 3];
                for (size_t channel = 0; channel < 3; ++channel)
                    row[x * 4 + channel] = (uint8_t)((row[x * 4 + channel] * black + 127) / 255);
                row[x * 4 + 3] = 255;
            }
        }
    }
    jpeg_finish_decompress(&context->info);
    image->pixels = context->pixels;
    image->count = bytes;
    image->width = (uint32_t)width;
    image->height = (uint32_t)height;
    context->pixels = NULL;
    context->status = MRJPEGSuccess;
cleanup:
    if (context->info.mem) jpeg_destroy_decompress(&context->info);
    free(context->pixels);
    MRJPEGStatus status = context->status;
    free(context);
    return status;
}

void mererun_jpeg_release(MRJPEGImage *image) {
    free(image->pixels);
    memset(image, 0, sizeof(*image));
}

const char *mererun_jpeg_version(void) { return "libjpeg-turbo 3.2.0 (scalar, private static decoder)"; }
