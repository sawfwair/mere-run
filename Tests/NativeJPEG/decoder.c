#include "CMereRunJPEG.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Exercise every truncation boundary under AddressSanitizer and UBSan.
int main(int argc, char **argv) {
    size_t attempts = 0;
    for (int index = 1; index < argc; ++index) {
        FILE *file = fopen(argv[index], "rb");
        assert(file);
        assert(fseek(file, 0, SEEK_END) == 0);
        long size = ftell(file);
        assert(size > 0);
        rewind(file);
        unsigned char *data = malloc((size_t)size);
        assert(data && fread(data, 1, (size_t)size, file) == (size_t)size);
        fclose(file);
        for (size_t count = 0; count <= (size_t)size; ++count) {
            MRJPEGImage image;
            MRJPEGStatus status = mererun_jpeg_decode(data, count, &image);
            if (count == (size_t)size) {
                assert(status == MRJPEGSuccess);
                assert(image.pixels && image.count == (size_t)image.width * image.height * 4);
            } else {
                assert(status != MRJPEGSuccess);
                assert(!image.pixels && image.count == 0);
            }
            mererun_jpeg_release(&image);
            mererun_jpeg_release(&image);
            ++attempts;
        }
        free(data);
    }
    printf("%zu decodes, including every truncated prefix, passed\n", attempts);
    return 0;
}
