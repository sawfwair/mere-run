# Native Pillow-compatible JPEG decoding

This private C target builds the decoder closure of libjpeg-turbo 3.2.0 at
`c85e6b905bf237038faa936dab160ebfc5da0344`. It links statically through `MediaIO`;
it needs neither Python nor a system JPEG installation. It has no public
SwiftPM product. Unlimited-OCR-2 is the first opt-in consumer.

`upstream.json` records source hashes and the 54 translation units selected by
linking a decompression-only client against the upstream scalar static build.
The original source files under `libjpeg/src/` are unmodified. Files under
`compile/` select those units; `jconfig.h` and `jconfigint.h` are generated for
the portable scalar build. `mere_jpeg_symbols.h` prefixes all upstream global
symbols to prevent collisions in host applications.

The wrapper uses full-resolution 8-bit decoding, integer slow DCT, fancy chroma
upsampling, and no color management, matching ordinary Pillow JPEG loading.
RGB, grayscale, progressive, and CMYK JPEGs produce opaque RGBA. CMYK conversion
matches Pillow's inverted JPEG sample convention and integer RGB conversion.
Truncated inputs fail instead of silently receiving a synthesized end marker.
Other sample precisions are rejected. EXIF orientation is separate from JPEG
sample decoding.

The upstream source is covered by the IJG, modified BSD, and zlib licenses in
`LICENSE.md` and `README.ijg`. This software is based in part on the work of the
Independent JPEG Group.

`bash Tests/NativeJPEG/check.sh` builds this exact decoder under AddressSanitizer
and UBSan and checks every truncated prefix of the independent JPEG fixtures.
Swift tests compare decoded bytes with Pillow goldens and exercise concurrent
decodes, signature detection, and non-JPEG fallback.
