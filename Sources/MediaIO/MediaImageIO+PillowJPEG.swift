import Foundation
import CMereRunJPEG

extension MediaImageIO {
    /// Uses Pillow-compatible full-resolution 8-bit JPEG decoding. Other image
    /// formats retain the platform decoder. No draft scaling or color management.
    public static func decodeWithPillowJPEG(_ url: URL) throws -> MediaImage {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.starts(with: [0xff, 0xd8]) else { return try decode(url) }
        var decoded = MRJPEGImage()
        let status = data.withUnsafeBytes { bytes in
            mererun_jpeg_decode(bytes.bindMemory(to: UInt8.self).baseAddress, bytes.count, &decoded)
        }
        defer { mererun_jpeg_release(&decoded) }
        guard status == MRJPEGSuccess, let pixels = decoded.pixels else {
            throw MediaIOError.imageDecodeFailed(url)
        }
        return try MediaImage(width: Int(decoded.width), height: Int(decoded.height),
                              rgba8: Array(UnsafeBufferPointer(start: pixels, count: decoded.count)))
    }
}
