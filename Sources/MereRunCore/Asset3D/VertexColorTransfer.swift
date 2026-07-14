import Foundation

/// sRGB transfer functions for vertex-color interchange.
///
/// Native reconstruction models regress display-referred pixel values, so
/// `colorsRGBA8` carries sRGB-encoded bytes. glTF 2.0 defines `COLOR_0` as
/// linear, so the GLB writers decode RGB through the exact piecewise sRGB
/// EOTF (IEC 61966-2-1) at export and widen to 16 bits because 8-bit linear
/// values band in the darks. OBJ and PLY keep the sRGB bytes verbatim: those
/// formats carry display-referred color by de-facto convention. Alpha is
/// linear coverage everywhere and is never transfer-coded.
public enum VertexColorTransfer {
    /// Exact piecewise sRGB EOTF, encoded [0, 1] to linear [0, 1].
    public static func linear(fromSRGB encoded: Float) -> Float {
        encoded <= 0.04045 ? encoded / 12.92 : pow((encoded + 0.055) / 1.055, 2.4)
    }

    /// Linear 16-bit normalized RGBA from sRGB-encoded RGBA8 vertex colors.
    /// RGB decodes through the sRGB EOTF; alpha rescales linearly.
    public static func linearRGBA16(fromSRGBA8 colors: [UInt8]) -> [UInt16] {
        var output = [UInt16](repeating: 0, count: colors.count)
        for index in colors.indices {
            let value = colors[index]
            output[index] = index % 4 == 3
                ? UInt16((Float(value) / 255 * 65535).rounded())
                : linearRGB16[Int(value)]
        }
        return output
    }

    private static let linearRGB16: [UInt16] = (0...255).map { value in
        UInt16((linear(fromSRGB: Float(value) / 255) * 65535).rounded())
    }
}
