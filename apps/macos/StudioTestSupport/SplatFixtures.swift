import Foundation

/// Gaussian splat PLY files written in-test, in the INRIA 3DGS layout IDOL exports
/// (`IdolGaussians.plyData`): a binary little-endian `vertex` element of float properties.
package enum SplatFixtures {
    /// One Gaussian as the file stores it: opacity as a logit, scales as logs, rotation as
    /// (w, x, y, z), color as degree-0 SH coefficients.
    package struct Row {
        package var position: SIMD3<Float>
        package var dc: SIMD3<Float>
        package var opacityLogit: Float
        package var logScale: SIMD3<Float>
        package var rotation: SIMD4<Float>

        package init(
            position: SIMD3<Float>, dc: SIMD3<Float> = .zero, opacityLogit: Float = 0,
            logScale: SIMD3<Float> = SIMD3(repeating: -4), rotation: SIMD4<Float> = SIMD4(1, 0, 0, 0)
        ) {
            self.position = position
            self.dc = dc
            self.opacityLogit = opacityLogit
            self.logScale = logScale
            self.rotation = rotation
        }

        /// The value of `name` in this row; zero for normals and anything the row doesn't model.
        func value(_ name: String) -> Float {
            switch name {
            case "x": return position.x
            case "y": return position.y
            case "z": return position.z
            case "f_dc_0": return dc.x
            case "f_dc_1": return dc.y
            case "f_dc_2": return dc.z
            case "opacity": return opacityLogit
            case "scale_0": return logScale.x
            case "scale_1": return logScale.y
            case "scale_2": return logScale.z
            case "rot_0": return rotation[0]
            case "rot_1": return rotation[1]
            case "rot_2": return rotation[2]
            case "rot_3": return rotation[3]
            default: return 0
            }
        }
    }

    /// IDOL's property order, normals included.
    package static let idolProperties = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"
    ]

    /// A PLY of `rows` with float `properties` in the given order. `extraHeader` lines go just
    /// before `end_header`, for a test that adds an element.
    package static func ply(
        _ rows: [Row], properties: [String] = idolProperties, format: String = "binary_little_endian",
        extraHeader: [String] = []
    ) -> Data {
        var header = "ply\nformat \(format) 1.0\ncomment written by SplatFixtures\nelement vertex \(rows.count)\n"
        header += properties.map { "property float \($0)\n" }.joined()
        header += extraHeader.map { "\($0)\n" }.joined()
        header += "end_header\n"
        var data = Data(header.utf8)
        if format == "ascii" {
            for row in rows {
                data.append(Data((properties.map { "\(row.value($0))" }.joined(separator: " ") + "\n").utf8))
            }
            return data
        }
        data.reserveCapacity(data.count + rows.count * properties.count * 4)
        for row in rows {
            for name in properties {
                var bits = row.value(name).bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
            }
        }
        return data
    }

    /// `count` Gaussians on a colored sphere shell of radius 0.4: a stand-in for an IDOL object
    /// with a recognizable shape, hue varying around the equator and brightness from top to bottom.
    package static func sphere(count: Int) -> [Row] {
        let golden = Float.pi * (3 - sqrt(5))
        return (0..<count).map { index in
            let y = 1 - 2 * (Float(index) + 0.5) / Float(count)
            let ring = sqrt(max(0, 1 - y * y))
            let angle = golden * Float(index)
            let direction = SIMD3(cos(angle) * ring, y, sin(angle) * ring)
            // File y is down, as 3DGS and IDOL write it: the bright cap is on top when drawn upright.
            let hue = SIMD3(0.5 + 0.5 * cos(angle), 0.5 + 0.5 * cos(angle - 2.1), 0.5 + 0.5 * cos(angle + 2.1))
            let shade = 0.35 * hue + SIMD3(repeating: 0.65 * (1 - y) / 2)
            return Row(
                position: 0.4 * direction,
                dc: (shade - SIMD3(repeating: 0.5)) / 0.282_094_79,
                opacityLogit: 3,
                logScale: SIMD3(repeating: log(1.6 * 0.4 * sqrt(4 * .pi / Float(count)) / 2))
            )
        }
    }
}
