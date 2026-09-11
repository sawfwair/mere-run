import Foundation

/// The affine mapping applied to one prediction, recorded so the normalized
/// artifact can be taken back to the values the model emitted.
public struct MarigoldV2DepthStatistics: Codable, Equatable, Sendable {
    /// Smallest and largest decoded value before normalization.
    public let rawMinimum: Float
    public let rawMaximum: Float
    /// Decoded values mapped to the near and far ends of the normalized range.
    public let normalizationNear: Float
    public let normalizationFar: Float
    /// Floor of the normalized range. Values are kept strictly positive so that
    /// downstream depth tooling does not read the nearest pixels as invalid.
    public let normalizedFloor: Float

    public init(
        rawMinimum: Float,
        rawMaximum: Float,
        normalizationNear: Float,
        normalizationFar: Float,
        normalizedFloor: Float
    ) {
        self.rawMinimum = rawMinimum
        self.rawMaximum = rawMaximum
        self.normalizationNear = normalizationNear
        self.normalizationFar = normalizationFar
        self.normalizedFloor = normalizedFloor
    }
}

/// Turns a decoded Marigold prediction into a normalized affine-relative depth map.
///
/// Marigold depth is affine-invariant: scale and shift are undetermined per image.
/// Normalizing to a fixed positive range therefore discards nothing beyond the
/// ambiguity the model already carries, and the applied mapping is recorded so the
/// decoded values remain recoverable.
public enum MarigoldV2DepthNormalizer {
    /// Lowest normalized value. Zero would be read as "no geometry" by the depth
    /// preview and EXR consumers, so the range starts just above it.
    public static let normalizedFloor: Float = 1.0 / 65_535

    /// Robust percentiles used for the mapping, matching the preview convention
    /// already used for geometry artifacts.
    public static let lowPercentile = 0.02
    public static let highPercentile = 0.98

    /// - Parameters:
    ///   - raw: decoded values, one per pixel.
    ///   - parameterization: how the checkpoint encodes distance. Disparity
    ///     decreases with distance and is flipped so the result always increases
    ///     away from the camera.
    /// - Returns: values in `[normalizedFloor, 1]` increasing away from the camera.
    public static func normalize(
        raw: [Float],
        parameterization: MarigoldV2DepthParameterization
    ) -> (values: [Float], statistics: MarigoldV2DepthStatistics) {
        let finite = raw.filter(\.isFinite)
        let rawMinimum = finite.min() ?? 0
        let rawMaximum = finite.max() ?? 0

        // Orient so that larger always means farther before choosing the range.
        let oriented = parameterization.increasesTowardCamera ? raw.map { -$0 } : raw
        let sorted = oriented.filter(\.isFinite).sorted()
        let near = percentile(sorted, lowPercentile)
        let far = percentile(sorted, highPercentile)
        let span = max(far - near, Float.ulpOfOne)

        let scale = 1 - normalizedFloor
        let values = oriented.map { value -> Float in
            guard value.isFinite else { return normalizedFloor }
            let position = min(1, max(0, (value - near) / span))
            return normalizedFloor + scale * position
        }

        // Report the mapping in the orientation the checkpoint actually emits.
        let reportedNear = parameterization.increasesTowardCamera ? -near : near
        let reportedFar = parameterization.increasesTowardCamera ? -far : far
        return (
            values,
            MarigoldV2DepthStatistics(
                rawMinimum: rawMinimum,
                rawMaximum: rawMaximum,
                normalizationNear: reportedNear,
                normalizationFar: reportedFar,
                normalizedFloor: normalizedFloor
            )
        )
    }

    static func percentile(_ sorted: [Float], _ fraction: Double) -> Float {
        guard let first = sorted.first, let last = sorted.last else { return 0 }
        guard sorted.count > 1 else { return first }
        let position = fraction * Double(sorted.count - 1)
        let lower = Int(position.rounded(.down))
        let upper = min(sorted.count - 1, lower + 1)
        let weight = Float(position - Double(lower))
        return sorted[lower] * (1 - weight) + sorted[upper] * weight
    }
}
