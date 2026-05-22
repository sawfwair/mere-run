import Foundation

/// Turbo timestep schedule for ACE-Step v1.5.
///
/// The turbo checkpoints use a fixed number of function evaluations (`fix_nfe`, default 8) and apply an
/// optional "linear" timestep shift:
///   `t = shift * s / (1 + (shift - 1) * s)`, where `s` is the base schedule in `[0, 1]`.
///
/// In the reference implementation, only discrete shifts `{1, 2, 3}` are supported and custom timesteps
/// are mapped to the nearest valid timestep.
public struct ACEStepTurboScheduler: Sendable, Hashable {
    public static let validShifts: [Float] = [1.0, 2.0, 3.0]
    public static let validReferenceTimesteps: [Float] = [
        1.0, 0.95454545, 0.93333334, 0.9, 0.875,
        0.85714287, 0.8333333, 0.7692308, 0.75,
        0.6666667, 0.64285713, 0.625, 0.54545456,
        0.5, 0.4, 0.375, 0.3, 0.25, 0.22222222, 0.125,
    ]

    /// Descending timesteps in `[0, 1]` (does not include the terminal `0`).
    public let timesteps: [Float]

    public init(fixNFE: Int = 8, shift: Float = 1.0, timesteps: [Float]? = nil) {
        self.timesteps = Self.makeTimesteps(fixNFE: fixNFE, shift: shift, custom: timesteps)
    }

    public static func makeTimesteps(
        fixNFE: Int,
        shift: Float,
        custom: [Float]?
    ) -> [Float] {
        let shiftRounded = roundShift(shift)
        let defaultSchedule = schedule(fixNFE: fixNFE, shift: shiftRounded)

        guard var custom else { return defaultSchedule }

        // Remove trailing zeros (the Python implementation treats an explicit terminal 0 as optional).
        while let last = custom.last, last == 0 {
            custom.removeLast()
        }
        guard !custom.isEmpty else { return defaultSchedule }

        let candidates = validReferenceTimesteps
        let maxCount = candidates.count
        if custom.count > maxCount {
            custom = Array(custom.prefix(maxCount))
        }

        return custom.map { nearestTimestep($0, candidates: candidates) }
    }

    public static func schedule(fixNFE: Int, shift: Float) -> [Float] {
        let base = baseTimesteps(fixNFE: fixNFE)
        return base.map { applyShift(base: $0, shift: shift) }
    }

    public static func baseTimesteps(fixNFE: Int) -> [Float] {
        precondition(fixNFE > 0, "fixNFE must be > 0 (got \(fixNFE))")
        let denom = Double(fixNFE)
        return (0..<fixNFE).map { i in
            Float(1.0 - Double(i) / denom)
        }
    }

    public static func applyShift(base: Float, shift: Float) -> Float {
        let s = Double(base)
        let mu = Double(shift)
        return Float((mu * s) / (1.0 + (mu - 1.0) * s))
    }

    public static func validTimesteps(fixNFE: Int) -> [Float] {
        var set = Set<Float>()
        set.reserveCapacity(fixNFE * validShifts.count)
        for shift in validShifts {
            for t in schedule(fixNFE: fixNFE, shift: shift) {
                set.insert(t)
            }
        }
        return set.sorted(by: >)
    }

    public static func roundShift(_ shift: Float) -> Float {
        validShifts.min { abs($0 - shift) < abs($1 - shift) } ?? 3.0
    }

    public static func nearestTimestep(_ t: Float, candidates: [Float]) -> Float {
        candidates.min { abs($0 - t) < abs($1 - t) } ?? t
    }
}
