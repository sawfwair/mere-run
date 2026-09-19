import Foundation
import MLX

public enum AuKVariant: String, Codable, Sendable, CaseIterable {
    case base, flash
    public var checkpoint: String { self == .base ? "auk_base.safetensors" : "auk_flash.safetensors" }
}

public enum AuKSampling {
    public static let flashGrid: [Float] = [0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1]

    public static func schedule(variant: AuKVariant, steps: Int) throws -> [Float] {
        guard (1...1000).contains(steps) else { throw AuKError.invalid("AuK steps must be in 1...1000") }
        if variant == .flash { return flashGrid }
        return (0...steps).map { index in
            let time = Double(index) / Double(steps)
            return Float(1 - cos(Double.pi * time / 2))
        }
    }

    public static func frameCount(seconds: Double) throws -> Int {
        guard seconds.isFinite, seconds > 0, seconds <= 300 else {
            throw AuKError.invalid("AuK duration must be finite and in (0, 300] seconds")
        }
        return max(1, Int(ceil(seconds * 24000 / 480)))
    }

    public static func integrate(initial: MLXArray, schedule: [Float],
                                 velocity: (MLXArray, Float) throws -> MLXArray,
                                 progress: (Int, Int) -> Void = { _, _ in }) throws -> MLXArray {
        guard schedule.count >= 2, schedule.first == 0, schedule.last == 1,
              zip(schedule, schedule.dropFirst()).allSatisfy({ $0.isFinite && $0 < $1 }) else {
            throw AuKError.invalid("AuK integration requires a strictly increasing grid from 0 to 1")
        }
        var state = initial
        for index in 0..<(schedule.count - 1) {
            try Task.checkCancellation()
            state = state + (try velocity(state, schedule[index])) * (schedule[index + 1] - schedule[index])
            eval(state)
            progress(index + 1, schedule.count - 1)
        }
        return state
    }
}
