import Foundation

public struct TrainingPhase: Sendable, Hashable {
    public let width: Int
    public let height: Int
    public let steps: Int

    public init(width: Int, height: Int, steps: Int) {
        self.width = width
        self.height = height
        self.steps = steps
    }
}

public enum TrainingSchedule {
    /// Builds a simple progressive resolution schedule biased toward lower resolutions.
    ///
    /// This is intentionally conservative: it returns a single `target` phase when progressive
    /// training is disabled, step counts are too small, or in benchmark mode.
    public static func progressivePhases(
        targetWidth: Int,
        targetHeight: Int,
        totalSteps: Int,
        enabled: Bool,
        benchmarkSteps: Int?,
        multiple: Int
    ) -> [TrainingPhase] {
        let target = TrainingPhase(width: targetWidth, height: targetHeight, steps: totalSteps)

        // Benchmark mode should measure a stable shape; keep a single phase.
        if benchmarkSteps != nil { return [target] }
        guard enabled, totalSteps >= 20 else { return [target] }

        func floorToMultiple(_ value: Int, multiple: Int) -> Int {
            max(multiple, (value / multiple) * multiple)
        }

        let lowW = floorToMultiple(Int(Double(targetWidth) * 0.5), multiple: multiple)
        let lowH = floorToMultiple(Int(Double(targetHeight) * 0.5), multiple: multiple)
        let midW = floorToMultiple(Int(Double(targetWidth) * 0.75), multiple: multiple)
        let midH = floorToMultiple(Int(Double(targetHeight) * 0.75), multiple: multiple)

        // Heavily bias toward lower resolutions: attention cost grows quickly with seqLen.
        // On Apple GPUs we see ~3s/step @ 512, ~7s/step @ 768, ~30s/step @ 1024, so we keep
        // the full-res phase small by default and let users opt out via --no-progressive.
        let maxSide = max(targetWidth, targetHeight)
        let (lowFrac, midFrac): (Double, Double) = {
            if maxSide >= 1024 { return (0.80, 0.18) }   // 2% full-res
            if maxSide >= 768 { return (0.75, 0.20) }    // 5% full-res
            return (0.70, 0.20)                          // 10% full-res
        }()

        var lowSteps = Int(Double(totalSteps) * lowFrac)
        var midSteps = Int(Double(totalSteps) * midFrac)
        var highSteps = totalSteps - lowSteps - midSteps
        if highSteps <= 0 {
            highSteps = totalSteps
            lowSteps = 0
            midSteps = 0
        }

        let candidates: [TrainingPhase] = [
            TrainingPhase(width: lowW, height: lowH, steps: lowSteps),
            TrainingPhase(width: midW, height: midH, steps: midSteps),
            TrainingPhase(width: targetWidth, height: targetHeight, steps: highSteps),
        ]

        var merged: [TrainingPhase] = []
        merged.reserveCapacity(3)

        for phase in candidates where phase.steps > 0 {
            if let last = merged.last, last.width == phase.width, last.height == phase.height {
                merged[merged.count - 1] = TrainingPhase(width: last.width, height: last.height, steps: last.steps + phase.steps)
            } else {
                merged.append(phase)
            }
        }

        // If scaling didn't change anything (e.g. tiny targets), fall back to a single phase.
        if merged.count == 1 { return merged }
        if merged.allSatisfy({ $0.width == targetWidth && $0.height == targetHeight }) {
            return [target]
        }

        return merged
    }
}

