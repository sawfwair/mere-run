import Foundation
import MereRunCore

final class ModelPullProgressPrinter: @unchecked Sendable {
    private let modelID: String
    private let now: @Sendable () -> Date
    private let lock = NSLock()
    private var state = ModelPullProgressFormatterState()

    init(modelID: String, now: @escaping @Sendable () -> Date = Date.init) {
        self.modelID = modelID
        self.now = now
    }

    func render(_ progress: ManagedModelResolver.InstallProgress) -> String {
        lock.lock()
        defer { lock.unlock() }
        return state.render(modelID: modelID, progress: progress, now: now())
    }
}

struct ModelPullProgressFormatterState {
    private var lastCompletedBytes: Int64?
    private var lastSampleDate: Date?
    private var smoothedSpeedBytesPerSecond: Double?

    mutating func render(
        modelID: String,
        progress: ManagedModelResolver.InstallProgress,
        now: Date
    ) -> String {
        switch progress {
        case .downloadingBytes(let completed, let total):
            return renderBytes(modelID: modelID, completed: completed, total: total, now: now)
        case .downloadingPercent(let percent, let speed):
            return renderPercent(modelID: modelID, percent: percent, speed: speed)
        case .extracting:
            return "[\(modelID)] extracting…"
        }
    }

    private mutating func renderBytes(
        modelID: String,
        completed: Int64,
        total: Int64?,
        now: Date
    ) -> String {
        let boundedCompleted = boundedCompletedBytes(completed, total: total)
        let speed = updateSpeed(completed: boundedCompleted, now: now)
        let completedText = formatBytes(boundedCompleted)
        var detailParts: [String] = []

        if let total, total > 0 {
            let totalText = formatBytes(total)
            detailParts.append(formatPercent(completed: boundedCompleted, total: total))
            detailParts.append("\(completedText) / \(totalText)")
            if let speed, speed > 0 {
                detailParts.append(formatRate(speed))
                let remainingBytes = max(total - boundedCompleted, 0)
                if remainingBytes > 0 {
                    detailParts.append("ETA \(formatDuration(Double(remainingBytes) / speed))")
                }
            }
        } else {
            detailParts.append(completedText)
            if let speed, speed > 0 {
                detailParts.append(formatRate(speed))
            }
        }

        return "[\(modelID)] \(detailParts.joined(separator: "  "))"
    }

    private func renderPercent(modelID: String, percent: Int, speed: Double?) -> String {
        let clamped = min(100, max(0, percent))
        guard let speed, speed > 0 else {
            return "[\(modelID)] \(clamped)%"
        }
        return "[\(modelID)] \(clamped)%  \(formatRate(speed))"
    }

    private mutating func updateSpeed(completed: Int64, now: Date) -> Double? {
        defer {
            lastCompletedBytes = completed
            lastSampleDate = now
        }

        guard let lastCompletedBytes, let lastSampleDate else {
            return nil
        }
        if completed < lastCompletedBytes {
            smoothedSpeedBytesPerSecond = nil
            return nil
        }

        let elapsed = now.timeIntervalSince(lastSampleDate)
        let delta = completed - lastCompletedBytes
        guard elapsed > 0, delta > 0 else {
            return smoothedSpeedBytesPerSecond
        }

        let instantSpeed = Double(delta) / elapsed
        if let smoothedSpeedBytesPerSecond {
            self.smoothedSpeedBytesPerSecond = (smoothedSpeedBytesPerSecond * 0.7) + (instantSpeed * 0.3)
        } else {
            smoothedSpeedBytesPerSecond = instantSpeed
        }
        return smoothedSpeedBytesPerSecond
    }

    private func boundedCompletedBytes(_ completed: Int64, total: Int64?) -> Int64 {
        let nonNegative = max(0, completed)
        guard let total, total > 0 else {
            return nonNegative
        }
        return min(nonNegative, total)
    }

    private func formatPercent(completed: Int64, total: Int64) -> String {
        guard total > 0 else { return "0%" }
        let clampedCompleted = min(max(completed, 0), total)
        if clampedCompleted == 0 {
            return "0%"
        }
        if clampedCompleted == total {
            return "100%"
        }

        let tenths = max(1, Int((Double(clampedCompleted) / Double(total) * 1_000).rounded(.down)))
        if tenths < 100 {
            return "\(tenths / 10).\(tenths % 10)%"
        }
        return "\(min(99, tenths / 10))%"
    }

    private func formatRate(_ bytesPerSecond: Double) -> String {
        let rounded = max(1, Int64(bytesPerSecond.rounded()))
        return "\(formatBytes(rounded))/s"
    }

    private func formatDuration(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds > 0 else {
            return "unknown"
        }
        let wholeSeconds = max(1, Int(seconds.rounded(.up)))
        if wholeSeconds < 60 {
            return "\(wholeSeconds)s"
        }
        let minutes = wholeSeconds / 60
        let remainingSeconds = wholeSeconds % 60
        if minutes < 60 {
            return "\(minutes)m \(remainingSeconds)s"
        }
        let hours = minutes / 60
        let remainingMinutes = minutes % 60
        return "\(hours)h \(remainingMinutes)m"
    }

    private func formatBytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
