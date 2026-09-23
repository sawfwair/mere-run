import StudioKit
import SwiftUI

/// The menu bar panel's "This Mac" section: CPU and decode rate as live tiles with two minutes of
/// history, and one memory bar that shows how much of the machine the API server holds beside
/// everything else. Machine readings come from `StudioMachineMonitor`, so the section works with
/// no server running; the decode tile and the server's share appear while one answers.
struct StudioMenuBarResources: View {
    @ObservedObject var machine: StudioMachineMonitor
    @ObservedObject var monitor: StudioServingMonitor
    let isServing: Bool

    var body: some View {
        if let sample = machine.latest {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    StudioLoadTile(
                        title: "CPU",
                        value: StudioMenuBarCopy.percent(sample.cpu),
                        values: machine.cpuHistory,
                        maximum: 1,
                        tint: MereRunTheme.accent
                    )
                    if isServing {
                        StudioLoadTile(
                            title: "Decode",
                            value: StudioMenuBarCopy.rate(monitor.throughputHistory.last),
                            values: monitor.throughputHistory,
                            maximum: max(monitor.throughputHistory.max() ?? 0, 1),
                            tint: MereRunTheme.green
                        )
                    }
                }
                StudioMemoryBreakdown(
                    serverBytes: isServing ? monitor.runtime?.memory?.currentBytes : nil,
                    usedBytes: sample.memoryUsedBytes,
                    totalBytes: sample.memoryTotalBytes
                )
            }
            .padding(.horizontal, 14)
            .padding(.bottom, 4)
        }
    }
}

/// One live number over its recent history: the title and value on top, a sparkline below.
private struct StudioLoadTile: View {
    let title: String
    let value: String
    let values: [Double]
    let maximum: Double
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(title.uppercased())
                    .font(.caption2.weight(.semibold))
                    .tracking(0.6)
                    .foregroundStyle(MereRunTheme.textMuted)
                Spacer(minLength: 6)
                Text(value)
                    .font(.callout.weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .contentTransition(.numericText())
            }
            StudioSparkline(values: values, maximum: maximum, tint: tint)
                .frame(height: 28)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                .fill(MereRunTheme.surfaceRaised.opacity(0.55))
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(title)
        .accessibilityValue(value)
    }
}

/// A line over a soft fill, newest reading at the right edge. The x axis always spans a full
/// history, so a young history grows in from the right rather than stretching across.
struct StudioSparkline: View {
    let values: [Double]
    let maximum: Double
    let tint: Color

    var body: some View {
        GeometryReader { geometry in
            let points = Self.points(values, maximum: maximum, in: geometry.size)
            ZStack {
                if let first = points.first, let last = points.last {
                    Path { path in
                        path.move(to: CGPoint(x: first.x, y: geometry.size.height))
                        points.forEach { path.addLine(to: $0) }
                        path.addLine(to: CGPoint(x: last.x, y: geometry.size.height))
                        path.closeSubpath()
                    }
                    .fill(LinearGradient(
                        colors: [tint.opacity(0.28), tint.opacity(0.02)],
                        startPoint: .top,
                        endPoint: .bottom
                    ))
                    Path { path in path.addLines(points) }
                        .stroke(tint, style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
                    Circle()
                        .fill(tint)
                        .frame(width: 4, height: 4)
                        .position(last)
                }
            }
        }
        .accessibilityHidden(true)
    }

    /// Each reading's point: the newest at the right edge, one slot per reading of a full history,
    /// the value scaled into the height with a 2pt margin so the line never clips.
    static func points(_ values: [Double], maximum: Double, in size: CGSize) -> [CGPoint] {
        let slots = max(StudioMachineMonitor.historyLength - 1, 1)
        let step = size.width / CGFloat(slots)
        let usable = size.height - 4
        return values.suffix(StudioMachineMonitor.historyLength).reversed().enumerated().map { index, value in
            let share = maximum > 0 ? min(max(value / maximum, 0), 1) : 0
            return CGPoint(x: size.width - CGFloat(index) * step, y: 2 + usable * (1 - CGFloat(share)))
        }.reversed()
    }
}

/// Memory as one bar: the API server's footprint, everything else in use, and what is free, with
/// the numbers beneath.
private struct StudioMemoryBreakdown: View {
    let serverBytes: UInt64?
    let usedBytes: UInt64
    let totalBytes: UInt64

    private var server: UInt64 { min(serverBytes ?? 0, usedBytes) }
    private var other: UInt64 { usedBytes - server }
    private var free: UInt64 { totalBytes > usedBytes ? totalBytes - usedBytes : 0 }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("MEMORY")
                    .font(.caption2.weight(.semibold))
                    .tracking(0.6)
                    .foregroundStyle(MereRunTheme.textMuted)
                Spacer(minLength: 6)
                Text(StudioMenuBarCopy.memoryUsage(used: usedBytes, total: totalBytes))
                    .font(.caption.weight(.medium))
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textSecondary)
            }
            GeometryReader { geometry in
                HStack(spacing: 2) {
                    if server > 0 {
                        segment(server, of: geometry.size.width, MereRunTheme.accent)
                    }
                    segment(other, of: geometry.size.width, MereRunTheme.textMuted.opacity(0.45))
                    Spacer(minLength: 0)
                }
                .background(Capsule().fill(MereRunTheme.surfaceRaised))
                .clipShape(Capsule())
            }
            .frame(height: 8)
            HStack(spacing: 12) {
                if server > 0 {
                    legend("Server", bytes: server, color: MereRunTheme.accent)
                }
                legend(server > 0 ? "Other" : "In use", bytes: other, color: MereRunTheme.textMuted.opacity(0.6))
                Spacer(minLength: 0)
                Text("\(StudioMenuBarCopy.size(free)) free")
                    .font(.caption2.weight(.medium))
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                .fill(MereRunTheme.surfaceRaised.opacity(0.55))
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Memory")
    }

    private func segment(_ bytes: UInt64, of width: CGFloat, _ color: Color) -> some View {
        let share = totalBytes > 0 ? CGFloat(Double(bytes) / Double(totalBytes)) : 0
        return Capsule()
            .fill(color)
            .frame(width: max(width * share, bytes > 0 ? 3 : 0))
    }

    private func legend(_ title: String, bytes: UInt64, color: Color) -> some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text("\(title) \(StudioMenuBarCopy.size(bytes))")
                .font(.caption2.weight(.medium))
                .monospacedDigit()
                .foregroundStyle(MereRunTheme.textSecondary)
                .fixedSize()
        }
    }
}
