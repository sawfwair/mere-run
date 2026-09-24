import StudioKit
import SwiftUI

/// The trainer dashboards' loss curve: step along the bottom, loss up the side, one stroke from
/// the first logged step to the latest. Drawn from the run's events file, so a finished run and
/// a run in flight read the same way.
struct StudioTrainingLossChart: View {
    let points: [(step: Int, loss: Double)]

    var body: some View {
        GeometryReader { proxy in
            Canvas { context, size in
                let inset: CGFloat = 26
                let rect = CGRect(
                    x: inset,
                    y: 12,
                    width: max(1, size.width - inset - 12),
                    height: max(1, size.height - inset - 18)
                )
                context.stroke(
                    Path { path in
                        path.move(to: CGPoint(x: rect.minX, y: rect.minY))
                        path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
                        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
                    },
                    with: .color(MereRunTheme.border),
                    lineWidth: 1
                )
                guard points.count > 1 else {
                    context.draw(
                        Text("Waiting for loss points")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted),
                        at: CGPoint(x: rect.midX, y: rect.midY)
                    )
                    return
                }
                let minStep = points.map(\.step).min() ?? 0
                let maxStep = points.map(\.step).max() ?? 1
                let minLoss = points.map(\.loss).min() ?? 0
                let maxLoss = points.map(\.loss).max() ?? 1
                func x(_ step: Int) -> CGFloat {
                    rect.minX + CGFloat(Double(step - minStep) / Double(max(maxStep - minStep, 1))) * rect.width
                }
                func y(_ loss: Double) -> CGFloat {
                    let range = max(maxLoss - minLoss, 0.000_000_1)
                    return rect.maxY - CGFloat((loss - minLoss) / range) * rect.height
                }
                var path = Path()
                for (index, point) in points.enumerated() {
                    let location = CGPoint(x: x(point.step), y: y(point.loss))
                    if index == 0 {
                        path.move(to: location)
                    } else {
                        path.addLine(to: location)
                    }
                }
                context.stroke(
                    path,
                    with: .linearGradient(
                        Gradient(colors: [MereRunTheme.accent, MereRunTheme.green]),
                        startPoint: CGPoint(x: rect.minX, y: rect.midY),
                        endPoint: CGPoint(x: rect.maxX, y: rect.midY)
                    ),
                    style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round)
                )
            }
        }
        .padding(8)
        .accessibilityElement()
        .accessibilityLabel("Training loss chart with \(points.count) points")
    }
}
