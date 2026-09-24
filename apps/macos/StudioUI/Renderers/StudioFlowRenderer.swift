import StudioKit
import SwiftUI

// Dense optical flow, as `vision flow` writes it: a field of (dx, dy) per pixel. The canvas
// draws it as direction-colored vectors on a sampled grid; the result panel says how much moved.

/// The flow field as vectors: hue by direction, length by log magnitude, one every few pixels.
struct StudioFlowFieldView: View {
    let field: StudioFlowField

    var body: some View {
        Canvas { context, size in
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black))
            let fitted = StudioAnalyzeGeometry.fittedRect(
                imageSize: CGSize(width: field.width, height: field.height), in: size
            )
            let step = max(1, min(field.width, field.height) / 28)
            let scaleX = fitted.width / CGFloat(field.width)
            let scaleY = fitted.height / CGFloat(field.height)
            for y in stride(from: 0, to: field.height, by: step) {
                for x in stride(from: 0, to: field.width, by: step) {
                    let vector = field.vectors[y * field.width + x]
                    let magnitude = hypot(Double(vector.x), Double(vector.y))
                    guard magnitude.isFinite, magnitude > 0.01 else { continue }
                    let angle = atan2(Double(vector.y), Double(vector.x))
                    let length = min(CGFloat(step) * 0.8, CGFloat(log1p(magnitude)) * 3 + 2)
                    let start = CGPoint(
                        x: fitted.minX + (CGFloat(x) + 0.5) * scaleX,
                        y: fitted.minY + (CGFloat(y) + 0.5) * scaleY
                    )
                    let end = CGPoint(x: start.x + cos(angle) * length, y: start.y + sin(angle) * length)
                    var path = Path()
                    path.move(to: start)
                    path.addLine(to: end)
                    let hue = (angle + .pi) / (2 * .pi)
                    context.stroke(path, with: .color(Color(hue: hue, saturation: 0.9, brightness: 1)), lineWidth: 1.2)
                }
            }
        }
        .aspectRatio(CGFloat(max(1, field.width)) / CGFloat(max(1, field.height)), contentMode: .fit)
        .overlay(alignment: .topLeading) {
            Text(field.summary)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(.white)
                .padding(8)
                .background(.black.opacity(0.55), in: Capsule())
                .padding(10)
        }
        .accessibilityLabel("Optical flow, \(field.summary)")
    }
}

/// The panel's reading of the field: its size and how far things moved.
struct StudioFlowStatisticsRows: View {
    let field: StudioFlowField

    var body: some View {
        let statistics = field.statistics
        VStack(spacing: 0) {
            StudioResultFactRow("Size", "\(field.width) × \(field.height) px")
            StudioResultFactRow("Mean", String(format: "%.2f px", statistics.meanMagnitude))
            StudioResultFactRow("Largest", String(format: "%.2f px", statistics.maximumMagnitude))
            StudioResultFactRow("Moving", String(format: "%.0f%% of pixels", statistics.movingFraction * 100))
        }
    }
}
