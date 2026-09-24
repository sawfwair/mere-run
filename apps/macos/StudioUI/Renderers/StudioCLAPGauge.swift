import StudioKit
import SwiftUI

/// Sound ▸ Score's result: the CLAP similarity as a ring, the number inside it, and the words it
/// was scored against. CLAP measures how well a prompt and a sound agree, so the gauge reads as
/// alignment, never as quality.
struct StudioCLAPGauge: View {
    let output: StudioCLAPScore.Output

    private static let diameter: CGFloat = 168
    private static let ringWidth: CGFloat = 14

    private var fraction: Double { min(1, max(0, output.score)) }

    var body: some View {
        VStack(spacing: 14) {
            ZStack {
                Circle()
                    .stroke(MereRunTheme.border.opacity(0.7), lineWidth: Self.ringWidth)
                Circle()
                    .trim(from: 0, to: fraction)
                    .stroke(MereRunTheme.accent, style: StrokeStyle(lineWidth: Self.ringWidth, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                VStack(spacing: 2) {
                    Text(output.score, format: .number.precision(.fractionLength(3)))
                        .font(.system(size: 34, weight: .bold, design: .rounded))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .monospacedDigit()
                    Text("alignment")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            .frame(width: Self.diameter, height: Self.diameter)
            if let prompt = output.prompt, !prompt.isBlank {
                Text("\u{201C}\(prompt)\u{201D}")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(3)
            }
            Text("How well the sound matches the words. Use it to rank candidates, not as an absolute quality score.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 14)
        .padding(.vertical, 18)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibilityLabel)
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.27))
            .frame(height: 1)
    }

    private var accessibilityLabel: String {
        let score = output.score.formatted(.number.precision(.fractionLength(3)))
        guard let prompt = output.prompt, !prompt.isBlank else { return "CLAP alignment \(score)" }
        return "CLAP alignment \(score) against \u{201C}\(prompt)\u{201D}"
    }
}
