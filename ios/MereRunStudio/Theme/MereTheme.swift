import SwiftUI
import UIKit

/// The paper-and-bronze palette from the macOS Studio (`MereRunTheme.swift`),
/// expressed as dynamic UIColors so light and dark resolve with the system
/// appearance. Values must stay in lockstep with `DESIGN.md`.
enum MereTheme {
    static let background = dynamicColor(light: 0xFAF8F3, dark: 0x171614)
    static let surface = dynamicColor(light: 0xFFFFFF, dark: 0x23211D)
    static let surfaceRaised = dynamicColor(light: 0xF1EDE3, dark: 0x302D27)
    static let border = dynamicColor(light: 0xD8D2C6, dark: 0x4E493F)
    static let textPrimary = dynamicColor(light: 0x211C13, dark: 0xF1EDE3)
    static let textSecondary = dynamicColor(light: 0x5C564A, dark: 0xC9C1B3)
    static let textMuted = dynamicColor(light: 0x8A8273, dark: 0x918A7C)
    static let accent = dynamicColor(light: 0x9C7A2E, dark: 0xC9A65D)
    static let success = dynamicColor(light: 0x3E7A3A, dark: 0x8FBF7F)
    static let caution = dynamicColor(light: 0x8A6D1F, dark: 0xD6B45C)
    static let failure = dynamicColor(light: 0x9C3A2E, dark: 0xD98B7F)

    enum Spacing {
        static let xs: CGFloat = 8
        static let s: CGFloat = 10
        static let m: CGFloat = 14
        static let l: CGFloat = 18
        static let xl: CGFloat = 22
        static let xxl: CGFloat = 28
    }

    enum Radius {
        static let field: CGFloat = 9
        static let panel: CGFloat = 10
        static let pill: CGFloat = 18
    }

    private static func dynamicColor(light: UInt32, dark: UInt32) -> Color {
        Color(UIColor { traits in
            traits.userInterfaceStyle == .dark ? UIColor(rgb: dark) : UIColor(rgb: light)
        })
    }
}

private extension UIColor {
    convenience init(rgb: UInt32) {
        self.init(
            red: CGFloat((rgb >> 16) & 0xFF) / 255,
            green: CGFloat((rgb >> 8) & 0xFF) / 255,
            blue: CGFloat(rgb & 0xFF) / 255,
            alpha: 1
        )
    }
}

extension View {
    /// The standard panel container: surface fill and hairline border.
    func merePanel() -> some View {
        background(
            RoundedRectangle(cornerRadius: MereTheme.Radius.panel)
                .fill(MereTheme.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: MereTheme.Radius.panel)
                        .stroke(MereTheme.border.opacity(0.6), lineWidth: 1)
                )
        )
    }
}

/// Compact status cluster: a dot and a word, never color alone.
struct MereStatusLabel: View {
    let text: String
    let color: Color

    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(text)
                .font(.footnote)
                .foregroundStyle(MereTheme.textSecondary)
        }
        .accessibilityElement(children: .combine)
    }
}
