import AppKit
import SwiftUI

enum MereRunTheme {
    // Semantic tokens resolve per the view's effective appearance (light/dark) so the whole app —
    // including AppKit-backed native controls — adapts when the system appearance changes. The dark
    // values are the original palette; the light values are a warm paper palette tuned for contrast.
    static let background = dynamic(light: "FAF8F3", dark: "171614")
    static let surface = dynamic(light: "FFFFFF", dark: "23211D")
    static let surfaceRaised = dynamic(light: "F1EDE3", dark: "302D27")
    static let border = dynamic(light: "D8D2C6", dark: "4E493F")
    static let textPrimary = dynamic(light: "211C13", dark: "F1EDE3")
    static let textSecondary = dynamic(light: "5C564A", dark: "C9C1B3")
    static let textMuted = dynamic(light: "8A8273", dark: "918A7C")
    static let accent = dynamic(light: "9C7A2E", dark: "C9A65D")
    static let green = dynamic(light: "5E7A45", dark: "8EAA74")
    static let yellow = dynamic(light: "9C7520", dark: "D2A24E")
    static let red = dynamic(light: "C2493B", dark: "D98072")

    /// Drop-shadow color tuned per appearance (soft neutral in light, deep black in dark).
    static let shadowColor = dynamicNSColor(
        light: NSColor(calibratedWhite: 0.35, alpha: 0.16),
        dark: NSColor(calibratedWhite: 0.0, alpha: 0.28)
    )

    static let titleFont = Font.system(size: 22, weight: .semibold)
    static let sectionFont = Font.system(size: 12, weight: .semibold)
    static let bodyFont = Font.system(size: 13, weight: .regular)
    static let captionFont = Font.system(size: 11, weight: .medium)
    static let monoFont = Font.system(size: 12, weight: .regular, design: .monospaced)

    static let cornerRadius: CGFloat = 8

    /// A SwiftUI color that resolves to `light` or `dark` against the current appearance.
    static func dynamic(light: String, dark: String) -> Color {
        dynamicNSColor(light: NSColor(hex: light), dark: NSColor(hex: dark))
    }

    static func dynamicNSColor(light: NSColor, dark: NSColor) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? dark : light
        })
    }
}

extension Color {
    init(hex: String) {
        let cleaned = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var value: UInt64 = 0
        Scanner(string: cleaned).scanHexInt64(&value)

        let red: UInt64
        let green: UInt64
        let blue: UInt64
        let alpha: UInt64

        switch cleaned.count {
        case 8:
            alpha = value >> 24
            red = value >> 16 & 0xff
            green = value >> 8 & 0xff
            blue = value & 0xff
        case 6:
            alpha = 255
            red = value >> 16 & 0xff
            green = value >> 8 & 0xff
            blue = value & 0xff
        default:
            alpha = 255
            red = 21
            green = 25
            blue = 26
        }

        self.init(
            .sRGB,
            red: Double(red) / 255,
            green: Double(green) / 255,
            blue: Double(blue) / 255,
            opacity: Double(alpha) / 255
        )
    }
}

extension NSColor {
    /// Mirrors `Color(hex:)` for the AppKit colors that back the dynamic theme tokens.
    convenience init(hex: String) {
        let cleaned = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var value: UInt64 = 0
        Scanner(string: cleaned).scanHexInt64(&value)

        let red: UInt64
        let green: UInt64
        let blue: UInt64
        let alpha: UInt64

        switch cleaned.count {
        case 8:
            alpha = value >> 24
            red = value >> 16 & 0xff
            green = value >> 8 & 0xff
            blue = value & 0xff
        case 6:
            alpha = 255
            red = value >> 16 & 0xff
            green = value >> 8 & 0xff
            blue = value & 0xff
        default:
            alpha = 255
            red = 21
            green = 25
            blue = 26
        }

        self.init(
            srgbRed: CGFloat(red) / 255,
            green: CGFloat(green) / 255,
            blue: CGFloat(blue) / 255,
            alpha: CGFloat(alpha) / 255
        )
    }
}

extension View {
    /// A theme-aware drop shadow used for raised surfaces (cards, popovers, the composer).
    func mereShadow(radius: CGFloat, y: CGFloat = 0) -> some View {
        shadow(color: MereRunTheme.shadowColor, radius: radius, x: 0, y: y)
    }

    func merePanel(cornerRadius: CGFloat = MereRunTheme.cornerRadius) -> some View {
        background {
            RoundedRectangle(cornerRadius: cornerRadius)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                }
        }
    }
}
