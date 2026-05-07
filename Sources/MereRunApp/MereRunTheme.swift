import SwiftUI

enum MereRunTheme {
    static let background = Color(hex: "171614")
    static let surface = Color(hex: "23211D")
    static let surfaceRaised = Color(hex: "302D27")
    static let border = Color(hex: "4E493F")
    static let textPrimary = Color(hex: "F1EDE3")
    static let textSecondary = Color(hex: "C9C1B3")
    static let textMuted = Color(hex: "918A7C")
    static let accent = Color(hex: "C9A65D")
    static let green = Color(hex: "8EAA74")
    static let yellow = Color(hex: "D2A24E")
    static let red = Color(hex: "D98072")

    static let titleFont = Font.system(size: 22, weight: .semibold)
    static let sectionFont = Font.system(size: 12, weight: .semibold)
    static let bodyFont = Font.system(size: 13, weight: .regular)
    static let captionFont = Font.system(size: 11, weight: .medium)
    static let monoFont = Font.system(size: 12, weight: .regular, design: .monospaced)

    static let cornerRadius: CGFloat = 8
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

extension View {
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
