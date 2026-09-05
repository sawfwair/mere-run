import AppKit
import CoreText
import StudioKit
import SwiftUI

package enum MereRunTheme {
    // Semantic tokens resolve per the view's effective appearance (light/dark) so the whole app —
    // including AppKit-backed native controls — adapts when the system appearance changes. The dark
    // values are the original palette; the light values are a warm paper palette tuned for contrast.
    package static let background = dynamic(light: "FAF8F3", dark: "171614")
    package static let surface = dynamic(light: "FFFFFF", dark: "23211D")
    package static let surfaceRaised = dynamic(light: "F1EDE3", dark: "302D27")
    package static let border = dynamic(light: "D8D2C6", dark: "4E493F")
    package static let textPrimary = dynamic(light: "211C13", dark: "F1EDE3")
    package static let textSecondary = dynamic(light: "5C564A", dark: "C9C1B3")
    package static let textMuted = dynamic(light: "716959", dark: "A49B8D")
    package static let accent = dynamic(light: "86671F", dark: "C9A65D")
    /// A solid bronze wash for selection fills and the user chat bubble — solid (not
    /// accent-at-opacity) so stacked layers never drift in tone.
    package static let accentSoft = dynamic(light: "F0E7D2", dark: "39331F")
    package static let green = dynamic(light: "5E7A45", dark: "8EAA74")
    package static let yellow = dynamic(light: "9C7520", dark: "D2A24E")
    package static let red = dynamic(light: "C2493B", dark: "D98072")
    /// Glyphs and labels drawn on a solid `accent` fill (the selected sidebar row, primary buttons).
    package static let onAccent = dynamic(light: "FFFFFF", dark: "1B160A")
    /// The raised, selected segment of a segmented control sitting on `surfaceRaised`.
    package static let segmentedSelection = dynamic(light: "FFFFFF", dark: "3A362E")
    /// The wordmark's period — the one green that is brand, not status. Same in both appearances.
    package static let wordmarkGreen = dynamic(light: "2D6A4F", dark: "2D6A4F")

    /// Transient pointer-hover fill for rows and icon buttons.
    package static let hoverFill = dynamicNSColor(
        light: NSColor(calibratedWhite: 0.1, alpha: 0.06),
        dark: NSColor(calibratedWhite: 1.0, alpha: 0.07)
    )

    /// Drop-shadow color tuned per appearance (soft neutral in light, deep black in dark).
    package static let shadowColor = dynamicNSColor(
        light: NSColor(calibratedWhite: 0.35, alpha: 0.16),
        dark: NSColor(calibratedWhite: 0.0, alpha: 0.28)
    )

    // Relative to Dynamic Type text styles (anchored near the original point sizes) so the whole app
    // scales with the user's Larger Text setting while keeping the existing token names.
    package static let titleFont = Font.system(.title, weight: .semibold)
    package static let sectionFont = Font.system(.subheadline, weight: .semibold)
    package static let bodyFont = Font.system(.body)
    package static let captionFont = Font.system(.caption, weight: .medium)
    package static let monoFont = Font.system(.callout, design: .monospaced)
    // New York serif is reserved for short display moments (empty-state headlines, the welcome
    // hero) — never controls or body copy. Serif-on-paper is a core identity element.
    package static let displayFont = Font.system(.largeTitle, design: .serif, weight: .medium)
    package static let displaySmallFont = Font.system(.title2, design: .serif, weight: .medium)

    /// The brand face: Caveat (SIL Open Font License 1.1), bundled under `Resources/Fonts` and
    /// registered with Core Text on first use. Only the sidebar wordmark uses it.
    package enum Brand {
        package static let familyName = "Caveat"
        package static let fontFileName = "Caveat[wght]"
        package static let wordmarkSize: CGFloat = 27

        /// Caveat Medium at `size`, or the system serif when the bundled face is unavailable so the
        /// wordmark still reads as a wordmark.
        package static func font(size: CGFloat = wordmarkSize) -> Font {
            if isAvailable {
                return Font.custom(familyName, fixedSize: size).weight(.medium)
            }
            return Font.system(size: size, weight: .medium, design: .serif)
        }

        /// Whether Caveat can be drawn by this process — registered from the bundle by `register()`
        /// or already installed on the machine.
        package static var isAvailable: Bool {
            register()
            return NSFont(name: familyName, size: wordmarkSize) != nil
        }

        /// Registers the bundled font for this process. Safe to call repeatedly; the work happens
        /// once. Returns true when Caveat is available afterwards.
        @discardableResult
        package static func register() -> Bool {
            registration
        }

        private static let registration: Bool = {
            if NSFont(name: familyName, size: wordmarkSize) != nil { return true }
            guard let url = fontURL() else { return false }
            var error: Unmanaged<CFError>?
            if CTFontManagerRegisterFontsForURL(url as CFURL, .process, &error) { return true }
            // Registered earlier by another bundle in this process still counts as available.
            if let error = error?.takeRetainedValue(),
               CFErrorGetCode(error) == CTFontManagerError.alreadyRegistered.rawValue {
                return true
            }
            return false
        }()

        /// The bundled font file, looked up in the SwiftPM resource bundle wherever this process
        /// finds it: `Contents/Resources` of the packaged app, or beside the executable (or the
        /// test bundle) in a SwiftPM build directory. Never `Bundle.module`, whose accessor traps
        /// when the bundle is missing; a missing font must degrade, not crash.
        package static func fontURL() -> URL? {
            let bundleName = "MereRun_StudioUI.bundle"
            var roots: [URL] = []
            if let resources = Bundle.main.resourceURL { roots.append(resources) }
            roots.append(Bundle.main.bundleURL)
            if let executable = Bundle.main.executableURL {
                roots.append(executable.deletingLastPathComponent())
            }
            let hostBundle = Bundle(for: BrandBundleLocator.self)
            roots.append(hostBundle.bundleURL.deletingLastPathComponent())
            if let hostResources = hostBundle.resourceURL { roots.append(hostResources) }
            for root in roots {
                let candidate = root
                    .appendingPathComponent(bundleName, isDirectory: true)
                    .appendingPathComponent("Fonts", isDirectory: true)
                    .appendingPathComponent(fontFileName, isDirectory: false)
                    .appendingPathExtension("ttf")
                if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            }
            return nil
        }
    }

    /// Consistent spacing scale for padding/stack spacing.
    package enum Spacing {
        package static let xs: CGFloat = 8
        package static let sm: CGFloat = 10
        package static let md: CGFloat = 14
        package static let lg: CGFloat = 18
        package static let xl: CGFloat = 22
        package static let xxl: CGFloat = 28
        package static let xxxl: CGFloat = 32
    }

    /// Consistent corner-radius scale.
    package enum Radius {
        package static let sm: CGFloat = 6
        package static let base: CGFloat = 8
        package static let md: CGFloat = 9
        package static let lg: CGFloat = 10
        /// Floating panels that overlay the shell (the Activity popover).
        package static let popover: CGFloat = 12
        package static let xl: CGFloat = 18
        package static let xxl: CGFloat = 20
    }

    package static let cornerRadius: CGFloat = Radius.base

    /// The app's motion vocabulary: exponential ease-outs for state fades, soft springs for
    /// selection and entrances. Nothing bounces.
    package enum Motion {
        package static let quick = Animation.easeOut(duration: 0.15)
        package static let standard = Animation.easeOut(duration: 0.22)
        package static let spring = Animation.spring(response: 0.32, dampingFraction: 0.86)
        package static let gentleSpring = Animation.spring(response: 0.5, dampingFraction: 0.9)
    }

    /// A SwiftUI color that resolves to `light` or `dark` against the current appearance.
    package static func dynamic(light: String, dark: String) -> Color {
        dynamicNSColor(light: NSColor(hex: light), dark: NSColor(hex: dark))
    }

    package static func dynamicNSColor(light: NSColor, dark: NSColor) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? dark : light
        })
    }
}

/// Anchors `Bundle(for:)` to this module: the app executable, or the test host it is linked into.
private final class BrandBundleLocator {}

extension Color {
    package init(hex: String) {
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
    package convenience init(hex: String) {
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
    package func mereShadow(radius: CGFloat, y: CGFloat = 0) -> some View {
        shadow(color: MereRunTheme.shadowColor, radius: radius, x: 0, y: y)
    }

    package func merePanel(cornerRadius: CGFloat = MereRunTheme.cornerRadius) -> some View {
        background {
            RoundedRectangle(cornerRadius: cornerRadius)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                }
        }
    }

    /// An accent focus treatment for the element that currently owns keyboard input (the
    /// composer): a warm ring plus a faint glow, fading with `active`.
    package func mereFocusRing(_ active: Bool, cornerRadius: CGFloat) -> some View {
        overlay {
            RoundedRectangle(cornerRadius: cornerRadius)
                .strokeBorder(MereRunTheme.accent.opacity(active ? 0.55 : 0), lineWidth: 1.5)
        }
        .shadow(
            color: active ? MereRunTheme.accent.opacity(0.16) : .clear,
            radius: active ? 9 : 0,
            y: 2
        )
        .animation(MereRunTheme.Motion.standard, value: active)
    }
}
