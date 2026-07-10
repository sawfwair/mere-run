import CoreGraphics

/// The Studio has two deliberate window presentations. Regular keeps the desktop three-column
/// workspace; compact promotes the canvas and composer while moving navigation into overlays.
enum StudioLayoutClass: Equatable {
    case compact
    case regular

    var isCompact: Bool { self == .compact }
}

enum StudioLayoutPolicy {
    static let sidebarWidth: CGFloat = 212
    static let libraryWidth: CGFloat = 272
    static let minimumCanvasWidth: CGFloat = 620

    /// The regular shell needs room for sidebar + library + canvas. Crossing this content-driven
    /// breakpoint changes the information architecture instead of squeezing those columns.
    static let compactBreakpoint: CGFloat = sidebarWidth + libraryWidth + minimumCanvasWidth + 16

    static let minimumWindowWidth: CGFloat = 480
    static let minimumWindowHeight: CGFloat = 440
    static let compactPanelInset: CGFloat = 12

    static func layoutClass(for width: CGFloat) -> StudioLayoutClass {
        width < compactBreakpoint ? .compact : .regular
    }

    static func compactPanelWidth(availableWidth: CGFloat, preferredWidth: CGFloat) -> CGFloat {
        min(preferredWidth, max(availableWidth - (compactPanelInset * 2), 0))
    }
}
