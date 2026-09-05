import CoreGraphics

/// Window and column sizing for the Studio shell. The sidebar is a native `NavigationSplitView`
/// column (collapsible, user-resizable); the Library is a fixed column inside the detail area.
package enum StudioLayoutPolicy {
    package static let sidebarWidth: CGFloat = 212
    package static let libraryWidth: CGFloat = 248
    /// The inspector column beside the feed (Main board).
    package static let inspectorWidth: CGFloat = 300
    /// The Command view column; wide enough for a flag label and a value field per row.
    package static let commandWidth: CGFloat = 440
    /// The feed's reading measure: cards never grow past this, whatever the window width.
    package static let feedMaxWidth: CGFloat = 640
    /// The prompt workspace needs this much room for the canvas and composer beside the Library.
    package static let minimumCanvasWidth: CGFloat = 520

    package static let defaultWindowWidth: CGFloat = 1_440
    package static let defaultWindowHeight: CGFloat = 820
    /// Wide enough for a collapsed sidebar, the Library column, and the minimum canvas.
    package static let minimumWindowWidth: CGFloat = libraryWidth + minimumCanvasWidth
    package static let minimumWindowHeight: CGFloat = 520

    package struct Presentation: Equatable {
        package let showsLibrary: Bool
        package let panelIsInline: Bool
        package let panelWidth: CGFloat
    }

    /// Space is allocated to the result before auxiliary columns. A narrow panel overlays the
    /// result while open; it never increases the window's minimum content width.
    package static func presentation(width: CGFloat, library: Bool, inspector: Bool, command: Bool) -> Presentation {
        let panelWidth = command ? Self.commandWidth : (inspector ? Self.inspectorWidth : 0)
        let hasPanel = panelWidth > 0
        let showsLibrary = library && width >= minimumCanvasWidth + libraryWidth + 1
        let occupied = showsLibrary ? libraryWidth + 1 : 0
        return Presentation(showsLibrary: showsLibrary,
                            panelIsInline: hasPanel && width >= occupied + minimumCanvasWidth + panelWidth + 1,
                            panelWidth: min(panelWidth, width))
    }
}
