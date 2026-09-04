import CoreGraphics

/// Window and column sizing for the Studio shell. The sidebar is a native `NavigationSplitView`
/// column (collapsible, user-resizable); the Library is a fixed column inside the detail area.
enum StudioLayoutPolicy {
    static let sidebarWidth: CGFloat = 212
    static let libraryWidth: CGFloat = 248
    /// The inspector column beside the feed (Main board).
    static let inspectorWidth: CGFloat = 300
    /// The Command view column; wide enough for a flag label and a value field per row.
    static let commandWidth: CGFloat = 520
    /// The feed's reading measure: cards never grow past this, whatever the window width.
    static let feedMaxWidth: CGFloat = 640
    /// The prompt workspace needs this much room for the canvas and composer beside the Library.
    static let minimumCanvasWidth: CGFloat = 520

    static let defaultWindowWidth: CGFloat = 1_280
    static let defaultWindowHeight: CGFloat = 820
    /// Wide enough for a collapsed sidebar, the Library column, and the minimum canvas.
    static let minimumWindowWidth: CGFloat = libraryWidth + minimumCanvasWidth
    static let minimumWindowHeight: CGFloat = 520
}
