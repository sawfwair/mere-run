import CoreGraphics

/// Window and column sizing for the Studio shell. The sidebar is a native `NavigationSplitView`
/// column (collapsible, user-resizable); the Library is a fixed column inside the detail area.
enum StudioLayoutPolicy {
    static let sidebarWidth: CGFloat = 212
    static let libraryWidth: CGFloat = 248
    /// The prompt workspace needs this much room for the canvas and composer beside the Library.
    static let minimumCanvasWidth: CGFloat = 520

    static let defaultWindowWidth: CGFloat = 1_280
    static let defaultWindowHeight: CGFloat = 820
    /// Wide enough for a collapsed sidebar, the Library column, and the minimum canvas.
    static let minimumWindowWidth: CGFloat = libraryWidth + minimumCanvasWidth
    static let minimumWindowHeight: CGFloat = 520
}
