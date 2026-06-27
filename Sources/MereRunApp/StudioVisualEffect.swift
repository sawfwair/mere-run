import AppKit
import SwiftUI

/// A thin SwiftUI bridge to NSVisualEffectView so the chromeless title-bar region reads with native
/// macOS material/vibrancy (translucent, blurs content behind the window) in both light and dark.
struct VisualEffectBackground: NSViewRepresentable {
    var material: NSVisualEffectView.Material = .headerView
    var blendingMode: NSVisualEffectView.BlendingMode = .behindWindow

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = PassthroughVisualEffectView()
        view.material = material
        view.blendingMode = blendingMode
        view.state = .followsWindowActiveState
        return view
    }

    func updateNSView(_ view: NSVisualEffectView, context: Context) {
        view.material = material
        view.blendingMode = blendingMode
    }
}

/// Decorative only: returns nil from hitTest so it never intercepts clicks or the window-drag
/// region of the chromeless title bar above it.
private final class PassthroughVisualEffectView: NSVisualEffectView {
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
}
