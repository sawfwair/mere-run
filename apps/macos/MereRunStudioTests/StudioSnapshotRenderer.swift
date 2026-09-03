import AppKit
import SwiftUI

/// Which system appearance a snapshot is rendered under.
enum StudioSnapshotAppearance: String, CaseIterable {
    case light
    case dark

    var nsAppearance: NSAppearance? {
        switch self {
        case .light: return NSAppearance(named: .aqua)
        case .dark: return NSAppearance(named: .darkAqua)
        }
    }
}

/// Renders a SwiftUI view to a PNG without presenting anything on screen.
///
/// The view is hosted in an `NSWindow` that is never ordered in: the harness never calls
/// `makeKeyAndOrderFront`, `orderFront`, or `NSApp.activate`, marks the window not visible, and
/// sets the test process's activation policy to `.prohibited` so no Dock icon appears either. The
/// window is titled with a unified toolbar rather than borderless so SwiftUI's `.toolbar` content
/// is bridged into a real `NSToolbar` and shows up in the render; visibility on screen is a matter
/// of ordering, not style, so this keeps the offscreen guarantee. The pixels come from
/// `cacheDisplay(in:to:)` on the window's frame view, which draws the window contents into a bitmap
/// without the window server ever seeing them.
///
/// Two fidelity gaps are known and accepted, both from macOS 26 compositing that only happens on
/// screen. The sidebar column and the toolbar platters are `NSGlassEffectView`s whose content the
/// window server composites, so an offscreen draw shows them as opaque white with no content;
/// before capturing, the renderer lifts each glass view's `contentView` out into the glass view's
/// superview at the same frame and hides the glass, so sidebar rows and toolbar items draw as plain
/// views over the window background without the translucent material. And a transparent title bar
/// (`titlebarAppearsTransparent`, what the app's `.hiddenTitleBar` style produces) makes every
/// `ScrollView` under it draw nothing offscreen, so the window keeps an opaque title bar and the
/// shell background does not extend under the toolbar the way it does in the app.
@MainActor
enum StudioSnapshotRenderer {
    /// Render `view` at `size` (points) under `appearance` and return the bitmap.
    ///
    /// `settle` bounds how long the main run loop is pumped so SwiftUI and AppKit finish laying
    /// out `NavigationSplitView`, toolbars, and any `.task` work that flips state on first appear.
    /// `afterAppear` runs once the view has appeared and had `settle / 2` to settle, before the
    /// final settle; use it to drive state the same way a user would (for example, navigating a
    /// `NavigationModel` to another domain) so `onAppear` restoration has already happened.
    static func render<Content: View>(
        _ view: Content,
        size: CGSize,
        appearance: StudioSnapshotAppearance,
        settle: TimeInterval = 1.5,
        afterAppear: (() -> Void)? = nil
    ) throws -> NSBitmapImageRep {
        prohibitActivation()

        let window = makeOffscreenWindow(size: size)
        defer {
            window.contentView = nil
            window.close()
        }
        window.appearance = appearance.nsAppearance

        let hostingView = NSHostingView(rootView: view)
        hostingView.sceneBridgingOptions = [.toolbars, .title]
        hostingView.frame = CGRect(origin: .zero, size: size)
        window.contentView = hostingView

        if let afterAppear {
            pumpMainRunLoop(for: settle / 2, window: window)
            afterAppear()
        }
        pumpMainRunLoop(for: settle, window: window)

        guard let frameView = window.contentView?.superview ?? window.contentView else {
            throw StudioSnapshotError.noContentView
        }
        if prepareForOffscreenCapture(frameView, appearance: appearance.nsAppearance) {
            pumpMainRunLoop(for: 0.3, window: window)
        }
        frameView.layoutSubtreeIfNeeded()
        guard let captured = frameView.bitmapImageRepForCachingDisplay(in: frameView.bounds) else {
            throw StudioSnapshotError.noBitmap
        }
        frameView.cacheDisplay(in: frameView.bounds, to: captured)
        return try flatten(captured, onto: .windowBackgroundColor, appearance: appearance.nsAppearance)
    }

    /// Render `view` and write it as PNG to `url`, creating parent directories as needed.
    @discardableResult
    static func writePNG<Content: View>(
        _ view: Content,
        size: CGSize,
        appearance: StudioSnapshotAppearance,
        to url: URL,
        settle: TimeInterval = 1.5
    ) throws -> URL {
        let rep = try render(view, size: size, appearance: appearance, settle: settle)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw StudioSnapshotError.pngEncodingFailed
        }
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url, options: .atomic)
        return url
    }

    /// Fraction of sampled pixels that are neither near-black nor near-white nor fully
    /// transparent. A real render of the shell is comfortably above zero; a blank window is not.
    static func nonBlankCoverage(of rep: NSBitmapImageRep, samplesPerAxis: Int = 48) -> Double {
        let width = rep.pixelsWide
        let height = rep.pixelsHigh
        guard width > 0, height > 0 else { return 0 }
        var distinct = Set<UInt32>()
        var informative = 0
        var total = 0
        for row in 0..<samplesPerAxis {
            for column in 0..<samplesPerAxis {
                let x = column * (width - 1) / max(samplesPerAxis - 1, 1)
                let y = row * (height - 1) / max(samplesPerAxis - 1, 1)
                guard let color = rep.colorAt(x: x, y: y) else { continue }
                total += 1
                let red = color.redComponent
                let green = color.greenComponent
                let blue = color.blueComponent
                let alpha = color.alphaComponent
                let key = UInt32(red * 255) << 24 | UInt32(green * 255) << 16 | UInt32(blue * 255) << 8 | UInt32(alpha * 255)
                distinct.insert(key)
                let luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                if alpha > 0.01, luminance > 0.02, luminance < 0.98 {
                    informative += 1
                }
            }
        }
        guard total > 0, distinct.count > 1 else { return 0 }
        return Double(informative) / Double(total)
    }

    // MARK: - Window-server-only effects

    /// Names of the private AppKit views that only composite on screen. Hiding them offscreen is
    /// what lets the views they decorate draw at all; nothing else about them is relied on.
    private static let scrollEdgeEffectClassNames: Set<String> = ["NSScrollPocket", "BackdropView"]
    private static let sidebarBackdropClassName = "NSBlurryAlleywayView"

    /// Neutralises everything under `root` that the window server would normally composite and
    /// that therefore draws as blank offscreen. Returns true when anything changed.
    ///
    /// - `NSGlassEffectView`: its `contentView` is lifted into the glass view's superview at the
    ///   same frame and the glass is hidden (macOS 26).
    /// - Scroll edge effects (`NSScrollPocket`, `BackdropView` under an `NSScrollView`): hidden.
    ///   Left in place they blank the whole scroll view, including the sidebar `List` (macOS 26).
    /// - The sidebar column's translucent backdrop (`NSBlurryAlleywayView`): replaced by an opaque
    ///   window-background view so the vibrant sidebar text and symbols blend against a surface
    ///   instead of against nothing.
    @discardableResult
    private static func prepareForOffscreenCapture(_ root: NSView, appearance: NSAppearance?) -> Bool {
        var changed = false
        for subview in root.subviews {
            if prepareForOffscreenCapture(subview, appearance: appearance) {
                changed = true
            }
        }
        let className = String(describing: type(of: root))

        if let scrollView = root as? NSScrollView {
            for subview in scrollView.subviews
            where scrollEdgeEffectClassNames.contains(String(describing: type(of: subview))) && !subview.isHidden {
                subview.isHidden = true
                changed = true
            }
        }

        if className == sidebarBackdropClassName, !root.isHidden, let superview = root.superview {
            let backing = NSView(frame: root.frame)
            backing.autoresizingMask = [.width, .height]
            backing.wantsLayer = true
            backing.layer?.backgroundColor = resolvedCGColor(.windowBackgroundColor, appearance: appearance)
            superview.addSubview(backing, positioned: .below, relativeTo: root)
            root.isHidden = true
            changed = true
        }

        if #available(macOS 26.0, *),
           let glass = root as? NSGlassEffectView,
           let content = glass.contentView,
           let superview = glass.superview {
            glass.contentView = nil
            content.translatesAutoresizingMaskIntoConstraints = false
            superview.addSubview(content, positioned: .above, relativeTo: glass)
            NSLayoutConstraint.activate([
                content.leadingAnchor.constraint(equalTo: superview.leadingAnchor, constant: glass.frame.minX),
                content.topAnchor.constraint(
                    equalTo: superview.topAnchor,
                    constant: superview.isFlipped ? glass.frame.minY : superview.bounds.height - glass.frame.maxY
                ),
                content.widthAnchor.constraint(equalToConstant: glass.frame.width),
                content.heightAnchor.constraint(equalToConstant: glass.frame.height)
            ])
            glass.isHidden = true
            changed = true
        }
        return changed
    }

    private static func resolvedCGColor(_ color: NSColor, appearance: NSAppearance?) -> CGColor {
        var resolved = color.cgColor
        appearance?.performAsCurrentDrawingAppearance {
            resolved = color.cgColor
        }
        return resolved
    }

    /// Composites `rep` over a solid fill so regions the offscreen draw leaves transparent (the
    /// window's rounded corners, backdrop layers) read as the window background, not black.
    private static func flatten(
        _ rep: NSBitmapImageRep,
        onto color: NSColor,
        appearance: NSAppearance?
    ) throws -> NSBitmapImageRep {
        guard let flattened = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: rep.pixelsWide,
            pixelsHigh: rep.pixelsHigh,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        ), let context = NSGraphicsContext(bitmapImageRep: flattened) else {
            throw StudioSnapshotError.noBitmap
        }
        let bounds = CGRect(x: 0, y: 0, width: rep.pixelsWide, height: rep.pixelsHigh)
        let draw = {
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = context
            color.setFill()
            bounds.fill()
            rep.draw(in: bounds, from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: false, hints: nil)
            NSGraphicsContext.restoreGraphicsState()
        }
        if let appearance {
            appearance.performAsCurrentDrawingAppearance(draw)
        } else {
            draw()
        }
        return flattened
    }

    // MARK: - Offscreen window

    private static var didProhibitActivation = false

    private static func prohibitActivation() {
        guard !didProhibitActivation else { return }
        didProhibitActivation = true
        let app = NSApplication.shared
        if app.activationPolicy() != .prohibited {
            app.setActivationPolicy(.prohibited)
        }
    }

    private static func makeOffscreenWindow(size: CGSize) -> NSWindow {
        let window = NSWindow(
            contentRect: CGRect(origin: .zero, size: size),
            styleMask: [.titled, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.isReleasedWhenClosed = false
        // Deliberately not `titlebarAppearsTransparent`: see the type comment.
        window.toolbarStyle = .unified
        window.isExcludedFromWindowsMenu = true
        window.collectionBehavior = [.transient, .ignoresCycle, .stationary]
        window.animationBehavior = .none
        window.hasShadow = false
        window.setIsVisible(false)
        window.orderOut(nil)
        return window
    }

    private static func pumpMainRunLoop(for duration: TimeInterval, window: NSWindow) {
        let deadline = Date().addingTimeInterval(duration)
        while Date() < deadline {
            window.layoutIfNeeded()
            window.contentView?.layoutSubtreeIfNeeded()
            RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        }
        window.layoutIfNeeded()
        window.contentView?.layoutSubtreeIfNeeded()
        window.displayIfNeeded()
    }
}

enum StudioSnapshotError: Error {
    case noContentView
    case noBitmap
    case pngEncodingFailed
}
