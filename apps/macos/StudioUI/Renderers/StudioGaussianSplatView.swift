import AppKit
import Metal
import QuartzCore
import StudioKit
import SwiftUI
import simd

// The splat viewer: a Gaussian splat PLY drawn with `StudioGaussianSplatRenderer` in a
// CAMetalLayer-backed view the reader can orbit (drag), zoom (pinch, or scroll), pan (Shift- or
// Option-drag), and reset (double-click). A 3D result card shows it in place of the Quick Look
// tile, which draws a splat PLY as uncolored points.

/// A loaded splat, ready to draw: the GPU upload plus what framing and the accessibility label
/// need. Loaded once per file and kept for a few files, so a feed re-render or scrolling a card
/// back into view doesn't decode 17 MB again.
final class StudioGaussianSplatScene: @unchecked Sendable {
    let renderer: StudioGaussianSplatRenderer
    let sphere: (center: SIMD3<Float>, radius: Float)
    let camera: StudioGaussianSplatCamera?
    let count: Int
    let fileCount: Int

    init(renderer: StudioGaussianSplatRenderer, splat: StudioGaussianSplat, camera: StudioGaussianSplatCamera?) {
        self.renderer = renderer
        sphere = splat.framingSphere
        self.camera = camera
        count = splat.count
        fileCount = splat.fileCount
    }

    /// "262,144 Gaussians", or "262,144 of 1,048,576 Gaussians" for a file read at a stride.
    var countText: String {
        count == fileCount ? "\(count.formatted()) Gaussians" : "\(count.formatted()) of \(fileCount.formatted()) Gaussians"
    }

    enum LoadError: LocalizedError {
        case noMetal
        var errorDescription: String? { "This Mac can't draw the splat with Metal." }
    }

    static func load(url: URL, cameraURL: URL?) async throws -> StudioGaussianSplatScene {
        let key = cacheKey(url)
        if let cached = await cached(key) { return cached }
        let scene = try await Task.detached(priority: .userInitiated) {
            let splat = try StudioGaussianSplat.load(from: url)
            guard let renderer = StudioGaussianSplatRenderer(splat: splat) else { throw LoadError.noMetal }
            return StudioGaussianSplatScene(renderer: renderer, splat: splat, camera: cameraURL.flatMap(StudioGaussianSplatCamera.load(from:)))
        }.value
        await store(scene, key)
        return scene
    }

    private static func cacheKey(_ url: URL) -> String {
        let values = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
        return "\(url.standardizedFileURL.path)|\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0)|\(values?.fileSize ?? 0)"
    }

    @MainActor private static var cache: [(key: String, scene: StudioGaussianSplatScene)] = []

    @MainActor private static func cached(_ key: String) -> StudioGaussianSplatScene? {
        cache.first { $0.key == key }?.scene
    }

    @MainActor private static func store(_ scene: StudioGaussianSplatScene, _ key: String) {
        cache.removeAll { $0.key == key }
        cache.insert((key, scene), at: 0)
        if cache.count > 3 { cache.removeLast(cache.count - 3) }
    }
}

/// A splat PLY, orbitable, with its Gaussian count as the accessibility label. Falls back to
/// Quick Look when the file can't be drawn as a splat.
struct StudioGaussianSplatView: View {
    let url: URL
    var cameraURL: URL?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phase: Phase = .loading
    @State private var commands = StudioGaussianSplatCommands()

    private enum Phase {
        case loading
        case loaded(StudioGaussianSplatScene)
        case failed(String)
    }

    var body: some View {
        Group {
            switch phase {
            case .loading:
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .accessibilityLabel("Loading Gaussian splat")
            case .loaded(let scene):
                StudioGaussianSplatCanvas(scene: scene, reduceMotion: reduceMotion, commands: commands)
                    .help("Drag to orbit, pinch or scroll to zoom, Shift-drag to pan, double-click to reset")
                    .accessibilityElement()
                    .accessibilityAddTraits(.isImage)
                    .accessibilityLabel("Gaussian splat, \(scene.countText)")
                    .accessibilityHint("Drag to orbit. Double-click to reset the view.")
                    .accessibilityAction(named: "Reset view") { commands.reset() }
                    .accessibilityAction(named: "Turn left") { commands.turn(by: -.pi / 6) }
                    .accessibilityAction(named: "Turn right") { commands.turn(by: .pi / 6) }
            case .failed(let message):
                StudioEmbeddedQuickLookPreview(url: url)
                    .help(message)
            }
        }
        .task(id: url) {
            do {
                phase = .loaded(try await StudioGaussianSplatScene.load(url: url, cameraURL: cameraURL))
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }
}

/// What the accessibility actions ask of the Metal view, which owns the camera.
@MainActor
final class StudioGaussianSplatCommands {
    fileprivate weak var view: StudioGaussianSplatMetalView?

    func reset() { view?.resetView() }
    func turn(by radians: Float) { view?.turn(by: radians) }
}

private struct StudioGaussianSplatCanvas: NSViewRepresentable {
    let scene: StudioGaussianSplatScene
    let reduceMotion: Bool
    let commands: StudioGaussianSplatCommands

    func makeNSView(context: Context) -> StudioGaussianSplatMetalView {
        let view = StudioGaussianSplatMetalView(scene: scene)
        view.reduceMotion = reduceMotion
        commands.view = view
        return view
    }

    func updateNSView(_ view: StudioGaussianSplatMetalView, context: Context) {
        view.reduceMotion = reduceMotion
        if view.scene !== scene { view.scene = scene }
        commands.view = view
    }
}

/// The CAMetalLayer-backed view. It draws only when something changed (the camera, a fresh
/// sort, the size) and sorts on a background queue: the newest finished order is drawn while
/// the next is computed, so a drag never waits on a sort.
final class StudioGaussianSplatMetalView: NSView {
    var scene: StudioGaussianSplatScene {
        didSet {
            order = nil
            framed = nil
            frameIfNeeded()
        }
    }
    var reduceMotion = false

    private var orbit: StudioGaussianSplatOrbit?
    private var framed: StudioGaussianSplatOrbit?
    private var order: StudioSplatOrder?
    private var sortInFlight = false
    private var sortPending = false
    private var resetTimer: Timer?

    isolated deinit {
        resetTimer?.invalidate()
    }

    private var metalLayer: CAMetalLayer? { layer as? CAMetalLayer }

    init(scene: StudioGaussianSplatScene) {
        self.scene = scene
        super.init(frame: .zero)
        wantsLayer = true
        layerContentsRedrawPolicy = .onSetNeedsDisplay
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    override func makeBackingLayer() -> CALayer {
        let layer = CAMetalLayer()
        layer.device = scene.renderer.device
        layer.pixelFormat = StudioGaussianSplatRenderer.pixelFormat
        layer.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
        layer.isOpaque = false
        layer.framebufferOnly = true
        return layer
    }

    override var wantsUpdateLayer: Bool { true }
    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func layout() {
        super.layout()
        updateDrawableSize()
        frameIfNeeded()
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        updateDrawableSize()
    }

    private var pixelSize: CGSize {
        let scale = window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2
        return CGSize(width: (bounds.width * scale).rounded(), height: (bounds.height * scale).rounded())
    }

    private func updateDrawableSize() {
        guard let metalLayer else { return }
        metalLayer.contentsScale = window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2
        let size = pixelSize
        if size.width > 0, size.height > 0, metalLayer.drawableSize != size {
            metalLayer.drawableSize = size
            needsDisplay = true
        }
    }

    /// Frames the splat once the view has a size, so the aspect is known.
    private func frameIfNeeded() {
        guard framed == nil, bounds.width > 0, bounds.height > 0 else { return }
        let start = StudioGaussianSplatOrbit.framing(
            sphere: scene.sphere, camera: scene.camera, aspect: Float(bounds.width / bounds.height)
        )
        framed = start
        setOrbit(start)
    }

    private func setOrbit(_ next: StudioGaussianSplatOrbit) {
        orbit = next
        requestSort()
        needsDisplay = true
    }

    // MARK: Drawing

    override func updateLayer() {
        guard let metalLayer, let orbit, let order, let drawable = metalLayer.nextDrawable(),
              let commandBuffer = scene.renderer.queue.makeCommandBuffer() else { return }
        scene.renderer.encode(commandBuffer, into: drawable.texture, orbit: orbit, order: order)
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    /// The bitmap path: `cacheDisplay(in:to:)` and printing draw views rather than composite
    /// their layers, and a CAMetalLayer's contents never reach that context. Draws one frame
    /// offscreen instead, so a snapshot or a print shows the splat.
    override func draw(_ dirtyRect: NSRect) {
        guard let context = NSGraphicsContext.current?.cgContext else { return }
        let start = orbit ?? StudioGaussianSplatOrbit.framing(
            sphere: scene.sphere, camera: scene.camera, aspect: Float(max(bounds.width, 1) / max(bounds.height, 1))
        )
        let size = pixelSize
        guard let image = scene.renderer.image(orbit: start, width: Int(size.width), height: Int(size.height), order: order) else {
            return
        }
        context.draw(image, in: bounds)
    }

    // MARK: Sorting

    private func requestSort() {
        guard let orbit else { return }
        if let order, simd_distance(order.eye, orbit.fileEye) <= orbit.distance * 1e-4 { return }
        guard !sortInFlight else {
            sortPending = true
            return
        }
        sortInFlight = true
        let renderer = scene.renderer
        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            let sorted = renderer.order(for: orbit)
            DispatchQueue.main.async {
                guard let self else { return }
                self.sortInFlight = false
                if let sorted, renderer === self.scene.renderer {
                    self.order = sorted
                    self.needsDisplay = true
                }
                if self.sortPending || renderer !== self.scene.renderer {
                    self.sortPending = false
                    self.requestSort()
                }
            }
        }
    }

    // MARK: Interaction

    override func mouseDown(with event: NSEvent) {
        if event.clickCount == 2 { resetView() }
    }

    override func mouseDragged(with event: NSEvent) {
        guard var next = orbit else { return }
        resetTimer?.invalidate()
        let delta = CGSize(width: event.deltaX, height: event.deltaY)
        if event.modifierFlags.contains(.shift) || event.modifierFlags.contains(.option) {
            next.pan(by: delta, height: bounds.height)
        } else {
            next.orbit(by: delta, height: bounds.height)
        }
        setOrbit(next)
    }

    /// Scrolling zooms. Inside a scroll view (the feed), a plain scroll still scrolls the page
    /// and Command- or Option-scroll zooms, so the viewer never traps the feed.
    override func scrollWheel(with event: NSEvent) {
        let zoomModifiers = event.modifierFlags.intersection([.command, .option])
        guard enclosingScrollView == nil || !zoomModifiers.isEmpty, var next = orbit, let framed else {
            super.scrollWheel(with: event)
            return
        }
        let delta = event.hasPreciseScrollingDeltas ? event.scrollingDeltaY : event.scrollingDeltaY * 10
        next.zoom(by: exp(-Float(delta) * 0.01), framedDistance: framed.distance)
        setOrbit(next)
    }

    override func magnify(with event: NSEvent) {
        guard var next = orbit, let framed else { return }
        next.zoom(by: 1 / max(1 + Float(event.magnification), 0.1), framedDistance: framed.distance)
        setOrbit(next)
    }

    func turn(by radians: Float) {
        guard var next = orbit else { return }
        next.yaw += radians
        setOrbit(next)
    }

    /// Back to the starting view: at once under Reduce Motion, else eased over a third of a second.
    func resetView() {
        resetTimer?.invalidate()
        guard let framed, let from = orbit else { return }
        guard !reduceMotion else {
            setOrbit(framed)
            return
        }
        let started = CACurrentMediaTime(), duration = 0.3
        resetTimer = Timer.scheduledTimer(withTimeInterval: 1 / 60, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                let linear = Float(min((CACurrentMediaTime() - started) / duration, 1))
                let eased = 1 - (1 - linear) * (1 - linear) * (1 - linear)
                self.setOrbit(from.interpolated(to: framed, progress: eased))
                if linear >= 1 { self.resetTimer?.invalidate() }
            }
        }
    }
}

/// A 3D result card's splat: the viewer, the picture IDOL reconstructed from beside it, and the
/// manifest's Gaussian count under both.
struct StudioGaussianSplatCard: View {
    let result: StudioGaussianSplatResult
    let summary: StudioMeshSummary?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 10) {
                viewer
                if let input = result.preparedInputURL {
                    StudioAsyncImagePreview(url: input, maxPixelSize: 360, contentMode: .fit, fallbackSystemImage: "photo")
                        .frame(width: 120, height: 120)
                        .background(MereRunTheme.surfaceRaised)
                        .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
                        .studioFileDrag(input)
                        .help("The picture IDOL reconstructed from")
                        .accessibilityLabel("Prepared input \(input.lastPathComponent)")
                }
            }
            if let summary {
                StudioMeshSummaryRow(summary: summary)
            }
        }
    }

    private var viewer: some View {
        StudioGaussianSplatView(url: result.splatURL, cameraURL: result.cameraURL)
            .frame(minWidth: 236, idealWidth: 482, maxWidth: 482)
            .frame(height: 320)
            .background(MereRunTheme.surfaceRaised)
            .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
            .studioOutputContextMenu(result.splatURL)
    }
}
