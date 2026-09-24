import AppKit
import StudioKit
import SwiftUI

// Drawing boxes and points on a picture instead of typing coordinates. `StudioRegionPromptLayer`
// is the interactive overlay that sits on a displayed image; `StudioRegionToolbar` picks what a
// click does; `StudioRegionPromptEditor` composes both around a picture for the surfaces that do
// not already draw one (a video frame, a subject's reference image). The math, the CLI text, and
// what a press does (`StudioRegionPress`) live in `StudioKit/StudioRegionPrompts.swift`.

// MARK: - Tools

extension StudioRegionTool {
    var title: String {
        switch self {
        case .box: return "Box"
        case .point: return "Point"
        case .negativePoint: return "Negative"
        }
    }

    var systemImage: String {
        switch self {
        case .box: return "rectangle.dashed"
        case .point: return "plus.circle"
        case .negativePoint: return "minus.circle"
        }
    }

    var help: String {
        switch self {
        case .box:
            return "Drag on the picture to draw a box around what to find. Click a box to select it, drag it to move it, and drag a corner to resize it"
        case .point:
            return "Click the picture, inside a box too, to mark a spot the mask must include"
        case .negativePoint:
            return "Click the picture to mark a spot the mask must leave out (Option-click does this in any tool)"
        }
    }

    /// The one-line reminder beside the toolbar.
    static let gestureHint = "Drag for a box, click for a point, Option-click for a negative point. The Box tool selects and moves boxes."
}

// MARK: - Toolbar

/// The compact Box / Point / Negative switch, a Clear button, and the count of what is drawn.
struct StudioRegionToolbar: View {
    @Binding var tool: StudioRegionTool
    @Binding var prompts: [StudioRegionPrompt]
    @Binding var selection: UUID?
    var isEnabled = true

    var body: some View {
        HStack(spacing: 8) {
            HStack(spacing: 2) {
                ForEach(StudioRegionTool.allCases, id: \.self) { candidate in
                    MereSegment(isSelected: candidate == tool, action: { tool = candidate }) {
                        Label(candidate.title, systemImage: candidate.systemImage)
                            .labelStyle(.titleAndIcon)
                    }
                    .help(candidate.help)
                    .accessibilityLabel(candidate.title)
                    .accessibilityHint(candidate.help)
                }
            }
            .padding(2)
            .background {
                RoundedRectangle(cornerRadius: 7)
                    .fill(MereRunTheme.surfaceRaised)
            }
            .fixedSize()
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Drawing tool")

            if !prompts.isEmpty {
                StudioAnalyzeChip(text: prompts.countDescription)
                    .fixedSize()
                    .accessibilityLabel("\(prompts.countDescription) drawn")
                Button {
                    selection = nil
                    prompts.removeAll()
                } label: {
                    Label("Clear", systemImage: "trash")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.mereSecondary)
                .fixedSize()
                .help("Remove every box and point")
            }
        }
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1 : 0.5)
    }
}

/// The toolbar with the one-line reminder of the gestures beside it, which yields first when the
/// column is too narrow for both.
struct StudioRegionToolbarRow: View {
    @Binding var tool: StudioRegionTool
    @Binding var prompts: [StudioRegionPrompt]
    @Binding var selection: UUID?
    var isEnabled = true
    var hint = StudioRegionTool.gestureHint

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 10) {
                toolbar
                Spacer(minLength: 8)
                Text(hint)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                    .fixedSize()
            }
            HStack(spacing: 10) {
                toolbar
                Spacer(minLength: 0)
            }
        }
    }

    private var toolbar: some View {
        StudioRegionToolbar(tool: $tool, prompts: $prompts, selection: $selection, isEnabled: isEnabled)
            .help(hint)
    }
}

// MARK: - The overlay

/// The boxes and points drawn over a displayed image, and the gestures that make them.
///
/// Sits in the coordinate space of the view showing the image; `fitted` is where the image's
/// pixels land in that space, so every press converts through `StudioRegionGeometry` into the
/// picture's own pixels. What a press starts is `StudioRegionPress.press(tool:hit:optionHeld:)`:
/// a drag on picture draws a box; a click adds a point (positive or negative per the tool,
/// negative with Option) or, with the Box tool, clears the selection; a press on a point, or on
/// a box with the Box tool, selects it and a drag moves it; the selected box shows corner handles
/// that resize it. Delete removes the selection and Escape clears it. The selection and tool are
/// the parent's state, so they outlive this layer's re-renders.
struct StudioRegionPromptLayer: View {
    @Binding var prompts: [StudioRegionPrompt]
    let imageSize: CGSize
    let fitted: CGRect
    @Binding var tool: StudioRegionTool
    @Binding var selection: UUID?
    var isEnabled = true
    /// Subjects take one box per selector; drawing another replaces it.
    var maximumBoxes: Int?

    @State private var drag: DragState?
    @State private var hover: StudioRegionHit?
    /// Option as the keyboard reports it while the pointer is over the layer, so a synthesized
    /// press that carries the flag and one whose flag arrived as a key both read as negative.
    @State private var optionHeld = false
    /// The prompt being moved or resized, as it is right now. It lives here rather than in the
    /// binding until the drag ends, so a drag does not write the draft (and persist it) on every
    /// pointer event.
    @State private var liveEdit: StudioRegionPrompt?
    @FocusState private var focused: Bool
    /// The window this layer is in, so the key monitor answers only its keys.
    @State private var hostWindow: NSWindow?
    /// This layer's claim on the key monitor; the layer whose prompt was pressed last owns Delete
    /// and Escape, so two editors on one page never both answer.
    @State private var keyOwner = UUID()

    private enum Metrics {
        static let handleSide: CGFloat = 9
        static let pointDiameter: CGFloat = 16
        /// A press that travels less than this is a click, not a box.
        static let clickSlop: CGFloat = 4
        static let tagHeight: CGFloat = 17
        /// Roughly what a tag's text measures, so its placement can keep it on the picture.
        static let tagCharacterWidth: CGFloat = 6.4
        static let tagPadding: CGFloat = 10
    }

    private enum DragState {
        case drawing(start: CGPoint, current: CGPoint, click: StudioRegionClick)
        case moving(id: UUID, original: StudioRegionPrompt, start: CGPoint)
        /// `anchor` is the corner that stays put, captured when the drag began.
        case resizing(id: UUID, anchor: CGPoint)
        /// A press on a prompt that has not moved yet: a click selects, a drag moves.
        case pressing(id: UUID, start: CGPoint)
    }

    /// What is drawn: the bound prompts, with the one mid-drag shown at its live position.
    private var displayedPrompts: [StudioRegionPrompt] {
        guard let liveEdit else { return prompts }
        return prompts.map { $0.id == liveEdit.id ? liveEdit : $0 }
    }

    var body: some View {
        ZStack(alignment: .topLeading) {
            Color.clear
                .contentShape(Rectangle())
            ForEach(Array(displayedPrompts.enumerated()), id: \.element.id) { index, prompt in
                promptView(prompt, ordinal: index + 1)
            }
            if case .drawing(let start, let current, _) = drag {
                drawingPreview(from: start, to: current)
            }
        }
        // `.edit` interactions take focus on a click whether or not Full Keyboard Access is on,
        // which is what lets Delete reach `onDeleteCommand` instead of the composer's text.
        .focusable(isEnabled, interactions: .edit)
        .focused($focused)
        .focusEffectDisabled()
        .gesture(pressGesture, including: isEnabled ? .all : .subviews)
        .onContinuousHover { phase in
            switch phase {
            case .active(let location):
                hover = isEnabled ? hit(at: location) : nil
            case .ended:
                hover = nil
            }
        }
        .onModifierKeysChanged(mask: .option) { _, modifiers in
            optionHeld = modifiers.contains(.option)
        }
        .pointerStyle(pointerStyle)
        .onDeleteCommand { removeSelection() }
        .onExitCommand { selection = nil }
        .onKeyPress(.delete) { removeSelectionKeyPress() }
        .onKeyPress(.deleteForward) { removeSelectionKeyPress() }
        .onChange(of: prompts.map(\.id)) { _, ids in
            if let selection, !ids.contains(selection) { self.selection = nil }
        }
        // A click that starts the drag gesture never makes the layer first responder, so the
        // keys that act on the selection arrive through a monitor rather than SwiftUI focus.
        .background(StudioHostWindowReader(window: $hostWindow))
        .onChange(of: selection != nil, initial: true) { _, hasSelection in
            if hasSelection { claimKeys() } else { StudioRegionKeyMonitor.shared.release(owner: keyOwner) }
        }
        .onDisappear { StudioRegionKeyMonitor.shared.release(owner: keyOwner) }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(prompts.isEmpty ? "No boxes or points drawn" : "\(prompts.countDescription) drawn")
        .accessibilityHint("Drag to draw a box. Click to add a point. Option-click for a negative point.")
    }

    // MARK: Drawing

    @ViewBuilder
    private func promptView(_ prompt: StudioRegionPrompt, ordinal: Int) -> some View {
        let isSelected = prompt.id == selection
        let isHovered = hover?.id == prompt.id
        Group {
            if let rect = prompt.rect {
                boxView(prompt, rect: rect, ordinal: ordinal, isSelected: isSelected, isHovered: isHovered)
            } else if let point = prompt.point {
                pointView(prompt, point: point, ordinal: ordinal, isSelected: isSelected, isHovered: isHovered)
            }
        }
        .allowsHitTesting(false)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(prompt.accessibilityDescription(ordinal: ordinal))
        .accessibilityAddTraits(isSelected ? .isSelected : [])
        .accessibilityAction(named: "Select") { selection = prompt.id }
        .accessibilityAction(named: "Remove") { remove(prompt.id) }
    }

    private func boxView(_ prompt: StudioRegionPrompt, rect: CGRect, ordinal: Int, isSelected: Bool, isHovered: Bool) -> some View {
        let viewRect = StudioRegionGeometry.viewRect(fromImage: rect, imageSize: imageSize, fitted: fitted)
        let width = max(viewRect.width, 2)
        let height = max(viewRect.height, 2)
        // Dashed until selected, so a prompt reads as what was asked and stays apart from the
        // solid boxes a result draws over the same picture. Selected, it is solid with a white
        // halo so the state reads at a glance over any picture.
        let stroke = isSelected
            ? StrokeStyle(lineWidth: 2.5)
            : StrokeStyle(lineWidth: 2, dash: [7, 4])
        let tagSide = StudioRegionTagPlacement.boxSide(viewRect: viewRect, fitted: fitted, tagHeight: Metrics.tagHeight + 3)
        return RoundedRectangle(cornerRadius: 3)
            .fill(MereRunTheme.accent.opacity(isSelected ? 0.14 : isHovered ? 0.07 : 0))
            .overlay {
                RoundedRectangle(cornerRadius: 3)
                    .strokeBorder(Color.black.opacity(0.18), style: StrokeStyle(lineWidth: 3.5, dash: stroke.dash))
                    .blendMode(.multiply)
            }
            .overlay {
                if isSelected {
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(Color.white.opacity(0.9), lineWidth: 1.5)
                        .padding(-2)
                }
            }
            .overlay {
                RoundedRectangle(cornerRadius: 3)
                    .strokeBorder(MereRunTheme.accent, style: stroke)
            }
            .frame(width: width, height: height)
            .overlay(alignment: .topLeading) {
                StudioRegionTag(text: tagText(prompt, ordinal: ordinal))
                    .fixedSize()
                    .offset(x: tagSide == .above ? -2 : 4, y: tagSide == .above ? -(Metrics.tagHeight + 3) : 4)
            }
            .overlay {
                if isSelected {
                    ForEach(StudioRegionBoxCorner.allCases, id: \.self) { corner in
                        let position = corner.point(of: CGRect(origin: .zero, size: CGSize(width: width, height: height)))
                        handle(isActive: hover == .handle(id: prompt.id, corner: corner))
                            .position(position)
                    }
                }
            }
            .offset(x: viewRect.minX, y: viewRect.minY)
            .animation(MereRunTheme.Motion.quick, value: isSelected)
    }

    private func handle(isActive: Bool) -> some View {
        RoundedRectangle(cornerRadius: 1.5)
            .fill(MereRunTheme.surface)
            .overlay {
                RoundedRectangle(cornerRadius: 1.5)
                    .strokeBorder(MereRunTheme.accent, lineWidth: 1.5)
            }
            .frame(width: Metrics.handleSide, height: Metrics.handleSide)
            .scaleEffect(isActive ? 1.3 : 1)
            .mereShadow(radius: 2, y: 1)
    }

    private func pointView(_ prompt: StudioRegionPrompt, point: CGPoint, ordinal: Int, isSelected: Bool, isHovered: Bool) -> some View {
        let center = StudioRegionGeometry.viewPoint(fromImage: point, imageSize: imageSize, fitted: fitted)
        let color = prompt.isPositivePoint ? MereRunTheme.accent : MereRunTheme.red
        let text = tagText(prompt, ordinal: ordinal)
        let tagSize = CGSize(
            width: CGFloat(text.count) * Metrics.tagCharacterWidth + Metrics.tagPadding,
            height: Metrics.tagHeight
        )
        let tagOffset = StudioRegionTagPlacement.pointOffset(
            center: center, fitted: fitted, tagSize: tagSize, reach: Metrics.pointDiameter / 2 + 6
        )
        return ZStack {
            if isSelected {
                Circle()
                    .fill(color.opacity(0.28))
                    .frame(width: Metrics.pointDiameter + 12, height: Metrics.pointDiameter + 12)
                Circle()
                    .strokeBorder(Color.white.opacity(0.9), lineWidth: 1.5)
                    .frame(width: Metrics.pointDiameter + 12, height: Metrics.pointDiameter + 12)
            }
            Circle()
                .fill(color)
                .overlay {
                    Circle().strokeBorder(Color.white.opacity(0.92), lineWidth: 1.5)
                }
                .overlay {
                    Image(systemName: prompt.isPositivePoint ? "plus" : "minus")
                        .font(.system(size: 8, weight: .heavy))
                        .foregroundStyle(Color.white)
                }
                .frame(width: Metrics.pointDiameter, height: Metrics.pointDiameter)
                .scaleEffect(isHovered || isSelected ? 1.12 : 1)
                .mereShadow(radius: 3, y: 1)
        }
        .overlay {
            StudioRegionTag(text: text, tint: color)
                .fixedSize()
                .offset(x: tagOffset.dx, y: tagOffset.dy)
        }
        .position(center)
        .animation(MereRunTheme.Motion.quick, value: isSelected)
        .animation(MereRunTheme.Motion.quick, value: isHovered)
    }

    private func tagText(_ prompt: StudioRegionPrompt, ordinal: Int) -> String {
        guard let label = prompt.label else { return String(ordinal) }
        return "\(ordinal) · \(label)"
    }

    private func drawingPreview(from start: CGPoint, to current: CGPoint) -> some View {
        let rect = StudioRegionGeometry.viewRect(
            fromImage: StudioRegionGeometry.imageRect(fromView: start, to: current, imageSize: imageSize, fitted: fitted),
            imageSize: imageSize,
            fitted: fitted
        )
        return RoundedRectangle(cornerRadius: 3)
            .fill(MereRunTheme.accent.opacity(0.1))
            .overlay {
                RoundedRectangle(cornerRadius: 3)
                    .strokeBorder(MereRunTheme.accent, style: StrokeStyle(lineWidth: 1.5, dash: [5, 4]))
            }
            .frame(width: max(rect.width, 1), height: max(rect.height, 1))
            .offset(x: rect.minX, y: rect.minY)
            .allowsHitTesting(false)
    }

    // MARK: Gestures

    private var pressGesture: some Gesture {
        DragGesture(minimumDistance: 0, coordinateSpace: .local)
            .onChanged { value in
                if drag == nil { begin(at: value.startLocation) }
                update(to: value.location)
            }
            .onEnded { value in
                // A click that never moved may arrive as an end alone.
                if drag == nil { begin(at: value.startLocation) }
                end(at: value.location)
            }
    }

    private func begin(at location: CGPoint) {
        focused = true
        claimKeys()
        switch StudioRegionPress.press(tool: tool, hit: hit(at: location), optionHeld: isOptionHeld) {
        case .resize(let id, let corner):
            selection = id
            guard let anchor = prompts.first(where: { $0.id == id })?.anchor(for: corner) else { return }
            drag = .resizing(id: id, anchor: anchor)
        case .grab(let id):
            selection = id
            drag = .pressing(id: id, start: location)
        case .draw(let click):
            drag = .drawing(start: location, current: location, click: click)
        }
    }

    private var isOptionHeld: Bool {
        optionHeld || NSEvent.modifierFlags.contains(.option)
    }

    private func update(to location: CGPoint) {
        switch drag {
        case .drawing(let start, _, let click):
            drag = .drawing(start: start, current: location, click: click)
        case .pressing(let id, let start):
            guard hypot(location.x - start.x, location.y - start.y) >= Metrics.clickSlop,
                  let original = prompts.first(where: { $0.id == id }) else { return }
            drag = .moving(id: id, original: original, start: start)
            move(id: id, original: original, from: start, to: location)
        case .moving(let id, let original, let start):
            move(id: id, original: original, from: start, to: location)
        case .resizing(let id, let anchor):
            guard let prompt = prompts.first(where: { $0.id == id }) else { return }
            let imagePoint = StudioRegionGeometry.imagePoint(fromView: location, imageSize: imageSize, fitted: fitted)
            liveEdit = prompt.resizingBox(anchor: anchor, to: imagePoint, within: imageSize)
        case nil:
            break
        }
    }

    private func end(at location: CGPoint) {
        defer {
            drag = nil
            liveEdit = nil
        }
        switch drag {
        case .drawing(let start, _, let click):
            if StudioRegionGeometry.isClick(from: start, to: location, imageSize: imageSize, fitted: fitted, slop: Metrics.clickSlop) {
                perform(click, at: start)
            } else {
                addBox(StudioRegionGeometry.imageRect(fromView: start, to: location, imageSize: imageSize, fitted: fitted))
            }
        case .moving, .resizing:
            // Commit the live position to the draft once, now that the drag is over.
            if let liveEdit, let index = prompts.firstIndex(where: { $0.id == liveEdit.id }) {
                prompts[index] = liveEdit
            }
        case .pressing, nil:
            // A click on a prompt only selects it, which `begin` already did.
            break
        }
    }

    private func perform(_ click: StudioRegionClick, at location: CGPoint) {
        switch click {
        case .addPoint(let isPositive):
            let imagePoint = StudioRegionGeometry.imagePoint(fromView: location, imageSize: imageSize, fitted: fitted)
            add(.point(imagePoint, isPositive: isPositive))
        case .clearSelection:
            selection = nil
        }
    }

    private func addBox(_ rect: CGRect) {
        if let maximumBoxes, prompts.boxes.count >= maximumBoxes,
           let oldest = prompts.firstIndex(where: \.isBox) {
            prompts.remove(at: oldest)
        }
        add(.box(rect))
    }

    private func add(_ prompt: StudioRegionPrompt) {
        prompts.append(prompt)
        selection = prompt.id
    }

    private func move(id: UUID, original: StudioRegionPrompt, from start: CGPoint, to location: CGPoint) {
        let scale = StudioRegionGeometry.scale(imageSize: imageSize, fitted: fitted)
        guard scale > 0 else { return }
        let delta = CGVector(dx: (location.x - start.x) / scale, dy: (location.y - start.y) / scale)
        liveEdit = original.moved(by: delta, within: imageSize)
    }

    /// Makes this layer the one the key monitor asks about Delete and Escape.
    private func claimKeys() {
        StudioRegionKeyMonitor.shared.claim(owner: keyOwner, window: hostWindow) { command in
            switch command {
            case .removeSelection: removeSelection()
            case .clearSelection: selection = nil
            }
        }
    }

    private func removeSelectionKeyPress() -> KeyPress.Result {
        guard selection != nil else { return .ignored }
        removeSelection()
        return .handled
    }

    private func removeSelection() {
        guard let selection else { return }
        remove(selection)
    }

    private func remove(_ id: UUID) {
        prompts.removeAll { $0.id == id }
        if selection == id { selection = nil }
    }

    /// Both reaches are a little larger than the marks they grab, so a handle or point does not
    /// have to be hit dead centre.
    private func hit(at location: CGPoint) -> StudioRegionHit? {
        StudioRegionHit.hit(
            in: prompts,
            at: location,
            imageSize: imageSize,
            fitted: fitted,
            selectedID: selection,
            handleRadius: Metrics.handleSide,
            pointRadius: Metrics.pointDiameter / 2 + 2
        )
    }

    private var pointerStyle: PointerStyle {
        guard isEnabled else { return .default }
        switch drag {
        case .moving: return .grabActive
        case .resizing: return .grabActive
        case .drawing: return .rectSelection
        case .pressing, nil: break
        }
        switch StudioRegionPress.press(tool: tool, hit: hover, optionHeld: isOptionHeld) {
        case .resize(_, let corner): return .frameResize(position: corner.resizePosition)
        case .grab: return .grabIdle
        case .draw: return tool == .box ? .rectSelection : .default
        }
    }
}

/// One local key-down monitor for every prompt layer, owned by whichever layer claimed it last.
///
/// The monitor exists only while a layer holds a selection. A key is answered when it lands in
/// the owner's window and nothing is editing text there (`StudioRegionKeyCommand` decides), and
/// an answered key is swallowed so the window does not beep or pass it on.
@MainActor
final class StudioRegionKeyMonitor {
    static let shared = StudioRegionKeyMonitor()

    private var monitor: Any?
    private var owner: UUID?
    private var window: NSWindow?
    private var handler: ((StudioRegionKeyCommand) -> Void)?

    func claim(owner: UUID, window: NSWindow?, handler: @escaping (StudioRegionKeyCommand) -> Void) {
        self.owner = owner
        self.window = window
        self.handler = handler
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self, self.handle(event) else { return event }
            return nil
        }
    }

    func release(owner: UUID) {
        guard self.owner == owner else { return }
        self.owner = nil
        window = nil
        handler = nil
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
    }

    /// Whether `event` acted on the owner's selection.
    private func handle(_ event: NSEvent) -> Bool {
        guard let handler, let eventWindow = event.window, window == nil || eventWindow === window else { return false }
        let command = StudioRegionKeyCommand.command(
            keyCode: event.keyCode,
            hasSelection: true,
            textIsEditing: eventWindow.firstResponder is NSText
        )
        guard let command else { return false }
        handler(command)
        return true
    }
}

/// Hands the window a SwiftUI view ends up in to the view, once AppKit has placed it.
private struct StudioHostWindowReader: NSViewRepresentable {
    @Binding var window: NSWindow?

    func makeNSView(context: Context) -> WindowReadingView {
        let view = WindowReadingView()
        view.onWindow = { window in
            DispatchQueue.main.async { self.window = window }
        }
        return view
    }

    func updateNSView(_ view: WindowReadingView, context: Context) {}

    final class WindowReadingView: NSView {
        var onWindow: ((NSWindow?) -> Void)?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            onWindow?(window)
        }
    }
}

private extension StudioRegionBoxCorner {
    var resizePosition: FrameResizePosition {
        switch self {
        case .topLeading: return .topLeading
        case .topTrailing: return .topTrailing
        case .bottomLeading: return .bottomLeading
        case .bottomTrailing: return .bottomTrailing
        }
    }
}

/// "1", "2 · saucer" — the numbered tab on a drawn prompt, in the result tab's shape.
private struct StudioRegionTag: View {
    let text: String
    var tint: Color = MereRunTheme.accent

    var body: some View {
        Text(text)
            .font(.system(size: 10.5, weight: .semibold))
            .foregroundStyle(Color.white)
            .lineLimit(1)
            .padding(.horizontal, 5)
            .padding(.vertical, 1.5)
            .background {
                RoundedRectangle(cornerRadius: 3)
                    .fill(tint)
            }
    }
}

// MARK: - Picture plus overlay

/// A picture with the prompt layer on it and the toolbar above, for surfaces that do not already
/// draw the image themselves: a clip's seed frame, a subject's reference image.
struct StudioRegionPromptEditor: View {
    /// The picture as shown (upright), or nil while it loads.
    let image: NSImage?
    /// The stored pixel space the prompts are in.
    let imageSize: CGSize
    /// How the stored pixels are turned to show `image` upright.
    var orientation = StudioImageOrientation.up
    @Binding var prompts: [StudioRegionPrompt]
    var maximumBoxes: Int?
    var maxHeight: CGFloat = 320
    var isEnabled = true
    var placeholder = "Loading picture…"

    @State private var tool = StudioRegionTool.box
    @State private var selection: UUID?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            StudioRegionToolbar(tool: $tool, prompts: $prompts, selection: $selection, isEnabled: isEnabled && image != nil)
            picture
                .mereMediaFrame()
        }
    }

    @ViewBuilder
    private var picture: some View {
        if let image {
            Image(nsImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxHeight: maxHeight)
                .overlay {
                    GeometryReader { geometry in
                        StudioRegionPromptLayer(
                            prompts: $prompts.inDisplaySpace(orientation, storedSize: imageSize),
                            imageSize: displaySize,
                            fitted: CGRect(origin: .zero, size: geometry.size),
                            tool: $tool,
                            selection: $selection,
                            isEnabled: isEnabled,
                            maximumBoxes: maximumBoxes
                        )
                    }
                }
        } else {
            Rectangle()
                .fill(MereRunTheme.surfaceRaised)
                .aspectRatio(aspect, contentMode: .fit)
                .frame(maxHeight: maxHeight)
                .overlay {
                    Text(placeholder)
                        .font(.system(size: 12))
                        .foregroundStyle(MereRunTheme.textMuted)
                }
        }
    }

    private var displaySize: CGSize {
        orientation.displaySize(ofStored: imageSize)
    }

    private var aspect: CGFloat {
        guard displaySize.width > 0, displaySize.height > 0 else { return 16.0 / 9 }
        return displaySize.width / displaySize.height
    }
}
