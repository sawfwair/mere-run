import AppKit
import StudioKit
import SwiftUI

// Drawing boxes and points on a picture instead of typing coordinates. `StudioRegionPromptLayer`
// is the interactive overlay that sits on a displayed image; `StudioRegionToolbar` picks what a
// click does; `StudioRegionPromptEditor` composes both around a picture for the surfaces that do
// not already draw one (a video frame, a subject's reference image). The math and the CLI text
// live in `StudioKit/StudioRegionPrompts.swift`.

// MARK: - Tools

/// What a click on empty picture does. A drag always draws a box, and Option-click always adds a
/// negative point, whichever tool is active.
enum StudioRegionTool: Hashable, CaseIterable {
    case box
    case point
    case negativePoint

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
        case .box: return "Drag on the picture to draw a box around what to find"
        case .point: return "Click the picture to mark a spot the mask must include"
        case .negativePoint: return "Click the picture to mark a spot the mask must leave out (Option-click does this in any tool)"
        }
    }
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
    var hint = "Drag for a box, click for a point, Option-click for a negative point."

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
/// picture's own pixels. A drag on empty picture draws a box; a click adds a point (positive or
/// negative per the tool, negative with Option); a press on a prompt selects it and drags move
/// it; the selected box shows corner handles that resize it; Delete removes the selection and
/// Escape clears it.
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
    /// The prompt being moved or resized, as it is right now. It lives here rather than in the
    /// binding until the drag ends, so a drag does not write the draft (and persist it) on every
    /// pointer event.
    @State private var liveEdit: StudioRegionPrompt?
    @FocusState private var focused: Bool

    private enum Metrics {
        static let handleSide: CGFloat = 8
        static let pointDiameter: CGFloat = 16
        /// A press that travels less than this is a click, not a box.
        static let clickSlop: CGFloat = 4
    }

    private enum DragState {
        case drawing(start: CGPoint, current: CGPoint, negative: Bool)
        case moving(id: UUID, original: StudioRegionPrompt, start: CGPoint)
        /// `anchor` is the corner that stays put, captured when the drag began.
        case resizing(id: UUID, anchor: CGPoint)
        /// A press on a prompt that has not moved yet: a click selects, a drag moves.
        case pressing(hit: StudioRegionHit, start: CGPoint)
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
        .focusable(isEnabled)
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
        .pointerStyle(pointerStyle)
        .onDeleteCommand { removeSelection() }
        .onExitCommand { selection = nil }
        .onChange(of: prompts.map(\.id)) { _, ids in
            if let selection, !ids.contains(selection) { self.selection = nil }
        }
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
        // solid boxes a result draws over the same picture.
        let stroke = isSelected
            ? StrokeStyle(lineWidth: 2.5)
            : StrokeStyle(lineWidth: 2, dash: [7, 4])
        return RoundedRectangle(cornerRadius: 3)
            .fill(MereRunTheme.accent.opacity(isSelected ? 0.12 : isHovered ? 0.07 : 0))
            .overlay {
                RoundedRectangle(cornerRadius: 3)
                    .strokeBorder(Color.black.opacity(0.18), style: StrokeStyle(lineWidth: 3.5, dash: stroke.dash))
                    .blendMode(.multiply)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 3)
                    .strokeBorder(MereRunTheme.accent, style: stroke)
            }
            .frame(width: width, height: height)
            .overlay(alignment: .topLeading) {
                StudioRegionTag(text: tagText(prompt, ordinal: ordinal))
                    .fixedSize()
                    .offset(x: -2, y: -20)
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
        return ZStack {
            if isSelected {
                Circle()
                    .fill(color.opacity(0.28))
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
        .overlay(alignment: .leading) {
            StudioRegionTag(text: tagText(prompt, ordinal: ordinal), tint: color)
                .fixedSize()
                .offset(x: Metrics.pointDiameter / 2 + 12, y: -1)
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
                end(at: value.location)
            }
    }

    private func begin(at location: CGPoint) {
        focused = true
        if let hit = hit(at: location) {
            selection = hit.id
            if case .handle(let id, let corner) = hit,
               let anchor = prompts.first(where: { $0.id == id })?.anchor(for: corner) {
                drag = .resizing(id: id, anchor: anchor)
            } else {
                drag = .pressing(hit: hit, start: location)
            }
        } else {
            selection = nil
            drag = .drawing(start: location, current: location, negative: NSEvent.modifierFlags.contains(.option))
        }
    }

    private func update(to location: CGPoint) {
        switch drag {
        case .drawing(let start, _, let negative):
            drag = .drawing(start: start, current: location, negative: negative)
        case .pressing(let hit, let start):
            guard hypot(location.x - start.x, location.y - start.y) >= Metrics.clickSlop,
                  let original = prompts.first(where: { $0.id == hit.id }) else { return }
            drag = .moving(id: hit.id, original: original, start: start)
            move(id: hit.id, original: original, from: start, to: location)
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
        case .drawing(let start, _, let negative):
            let travelled = max(abs(location.x - start.x), abs(location.y - start.y))
            let rect = StudioRegionGeometry.imageRect(fromView: start, to: location, imageSize: imageSize, fitted: fitted)
            // A press that barely moved on screen, or moved less than a pixel of the picture
            // (a zoomed-out image), is a click.
            if travelled < Metrics.clickSlop || rect.width < 1 || rect.height < 1 {
                click(at: start, negative: negative)
            } else {
                addBox(rect)
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

    private func click(at location: CGPoint, negative: Bool) {
        let imagePoint = StudioRegionGeometry.imagePoint(fromView: location, imageSize: imageSize, fitted: fitted)
        switch (tool, negative) {
        case (_, true), (.negativePoint, _):
            add(.point(imagePoint, isPositive: false))
        case (.point, false):
            add(.point(imagePoint, isPositive: true))
        case (.box, false):
            // The box tool draws; a bare click is how you deselect.
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
        switch hover {
        case .handle(_, let corner): return .frameResize(position: corner.resizePosition)
        case .box, .point: return .grabIdle
        case nil: return tool == .box ? .rectSelection : .default
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
    /// The picture, or nil while it loads.
    let image: NSImage?
    /// The pixel space the prompts are in; the image is displayed aspect-fitted to it.
    let imageSize: CGSize
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
                            prompts: $prompts,
                            imageSize: imageSize,
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

    private var aspect: CGFloat {
        guard imageSize.width > 0, imageSize.height > 0 else { return 16.0 / 9 }
        return imageSize.width / imageSize.height
    }
}
