import CoreGraphics
import Foundation

// Boxes and points a person draws on an image to say what to segment or track, the geometry that
// maps them between the picture on screen and the picture's own pixels, and the text the CLI reads
// them as. The views in `StudioUI/StudioRegionPromptEditor.swift` draw and edit these; nothing
// here knows about SwiftUI, so the coordinate math and the encoding are testable on their own.

// MARK: - One drawn prompt

/// A box or a point drawn on the input, in the input's own pixel space (origin top-left).
///
/// The CLI's `vision segment` and `vision track` take these as `--box x1,y1,x2,y2[,label]` and
/// `--point x,y,positive|negative[,label]` in image pixels, and Video ▸ Subjects writes the same
/// shapes into its mask plan. The label is optional and rides into the result document.
package struct StudioRegionPrompt: Identifiable, Codable, Equatable, Sendable {
    package enum Shape: Codable, Equatable, Sendable {
        /// Corners in xyxy order; `StudioRegionPrompt.box` normalizes them so x1 ≤ x2 and y1 ≤ y2.
        case box(x1: Double, y1: Double, x2: Double, y2: Double)
        /// A point the mask must include (positive) or avoid (negative).
        case point(x: Double, y: Double, isPositive: Bool)
    }

    package var id: UUID
    package var shape: Shape
    /// What this prompt is for ("coffee cup"); carried into the CLI's result document. Commas are
    /// never stored because the CLI splits the prompt text on them.
    package var label: String?

    package init(id: UUID = UUID(), shape: Shape, label: String? = nil) {
        self.id = id
        self.shape = shape
        self.label = Self.sanitizedLabel(label)
    }

    /// A box covering `rect`, whichever way it was dragged.
    package static func box(_ rect: CGRect, label: String? = nil, id: UUID = UUID()) -> StudioRegionPrompt {
        let standardized = rect.standardized
        return StudioRegionPrompt(
            id: id,
            shape: .box(
                x1: standardized.minX, y1: standardized.minY,
                x2: standardized.maxX, y2: standardized.maxY
            ),
            label: label
        )
    }

    package static func point(_ point: CGPoint, isPositive: Bool, label: String? = nil, id: UUID = UUID()) -> StudioRegionPrompt {
        StudioRegionPrompt(id: id, shape: .point(x: point.x, y: point.y, isPositive: isPositive), label: label)
    }

    package var isBox: Bool {
        if case .box = shape { return true }
        return false
    }

    package var isPoint: Bool { !isBox }

    package var isPositivePoint: Bool {
        if case .point(_, _, let isPositive) = shape { return isPositive }
        return false
    }

    package var isNegativePoint: Bool {
        if case .point(_, _, let isPositive) = shape { return !isPositive }
        return false
    }

    /// The box in pixel space, for a box prompt.
    package var rect: CGRect? {
        guard case .box(let x1, let y1, let x2, let y2) = shape else { return nil }
        return CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
    }

    /// The point in pixel space, for a point prompt.
    package var point: CGPoint? {
        guard case .point(let x, let y, _) = shape else { return nil }
        return CGPoint(x: x, y: y)
    }

    /// The same prompt shifted by `delta` pixels, kept inside `imageSize`.
    package func moved(by delta: CGVector, within imageSize: CGSize) -> StudioRegionPrompt {
        switch shape {
        case .box:
            guard let rect else { return self }
            return StudioRegionPrompt.box(
                StudioRegionGeometry.clampedRect(rect.offsetBy(dx: delta.dx, dy: delta.dy), within: imageSize),
                label: label, id: id
            )
        case .point(_, _, let isPositive):
            guard let point else { return self }
            let shifted = CGPoint(x: point.x + delta.dx, y: point.y + delta.dy)
            return StudioRegionPrompt.point(
                StudioRegionGeometry.clampedPoint(shifted, within: imageSize),
                isPositive: isPositive, label: label, id: id
            )
        }
    }

    /// A box prompt spanning `anchor` and `imagePoint`: the corner being dragged lands on the
    /// point, the opposite one stays on the anchor. The anchor is captured when the drag begins
    /// (`anchor(for:)`) rather than re-read from the box, so dragging a corner across the box
    /// flips it cleanly instead of collapsing it. Points have no corners and come back unchanged.
    package func resizingBox(anchor: CGPoint, to imagePoint: CGPoint, within imageSize: CGSize) -> StudioRegionPrompt {
        guard isBox else { return self }
        let moved = StudioRegionGeometry.clampedPoint(imagePoint, within: imageSize)
        return StudioRegionPrompt.box(StudioRegionGeometry.rect(from: anchor, to: moved), label: label, id: id)
    }

    /// The corner that stays put while `corner` is dragged.
    package func anchor(for corner: StudioRegionBoxCorner) -> CGPoint? {
        rect.map { corner.opposite.point(of: $0) }
    }

    /// What VoiceOver reads for this prompt: "Box 1, 120 by 80 at 40, 30", "Positive point 2 at
    /// 400, 260". `ordinal` is its 1-based position among all the prompts on the image.
    package func accessibilityDescription(ordinal: Int) -> String {
        let name = label.map { ", \($0)" } ?? ""
        switch shape {
        case .box:
            guard let rect else { return "Box \(ordinal)" }
            return "Box \(ordinal)\(name), \(Self.pixels(rect.width)) by \(Self.pixels(rect.height)) at "
                + "\(Self.pixels(rect.minX)), \(Self.pixels(rect.minY))"
        case .point(let x, let y, let isPositive):
            return "\(isPositive ? "Positive" : "Negative") point \(ordinal)\(name) at \(Self.pixels(x)), \(Self.pixels(y))"
        }
    }

    static func pixels(_ value: Double) -> String {
        String(Int(value.rounded()))
    }

    /// Commas become spaces and whitespace runs collapse, so "cup, saucer" is stored as
    /// "cup saucer" and never splits the CLI's comma-separated prompt.
    private static func sanitizedLabel(_ label: String?) -> String? {
        guard let label else { return nil }
        let cleaned = label.replacingOccurrences(of: ",", with: " ")
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
        return cleaned.isEmpty ? nil : cleaned
    }
}

extension StudioDraft {
    /// Points the draft at a different input. Boxes and points are in the previous picture's
    /// pixels and Track's frames belong to the previous clip, so they go with it; an unchanged
    /// path keeps them. Every way an input arrives (the well, a drop or paste, a Library row, a
    /// handoff) goes through here.
    package mutating func replaceInput(_ path: String) {
        guard path != inputPath else { return }
        inputPath = path
        visionRegionPrompts = nil
        visionInitFrame = nil
        visionEndFrame = nil
    }

    /// The `--box` values as one newline-separated text, the way the Command view carries a
    /// repeated option. Writing it replaces the boxes and keeps the points.
    package var visionBoxPromptsText: String {
        get { StudioRegionPromptText.boxText(visionRegionPrompts ?? []) }
        set { replaceRegionPrompts(boxes: StudioRegionPromptText.prompts(boxText: newValue, pointText: ""), points: nil) }
    }

    /// The `--point` values, likewise; writing it replaces the points and keeps the boxes.
    package var visionPointPromptsText: String {
        get { StudioRegionPromptText.pointText(visionRegionPrompts ?? []) }
        set { replaceRegionPrompts(boxes: nil, points: StudioRegionPromptText.prompts(boxText: "", pointText: newValue)) }
    }

    /// `--init-frame` as the Command view's integer; 0 is the CLI's default and reads as unset.
    package var visionInitFrameValue: Int {
        get { visionInitFrame ?? 0 }
        set { visionInitFrame = newValue == 0 ? nil : newValue }
    }

    /// `--end-frame` as text, blank meaning the end of the clip.
    package var visionEndFrameText: String {
        get { visionEndFrame.map(String.init) ?? "" }
        set { visionEndFrame = Int(newValue.trimmingCharacters(in: .whitespaces)) }
    }

    private mutating func replaceRegionPrompts(boxes: [StudioRegionPrompt]?, points: [StudioRegionPrompt]?) {
        let current = visionRegionPrompts ?? []
        let merged = (boxes ?? current.boxes) + (points ?? current.points)
        visionRegionPrompts = merged.isEmpty ? nil : merged
    }
}

extension Array where Element == StudioRegionPrompt {
    package var boxes: [StudioRegionPrompt] { filter(\.isBox) }
    package var points: [StudioRegionPrompt] { filter(\.isPoint) }

    /// "2 boxes · 1 point", or "" for an empty set — the toolbar's count chip.
    package var countDescription: String {
        var parts: [String] = []
        let boxCount = boxes.count
        let pointCount = points.count
        if boxCount > 0 { parts.append(boxCount == 1 ? "1 box" : "\(boxCount) boxes") }
        if pointCount > 0 { parts.append(pointCount == 1 ? "1 point" : "\(pointCount) points") }
        return parts.joined(separator: " · ")
    }
}

/// One corner of a box prompt, as a resize handle names it.
package enum StudioRegionBoxCorner: CaseIterable, Hashable, Sendable {
    case topLeading
    case topTrailing
    case bottomLeading
    case bottomTrailing

    package var opposite: StudioRegionBoxCorner {
        switch self {
        case .topLeading: return .bottomTrailing
        case .topTrailing: return .bottomLeading
        case .bottomLeading: return .topTrailing
        case .bottomTrailing: return .topLeading
        }
    }

    package func point(of rect: CGRect) -> CGPoint {
        switch self {
        case .topLeading: return CGPoint(x: rect.minX, y: rect.minY)
        case .topTrailing: return CGPoint(x: rect.maxX, y: rect.minY)
        case .bottomLeading: return CGPoint(x: rect.minX, y: rect.maxY)
        case .bottomTrailing: return CGPoint(x: rect.maxX, y: rect.maxY)
        }
    }
}

// MARK: - Upright versus stored

/// The EXIF orientation of a picture: how a viewer turns the stored pixels to show it upright.
///
/// The CLI decodes without the transform and keeps every coordinate in the stored pixels, while
/// Studio shows the picture upright, so prompts drawn on screen and results drawn back onto it
/// pass through this mapping. `displayTransform` takes a stored point (origin top-left) to its
/// upright position (origin top-left); the eight cases are the EXIF values 1…8.
package enum StudioImageOrientation: Int, Codable, Equatable, Sendable, CaseIterable {
    case up = 1
    case upMirrored = 2
    case down = 3
    case downMirrored = 4
    /// Transposed: mirrored and turned a quarter turn.
    case leftMirrored = 5
    /// Turned a quarter turn clockwise to show upright — the portrait phone photo.
    case right = 6
    case rightMirrored = 7
    /// Turned a quarter turn counter-clockwise to show upright.
    case left = 8

    package init?(exif: Int) {
        self.init(rawValue: exif)
    }

    /// Whether the upright picture swaps the stored width and height.
    package var swapsAxes: Bool { rawValue >= 5 }

    package func displaySize(ofStored size: CGSize) -> CGSize {
        swapsAxes ? CGSize(width: size.height, height: size.width) : size
    }

    /// Stored point → upright point, both origin top-left.
    package func displayTransform(storedSize: CGSize) -> CGAffineTransform {
        let width = storedSize.width
        let height = storedSize.height
        switch self {
        case .up: return .identity
        case .upMirrored: return CGAffineTransform(a: -1, b: 0, c: 0, d: 1, tx: width, ty: 0)
        case .down: return CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: width, ty: height)
        case .downMirrored: return CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: height)
        case .leftMirrored: return CGAffineTransform(a: 0, b: 1, c: 1, d: 0, tx: 0, ty: 0)
        case .right: return CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: height, ty: 0)
        case .rightMirrored: return CGAffineTransform(a: 0, b: -1, c: -1, d: 0, tx: height, ty: width)
        case .left: return CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: width)
        }
    }

    package func displayPoint(fromStored point: CGPoint, storedSize: CGSize) -> CGPoint {
        point.applying(displayTransform(storedSize: storedSize))
    }

    package func storedPoint(fromDisplay point: CGPoint, storedSize: CGSize) -> CGPoint {
        point.applying(displayTransform(storedSize: storedSize).inverted())
    }

    /// A stored rect's upright bounds (a quarter turn swaps its sides).
    package func displayRect(fromStored rect: CGRect, storedSize: CGSize) -> CGRect {
        rect.applying(displayTransform(storedSize: storedSize))
    }

    package func storedRect(fromDisplay rect: CGRect, storedSize: CGSize) -> CGRect {
        rect.applying(displayTransform(storedSize: storedSize).inverted())
    }
}

extension StudioRegionPrompt {
    /// This prompt in the upright picture's pixels, for drawing and hit testing on screen.
    package func inDisplaySpace(_ orientation: StudioImageOrientation, storedSize: CGSize) -> StudioRegionPrompt {
        mapped { orientation.displayPoint(fromStored: $0, storedSize: storedSize) }
    }

    /// This prompt back in the stored pixels the CLI reads.
    package func inStoredSpace(_ orientation: StudioImageOrientation, storedSize: CGSize) -> StudioRegionPrompt {
        mapped { orientation.storedPoint(fromDisplay: $0, storedSize: storedSize) }
    }

    private func mapped(_ transform: (CGPoint) -> CGPoint) -> StudioRegionPrompt {
        switch shape {
        case .box:
            guard let rect else { return self }
            let start = transform(CGPoint(x: rect.minX, y: rect.minY))
            let end = transform(CGPoint(x: rect.maxX, y: rect.maxY))
            return .box(StudioRegionGeometry.rect(from: start, to: end), label: label, id: id)
        case .point(_, _, let isPositive):
            guard let point else { return self }
            return .point(transform(point), isPositive: isPositive, label: label, id: id)
        }
    }
}

extension Array where Element == StudioRegionPrompt {
    package func inDisplaySpace(_ orientation: StudioImageOrientation, storedSize: CGSize) -> [StudioRegionPrompt] {
        map { $0.inDisplaySpace(orientation, storedSize: storedSize) }
    }

    package func inStoredSpace(_ orientation: StudioImageOrientation, storedSize: CGSize) -> [StudioRegionPrompt] {
        map { $0.inStoredSpace(orientation, storedSize: storedSize) }
    }
}

// MARK: - Between the screen and the pixels

/// Mapping between the view that shows an image and the image's own pixels.
///
/// The image is aspect-fitted into the view, so `fitted` is where its pixels land
/// (`StudioAnalyzeGeometry.fittedRect`, letterboxed when the view's aspect differs). One uniform
/// scale maps both axes; every conversion clamps to the image so a drag that leaves the picture
/// still produces a prompt the CLI accepts. The pixel space here is the picture as shown; a
/// rotated photo's prompts go through `StudioImageOrientation` on their way to the stored pixels
/// the CLI reads.
package enum StudioRegionGeometry {
    /// Points per pixel inside `fitted`.
    package static func scale(imageSize: CGSize, fitted: CGRect) -> CGFloat {
        guard imageSize.width > 0, imageSize.height > 0 else { return 1 }
        return fitted.width / imageSize.width
    }

    /// The pixel under a view point, clamped to the image.
    package static func imagePoint(fromView viewPoint: CGPoint, imageSize: CGSize, fitted: CGRect) -> CGPoint {
        let scale = scale(imageSize: imageSize, fitted: fitted)
        guard scale > 0 else { return .zero }
        let raw = CGPoint(x: (viewPoint.x - fitted.minX) / scale, y: (viewPoint.y - fitted.minY) / scale)
        return clampedPoint(raw, within: imageSize)
    }

    /// Where a pixel lands in the view.
    package static func viewPoint(fromImage imagePoint: CGPoint, imageSize: CGSize, fitted: CGRect) -> CGPoint {
        let scale = scale(imageSize: imageSize, fitted: fitted)
        return CGPoint(x: fitted.minX + imagePoint.x * scale, y: fitted.minY + imagePoint.y * scale)
    }

    /// Where a pixel rect lands in the view.
    package static func viewRect(fromImage imageRect: CGRect, imageSize: CGSize, fitted: CGRect) -> CGRect {
        let scale = scale(imageSize: imageSize, fitted: fitted)
        return CGRect(
            x: fitted.minX + imageRect.minX * scale,
            y: fitted.minY + imageRect.minY * scale,
            width: imageRect.width * scale,
            height: imageRect.height * scale
        )
    }

    /// The pixel rect between two view points, whichever way they were dragged, clamped to the image.
    package static func imageRect(fromView start: CGPoint, to end: CGPoint, imageSize: CGSize, fitted: CGRect) -> CGRect {
        rect(
            from: imagePoint(fromView: start, imageSize: imageSize, fitted: fitted),
            to: imagePoint(fromView: end, imageSize: imageSize, fitted: fitted)
        )
    }

    /// The rect spanning two corners in any order.
    package static func rect(from start: CGPoint, to end: CGPoint) -> CGRect {
        CGRect(
            x: min(start.x, end.x),
            y: min(start.y, end.y),
            width: abs(end.x - start.x),
            height: abs(end.y - start.y)
        )
    }

    package static func clampedPoint(_ point: CGPoint, within imageSize: CGSize) -> CGPoint {
        CGPoint(
            x: min(max(0, point.x), max(0, imageSize.width)),
            y: min(max(0, point.y), max(0, imageSize.height))
        )
    }

    /// The rect slid (not shrunk) back inside the image, so moving a box never distorts it.
    package static func clampedRect(_ rect: CGRect, within imageSize: CGSize) -> CGRect {
        var moved = rect.standardized
        moved.size.width = min(moved.width, imageSize.width)
        moved.size.height = min(moved.height, imageSize.height)
        moved.origin.x = min(max(0, moved.minX), imageSize.width - moved.width)
        moved.origin.y = min(max(0, moved.minY), imageSize.height - moved.height)
        return moved
    }
}

// MARK: - What is under the pointer

/// Which part of which prompt a view point lands on, so a press knows whether it resizes, moves,
/// selects, or starts something new. Handles win over their box, and points win over boxes,
/// because they are smaller.
package enum StudioRegionHit: Equatable {
    case handle(id: UUID, corner: StudioRegionBoxCorner)
    case point(id: UUID)
    case box(id: UUID)

    package var id: UUID {
        switch self {
        case .handle(let id, _), .point(let id), .box(let id): return id
        }
    }

    /// The topmost hit at `viewPoint`, later prompts first so the one drawn last wins.
    ///
    /// - Parameters:
    ///   - handleRadius: how far from a handle's centre, in view points, a press still grabs it;
    ///     only a selected box shows handles, so `selectedID` decides which box offers them.
    ///   - pointRadius: how far from a point marker's centre a press still grabs it.
    package static func hit(
        in prompts: [StudioRegionPrompt],
        at viewPoint: CGPoint,
        imageSize: CGSize,
        fitted: CGRect,
        selectedID: UUID?,
        handleRadius: CGFloat,
        pointRadius: CGFloat
    ) -> StudioRegionHit? {
        if let selectedID, let selected = prompts.first(where: { $0.id == selectedID }), let rect = selected.rect {
            let viewRect = StudioRegionGeometry.viewRect(fromImage: rect, imageSize: imageSize, fitted: fitted)
            for corner in StudioRegionBoxCorner.allCases {
                let center = corner.point(of: viewRect)
                if abs(center.x - viewPoint.x) <= handleRadius, abs(center.y - viewPoint.y) <= handleRadius {
                    return .handle(id: selectedID, corner: corner)
                }
            }
        }
        for prompt in prompts.reversed() {
            guard let point = prompt.point else { continue }
            let center = StudioRegionGeometry.viewPoint(fromImage: point, imageSize: imageSize, fitted: fitted)
            if hypot(center.x - viewPoint.x, center.y - viewPoint.y) <= pointRadius {
                return .point(id: prompt.id)
            }
        }
        for prompt in prompts.reversed() {
            guard let rect = prompt.rect else { continue }
            let viewRect = StudioRegionGeometry.viewRect(fromImage: rect, imageSize: imageSize, fitted: fitted)
            if viewRect.insetBy(dx: -2, dy: -2).contains(viewPoint) {
                return .box(id: prompt.id)
            }
        }
        return nil
    }
}

// MARK: - The text the CLI reads

/// Encoding prompts as the `--box` and `--point` values `vision segment` and `vision track` parse,
/// one per line as `CommandDraft.visionBoxPrompts` / `visionPointPrompts` hold them.
///
/// The formats are `VisionSegment.parseBoxPrompt` and `parsePointPrompt` in
/// `Sources/MereRunCLI/Commands/VisionSegmentCommand.swift`: four or five comma-separated fields
/// for a box (`x1,y1,x2,y2[,label]`), three or four for a point (`x,y,positive[,label]`, where the
/// polarity field accepts `positive`, `pos`, `p`, `1`, `negative`, `neg`, `n`, or `0`), every
/// coordinate a `Float`. Coordinates are written as whole pixels, which is what the CLI's help
/// promises ("in image pixels") and what the result documents report back.
package enum StudioRegionPromptText {
    /// One `--box` value per box prompt, with the labels `associating` gives.
    package static func boxLines(_ prompts: [StudioRegionPrompt]) -> [String] {
        associating(prompts).boxes.compactMap { prompt in
            guard let rect = prompt.rect else { return nil }
            var fields = [rect.minX, rect.minY, rect.maxX, rect.maxY].map(StudioRegionPrompt.pixels)
            if let label = prompt.label { fields.append(label) }
            return fields.joined(separator: ",")
        }
    }

    /// One `--point` value per point prompt, labeled after the box it refines (`associating`).
    package static func pointLines(_ prompts: [StudioRegionPrompt]) -> [String] {
        associating(prompts).points.compactMap { prompt in
            guard case .point(let x, let y, let isPositive) = prompt.shape else { return nil }
            var fields = [StudioRegionPrompt.pixels(x), StudioRegionPrompt.pixels(y), isPositive ? "positive" : "negative"]
            if let label = prompt.label { fields.append(label) }
            return fields.joined(separator: ",")
        }
    }

    /// The prompts as the CLI should read them, so each point stays with the box it was drawn on.
    ///
    /// The CLI groups a labeled point with the `--box` of the same label that contains it (else
    /// the first of that label), and unlabeled points with the one unlabeled box when there is
    /// exactly one (`SAM31PromptSet.normalized`). So a point drawn without a label of its own takes
    /// the label of the box it belongs to: the only box, or the smallest box containing it. With
    /// one box, an unlabeled box passes no label on and the CLI's single-unlabeled-box rule joins
    /// them. With several boxes and at least one point, an unlabeled box is named "object 1",
    /// "object 2", … by its place among the boxes, so the association survives the command line
    /// (and the result document names the objects the same way). A point outside every box, or
    /// inside none of several, stays unlabeled and forms one object with the other unlabeled
    /// points. Read back from the command line (`prompts(boxText:pointText:)`), a point keeps the
    /// box label it was sent with.
    package static func associating(_ prompts: [StudioRegionPrompt]) -> [StudioRegionPrompt] {
        var boxOrdinal = 0
        let needsNames = prompts.boxes.count > 1 && !prompts.points.isEmpty
        let named: [StudioRegionPrompt] = prompts.map { prompt in
            guard prompt.isBox else { return prompt }
            boxOrdinal += 1
            guard needsNames, prompt.label == nil else { return prompt }
            var labeled = prompt
            labeled.label = "object \(boxOrdinal)"
            return labeled
        }
        let boxes = named.boxes
        return named.map { prompt in
            guard let point = prompt.point, prompt.label == nil,
                  let label = refinedBox(for: point, in: boxes)?.label else { return prompt }
            var labeled = prompt
            labeled.label = label
            return labeled
        }
    }

    /// The box a point refines: the only box, or the smallest one containing the point.
    package static func refinedBox(for point: CGPoint, in boxes: [StudioRegionPrompt]) -> StudioRegionPrompt? {
        if boxes.count == 1 { return boxes[0] }
        return boxes
            .compactMap { box in box.rect.map { (box: box, rect: $0) } }
            .filter { $0.rect.contains(point) }
            .min { $0.rect.width * $0.rect.height < $1.rect.width * $1.rect.height }?
            .box
    }

    /// The `visionBoxPrompts` field: box lines joined with newlines.
    package static func boxText(_ prompts: [StudioRegionPrompt]) -> String {
        boxLines(prompts).joined(separator: "\n")
    }

    /// The `visionPointPrompts` field: point lines joined with newlines.
    package static func pointText(_ prompts: [StudioRegionPrompt]) -> String {
        pointLines(prompts).joined(separator: "\n")
    }

    /// The prompts a pair of draft fields describe, boxes first, skipping any line the CLI would
    /// reject the same way it would.
    package static func prompts(boxText: String, pointText: String) -> [StudioRegionPrompt] {
        lines(of: boxText).compactMap(box) + lines(of: pointText).compactMap(point)
    }

    /// `x1,y1,x2,y2[,label]`, or nil for anything `VisionSegment.parseBoxPrompt` would reject.
    package static func box(_ line: String) -> StudioRegionPrompt? {
        let parts = fields(of: line)
        guard parts.count == 4 || parts.count == 5,
              let x1 = Float(parts[0]), let y1 = Float(parts[1]),
              let x2 = Float(parts[2]), let y2 = Float(parts[3]) else { return nil }
        let rect = StudioRegionGeometry.rect(
            from: CGPoint(x: Double(x1), y: Double(y1)),
            to: CGPoint(x: Double(x2), y: Double(y2))
        )
        return .box(rect, label: parts.count == 5 ? parts[4] : nil)
    }

    /// `x,y,positive[,label]`, or nil for anything `VisionSegment.parsePointPrompt` would reject.
    package static func point(_ line: String) -> StudioRegionPrompt? {
        let parts = fields(of: line)
        guard parts.count == 3 || parts.count == 4,
              let x = Float(parts[0]), let y = Float(parts[1]) else { return nil }
        let isPositive: Bool
        switch parts[2].lowercased() {
        case "positive", "pos", "p", "1": isPositive = true
        case "negative", "neg", "n", "0": isPositive = false
        default: return nil
        }
        return .point(
            CGPoint(x: Double(x), y: Double(y)),
            isPositive: isPositive,
            label: parts.count == 4 ? parts[3] : nil
        )
    }

    private static func fields(of line: String) -> [String] {
        line.split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    }

    private static func lines(of text: String) -> [String] {
        text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }
}

/// The selector text Video ▸ Subjects keeps per subject and per correction, and writes into the
/// mask plan as `SCAIL2MaskSelector`: one box as `x1,y1,x2,y2`, and point lists as `x,y; x,y`
/// (a newline works as a separator too). A selector takes at most one box, so the first wins.
package enum StudioSubjectSelectorText {
    /// The prompts three selector fields describe.
    package static func prompts(box: String, positivePoints: String, negativePoints: String) -> [StudioRegionPrompt] {
        var prompts: [StudioRegionPrompt] = []
        if let rect = self.box(box) { prompts.append(.box(rect)) }
        prompts += points(positivePoints).map { StudioRegionPrompt.point($0, isPositive: true) }
        prompts += points(negativePoints).map { StudioRegionPrompt.point($0, isPositive: false) }
        return prompts
    }

    /// The first box as `x1,y1,x2,y2`, or "" when there is none.
    package static func boxText(_ prompts: [StudioRegionPrompt]) -> String {
        guard let rect = prompts.first(where: \.isBox)?.rect else { return "" }
        return [rect.minX, rect.minY, rect.maxX, rect.maxY].map(StudioRegionPrompt.pixels).joined(separator: ",")
    }

    /// The positive points as `x,y; x,y`.
    package static func positivePointsText(_ prompts: [StudioRegionPrompt]) -> String {
        pointsText(prompts.filter(\.isPositivePoint))
    }

    /// The negative points as `x,y; x,y`.
    package static func negativePointsText(_ prompts: [StudioRegionPrompt]) -> String {
        pointsText(prompts.filter(\.isNegativePoint))
    }

    private static func pointsText(_ prompts: [StudioRegionPrompt]) -> String {
        prompts.compactMap(\.point)
            .map { "\(StudioRegionPrompt.pixels($0.x)),\(StudioRegionPrompt.pixels($0.y))" }
            .joined(separator: "; ")
    }

    private static func box(_ raw: String) -> CGRect? {
        let values = raw.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        guard values.count == 4 else { return nil }
        return StudioRegionGeometry.rect(
            from: CGPoint(x: Double(values[0]), y: Double(values[1])),
            to: CGPoint(x: Double(values[2]), y: Double(values[3]))
        )
    }

    private static func points(_ raw: String) -> [CGPoint] {
        raw.replacingOccurrences(of: "\n", with: ";")
            .split(separator: ";")
            .compactMap { pair in
                let values = pair.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
                guard values.count == 2 else { return nil }
                return CGPoint(x: Double(values[0]), y: Double(values[1]))
            }
    }
}

// MARK: - Frames of a clip

/// Frame indices of a clip, as `vision track --init-frame` / `--end-frame` count them: frame `i`
/// is the picture shown from `i / fps` seconds. Studio asks the decoder for the middle of that
/// interval so a zero-tolerance seek lands inside frame `i` rather than on its edge.
///
/// The count mirrors the CLI's extraction (`AppleMediaVideoIO.extractFrames`): the track's
/// nominal rate, or 30 when it declares none, floored at 1 fps
/// (`MediaVideoFrameRateResolver.resolve(_:fallbackFPS: 30)`), and `duration × fps` rounded to
/// nearest, ties to even.
package struct StudioVideoFrameGrid: Equatable, Sendable {
    package static let fallbackFrameRate = 30.0

    package let frameCount: Int
    package let frameRate: Double

    /// The grid of a clip of `duration` seconds at `frameRate` (0 or non-finite for a clip that
    /// declares none); a clip always has at least one frame.
    package init(duration: TimeInterval, frameRate: Double) {
        let declared = frameRate.isFinite && frameRate > 0 ? frameRate : Self.fallbackFrameRate
        self.frameRate = max(1, declared)
        let counted = duration.isFinite && duration > 0
            ? Int((duration * self.frameRate).rounded(.toNearestOrEven))
            : 0
        frameCount = max(1, counted)
    }

    package var lastFrame: Int { frameCount - 1 }

    package func clamped(_ frame: Int) -> Int {
        min(max(0, frame), lastFrame)
    }

    /// The moment to decode for `frame`: the middle of its display interval.
    package func time(ofFrame frame: Int) -> TimeInterval {
        (Double(clamped(frame)) + 0.5) / frameRate
    }

    /// The moment to decode for plan frame `planTime` seconds into a clip that `video
    /// prepare-masks` resamples: the CLI picks source frame `round(t × sourceFPS)`
    /// (`SCAIL2MaskPreparer`, the normalized-frames loop), so this is the middle of that frame.
    package static func sourceTime(forPlanTime planTime: TimeInterval, sourceFrameRate: Double) -> TimeInterval {
        let rate = max(1, sourceFrameRate.isFinite && sourceFrameRate > 0 ? sourceFrameRate : fallbackFrameRate)
        return ((planTime * rate).rounded() + 0.5) / rate
    }

    /// "Prompts on frame 10 · tracks frames 0–30": what `vision track --init-frame 10 --end-frame
    /// 30` does. The tracker segments the prompts on the init frame, then propagates through the
    /// whole clip from frame 0 to the end frame (`SAM31VideoTracker.track` walks both directions
    /// from the seed), so the range never starts at the prompt frame.
    package func trackRangeDescription(promptFrame: Int, endFrame: Int?) -> String {
        let prompts = "Prompts on frame \(clamped(promptFrame))"
        guard let endFrame else { return "\(prompts) · tracks all \(frameCount) frames" }
        return "\(prompts) · tracks frames 0–\(clamped(endFrame))"
    }

    /// "0:04.5" — the clock the scrubber shows beside the frame number.
    package func timeDescription(ofFrame frame: Int) -> String {
        let seconds = Double(clamped(frame)) / frameRate
        let whole = Int(seconds)
        let tenths = Int(((seconds - Double(whole)) * 10).rounded(.down))
        return String(format: "%d:%02d.%d", whole / 60, whole % 60, tenths)
    }
}

/// Where a driving-clip frame lands in the mask plan's `width × height` canvas.
///
/// `video prepare-masks` decodes each driving frame and center-crops it to the plan size
/// (`MediaImageIO.centerCropped`: scale by the larger ratio, then crop the overflow evenly), so a
/// driving selector's coordinates are in that canvas, not the source clip. Drawing the frame into
/// a canvas with this rect reproduces the space the CLI segments in.
package enum StudioSubjectFrameGeometry {
    package static func coverRect(source: CGSize, canvas: CGSize) -> CGRect {
        guard source.width > 0, source.height > 0, canvas.width > 0, canvas.height > 0 else { return .zero }
        let scale = max(canvas.width / source.width, canvas.height / source.height)
        let scaled = CGSize(
            width: max(canvas.width, (source.width * scale).rounded()),
            height: max(canvas.height, (source.height * scale).rounded())
        )
        return CGRect(
            x: -((scaled.width - canvas.width) / 2).rounded(.down),
            y: -((scaled.height - canvas.height) / 2).rounded(.down),
            width: scaled.width,
            height: scaled.height
        )
    }
}
