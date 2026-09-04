import AppKit
import SwiftUI

// The pieces the Analyze canvas is built from: the input chip, the media frame, the native result
// renderings (boxes, masks, track spans, transcripts, raw documents), and the two panels of the
// result column.

// MARK: - Shared chrome

extension View {
    /// The board's media frame: radius 10 on a hairline border, with the panel shadow.
    func mereMediaFrame() -> some View {
        clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg))
            .overlay {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .strokeBorder(MereRunTheme.border, lineWidth: 1)
            }
            .mereShadow(radius: 24, y: 8)
    }

    /// The board's `panel`: `surface` on a 1pt border at 80%, radius 10.
    func mereAnalyzePanel() -> some View {
        background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                }
        }
    }
}

/// The board's `chip`: 24pt tall on `surfaceRaised`, radius 6, 11.5pt medium. The compact variant
/// (20pt, 10.5pt) is what the parameter strips use.
struct StudioAnalyzeChip: View {
    let text: String
    var isCompact = false

    var body: some View {
        Text(text)
            .font(.system(size: isCompact ? 10.5 : 11.5, weight: .medium))
            .foregroundStyle(MereRunTheme.textPrimary)
            .lineLimit(1)
            .truncationMode(.middle)
            .padding(.horizontal, 9)
            .frame(height: isCompact ? 20 : 24)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                    .fill(MereRunTheme.surfaceRaised)
            }
    }
}

// MARK: - Image with overlays

/// The input image at its own aspect ratio, with the run's result drawn over it.
///
/// The frame hugs the picture (`aspectRatio(.fit)` reports the fitted size), so the overlay's
/// coordinate space *is* the displayed image and one uniform scale maps result pixels onto it.
struct StudioAnalyzeImageView: View {
    let url: URL
    let maxHeight: CGFloat
    /// Drawn as 2pt accent rectangles with a label tab.
    let detections: [StudioAnalyzeDetection]
    /// Composited as a soft accent tint.
    let masks: [StudioAnalyzeDetection]
    /// The image's true pixel size, which the result's coordinates are in.
    let imageSize: CGSize?

    @State private var image: NSImage?
    @State private var didLoad = false

    private var pixelSize: CGSize {
        if let imageSize, imageSize.width > 0, imageSize.height > 0 { return imageSize }
        if let image, image.size.width > 0, image.size.height > 0 { return image.size }
        return CGSize(width: 1, height: 1)
    }

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: maxHeight)
                    .overlay { overlay }
            } else {
                Rectangle()
                    .fill(MereRunTheme.surfaceRaised)
                    .aspectRatio(1, contentMode: .fit)
                    .frame(maxHeight: maxHeight)
                    .overlay {
                        if didLoad {
                            Image(systemName: "photo")
                                .font(.system(size: 32, weight: .semibold))
                                .foregroundStyle(MereRunTheme.textMuted)
                        } else {
                            ProgressView().controlSize(.small)
                        }
                    }
            }
        }
        .task(id: url) {
            didLoad = false
            image = nil
            let loaded = await Task.detached(priority: .userInitiated) {
                StudioImagePreviewLoader.downsampledImage(from: url, maxPixelSize: 1_600)
            }.value
            guard !Task.isCancelled else { return }
            image = loaded?.image
            didLoad = true
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibilityDescription)
    }

    private var accessibilityDescription: String {
        guard !detections.isEmpty else { return "Input image \(url.lastPathComponent)" }
        return "Input image with \(detections.count) results: "
            + detections.map(\.label).joined(separator: ", ")
    }

    private var overlay: some View {
        GeometryReader { geometry in
            ZStack(alignment: .topLeading) {
                ForEach(masks) { detection in
                    maskLayer(detection, in: geometry.size)
                }
                ForEach(detections) { detection in
                    box(detection, in: geometry.size)
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func box(_ detection: StudioAnalyzeDetection, in size: CGSize) -> some View {
        let rect = StudioAnalyzeGeometry.viewRect(
            for: detection.box, imageSize: pixelSize, displaySize: size
        )
        return RoundedRectangle(cornerRadius: 4)
            .strokeBorder(MereRunTheme.accent, lineWidth: 2)
            // The design's inset hairline: it keeps the accent edge readable over a pale subject.
            .overlay {
                RoundedRectangle(cornerRadius: 3)
                    .strokeBorder(Color.black.opacity(0.25), lineWidth: 1)
                    .padding(2)
            }
            .frame(width: max(rect.width, 2), height: max(rect.height, 2))
            .overlay(alignment: .topLeading) {
                StudioAnalyzeBoxLabel(text: detection.tabLabel)
                    .fixedSize()
                    .offset(x: -2, y: -22)
            }
            .offset(x: rect.minX, y: rect.minY)
    }

    /// The mask PNG the run wrote, tinted; a detection with no mask file falls back to a tinted
    /// box so the Masks view never comes up empty.
    @ViewBuilder
    private func maskLayer(_ detection: StudioAnalyzeDetection, in size: CGSize) -> some View {
        if let maskURL = detection.maskURL {
            StudioAnalyzeMaskLayer(url: maskURL)
        } else {
            let rect = StudioAnalyzeGeometry.viewRect(
                for: detection.box, imageSize: pixelSize, displaySize: size
            )
            RoundedRectangle(cornerRadius: 4)
                .fill(MereRunTheme.accent.opacity(0.45))
                .frame(width: max(rect.width, 2), height: max(rect.height, 2))
                .offset(x: rect.minX, y: rect.minY)
        }
    }
}

/// "coffee cup 0.94" on an accent tab that sits on top of its box.
private struct StudioAnalyzeBoxLabel: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(MereRunTheme.onAccent)
            .lineLimit(1)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background {
                UnevenRoundedRectangle(
                    cornerRadii: .init(topLeading: 4, bottomLeading: 0, bottomTrailing: 0, topTrailing: 4)
                )
                .fill(MereRunTheme.accent)
            }
    }
}

extension StudioAnalyzeDetection {
    /// What the box tab reads: "coffee cup 0.94", or just the label when there is no score.
    var tabLabel: String {
        [label, confidenceDescription].compactMap { $0 }.joined(separator: " ")
    }
}

/// One mask PNG composited over the image. The file is white-on-black (or white on transparent),
/// so its luminance becomes the alpha of an accent wash.
private struct StudioAnalyzeMaskLayer: View {
    let url: URL

    @State private var mask: NSImage?

    var body: some View {
        Group {
            if let mask {
                Rectangle()
                    .fill(MereRunTheme.accent)
                    .mask {
                        Image(nsImage: mask)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .luminanceToAlpha()
                    }
                    .opacity(0.45)
            } else {
                Color.clear
            }
        }
        .task(id: url) {
            let loaded = await Task.detached(priority: .userInitiated) {
                StudioImagePreviewLoader.downsampledImage(from: url, maxPixelSize: 1_600)
            }.value
            guard !Task.isCancelled else { return }
            mask = loaded?.image
        }
    }
}

// MARK: - Track spans

/// Where each tracked object is visible across the clip, under the video's own scrubber.
struct StudioAnalyzeTrackScrubber: View {
    let document: StudioVisionTrackDocument

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            ForEach(document.objects, id: \.objectID) { object in
                HStack(spacing: 8) {
                    Text(object.label)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .lineLimit(1)
                        .frame(width: 92, alignment: .leading)
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            Capsule().fill(MereRunTheme.surfaceRaised)
                            ForEach(Array(spans(of: object.objectID).enumerated()), id: \.offset) { _, span in
                                Capsule()
                                    .fill(MereRunTheme.accent)
                                    .frame(width: max(2, geometry.size.width * span.width))
                                    .offset(x: geometry.size.width * span.start)
                            }
                        }
                    }
                    .frame(height: 6)
                    Text(StudioTimeFormat.string(document.duration))
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel("\(object.label) visible in \(visibleFrameCount(of: object.objectID)) frames")
            }
        }
    }

    private struct Span {
        let start: Double
        let width: Double
    }

    private func visibleFrameCount(of objectID: String) -> Int {
        document.frames.filter { frame in
            frame.detections.contains { $0.objectID == objectID && $0.visible }
        }.count
    }

    /// The runs of consecutive frames where the object is visible, as fractions of the clip.
    private func spans(of objectID: String) -> [Span] {
        let total = Double(document.frames.count)
        guard total > 0 else { return [] }
        var spans: [Span] = []
        var runStart: Int?
        for (index, frame) in document.frames.enumerated() {
            let visible = frame.detections.contains { $0.objectID == objectID && $0.visible }
            if visible, runStart == nil {
                runStart = index
            } else if !visible, let start = runStart {
                spans.append(Span(start: Double(start) / total, width: Double(index - start) / total))
                runStart = nil
            }
        }
        if let start = runStart {
            spans.append(Span(start: Double(start) / total, width: (total - Double(start)) / total))
        }
        return spans
    }
}

// MARK: - Raw document

/// The result document as the run wrote it, monospaced.
struct StudioAnalyzeDocumentView: View {
    let url: URL?
    let text: String?

    var body: some View {
        ScrollView([.vertical, .horizontal]) {
            Text(text ?? "This run left no result document.")
                .font(.system(size: 11.5, design: .monospaced))
                .foregroundStyle(text == nil ? MereRunTheme.textMuted : MereRunTheme.textPrimary)
                .textSelection(.enabled)
                .padding(MereRunTheme.Spacing.md)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(MereRunTheme.surface)
        .accessibilityLabel(url.map { "Result document \($0.lastPathComponent)" } ?? "Result document")
    }
}

// MARK: - Result panel

/// What the model found: a header that names it, one row per result, and the steps that continue
/// from it.
struct StudioAnalyzeResultPanel: View {
    let item: StudioLibraryItem
    let document: StudioAnalyzeDocument?
    let detections: [StudioAnalyzeDetection]
    let speechSegments: [StudioAnalyzeSpeechSegment]
    let outputText: String?
    let view: StudioAnalyzeResultView
    let nextActions: [StudioAnalyzeNextAction]
    let onOpenTask: (StudioTask) -> Void
    let onSave: (StudioAnalyzeSaveKind) -> Void

    private static let rowsMaxHeight: CGFloat = 320
    /// More rows than this and the list scrolls at `rowsMaxHeight` rather than growing.
    private static let scrollingRowThreshold = 7

    var body: some View {
        VStack(spacing: 0) {
            header
            hairline(0.4)
            rows
            actionRow
        }
        .mereAnalyzePanel()
        .accessibilityElement(children: .contain)
        .accessibilityLabel(summary)
    }

    private var summary: String {
        if let document { return document.summary(detectionCount: detections.count) }
        if outputText?.isBlank == false { return "Result" }
        return "No result"
    }

    private var meta: String {
        var parts: [String] = []
        if let model = document?.modelID ?? item.commandDraft?.model, !model.isBlank {
            parts.append(StudioComposer.displayModelName(model))
        }
        let elapsed = item.updatedAt.timeIntervalSince(item.createdAt)
        if elapsed >= 0.05 { parts.append(String(format: "%.1f s", elapsed)) }
        return parts.joined(separator: " · ")
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text(summary)
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.textPrimary)
            Spacer(minLength: 8)
            Text(meta)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }

    private func hairline(_ opacity: Double) -> some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(opacity))
            .frame(height: 1)
    }

    @ViewBuilder
    private var rows: some View {
        switch view {
        case .transcript, .timeline:
            speechRows
        case .text, .score:
            textRow
        default:
            if detections.isEmpty {
                textRow
            } else {
                detectionRows
            }
        }
    }

    private var detectionRows: some View {
        boundedRows(count: detections.count) {
            ForEach(detections) { detection in
                HStack(spacing: 10) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(MereRunTheme.accent)
                        .frame(width: 10, height: 10)
                    Text(detection.label)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if let confidence = detection.confidenceDescription {
                        Text(confidence)
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(MereRunTheme.textSecondary)
                            .fixedSize()
                    }
                    Text(detection.boxDescription)
                        .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                        .fixedSize()
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .accessibilityElement(children: .combine)
                hairline(0.27)
            }
        }
    }

    private var speechRows: some View {
        boundedRows(count: speechSegments.count) {
            ForEach(speechSegments) { segment in
                HStack(alignment: .top, spacing: 10) {
                    Text(segment.startDescription)
                        .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .frame(width: 44, alignment: .leading)
                    VStack(alignment: .leading, spacing: 2) {
                        if let speaker = segment.speaker {
                            Text(speaker)
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(MereRunTheme.accent)
                        }
                        if !segment.text.isEmpty {
                            Text(segment.text)
                                .font(.system(size: 13))
                                .foregroundStyle(MereRunTheme.textPrimary)
                                .fixedSize(horizontal: false, vertical: true)
                                .textSelection(.enabled)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .accessibilityElement(children: .combine)
                hairline(0.27)
            }
        }
    }

    @ViewBuilder
    private var textRow: some View {
        if let text = outputText?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
            ScrollView {
                Text(text)
                    .font(.system(size: 13))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 12)
            }
            .frame(maxHeight: Self.rowsMaxHeight)
            hairline(0.27)
        } else {
            Text("This run left no readable result.")
                .font(.system(size: 12.5))
                .foregroundStyle(MereRunTheme.textMuted)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
            hairline(0.27)
        }
    }

    /// Rows size to their content; a long list scrolls inside the panel instead of pushing the
    /// action row off the column.
    @ViewBuilder
    private func boundedRows<Content: View>(
        count: Int,
        @ViewBuilder content: () -> Content
    ) -> some View {
        if count > Self.scrollingRowThreshold {
            ScrollView {
                VStack(spacing: 0) { content() }
            }
            .frame(height: Self.rowsMaxHeight)
        } else {
            VStack(spacing: 0) { content() }
        }
    }

    private var actionRow: some View {
        HStack(spacing: 6) {
            ForEach(nextActions) { action in
                if case .openTask(let task) = action.kind {
                    Button(action.title) { onOpenTask(task) }
                        .buttonStyle(.mereSecondary)
                        .help("Open \(task.domain.title) ▸ \(task.title) with this input")
                }
            }
            Spacer(minLength: 8)
            ForEach(nextActions) { action in
                if case .save(let kind) = action.kind {
                    Button(action.title) { onSave(kind) }
                        .buttonStyle(.mereSecondary)
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }
}

// MARK: - Prompt panel

/// What was asked, and the settings it ran with.
struct StudioAnalyzePromptPanel: View {
    let item: StudioLibraryItem

    private var prompt: String {
        item.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The run's own chips, capitalized the way the board draws them.
    private var chips: [String] {
        StudioFeedChips.chips(for: item).map { chip in
            guard let first = chip.first else { return chip }
            return first.uppercased() + chip.dropFirst()
        }
    }

    var body: some View {
        if !prompt.isEmpty || !chips.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text(prompt.isEmpty ? "Run" : "Prompt")
                    .font(.system(size: 11, weight: .semibold))
                    .kerning(0.55)
                    .textCase(.uppercase)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .accessibilityAddTraits(.isHeader)
                if !prompt.isEmpty {
                    Text(prompt)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
                if !chips.isEmpty {
                    HStack(spacing: 4) {
                        ForEach(chips, id: \.self) { chip in
                            StudioAnalyzeChip(text: chip, isCompact: true)
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .mereAnalyzePanel()
        }
    }
}
