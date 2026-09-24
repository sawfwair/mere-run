import AVFoundation
import AppKit
import StudioKit
import SwiftUI

// Picking frames on the clip instead of typing their numbers: Track's prompt frame (the seed the
// prompts are drawn on) and its optional end frame, on a scrubber over the decoded frame.

/// Single frames of one clip, decoded on demand.
///
/// One asset and two generators per clip: an exact one for the frame that stays on screen and a
/// tolerant one for scrubbing, which lets the decoder hand back the nearest keyframe instead of
/// decoding forward from it for every pointer move. Decoding is `async`, so a scrub that moves on
/// cancels the frame it no longer needs.
final class StudioVideoFrameLoader: @unchecked Sendable {
    let url: URL
    private let exactGenerator: AVAssetImageGenerator
    private let scrubGenerator: AVAssetImageGenerator

    init(url: URL, maxPixelSize: CGFloat = 1_600) {
        self.url = url
        let asset = AVURLAsset(url: url)
        func generator(tolerance: CMTime) -> AVAssetImageGenerator {
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: maxPixelSize, height: maxPixelSize)
            generator.requestedTimeToleranceBefore = tolerance
            generator.requestedTimeToleranceAfter = tolerance
            return generator
        }
        exactGenerator = generator(tolerance: .zero)
        scrubGenerator = generator(tolerance: CMTime(seconds: 0.5, preferredTimescale: 600))
    }

    /// The frame shown at `time`; `exact` decodes that very frame, otherwise a nearby one.
    func image(at time: TimeInterval, exact: Bool) async throws -> NSImage {
        let generator = exact ? exactGenerator : scrubGenerator
        let (cgImage, _) = try await generator.image(at: CMTime(seconds: time, preferredTimescale: 600))
        return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }

    /// The clip's frame rate as the container declares it, or nil when it declares none.
    static func frameRate(of url: URL) -> Double? {
        let rate = AVURLAsset(url: url).tracks(withMediaType: .video).first?.nominalFrameRate
        guard let rate, rate > 0 else { return nil }
        return Double(rate)
    }
}

/// Track's input: the clip's frames on a scrubber, the prompts drawn on the prompt frame, and the
/// buttons that make the shown frame the prompt frame or the end of tracking.
///
/// Prompts belong to the prompt frame (`--init-frame`) — that is the picture the tracker segments
/// before following each object — so they are only editable there. Scrubbing elsewhere shows them
/// faded, and "Draw prompts on this frame" moves them to the frame in view. The tracker then
/// covers the clip from frame 0 to the end frame whatever frame the prompts are on
/// (`StudioVideoFrameGrid.trackRangeDescription`), so the scrubber shades that whole span.
struct StudioTrackFrameEditor: View {
    let url: URL
    /// The clip's pixel size, which the prompts are in.
    let frameSize: CGSize
    let grid: StudioVideoFrameGrid
    @Binding var prompts: [StudioRegionPrompt]
    @Binding var initFrame: Int
    @Binding var endFrame: Int?
    var maxHeight: CGFloat = 480

    @State private var shownFrame: Int?
    @State private var isScrubbing = false
    @State private var loader: StudioVideoFrameLoader?
    @State private var image: NSImage?
    @State private var loadedFrame: Int?
    @State private var tool = StudioRegionTool.box
    @State private var selection: UUID?

    private var currentFrame: Int { grid.clamped(shownFrame ?? initFrame) }
    private var onSeedFrame: Bool { currentFrame == grid.clamped(initFrame) }
    /// The frame in view cannot end tracking before it starts.
    private var canEndHere: Bool { endFrame != currentFrame && currentFrame >= grid.clamped(initFrame) }

    /// Which decode is wanted: the frame in view of this clip, tolerant while the knob is moving
    /// and exact once it settles. The clip is part of the key so a replaced input decodes anew.
    private struct FrameRequest: Hashable {
        let url: URL
        let frame: Int
        let exact: Bool
    }

    private var request: FrameRequest {
        FrameRequest(url: url, frame: currentFrame, exact: !isScrubbing)
    }

    /// What the editor adds around the frame: the toolbar, the scrubber, the range line, the
    /// mark buttons, and the gaps between them. The canvas takes this out of the media height
    /// so the whole editor fits above the composer.
    static let rowsHeight: CGFloat = 38 + 28 + 24 + 36

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            StudioRegionToolbarRow(
                tool: $tool, prompts: $prompts, selection: $selection,
                isEnabled: onSeedFrame && image != nil, hint: hint
            )
            frameView
                .mereMediaFrame()
            scrubberRow
            Text(rangeDescription)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
            markRow
        }
        .task(id: request) { await loadFrame(request) }
        .onChange(of: initFrame) { _, seed in
            // Moving the seed from elsewhere (a Library row, the Command view) follows it.
            if shownFrame == nil || !onSeedFrame { shownFrame = grid.clamped(seed) }
        }
    }

    private var hint: String {
        if onSeedFrame { return StudioRegionTool.gestureHint }
        return "Prompts are drawn on frame \(grid.clamped(initFrame)); tracking covers the range below."
    }

    // MARK: Frame

    @ViewBuilder
    private var frameView: some View {
        if let image {
            Image(nsImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxHeight: maxHeight)
                .overlay {
                    GeometryReader { geometry in
                        StudioRegionPromptLayer(
                            prompts: $prompts,
                            imageSize: frameSize,
                            fitted: CGRect(origin: .zero, size: geometry.size),
                            tool: $tool,
                            selection: $selection,
                            isEnabled: onSeedFrame
                        )
                        .opacity(onSeedFrame ? 1 : 0.35)
                    }
                }
                .overlay(alignment: .topTrailing) {
                    frameBadge
                        .padding(10)
                }
                // Decoding a new frame keeps the previous one on screen, so scrubbing never blinks.
                .opacity(loadedFrame == currentFrame ? 1 : 0.85)
        } else {
            Rectangle()
                .fill(MereRunTheme.surfaceRaised)
                .aspectRatio(aspect, contentMode: .fit)
                .frame(maxHeight: maxHeight)
                .overlay { ProgressView().controlSize(.small) }
        }
    }

    private var frameBadge: some View {
        HStack(spacing: 6) {
            if onSeedFrame {
                Image(systemName: "flag.fill")
                    .font(.system(size: 10, weight: .semibold))
            }
            Text(onSeedFrame ? "Prompt frame \(currentFrame)" : "Frame \(currentFrame)")
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
        }
        .foregroundStyle(onSeedFrame ? MereRunTheme.onAccent : Color.white)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background {
            Capsule().fill(onSeedFrame ? MereRunTheme.accent : Color.black.opacity(0.55))
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    private var aspect: CGFloat {
        guard frameSize.width > 0, frameSize.height > 0 else { return 16.0 / 9 }
        return frameSize.width / frameSize.height
    }

    // MARK: Scrubber

    private var scrubberRow: some View {
        HStack(spacing: 10) {
            StudioFrameScrubber(
                grid: grid,
                frame: Binding(get: { currentFrame }, set: { shownFrame = grid.clamped($0) }),
                isScrubbing: $isScrubbing,
                promptFrame: grid.clamped(initFrame),
                endFrame: endFrame.map(grid.clamped)
            )
            Text("\(currentFrame) / \(grid.lastFrame) · \(grid.timeDescription(ofFrame: currentFrame))")
                .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
                .fixedSize()
        }
    }

    /// The buttons that make the frame in view the prompt frame or the end of tracking, and the
    /// chip that says when it already is. Short titles so the row fits beside the frame at the
    /// default window; the full sentence is each control's help. When even the titles do not
    /// fit, the buttons keep their icons alone.
    private var markRow: some View {
        ViewThatFits(in: .horizontal) {
            markControls(iconsOnly: false)
            markControls(iconsOnly: true)
        }
    }

    @ViewBuilder
    private func markControls(iconsOnly: Bool) -> some View {
        HStack(spacing: 6) {
            if onSeedFrame {
                markState("Prompt frame", systemImage: "flag.fill", iconsOnly: iconsOnly)
                    .help("The prompts are drawn on this frame; the tracker segments them here")
            } else {
                markButton("Prompts here", systemImage: "flag", iconsOnly: iconsOnly) {
                    initFrame = currentFrame
                    if let endFrame, endFrame < currentFrame { self.endFrame = nil }
                }
                .help("Draw the prompts on this frame instead: the tracker segments them here, then follows them through the whole range")
            }

            if endFrame == currentFrame {
                markState("Ends here", systemImage: "flag.checkered", iconsOnly: iconsOnly)
                    .help("Tracking stops after this frame")
            } else {
                markButton("End here", systemImage: "flag.checkered", iconsOnly: iconsOnly) {
                    endFrame = currentFrame
                }
                .disabled(!canEndHere)
                .help(
                    canEndHere
                        ? "Stop tracking after this frame instead of at the end of the clip"
                        : "Tracking cannot end before the prompt frame; move the prompts here first"
                )
            }

            if endFrame != nil {
                markButton("Track to end", systemImage: "xmark.circle", iconsOnly: iconsOnly) {
                    endFrame = nil
                }
                .help("Clear the end frame and track to the end of the clip")
            }
            Spacer(minLength: 0)
        }
    }

    private func markButton(_ title: String, systemImage: String, iconsOnly: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            markLabel(title, systemImage: systemImage, iconsOnly: iconsOnly)
        }
        .buttonStyle(.mereSecondary)
        .fixedSize()
        .accessibilityLabel(title)
    }

    /// What the frame in view already is, in the place its button would be.
    private func markState(_ text: String, systemImage: String, iconsOnly: Bool) -> some View {
        markLabel(text, systemImage: systemImage, iconsOnly: iconsOnly)
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(MereRunTheme.accent)
            .padding(.horizontal, 10)
            .frame(height: 28)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                    .fill(MereRunTheme.accentSoft)
            }
            .fixedSize()
            .accessibilityLabel(text)
            .accessibilityAddTraits(.isStaticText)
    }

    @ViewBuilder
    private func markLabel(_ title: String, systemImage: String, iconsOnly: Bool) -> some View {
        if iconsOnly {
            Image(systemName: systemImage)
        } else {
            Label(title, systemImage: systemImage)
                .labelStyle(.titleAndIcon)
        }
    }

    private var rangeDescription: String {
        grid.trackRangeDescription(promptFrame: initFrame, endFrame: endFrame)
    }

    // MARK: Loading

    private func loadFrame(_ request: FrameRequest) async {
        // One loader per clip, made the first time that clip is asked for; a replaced clip
        // drops the old picture rather than showing it under the new prompts.
        let loader: StudioVideoFrameLoader
        if let current = self.loader, current.url == request.url {
            loader = current
        } else {
            loader = StudioVideoFrameLoader(url: request.url)
            self.loader = loader
            image = nil
            loadedFrame = nil
        }
        // A frame already decoded exactly needs no tolerant re-decode.
        if !request.exact, loadedFrame == request.frame { return }
        guard let decoded = try? await loader.image(at: grid.time(ofFrame: request.frame), exact: request.exact),
              !Task.isCancelled else { return }
        image = decoded
        loadedFrame = request.frame
    }
}

/// A frame slider: the clip as a track, the tracked span in accent from frame 0 to the end frame
/// (or the clip's end) with a marker on the prompt frame, and a knob for the frame in view. Arrow
/// keys step a frame when it has focus. `isScrubbing` is true while the knob is being dragged.
struct StudioFrameScrubber: View {
    let grid: StudioVideoFrameGrid
    @Binding var frame: Int
    @Binding var isScrubbing: Bool
    let promptFrame: Int
    let endFrame: Int?

    @FocusState private var focused: Bool
    @State private var hovering = false
    /// True only while a drag is live; SwiftUI resets it when the gesture ends or is cancelled,
    /// so a cancelled drag can never leave the editor decoding tolerant frames.
    @GestureState private var dragging = false

    private enum Metrics {
        static let trackHeight: CGFloat = 6
        static let knobDiameter: CGFloat = 14
        static let height: CGFloat = 20
    }

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let usable = max(1, width - Metrics.knobDiameter)
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(MereRunTheme.surfaceRaised)
                    .frame(height: Metrics.trackHeight)
                Capsule()
                    .fill(MereRunTheme.accent.opacity(endFrame == nil ? 0.45 : 0.8))
                    .frame(width: spanWidth(usable: usable), height: Metrics.trackHeight)
                    .offset(x: Metrics.knobDiameter / 2)
                marker(at: promptFrame, usable: usable)
                if let endFrame { marker(at: endFrame, usable: usable) }
                Circle()
                    .fill(MereRunTheme.surface)
                    .overlay {
                        Circle().strokeBorder(focused || hovering ? MereRunTheme.accent : MereRunTheme.border, lineWidth: 1.5)
                    }
                    .frame(width: Metrics.knobDiameter, height: Metrics.knobDiameter)
                    .mereShadow(radius: 2, y: 1)
                    .offset(x: usable * fraction(of: frame))
            }
            .frame(height: geometry.size.height)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .updating($dragging) { _, state, _ in state = true }
                    .onChanged { value in
                        focused = true
                        let position = min(1, max(0, (value.location.x - Metrics.knobDiameter / 2) / usable))
                        frame = grid.clamped(Int((position * Double(grid.lastFrame)).rounded()))
                    }
            )
        }
        .frame(height: Metrics.height)
        .frame(maxWidth: .infinity)
        .focusable()
        .focused($focused)
        .focusEffectDisabled()
        .onHover { hovering = $0 }
        .onChange(of: dragging) { _, dragging in isScrubbing = dragging }
        .onMoveCommand { direction in
            switch direction {
            case .left: frame = grid.clamped(frame - 1)
            case .right: frame = grid.clamped(frame + 1)
            default: break
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Frame")
        .accessibilityValue("\(frame) of \(grid.lastFrame)")
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: frame = grid.clamped(frame + 1)
            case .decrement: frame = grid.clamped(frame - 1)
            @unknown default: break
            }
        }
    }

    private func fraction(of frame: Int) -> CGFloat {
        guard grid.lastFrame > 0 else { return 0 }
        return CGFloat(grid.clamped(frame)) / CGFloat(grid.lastFrame)
    }

    private func spanWidth(usable: CGFloat) -> CGFloat {
        usable * fraction(of: endFrame ?? grid.lastFrame)
    }

    private func marker(at frame: Int, usable: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: 1)
            .fill(MereRunTheme.accent)
            .frame(width: 2, height: Metrics.trackHeight + 6)
            .offset(x: Metrics.knobDiameter / 2 - 1 + usable * fraction(of: frame))
    }
}
