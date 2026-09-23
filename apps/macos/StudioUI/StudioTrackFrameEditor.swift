import AVFoundation
import AppKit
import StudioKit
import SwiftUI

// Picking frames on the clip instead of typing their numbers: Track's seed frame (where the
// prompts are drawn) and its optional end frame, on a scrubber over the decoded frame.

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

/// Track's input: the clip's frames on a scrubber, the prompts drawn on the seed frame, and the
/// buttons that make the shown frame the start or the end of tracking.
///
/// Prompts belong to the seed frame — that is the picture the tracker segments before following
/// each object — so they are only editable there. Scrubbing elsewhere shows them faded, and
/// "Start tracking here" moves the seed (prompts included) to the frame in view.
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

    /// Which decode is wanted: the frame in view, tolerant while the knob is moving and exact
    /// once it settles.
    private struct FrameRequest: Hashable {
        let frame: Int
        let exact: Bool
    }

    private var request: FrameRequest {
        FrameRequest(frame: currentFrame, exact: !isScrubbing)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            StudioRegionToolbarRow(
                tool: $tool, prompts: $prompts, selection: $selection,
                isEnabled: onSeedFrame && image != nil, hint: hint
            )
            frameView
                .mereMediaFrame()
            scrubberRow
            markRow
        }
        .task(id: url) {
            loader = StudioVideoFrameLoader(url: url)
            image = nil
            loadedFrame = nil
        }
        .task(id: request) { await loadFrame(request) }
        .onChange(of: initFrame) { _, seed in
            // Moving the seed from elsewhere (a Library row, the Command view) follows it.
            if shownFrame == nil || !onSeedFrame { shownFrame = grid.clamped(seed) }
        }
    }

    private var hint: String {
        if onSeedFrame { return "Drag for a box, click for a point, Option-click for a negative point." }
        return "Prompts are drawn on frame \(grid.clamped(initFrame)), where tracking starts."
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
            Text(onSeedFrame ? "Start frame \(currentFrame)" : "Frame \(currentFrame)")
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
                startFrame: grid.clamped(initFrame),
                endFrame: endFrame.map(grid.clamped)
            )
            Text("\(currentFrame) / \(grid.lastFrame) · \(grid.timeDescription(ofFrame: currentFrame))")
                .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
                .fixedSize()
        }
    }

    private var markRow: some View {
        HStack(spacing: 6) {
            if onSeedFrame {
                markState("Tracking starts here", systemImage: "flag.fill")
            } else {
                Button {
                    initFrame = currentFrame
                    if let endFrame, endFrame < currentFrame { self.endFrame = nil }
                } label: {
                    Label("Start tracking here", systemImage: "flag")
                }
                .buttonStyle(.mereSecondary)
                .help("Seed the tracker on this frame; prompts are drawn on it")
            }

            if endFrame == currentFrame {
                markState("Tracking ends here", systemImage: "flag.checkered")
            } else {
                Button {
                    endFrame = currentFrame
                } label: {
                    Label("End tracking here", systemImage: "flag.checkered")
                }
                .buttonStyle(.mereSecondary)
                .disabled(!canEndHere)
                .help(
                    canEndHere
                        ? "Stop tracking after this frame instead of at the end of the clip"
                        : "Tracking cannot end before it starts; move the start frame here first"
                )
            }

            if endFrame != nil {
                Button {
                    endFrame = nil
                } label: {
                    Label("Track to the end", systemImage: "xmark.circle")
                }
                .buttonStyle(.mereSecondary)
                .help("Clear the end frame")
            }
            Spacer(minLength: 8)
            Text(rangeDescription)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
        }
    }

    /// What the frame in view already is, in the place its button would be.
    private func markState(_ text: String, systemImage: String) -> some View {
        Label(text, systemImage: systemImage)
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(MereRunTheme.accent)
            .padding(.horizontal, 10)
            .frame(height: 28)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                    .fill(MereRunTheme.accentSoft)
            }
            .accessibilityAddTraits(.isStaticText)
    }

    private var rangeDescription: String {
        let start = grid.clamped(initFrame)
        if let endFrame {
            return "Frames \(start)–\(grid.clamped(endFrame)) of \(grid.frameCount)"
        }
        return start == 0 ? "The whole clip, \(grid.frameCount) frames" : "Frame \(start) to the end of the clip"
    }

    // MARK: Loading

    private func loadFrame(_ request: FrameRequest) async {
        guard let loader else { return }
        // A frame already decoded exactly needs no tolerant re-decode.
        if !request.exact, loadedFrame == request.frame { return }
        guard let decoded = try? await loader.image(at: grid.time(ofFrame: request.frame), exact: request.exact),
              !Task.isCancelled else { return }
        image = decoded
        loadedFrame = request.frame
    }
}

/// A frame slider: the clip as a track, the tracked span in accent from the start frame to the
/// end frame (or the clip's end), and a knob for the frame in view. Arrow keys step a frame when
/// it has focus. `isScrubbing` is true while the knob is being dragged.
struct StudioFrameScrubber: View {
    let grid: StudioVideoFrameGrid
    @Binding var frame: Int
    @Binding var isScrubbing: Bool
    let startFrame: Int
    let endFrame: Int?

    @FocusState private var focused: Bool
    @State private var hovering = false

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
                    .offset(x: Metrics.knobDiameter / 2 + usable * fraction(of: startFrame))
                marker(at: startFrame, usable: usable)
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
                    .onChanged { value in
                        focused = true
                        isScrubbing = true
                        let position = min(1, max(0, (value.location.x - Metrics.knobDiameter / 2) / usable))
                        frame = grid.clamped(Int((position * Double(grid.lastFrame)).rounded()))
                    }
                    .onEnded { _ in isScrubbing = false }
            )
        }
        .frame(height: Metrics.height)
        .frame(maxWidth: .infinity)
        .focusable()
        .focused($focused)
        .focusEffectDisabled()
        .onHover { hovering = $0 }
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
        let end = endFrame ?? grid.lastFrame
        return max(0, usable * (fraction(of: end) - fraction(of: startFrame)))
    }

    private func marker(at frame: Int, usable: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: 1)
            .fill(MereRunTheme.accent)
            .frame(width: 2, height: Metrics.trackHeight + 6)
            .offset(x: Metrics.knobDiameter / 2 - 1 + usable * fraction(of: frame))
    }
}
