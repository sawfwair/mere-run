import AVFoundation
import AppKit
import ImageIO
import SwiftUI

/// What the Analyze canvas asks the workspace to do beyond the shared feed actions.
struct StudioAnalyzeActions {
    /// Pick a different input; writes the composer's well so the two never disagree.
    let replaceInput: () -> Void
    /// Continue in a sibling task, carrying this input and prompt.
    let openTask: (StudioTask) -> Void
    /// Write part of the result somewhere the user picks.
    let save: (StudioAnalyzeSaveKind) -> Void
}

/// The Analyze archetype: one input on the left, what the model found on the right.
///
/// Input-first tasks are not a feed — you point them at a file, and the answer belongs beside the
/// thing it is about. So the canvas shows the current input large with the result drawn over it,
/// a result column that names what was found and what to do next, and the run in flight (or the
/// failure) as the same cards the Generate feed uses. Earlier runs stay in the Library column.
struct StudioAnalyzeCanvas: View {
    let archetype: StudioAnalyzeArchetype
    let mode: StudioMode
    /// Every row of this mode, oldest first, as the feed builds them.
    let cards: [StudioFeedCard]
    /// The Library row the user picked, when they picked one.
    let selectedID: UUID?
    /// The composer's well: the file the next run would read.
    let inputPath: String
    let readiness: ModelReadinessState
    let pullJob: Job?
    let actions: StudioFeedActions
    let analyze: StudioAnalyzeActions

    @State private var chosenView: StudioAnalyzeResultView?
    @State private var loaded: StudioAnalyzeLoadedResult?
    @State private var inputSize: CGSize?
    @State private var inputDuration: TimeInterval?

    private enum Metrics {
        static let contentWidth: CGFloat = 940
        static let resultColumnWidth: CGFloat = 360
        static let columnSpacing: CGFloat = 20
        static let mediaMaxHeight: CGFloat = 520
        static let insets = EdgeInsets(top: 22, leading: 24, bottom: 6, trailing: 24)
    }

    // MARK: Derived state

    private var inputURL: URL? {
        let trimmed = inputPath.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : URL(fileURLWithPath: trimmed)
    }

    /// The run whose result is on screen: the picked Library row when it finished, else the most
    /// recent finished run of this task.
    private var resultCard: StudioFeedCard? {
        if let selectedID, let picked = cards.first(where: { $0.id == selectedID }), picked.kind == .generation {
            return picked
        }
        return cards.last { $0.kind == .generation }
    }

    /// The cards that sit above the result: everything still in flight, plus the newest failure.
    private var pendingCards: [StudioFeedCard] {
        var pending = cards.filter { $0.kind == .running || $0.kind == .queued }
        if let last = cards.last, last.kind == .failed { pending.append(last) }
        return pending
    }

    private var showsReadinessCard: Bool {
        readiness.blocksRun || pullJob != nil
    }

    /// An input-first task shows its input the moment there is one: attaching a picture and then
    /// still being told to "Choose image…" would be absurd, so the serif empty state is only for
    /// an empty well with nothing to report.
    private var hasBody: Bool {
        inputURL != nil || resultCard != nil || !pendingCards.isEmpty || showsReadinessCard
    }

    private var view: StudioAnalyzeResultView {
        guard let chosenView, archetype.views.contains(chosenView) else { return archetype.defaultView }
        return chosenView
    }

    private var document: StudioAnalyzeDocument? {
        guard let loaded, loaded.itemID == resultCard?.id else { return nil }
        return loaded.document
    }

    private var detections: [StudioAnalyzeDetection] {
        guard let document else { return [] }
        let size = inputSize ?? document.reportedInputSize ?? CGSize(width: 1, height: 1)
        return document.detections(imageSize: size)
    }

    /// Whether the result on screen is about the input on screen. Replacing the input must not
    /// leave the previous run's boxes drawn over a different picture, so the overlays wait for
    /// the next run while the panel keeps listing what the last one found.
    private var resultDescribesInput: Bool {
        guard let itemInput = resultCard?.item.inputURL else { return true }
        guard let inputURL else { return false }
        return itemInput.standardizedFileURL == inputURL.standardizedFileURL
    }

    private var overlayDetections: [StudioAnalyzeDetection] {
        resultDescribesInput ? detections : []
    }

    var body: some View {
        Group {
            if hasBody {
                content
            } else {
                StudioEmptyState(mode: mode, onUseExample: actions.useExample, onAttach: actions.attach)
                    .padding(MereRunTheme.Spacing.xxxl)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: resultCard?.id) { await loadDocument() }
        .task(id: inputPath) { await measureInput() }
    }

    private var content: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                inputStrip
                    .padding(.bottom, 10)
                HStack(alignment: .top, spacing: Metrics.columnSpacing) {
                    inputColumn
                        .frame(maxWidth: .infinity, alignment: .leading)
                    resultColumn
                        .frame(width: Metrics.resultColumnWidth)
                }
            }
            .padding(Metrics.insets)
            .frame(maxWidth: Metrics.contentWidth)
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: - Input strip

    private var inputStrip: some View {
        HStack(spacing: 10) {
            Text("Input")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
            StudioAnalyzeChip(text: inputDescription)
                .help(inputURL?.path ?? "No input attached")
            Button(inputURL == nil ? "Choose…" : "Replace", action: analyze.replaceInput)
                .buttonStyle(.mereSecondary)
                .help("Pick a different \(archetype.inputKind.noun)")
            Spacer(minLength: 8)
            // Nothing has been found yet, so there is nothing to switch between.
            if archetype.views.count > 1, resultCard != nil {
                MereSegmentedControl(
                    archetype.views,
                    selection: Binding(get: { view }, set: { chosenView = $0 }),
                    accessibilityLabel: "Result view"
                ) { $0.title }
            }
        }
        .accessibilityElement(children: .contain)
    }

    private var inputDescription: String {
        guard let inputURL else { return "No input" }
        var parts = [inputURL.lastPathComponent]
        if let inputSize {
            parts.append("\(Int(inputSize.width))×\(Int(inputSize.height))")
        } else if let inputDuration {
            parts.append(StudioTimeFormat.string(inputDuration))
        }
        return parts.joined(separator: " · ")
    }

    // MARK: - Input column

    @ViewBuilder
    private var inputColumn: some View {
        switch view {
        case .json:
            StudioAnalyzeDocumentView(url: loaded?.url, text: loaded?.raw)
                .frame(height: Metrics.mediaMaxHeight)
                .mereMediaFrame()
        case .transcript, .timeline, .text, .score, .stems, .vectors:
            mediaView
        default:
            mediaView
        }
    }

    @ViewBuilder
    private var mediaView: some View {
        switch archetype.inputKind {
        case .image:
            imageView
        case .video:
            videoView
        case .audio:
            audioView
        case .file:
            fileView
        }
    }

    @ViewBuilder
    private var imageView: some View {
        if let inputURL {
            StudioAnalyzeImageView(
                url: inputURL,
                maxHeight: Metrics.mediaMaxHeight,
                detections: view == .masks ? [] : overlayDetections,
                masks: view == .masks ? overlayDetections : [],
                imageSize: inputSize
            )
            .mereMediaFrame()
        } else {
            missingInput
        }
    }

    @ViewBuilder
    private var videoView: some View {
        if let inputURL {
            VStack(spacing: 8) {
                StudioVideoPlayerView(url: playableVideoURL ?? inputURL)
                    .aspectRatio(videoAspect, contentMode: .fit)
                    .frame(maxHeight: Metrics.mediaMaxHeight - 26)
                    .mereMediaFrame()
                if case .tracking(let tracking) = document {
                    StudioAnalyzeTrackScrubber(document: tracking)
                }
            }
        } else {
            missingInput
        }
    }

    /// Track writes an annotated clip; that is the one worth playing when it exists.
    private var playableVideoURL: URL? {
        guard let outputURL = resultCard?.item.outputURL,
              StudioOutputFileKind.classify(outputURL) == .video else { return nil }
        return outputURL
    }

    private var videoAspect: CGFloat {
        guard let size = inputSize ?? document?.reportedInputSize, size.height > 0 else { return 16.0 / 9 }
        return size.width / size.height
    }

    @ViewBuilder
    private var audioView: some View {
        if let inputURL {
            StudioAudioPlayerView(url: playableAudioURL ?? inputURL)
                .frame(height: 220)
                .mereMediaFrame()
        } else {
            missingInput
        }
    }

    /// Enhance and Separate write new audio; the player should offer the result, not the source.
    private var playableAudioURL: URL? {
        guard archetype.views.contains(.audio) || archetype.views.contains(.stems),
              let outputURL = resultCard?.item.outputURL,
              StudioOutputFileKind.classify(outputURL) == .audio else { return nil }
        return outputURL
    }

    @ViewBuilder
    private var fileView: some View {
        if let inputURL {
            VStack(spacing: MereRunTheme.Spacing.sm) {
                Image(systemName: "doc.text")
                    .font(.system(size: 34, weight: .medium))
                    .foregroundStyle(MereRunTheme.accent)
                Text(inputURL.lastPathComponent)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 220)
            .mereMediaFrame()
        } else {
            missingInput
        }
    }

    private var missingInput: some View {
        VStack(spacing: MereRunTheme.Spacing.sm) {
            Image(systemName: "paperclip")
                .font(.system(size: 26, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            Text("No \(archetype.inputKind.noun) attached.")
                .font(.system(size: 13))
                .foregroundStyle(MereRunTheme.textSecondary)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 220)
        .mereMediaFrame()
    }

    // MARK: - Result column

    private var resultColumn: some View {
        VStack(alignment: .leading, spacing: 14) {
            ForEach(pendingCards) { card in
                pendingCard(card)
            }
            if showsReadinessCard {
                StudioReadinessCard(
                    readiness: readiness,
                    pullJob: pullJob,
                    onPullModel: actions.pullModel,
                    onShowDetails: actions.showDetails,
                    onCancelPull: { actions.cancel($0) }
                )
            }
            if let resultCard {
                resultPanel(resultCard)
                StudioAnalyzePromptPanel(item: resultCard.item)
            }
        }
    }

    @ViewBuilder
    private func pendingCard(_ card: StudioFeedCard) -> some View {
        switch card.kind {
        case .running:
            if let job = card.job {
                StudioRunningCard(item: card.item, job: job, isHighlighted: false) { actions.cancel(job) }
            } else {
                StudioFailureCard(item: card.item, job: nil, isHighlighted: false, actions: actions)
            }
        case .queued:
            StudioQueuedRow(
                item: card.item,
                position: StudioFeedCardBuilder.queuePosition(of: card, in: cards) ?? 0,
                isHighlighted: false
            ) {
                actions.remove(card)
            }
        case .failed:
            StudioFailureCard(item: card.item, job: card.job, isHighlighted: false, actions: actions)
        case .generation:
            EmptyView()
        }
    }

    @ViewBuilder
    private func resultPanel(_ card: StudioFeedCard) -> some View {
        StudioAnalyzeResultPanel(
            item: card.item,
            document: document,
            detections: detections,
            speechSegments: document?.speechSegments ?? [],
            outputText: card.item.outputText,
            view: view,
            nextActions: archetype.nextActions,
            onOpenTask: analyze.openTask,
            onSave: analyze.save
        )
    }

    // MARK: - Loading

    private func loadDocument() async {
        guard let item = resultCard?.item else {
            loaded = nil
            return
        }
        let itemID = item.id
        let url = StudioAnalyzeDocumentSource.url(for: item)
        let fallbackText = item.outputText
        let result = await Task.detached(priority: .userInitiated) {
            StudioAnalyzeLoadedResult.load(itemID: itemID, url: url, fallbackText: fallbackText)
        }.value
        guard !Task.isCancelled else { return }
        loaded = result
    }

    private func measureInput() async {
        guard let inputURL else {
            inputSize = nil
            inputDuration = nil
            return
        }
        let kind = archetype.inputKind
        let measured = await Task.detached(priority: .userInitiated) {
            StudioAnalyzeMediaInfo.measure(inputURL, kind: kind)
        }.value
        guard !Task.isCancelled else { return }
        inputSize = measured.size
        inputDuration = measured.duration
    }
}

extension StudioAnalyzeInputKind {
    /// "image", "video", "audio file", "file" — the word the strip and empty state use.
    var noun: String {
        switch self {
        case .image: return "image"
        case .video: return "video"
        case .audio: return "audio file"
        case .file: return "file"
        }
    }
}

// MARK: - Loading the result document

/// Where a run's result document lives.
enum StudioAnalyzeDocumentSource {
    private static let documentExtensions = ["json", "txt", "vtt", "srt", "jsonl"]
    /// The receipt roles that name a structured result, most specific first.
    private static let documentRoles = [StudioArtifactRole.detections, StudioArtifactRole.tracking]

    /// The artifact holding the run's structured result: the `--json-output` sidecar for the
    /// vision tasks, the transcript the speech tasks write. A run whose receipt named its
    /// sidecars says which file that is; extension order is the fallback for the rest.
    static func url(for item: StudioLibraryItem) -> URL? {
        let artifacts = item.allArtifactURLs
        for role in documentRoles {
            if let match = artifacts.first(where: { item.artifactRole(for: $0) == role }) {
                return match
            }
        }
        for pathExtension in documentExtensions {
            if let match = artifacts.first(where: { $0.pathExtension.lowercased() == pathExtension }) {
                return match
            }
        }
        return nil
    }
}

/// A decoded result document, tied to the run it came from so a stale one is never drawn.
struct StudioAnalyzeLoadedResult: Equatable {
    let itemID: UUID
    let url: URL?
    /// The document as written, for the JSON view.
    let raw: String?
    let document: StudioAnalyzeDocument?

    static func load(itemID: UUID, url: URL?, fallbackText: String?) -> StudioAnalyzeLoadedResult {
        if let url, let data = try? Data(contentsOf: url), !data.isEmpty {
            return StudioAnalyzeLoadedResult(
                itemID: itemID,
                url: url,
                raw: String(data: data, encoding: .utf8),
                document: StudioAnalyzeDocument.decode(data)
            )
        }
        guard let fallbackText, !fallbackText.isBlank, let data = fallbackText.data(using: .utf8) else {
            return StudioAnalyzeLoadedResult(itemID: itemID, url: url, raw: nil, document: nil)
        }
        return StudioAnalyzeLoadedResult(
            itemID: itemID,
            url: nil,
            raw: fallbackText,
            document: StudioAnalyzeDocument.decode(data)
        )
    }
}

/// What the input strip can say about a file without decoding all of it.
enum StudioAnalyzeMediaInfo {
    struct Measurement: Equatable {
        var size: CGSize?
        var duration: TimeInterval?
    }

    static func measure(_ url: URL, kind: StudioAnalyzeInputKind) -> Measurement {
        switch kind {
        case .image:
            return Measurement(size: pixelSize(of: url), duration: nil)
        case .video:
            let asset = AVURLAsset(url: url)
            let track = asset.tracks(withMediaType: .video).first
            let size = track.map { $0.naturalSize.applying($0.preferredTransform) }
                .map { CGSize(width: abs($0.width), height: abs($0.height)) }
            return Measurement(size: size, duration: CMTimeGetSeconds(asset.duration))
        case .audio:
            let asset = AVURLAsset(url: url)
            let duration = CMTimeGetSeconds(asset.duration)
            return Measurement(size: nil, duration: duration.isFinite ? duration : nil)
        case .file:
            return Measurement()
        }
    }

    /// The image's own pixel dimensions, read from its metadata without decoding the pixels.
    static func pixelSize(of url: URL) -> CGSize? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0 else {
            return nil
        }
        return CGSize(width: width, height: height)
    }
}
