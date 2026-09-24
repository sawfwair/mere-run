import AVFoundation
import AppKit
import ImageIO
import StudioKit
import SwiftUI

/// What the Analyze canvas asks the workspace to do beyond the shared feed actions.
struct StudioAnalyzeActions {
    /// Pick a different input; writes the composer's well so the two never disagree.
    let replaceInput: () -> Void
    /// Continue in a sibling task, carrying this input, prompt, and what the result found on it.
    let openTask: (StudioTask, [StudioAnalyzeDetection]) -> Void
    /// Write part of the result somewhere the user picks.
    let save: (StudioAnalyzeSaveKind) -> Void
}

/// The draft fields Segment and Track edit on the picture itself: the boxes and points drawn on
/// the input, and Track's seed and end frames. Nil for the tasks that take no drawn prompts.
struct StudioAnalyzePromptEditing {
    var regionPrompts: Binding<[StudioRegionPrompt]>
    var initFrame: Binding<Int>
    var endFrame: Binding<Int?>
}

/// The Analyze archetype: one input on the left, what the model found on the right.
///
/// Input-first tasks are not a feed — you point them at a file, and the answer belongs beside the
/// thing it is about. So the canvas shows the current input large with the result drawn over it,
/// a result column that names what was found and what to do next, and the run in flight (or the
/// failure) as the same cards the Generate feed uses. Earlier runs stay in the Library column.
struct StudioAnalyzeCanvas: View {
    let archetype: StudioAnalyzeArchetype
    /// The task's glyph, empty-state words, and examples: the mode's for a prompt task, the
    /// task's own on the shared task workspace.
    let presentation: StudioTaskPresentation
    /// Every row of this task, oldest first, as the feed builds them.
    let cards: [StudioFeedCard]
    /// The Library row the user picked, when they picked one.
    let selectedID: UUID?
    /// The composer's well: the file the next run would read.
    let inputPath: String
    let readiness: ModelReadinessState
    let pullJob: Job?
    let actions: StudioFeedActions
    let readinessActions: StudioReadinessActions
    let analyze: StudioAnalyzeActions
    var editing: StudioAnalyzePromptEditing?
    /// The variant the task draft runs, when the task has several; picks the input kind and
    /// views from the archetype's `variants`.
    var templateID: CommandTemplateID?
    /// The typed input of a `.text` task (Embeddings, Anonymize): the input column is this
    /// editor, bound to the command's positional.
    var textInput: Binding<String>?

    @State private var chosenView: StudioAnalyzeResultView?
    @State private var loaded: StudioAnalyzeLoadedResult?
    /// The input's stored pixel size — the space the CLI's coordinates are in.
    @State private var inputSize: CGSize?
    @State private var inputDuration: TimeInterval?
    @State private var inputFrameRate: Double?
    /// How the input photo is turned to show upright; results and prompts map through it.
    @State private var inputOrientation = StudioImageOrientation.up
    @State private var regionTool = StudioRegionTool.box
    @State private var regionSelection: UUID?
    /// Track shows its finished clip once there is one; this brings the seed-frame editor back.
    @State private var editsTrackPrompts = false
    /// The canvas's own height: what the column has above the composer, which the input's
    /// picture fits into. Starts unbounded so the first layout uses the cap, then follows the
    /// window.
    @State private var availableHeight: CGFloat = .infinity

    private enum Metrics {
        static let contentWidth: CGFloat = 940
        static let resultColumnWidth: CGFloat = 360
        static let columnSpacing: CGFloat = 20
        static let insets = EdgeInsets(top: 22, leading: 24, bottom: 6, trailing: 24)
        /// The input strip's row and its gap to the columns.
        static let inputStripHeight: CGFloat = 28 + 10
        /// The drawing toolbar's row and its gap to the picture.
        static let toolbarRowHeight: CGFloat = 30 + 8
    }

    /// How tall the input may be: the column above the composer less the rows around the
    /// picture (`StudioAnalyzeMediaLayout`). Track's frame editor takes its own rows out of this.
    private var mediaHeight: CGFloat {
        var chrome = Metrics.insets.top + Metrics.insets.bottom + Metrics.inputStripHeight
        if editing != nil, inputKind == .image { chrome += Metrics.toolbarRowHeight }
        return StudioAnalyzeMediaLayout.mediaHeight(availableHeight: availableHeight, chromeHeight: chrome)
    }

    // MARK: Derived state

    private var inputKind: StudioAnalyzeInputKind {
        archetype.inputKind(for: templateID)
    }

    private var views: [StudioAnalyzeResultView] {
        archetype.views(for: templateID)
    }

    private var inputURL: URL? {
        guard inputKind != .text, inputKind != .none else { return nil }
        let trimmed = inputPath.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : URL(fileURLWithPath: trimmed)
    }

    /// A typed input counts as attached once there is text in it.
    private var hasTypedInput: Bool {
        inputKind == .text && !(textInput?.wrappedValue.isBlank ?? true)
    }

    /// The run whose result is on screen: the picked Library row when it finished, else the most
    /// recent finished run of this task. On a task with variants, only a run of the chosen
    /// variant: Datasets ▸ Validate does not show the last Discover's candidates.
    private var resultCard: StudioFeedCard? {
        let finished = cards.filter { $0.kind == .generation && (templateID == nil || $0.item.templateID == templateID) }
        if let selectedID, let picked = finished.first(where: { $0.id == selectedID }) {
            return picked
        }
        return finished.last
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
    /// an empty well with nothing to report. A typed input is entered on the canvas itself, so
    /// its editor is always up.
    private var hasBody: Bool {
        inputURL != nil || inputKind == .text || inputKind == .none || resultCard != nil || !pendingCards.isEmpty
            || showsReadinessCard
    }

    private var view: StudioAnalyzeResultView {
        guard let chosenView, views.contains(chosenView) else { return views.first ?? .json }
        return chosenView
    }

    private var document: StudioAnalyzeDocument? {
        guard resultDescribesInput, let loaded, loaded.itemID == resultCard?.id else { return nil }
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
        guard let item = resultCard?.item else { return false }
        // A typed or absent input has no file identity to compare; the result stands as is.
        guard inputKind != .text, inputKind != .none else { return true }
        return StudioInputIdentity.matches(item: item, input: inputURL)
    }

    private var overlayDetections: [StudioAnalyzeDetection] {
        resultDescribesInput ? detections : []
    }

    var body: some View {
        Group {
            if hasBody {
                content
            } else {
                StudioEmptyState(presentation: presentation, onUseExample: actions.useExample, onAttach: actions.attach)
                    .padding(MereRunTheme.Spacing.xxxl)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: resultCard?.id) { await loadDocument() }
        .task(id: inputURL?.path ?? "") { await measureInput() }
        .onChange(of: resultCard?.id) { _, _ in editsTrackPrompts = false }
    }

    private var content: some View {
        GeometryReader { geometry in
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
            .onChange(of: geometry.size.height, initial: true) { _, height in
                availableHeight = height
            }
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
            if inputKind != .text, inputKind != .none {
                Button(inputURL == nil ? "Choose…" : "Replace", action: analyze.replaceInput)
                    .buttonStyle(.mereSecondary)
                    .help("Pick a different \(inputKind.noun)")
            }
            Spacer(minLength: 8)
            // Nothing has been found yet, so there is nothing to switch between.
            if views.count > 1, resultCard != nil {
                MereSegmentedControl(
                    views,
                    selection: Binding(get: { view }, set: { chosenView = $0 }),
                    accessibilityLabel: "Result view"
                ) { $0.title }
            }
        }
        .accessibilityElement(children: .contain)
    }

    private var inputDescription: String {
        switch inputKind {
        case .text:
            let lines = (textInput?.wrappedValue ?? "").components(separatedBy: .newlines).filter { !$0.isBlank }
            return lines.isEmpty ? "No text yet" : (lines.count == 1 ? "1 line" : "\(lines.count) lines")
        case .none:
            return "No input needed"
        case .image, .video, .audio, .file, .directory:
            break
        }
        guard let inputURL else { return "No input" }
        var parts = [inputURL.lastPathComponent]
        if let inputSize {
            // The size as the picture is shown, which for a portrait phone photo is the
            // stored size turned on its side.
            let shown = inputOrientation.displaySize(ofStored: inputSize)
            parts.append("\(Int(shown.width))×\(Int(shown.height))")
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
                .frame(height: mediaHeight)
                .mereMediaFrame()
        default:
            // A view about the result rather than the input (landmarks, a flow field, a depth
            // map, a scene) takes the column from a registered renderer.
            if let rendering = StudioResultRenderers.canvasRendering(
                for: view, document: document, item: resultDescribesInput ? resultCard?.item : nil, inputURL: inputURL
            ) {
                StudioResultCanvasView(rendering: rendering, maxHeight: mediaHeight)
            } else {
                mediaView
            }
        }
    }

    @ViewBuilder
    private var mediaView: some View {
        switch inputKind {
        case .image:
            imageView
        case .video:
            videoView
        case .audio:
            audioView
        case .file, .directory:
            fileView
        case .text:
            textView
        case .none:
            // Nothing to show on the left; the result document takes the column.
            StudioAnalyzeDocumentView(url: loaded?.url, text: loaded?.raw)
                .frame(height: mediaHeight)
                .mereMediaFrame()
        }
    }

    /// The typed input, edited in place: one text per line for Embeddings, the passage to
    /// protect for Anonymize.
    @ViewBuilder
    private var textView: some View {
        if let textInput {
            TextEditor(text: textInput)
                .font(.system(size: 13))
                .foregroundStyle(MereRunTheme.textPrimary)
                .scrollContentBackground(.hidden)
                .padding(MereRunTheme.Spacing.sm)
                .frame(height: min(mediaHeight, 320))
                .background(MereRunTheme.surface)
                .overlay(alignment: .topLeading) {
                    // The task's placeholder, where the editor has none of its own.
                    if !hasTypedInput, !presentation.promptPlaceholder.isEmpty {
                        Text(presentation.promptPlaceholder)
                            .font(.system(size: 13))
                            .foregroundStyle(MereRunTheme.textMuted)
                            .padding(.horizontal, MereRunTheme.Spacing.sm + 5)
                            .padding(.vertical, MereRunTheme.Spacing.sm + 1)
                            .allowsHitTesting(false)
                            .accessibilityHidden(true)
                    }
                }
                .mereMediaFrame()
                .accessibilityLabel(presentation.promptPlaceholder.isEmpty ? "Input text" : presentation.promptPlaceholder)
        } else {
            missingInput
        }
    }

    @ViewBuilder
    private var imageView: some View {
        if let inputURL {
            VStack(alignment: .leading, spacing: 8) {
                if let editing {
                    StudioRegionToolbarRow(
                        tool: $regionTool, prompts: editing.regionPrompts, selection: $regionSelection
                    )
                }
                StudioAnalyzeImageView(
                    url: inputURL,
                    maxHeight: mediaHeight,
                    detections: view == .masks ? [] : overlayDetections,
                    masks: view == .masks ? overlayDetections : [],
                    imageSize: inputSize,
                    orientation: inputOrientation,
                    editing: editing.map {
                        StudioAnalyzeImageEditing(prompts: $0.regionPrompts, tool: $regionTool, selection: $regionSelection)
                    }
                )
                .mereMediaFrame()
            }
        } else {
            missingInput
        }
    }

    @ViewBuilder
    private var videoView: some View {
        if let inputURL {
            if let editing, !resultDescribesInput || editsTrackPrompts {
                seedFrameEditor(url: inputURL, editing: editing)
            } else {
                VStack(spacing: 8) {
                    StudioVideoPlayerView(url: playableVideoURL ?? inputURL)
                        .aspectRatio(videoAspect, contentMode: .fit)
                        .frame(maxHeight: mediaHeight - 26)
                        .mereMediaFrame()
                    if case .tracking(let tracking) = document {
                        StudioAnalyzeTrackScrubber(document: tracking)
                    }
                    if editing != nil {
                        HStack {
                            Button {
                                editsTrackPrompts = true
                            } label: {
                                Label("Adjust prompts and frames", systemImage: "rectangle.dashed")
                            }
                            .buttonStyle(.mereSecondary)
                            .help("Draw on the start frame and pick where tracking starts and ends")
                            Spacer()
                        }
                    }
                }
            }
        } else {
            missingInput
        }
    }

    /// Track's input as frames to draw on and pick from. The clip's size and rate arrive from
    /// `measureInput`; until then there is nothing to map a drawing onto.
    @ViewBuilder
    private func seedFrameEditor(url: URL, editing: StudioAnalyzePromptEditing) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if resultDescribesInput {
                HStack {
                    Button {
                        editsTrackPrompts = false
                    } label: {
                        Label("Show result", systemImage: "play.rectangle")
                    }
                    .buttonStyle(.mereSecondary)
                    Spacer()
                }
            }
            if let frameSize = inputSize ?? document?.reportedInputSize {
                StudioTrackFrameEditor(
                    url: url,
                    frameSize: frameSize,
                    grid: StudioVideoFrameGrid(duration: inputDuration ?? 0, frameRate: inputFrameRate ?? 0),
                    prompts: editing.regionPrompts,
                    initFrame: editing.initFrame,
                    endFrame: editing.endFrame,
                    maxHeight: seedFrameHeight
                )
            } else {
                Rectangle()
                    .fill(MereRunTheme.surfaceRaised)
                    .aspectRatio(videoAspect, contentMode: .fit)
                    .frame(maxHeight: seedFrameHeight)
                    .overlay { ProgressView().controlSize(.small) }
                    .mereMediaFrame()
            }
        }
    }

    /// The frame's share of the media height once the editor's own rows, and the Show result
    /// row when a result is on file, have theirs.
    private var seedFrameHeight: CGFloat {
        mediaHeight - StudioTrackFrameEditor.rowsHeight - (resultDescribesInput ? 36 : 0)
    }

    /// Track writes an annotated clip; that is the one worth playing when it exists.
    private var playableVideoURL: URL? {
        guard resultDescribesInput, let outputURL = resultCard?.item.outputURL,
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
                Image(systemName: inputKind == .directory ? "folder" : "doc.text")
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
            Text("No \(inputKind.noun) attached.")
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
                    actions: readinessActions,
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
                StudioFailureCard(item: card.item, job: nil, isHighlighted: false, actions: actions,
                                  modelInventory: readinessActions.modelInventory)
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
            StudioFailureCard(item: card.item, job: card.job, isHighlighted: false, actions: actions,
                              modelInventory: readinessActions.modelInventory)
        case .generation:
            EmptyView()
        }
    }

    @ViewBuilder
    private func resultPanel(_ card: StudioFeedCard) -> some View {
        if resultDescribesInput {
            StudioAnalyzeResultPanel(
                item: card.item,
                document: document,
                detections: detections,
                speechSegments: document?.speechSegments ?? [],
                outputText: card.item.outputText,
                view: view,
                nextActions: archetype.nextActions,
                onOpenTask: { analyze.openTask($0, detections) },
                onSave: analyze.save
            )
        } else {
            ContentUnavailableView("Input changed", systemImage: "arrow.triangle.2.circlepath",
                description: Text("Run this task again to analyze the selected input. The earlier result remains in Library."))
        }
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
        let derived = StudioAnalyzeDocument.derived(from: item)
        let result = await Task.detached(priority: .userInitiated) {
            StudioAnalyzeLoadedResult.load(itemID: itemID, url: url, fallbackText: fallbackText, derived: derived)
        }.value
        guard !Task.isCancelled else { return }
        loaded = result
    }

    private func measureInput() async {
        guard let inputURL else {
            inputSize = nil
            inputDuration = nil
            inputFrameRate = nil
            inputOrientation = .up
            return
        }
        let kind = inputKind
        let measured = await Task.detached(priority: .userInitiated) {
            StudioAnalyzeMediaInfo.measure(inputURL, kind: kind)
        }.value
        guard !Task.isCancelled else { return }
        inputSize = measured.size
        inputDuration = measured.duration
        inputFrameRate = measured.frameRate
        inputOrientation = measured.orientation
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
        case .directory: return "folder"
        case .text: return "text"
        case .none: return "input"
        }
    }
}

// MARK: - Loading the result document

/// A decoded result document, tied to the run it came from so a stale one is never drawn.
struct StudioAnalyzeLoadedResult: Equatable {
    let itemID: UUID
    let url: URL?
    /// The document as written, for the JSON view.
    let raw: String?
    let document: StudioAnalyzeDocument?

    /// - Parameter derived: the document the run's row alone stands for
    ///   (`StudioAnalyzeDocument.derived(from:)`), which wins over reading its output.
    static func load(itemID: UUID, url: URL?, fallbackText: String?, derived: StudioAnalyzeDocument? = nil) -> StudioAnalyzeLoadedResult {
        if let derived {
            return StudioAnalyzeLoadedResult(itemID: itemID, url: nil, raw: fallbackText, document: derived)
        }
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
        /// Stored pixels for an image, the displayed frame for a clip.
        var size: CGSize?
        var duration: TimeInterval?
        /// The clip's declared frame rate, which Track's frame numbers count in.
        var frameRate: Double?
        /// The photo's EXIF orientation; clips are already shown the way their track transform says.
        var orientation = StudioImageOrientation.up
    }

    static func measure(_ url: URL, kind: StudioAnalyzeInputKind) -> Measurement {
        switch kind {
        case .image:
            let metadata = StudioImageMetadata.read(url)
            return Measurement(size: metadata?.storedSize, duration: nil, orientation: metadata?.orientation ?? .up)
        case .video:
            let asset = AVURLAsset(url: url)
            let track = asset.tracks(withMediaType: .video).first
            let size = track.map { $0.naturalSize.applying($0.preferredTransform) }
                .map { CGSize(width: abs($0.width), height: abs($0.height)) }
            let rate = track.map { Double($0.nominalFrameRate) }.flatMap { $0 > 0 ? $0 : nil }
            return Measurement(size: size, duration: CMTimeGetSeconds(asset.duration), frameRate: rate)
        case .audio:
            let asset = AVURLAsset(url: url)
            let duration = CMTimeGetSeconds(asset.duration)
            return Measurement(size: nil, duration: duration.isFinite ? duration : nil)
        case .file, .directory, .text, .none:
            return Measurement()
        }
    }

    /// The image's stored pixel dimensions, read from its metadata without decoding the pixels.
    static func pixelSize(of url: URL) -> CGSize? {
        StudioImageMetadata.read(url)?.storedSize
    }
}
