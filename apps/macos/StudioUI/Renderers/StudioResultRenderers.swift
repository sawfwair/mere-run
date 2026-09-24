import StudioKit
import SwiftUI

/// What a bespoke renderer draws: one case per (view, document) pairing the registry knows.
/// Typed rather than `AnyView` so the panel's switch is exhaustive and a new renderer is one
/// case plus one view.
enum StudioResultRendering: Equatable {
    /// A tensor file's header, for Earth's safetensors, `sfx ae encode`'s `.npy`, and the
    /// conditioning tensors `sfx condition text` exports.
    case tensor(StudioTensorHeader)
    /// The CLAP alignment gauge over `sfx clap score`'s printed result.
    case clap(StudioCLAPScore.Output)
    /// Video Foley's picture over the waveform it produced.
    case syncReview(video: URL, audio: URL)
    /// An audio result the input column cannot play for the run (`sfx ae decode` takes a
    /// latents file, not audio): the waveform player over the file's row.
    case audioOutput(URL)
    /// What `music analyze` understood about a recording.
    case musicAnalysis(StudioMusicAnalysisDocument)
    /// The notes `music transcribe` wrote to MIDI.
    case pianoRoll(StudioMIDISummary)
    /// One row per body, hand, or face a pose run found.
    case poseSubjects(StudioPoseOverlayResult)
    /// How far things moved in a flow field.
    case flowStatistics(StudioFlowField)
    /// Which face was embedded and the vector's size.
    case faceEmbedding(StudioFaceEmbeddingDocument)
    /// A comparison's similarity and the two faces it compared.
    case faceComparison(StudioFaceComparisonDocument)
    /// One row per picture of a face batch.
    case faceBatch(StudioFaceBatchDocument)
    /// A depth run's manifest: size, inference size, checkpoint, range.
    case depthManifest(StudioDepthManifest)
    /// The stems `music separate` wrote, from its manifest when the run's document is one.
    case stems(StudioSeparationManifest?)
    /// Text ▸ Embeddings' vectors and their cosine similarity.
    case embeddings(StudioEmbeddingDocument)
    /// Text ▸ Anonymize's inputs beside their protected text and marked spans.
    case anonymizationSpans(StudioAnonymizationDocument)
    /// Text ▸ Anonymize's protected text alone.
    case anonymizedText(StudioAnonymizationDocument)
    /// Image ▸ Datasets ▸ Discover's candidate folders, each with "Train on it".
    case datasetCandidates(StudioDatasetDiscoveryDocument)
    /// Image ▸ Datasets ▸ Run plan's preflight or materialization report.
    case runPlan(StudioRunPlanReport)
    /// Image ▸ Datasets ▸ Validate's artifact folder.
    case validationArtifacts(StudioImageValidationReport)
    /// The counts a 3D run's manifest reports, under the mesh tile on its feed card.
    case meshSummary(StudioMeshSummary)
}

/// What a renderer draws in the Analyze input column, in place of the input, when the chosen
/// view is about the result rather than the picture it came from.
enum StudioResultCanvasRendering: Equatable {
    /// The input picture with face boxes and landmarks, or pose landmarks, over it.
    case overlay(image: URL, StudioVisionOverlay)
    /// A dense flow field as vectors.
    case flow(StudioFlowField)
    /// The depth previews and review clip a run wrote.
    case depth(StudioVisionRunArtifacts)
    /// The point cloud and previews a geometry run wrote.
    case scene(StudioVisionRunArtifacts)
}

/// A rendering on a finished feed card and where it sits: in place of the output grid, when
/// the run's files have no picture to tile (a WAV reviewed against its clip, a tensor's
/// header), or under the tiles, when it says more about what they show (a mesh's counts).
struct StudioCardRendering: Equatable {
    enum Placement: Equatable {
        case replacesOutputs
        case belowOutputs
    }

    let rendering: StudioResultRendering
    let placement: Placement
}

/// The registry the Analyze result panel asks before drawing its own rows: given the view the
/// strip has selected and the decoded document, the bespoke rendering for that pair, or nil
/// when the panel's generic rows (detections, speech turns, text) already say it.
///
/// A page PR moves its renderer into this folder, adds a `StudioResultRendering` case, a match
/// here, and a branch in `StudioResultRendererView`; the panel, the canvas, and the schema never
/// change for it. Rendering is keyed by `(view, document)` rather than by task, so a document
/// drawn the same way for two tasks (a tensor header from Earth or from `sfx ae encode`) is
/// drawn once.
enum StudioResultRenderers {
    static func rendering(
        for view: StudioAnalyzeResultView,
        document: StudioAnalyzeDocument?,
        item: StudioLibraryItem
    ) -> StudioResultRendering? {
        switch (view, document) {
        case (.tensor, .tensor(let header)):
            return .tensor(header)
        case (.score, .clap(let output)):
            return .clap(output)
        case (.audio, nil):
            // The canvas plays an audio task's output in the input column; a run whose input
            // was not audio has nowhere else to be heard. Tasks on the shared workspace only,
            // like the panel's own output rows.
            guard item.templateID?.studioTask.usesTaskDraft == true,
                  item.inputURL.map({ StudioOutputFileKind.classify($0) }) != .audio,
                  let audio = item.allArtifactURLs.first(where: {
                      StudioOutputFileKind.classify($0) == .audio && FileManager.default.fileExists(atPath: $0.path)
                  }) else { return nil }
            return .audioOutput(audio)
        case (.analysis, .musicAnalysis(let analysis)):
            return .musicAnalysis(analysis)
        case (.notes, .midi(let summary)):
            return .pianoRoll(summary)
        case (_, .pose(let result)):
            return .poseSubjects(result)
        case (_, .flow(let field)):
            return .flowStatistics(field)
        case (_, .faceEmbedding(let document)):
            return .faceEmbedding(document)
        case (_, .faceComparison(let document)):
            return .faceComparison(document)
        case (_, .faceBatch(let document)):
            return .faceBatch(document)
        case (_, .depthManifest(let manifest)):
            return .depthManifest(manifest)
        case (.stems, .separation(let manifest)):
            return .stems(manifest)
        case (.stems, _):
            return .stems(nil)
        // The JSON view puts the raw document in the input column; the panel beside it keeps
        // the document's own rows rather than falling to the detection rows it has none of.
        case (.vectors, .embeddings(let document)), (.json, .embeddings(let document)):
            return .embeddings(document)
        case (.spans, .anonymization(let document)):
            return .anonymizationSpans(document)
        case (.text, .anonymization(let document)):
            return .anonymizedText(document)
        case (.candidates, .datasetDiscovery(let document)), (.json, .datasetDiscovery(let document)):
            return .datasetCandidates(document)
        case (.report, .runPlan(let report)), (.json, .runPlan(let report)):
            return .runPlan(report)
        case (.json, .validation(let report)):
            // Validate's only view; the CLI's words are in the input column, the folder is the result.
            return .validationArtifacts(report)
        default:
            return nil
        }
    }

    /// A run's directory listing, read once per row version: the canvas asks for its rendering
    /// on every body evaluation, and walking the output folder each time would not do.
    @MainActor private static var artifactsCache: [UUID: (updatedAt: Date, artifacts: StudioVisionRunArtifacts)] = [:]

    @MainActor
    static func runArtifacts(for item: StudioLibraryItem) -> StudioVisionRunArtifacts {
        if let cached = artifactsCache[item.id], cached.updatedAt == item.updatedAt { return cached.artifacts }
        let artifacts = StudioVisionRunArtifacts.read(item: item)
        if artifactsCache.count > 32 { artifactsCache.removeAll() }
        artifactsCache[item.id] = (item.updatedAt, artifacts)
        return artifacts
    }

    /// The rendering that takes the input column for `view`, or nil when the column shows the
    /// input itself. `item` is the run on screen only while its result describes the input on
    /// screen; a directory output (depth, geometry) is read from the row's artifacts.
    @MainActor
    static func canvasRendering(
        for view: StudioAnalyzeResultView,
        document: StudioAnalyzeDocument?,
        item: StudioLibraryItem?,
        inputURL: URL?
    ) -> StudioResultCanvasRendering? {
        switch (view, document) {
        case (.points, .faces(let faces)):
            return inputURL.map { .overlay(image: $0, .faces(faces)) }
        case (.points, .pose(let pose)):
            return inputURL.map { .overlay(image: $0, .pose(pose)) }
        case (.vectors, .flow(let field)):
            return .flow(field)
        case (.depth, _), (.video, _):
            guard let item, item.templateID == .visionDepth || item.templateID == .visionDepthVideo else { return nil }
            let artifacts = runArtifacts(for: item)
            return artifacts.isEmpty ? nil : .depth(artifacts)
        case (.scene, _):
            guard let item, item.templateID == .visionGeometry || item.templateID == .visionGeometryMultiview else { return nil }
            let artifacts = runArtifacts(for: item)
            return artifacts.isEmpty ? nil : .scene(artifacts)
        default:
            return nil
        }
    }

    /// The generation feed's counterpart: what a finished card draws for the run's outputs and
    /// where, keyed by what the run wrote rather than by task. Video Foley's WAV is reviewed
    /// against the clip it was made for and a tensor output (`sfx condition text`) shows its
    /// header, each in place of the output grid a WAV or a `.safetensors` file has no picture
    /// for; a 3D run's manifests are summarized under its mesh tile. Nil leaves the grid alone.
    static func cardRendering(for item: StudioLibraryItem, files: [URL]) -> StudioCardRendering? {
        if item.templateID?.studioTask == .threeDFromImage, let summary = StudioMeshSummary.load(item: item) {
            return StudioCardRendering(rendering: .meshSummary(summary), placement: .belowOutputs)
        }
        guard let output = item.outputURL, files.contains(output) else { return nil }
        if item.templateID == .sfxVideo, let video = item.inputURL, StudioOutputFileKind.classify(video) == .video,
           StudioOutputFileKind.classify(output) == .audio, FileManager.default.fileExists(atPath: video.path) {
            return StudioCardRendering(rendering: .syncReview(video: video, audio: output), placement: .replacesOutputs)
        }
        if StudioTensorHeader.fileExtensions.contains(output.pathExtension.lowercased()), let header = StudioTensorHeader.load(from: output) {
            return StudioCardRendering(rendering: .tensor(header), placement: .replacesOutputs)
        }
        return nil
    }

    /// The files a card rendering already shows, so the card lists neither as a sidecar.
    static func renderedFiles(of rendering: StudioResultRendering, item: StudioLibraryItem) -> [URL] {
        switch rendering {
        case .tensor:
            return item.outputURL.map { [$0] } ?? []
        case .clap, .musicAnalysis, .pianoRoll, .poseSubjects, .flowStatistics, .faceEmbedding, .faceComparison,
             .faceBatch, .depthManifest, .stems, .embeddings, .anonymizationSpans, .anonymizedText, .datasetCandidates,
             .runPlan, .validationArtifacts, .meshSummary:
            return []
        case .syncReview(_, let audio), .audioOutput(let audio):
            return [audio]
        }
    }
}

/// A card rendering in the generation feed's chrome: rows (a tensor header) sit on the raised
/// panel the card gives a text preview; the sync review is its own tiles.
struct StudioCardRenderingView: View {
    let rendering: StudioResultRendering
    let item: StudioLibraryItem

    var body: some View {
        switch rendering {
        case .tensor(let header):
            // The header's rows, then the file itself, since the card lists it as no chip.
            VStack(spacing: 0) {
                StudioTensorInspector(header: header)
                if let url = item.outputURL {
                    StudioResultFileRow(url: url, glyph: "square.stack.3d.down.forward", quickLook: true)
                }
            }
            .background(MereRunTheme.surfaceRaised.opacity(0.6))
            .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
        case .clap, .audioOutput, .musicAnalysis, .pianoRoll, .poseSubjects, .flowStatistics, .faceEmbedding,
             .faceComparison, .faceBatch, .depthManifest, .stems, .embeddings, .anonymizationSpans, .anonymizedText,
             .datasetCandidates, .runPlan, .validationArtifacts, .meshSummary:
            StudioResultRendererView(rendering: rendering, item: item)
                .background(MereRunTheme.surfaceRaised.opacity(0.6))
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
        case .syncReview:
            StudioResultRendererView(rendering: rendering, item: item)
        }
    }
}

/// The rows a rendering contributes inside the result panel, under its header and above its
/// action row, or inside a generation card in place of the output grid.
struct StudioResultRendererView: View {
    let rendering: StudioResultRendering
    let item: StudioLibraryItem

    var body: some View {
        switch rendering {
        case .tensor(let header):
            StudioTensorInspector(header: header)
        case .clap(let output):
            StudioCLAPGauge(output: output)
        case .syncReview(let video, let audio):
            StudioSyncReviewTile(videoURL: video, audioURL: audio)
        case .audioOutput(let audio):
            StudioAudioOutputRows(url: audio)
        case .musicAnalysis(let analysis):
            StudioMusicAnalysisRenderer(analysis: analysis)
        case .pianoRoll(let summary):
            StudioPianoRollRenderer(summary: summary, midiURL: StudioAnalyzeDocumentSource.url(for: item))
        case .poseSubjects(let result):
            StudioPoseSubjectRows(result: result)
        case .flowStatistics(let field):
            StudioFlowStatisticsRows(field: field)
        case .faceEmbedding(let document):
            StudioFaceEmbeddingRows(document: document)
        case .faceComparison(let document):
            StudioFaceComparisonRows(document: document)
        case .faceBatch(let document):
            StudioFaceBatchRows(document: document)
        case .depthManifest(let manifest):
            StudioDepthManifestRows(manifest: manifest)
        case .stems(let manifest):
            StudioStemsList(item: item, manifest: manifest)
        case .embeddings(let document):
            StudioEmbeddingsMatrix(document: document)
        case .anonymizationSpans(let document):
            StudioAnonymizationSpans(document: document)
        case .anonymizedText(let document):
            StudioAnonymizedText(document: document)
        case .datasetCandidates(let document):
            StudioDatasetCandidatesRow(document: document)
        case .runPlan(let report):
            StudioRunPlanReportView(report: report)
        case .validationArtifacts(let report):
            StudioImageValidationArtifacts(report: report)
        case .meshSummary(let summary):
            StudioMeshSummaryRow(summary: summary)
        }
    }
}

/// The decoded audio as the result panel's rows: the player, then the file's name with Reveal.
private struct StudioAudioOutputRows: View {
    let url: URL

    var body: some View {
        VStack(spacing: 0) {
            StudioAudioPlayerView(url: url)
                .frame(height: 210)
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
            Rectangle()
                .fill(MereRunTheme.border.opacity(0.27))
                .frame(height: 1)
            StudioResultFileRow(url: url)
        }
    }
}
/// A rendering in the Analyze input column, framed the way the input media is.
struct StudioResultCanvasView: View {
    let rendering: StudioResultCanvasRendering
    let maxHeight: CGFloat

    var body: some View {
        switch rendering {
        case .overlay(let image, let overlay):
            StudioVisionOverlayView(imageURL: image, overlay: overlay)
                .frame(maxHeight: maxHeight)
                .mereMediaFrame()
        case .flow(let field):
            StudioFlowFieldView(field: field)
                .frame(maxHeight: maxHeight)
                .mereMediaFrame()
        case .depth(let artifacts):
            StudioDepthPreviewView(artifacts: artifacts, maxHeight: maxHeight)
        case .scene(let artifacts):
            StudioGeometrySceneView(artifacts: artifacts, maxHeight: maxHeight)
        }
    }
}

/// The candidates renderer with its "Train on it" wired to the window's navigation and the
/// Training page's parked draft. Kept apart from `StudioResultRendererView` so the panel itself
/// never requires a navigation model in its environment.
private struct StudioDatasetCandidatesRow: View {
    let document: StudioDatasetDiscoveryDocument
    @EnvironmentObject private var navigation: NavigationModel
    @Environment(\.studioTaskSessions) private var sessions

    var body: some View {
        StudioDatasetCandidates(document: document) { candidate in
            StudioDatasetTrainingHandoff.open(candidate, navigation: navigation, sessions: sessions)
        }
    }
}
