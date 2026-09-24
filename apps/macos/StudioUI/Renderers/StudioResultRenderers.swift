import StudioKit
import SwiftUI

/// What a bespoke renderer draws: one case per (view, document) pairing the registry knows.
/// Typed rather than `AnyView` so the panel's switch is exhaustive and a new renderer is one
/// case plus one view.
enum StudioResultRendering: Equatable {
    /// A tensor file's header, for Earth's safetensors and `sfx ae encode`'s `.npy`.
    case tensor(StudioTensorHeader)
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
    static func rendering(for view: StudioAnalyzeResultView, document: StudioAnalyzeDocument?) -> StudioResultRendering? {
        switch (view, document) {
        case (.tensor, .tensor(let header)):
            return .tensor(header)
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
        default:
            return nil
        }
    }

    /// The rendering that takes the input column for `view`, or nil when the column shows the
    /// input itself. `item` is the run on screen only while its result describes the input on
    /// screen; a directory output (depth, geometry) is read from the row's artifacts.
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
            let artifacts = StudioVisionRunArtifacts.read(item: item)
            return artifacts.isEmpty ? nil : .depth(artifacts)
        case (.scene, _):
            guard let item, item.templateID == .visionGeometry || item.templateID == .visionGeometryMultiview else { return nil }
            let artifacts = StudioVisionRunArtifacts.read(item: item)
            return artifacts.isEmpty ? nil : .scene(artifacts)
        default:
            return nil
        }
    }
}

/// The rows a rendering contributes inside the result panel, under its header and above its
/// action row.
struct StudioResultRendererView: View {
    let rendering: StudioResultRendering
    let item: StudioLibraryItem

    var body: some View {
        switch rendering {
        case .tensor(let header):
            StudioTensorInspector(header: header)
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

// MARK: - Shared rows

/// One labelled fact in the result panel, the way the tensor inspector and the run plan report
/// lay theirs out.
struct StudioResultFactRow: View {
    let label: String
    let value: String

    init(_ label: String, _ value: String) {
        self.label = label
        self.value = value
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(label)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .frame(width: 72, alignment: .leading)
                Text(value)
                    .font(.system(size: 12.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .accessibilityElement(children: .combine)
            StudioResultHairline()
        }
    }
}

/// The hairline under each panel row.
struct StudioResultHairline: View {
    var body: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.27))
            .frame(height: 1)
    }
}

/// Rows that size to their content, scrolling inside the panel past a handful so the action
/// row stays in reach: the panel's own rule.
struct StudioResultRowList<Content: View>: View {
    let count: Int
    @ViewBuilder let content: () -> Content

    private static var rowsMaxHeight: CGFloat { 320 }
    private static var scrollingRowThreshold: Int { 7 }

    var body: some View {
        if count > Self.scrollingRowThreshold {
            ScrollView {
                VStack(spacing: 0) { content() }
            }
            .frame(height: Self.rowsMaxHeight)
        } else {
            VStack(spacing: 0) { content() }
        }
    }
}
