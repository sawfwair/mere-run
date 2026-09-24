import StudioKit
import SwiftUI

/// What a bespoke renderer draws: one case per (view, document) pairing the registry knows.
/// Typed rather than `AnyView` so the panel's switch is exhaustive and a new renderer is one
/// case plus one view.
enum StudioResultRendering: Equatable {
    /// A tensor file's header, for Earth's safetensors and `sfx ae encode`'s `.npy`.
    case tensor(StudioTensorHeader)
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
        // The JSON view puts the raw document in the input column; the panel beside it keeps
        // the document's own rows rather than falling to the detection rows it has none of.
        switch (view, document) {
        case (.tensor, .tensor(let header)):
            return .tensor(header)
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
