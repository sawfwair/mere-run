import StudioKit
import SwiftUI

/// What a bespoke renderer draws: one case per (view, document) pairing the registry knows.
/// Typed rather than `AnyView` so the panel's switch is exhaustive and a new renderer is one
/// case plus one view.
enum StudioResultRendering: Equatable {
    /// A tensor file's header, for Earth's safetensors and `sfx ae encode`'s `.npy`.
    case tensor(StudioTensorHeader)
    /// The counts a 3D run's manifest reports, under the mesh tile on its feed card.
    case meshSummary(StudioMeshSummary)
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
        default:
            return nil
        }
    }

    /// The rendering a Generate feed card adds under its outputs for one finished row, read
    /// from the files the run left: a mesh summary for a 3D run's manifests. Nil for a row whose
    /// tiles say everything.
    static func cardRendering(for item: StudioLibraryItem) -> StudioResultRendering? {
        if item.templateID?.studioTask == .threeDFromImage, let summary = StudioMeshSummary.load(item: item) {
            return .meshSummary(summary)
        }
        return nil
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
        case .meshSummary(let summary):
            StudioMeshSummaryRow(summary: summary)
        }
    }
}
