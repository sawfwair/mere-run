import StudioKit
import SwiftUI

/// What a bespoke input view draws in the Analyze canvas's input column: one case per template
/// family that can say more about its input file than its name. Empty until the first page
/// registers one.
enum StudioInputRendering: Equatable {}

/// The registry the Analyze canvas asks before drawing a `.file` input as an icon and a name:
/// given the draft's template and the attached file, the bespoke rendering for it, or nil when
/// the generic file block already says everything there is to say. Keyed by template rather
/// than by task, like `StudioResultRenderers`, so a page PR adds a case, a match, and a branch
/// in `StudioInputRendererView` without touching the canvas.
enum StudioInputRenderers {
    static func rendering(for templateID: CommandTemplateID?, url: URL?) -> StudioInputRendering? {
        nil
    }
}

struct StudioInputRendererView: View {
    let rendering: StudioInputRendering

    var body: some View {
        // Nothing is registered yet; the first page PR turns this into a switch over its case.
        EmptyView()
    }
}
