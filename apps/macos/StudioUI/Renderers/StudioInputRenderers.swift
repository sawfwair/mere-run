import StudioKit
import SwiftUI

/// What a bespoke input view draws in the Analyze canvas's input column: one case per template
/// family that can say more about its input file than its name.
enum StudioInputRendering: Equatable {
    /// The Earth commands' safetensors bundle read against the tensors its command requires;
    /// nil `url` is the empty well, which shows the requirement as a hint.
    case earthBundle(StudioEarthInputRequirement, URL?)
}

/// The registry the Analyze canvas asks before drawing a `.file` input as an icon and a name:
/// given the draft's template and the attached file, the bespoke rendering for it, or nil when
/// the generic file block already says everything there is to say. Keyed by template rather
/// than by task, like `StudioResultRenderers`.
enum StudioInputRenderers {
    static func rendering(for templateID: CommandTemplateID?, url: URL?) -> StudioInputRendering? {
        guard let templateID, let requirement = StudioEarthInputRequirement.requirement(for: templateID) else { return nil }
        return .earthBundle(requirement, url)
    }
}

struct StudioInputRendererView: View {
    let rendering: StudioInputRendering

    var body: some View {
        switch rendering {
        case .earthBundle(let requirement, let url):
            StudioEarthInputChecklist(requirement: requirement, url: url)
        }
    }
}
