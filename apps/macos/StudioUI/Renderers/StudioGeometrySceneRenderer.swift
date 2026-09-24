import StudioKit
import SwiftUI

// What a geometry run wrote: a point cloud (GLB, with the PLY beside it) and the depth, normal,
// and confidence previews. The canvas embeds Quick Look for the cloud, as the page did, and the
// strip switches to the previews.

/// The scene of a geometry run in the Analyze input column: Quick Look over the point cloud,
/// or the preview picked from the strip.
struct StudioGeometrySceneView: View {
    let artifacts: StudioVisionRunArtifacts
    let maxHeight: CGFloat

    @State private var selection: URL?

    private var choices: [URL] { artifacts.scenes + artifacts.previews }

    private var active: URL? {
        if let selection, choices.contains(selection) { return selection }
        return choices.first
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let active {
                Group {
                    if StudioOutputFileKind.classify(active) == .model3D {
                        StudioEmbeddedQuickLookPreview(url: active)
                    } else {
                        StudioAsyncImagePreview(url: active, maxPixelSize: 2_000, contentMode: .fit, fallbackSystemImage: "photo")
                    }
                }
                .frame(maxWidth: .infinity)
                .frame(height: choices.count > 1 ? maxHeight - 40 : maxHeight)
                .background(MereRunTheme.surfaceRaised)
                .mereMediaFrame()
                .accessibilityLabel("Scene \(active.lastPathComponent)")
            }
            if choices.count > 1 {
                StudioResultFileStrip(files: choices, selection: Binding(get: { active }, set: { selection = $0 }))
            }
        }
    }
}
