import StudioKit
import SwiftUI

/// The one line under a 3D result's tile: what the run's manifest counted in the mesh. A tile
/// shows the shape; this says how much of it there is, so a TRELLIS.2 asset and a quick TripoSR
/// pass read differently at a glance.
struct StudioMeshSummaryRow: View {
    let summary: StudioMeshSummary

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(MereRunTheme.textMuted)
            Text(summary.text)
                .font(.caption.weight(.medium))
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Mesh: \(summary.text)")
    }
}
