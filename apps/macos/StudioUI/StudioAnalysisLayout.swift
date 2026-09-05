import SwiftUI

/// Keeps input controls and the result usable in both a desk-sized window and a narrow one.
struct StudioAnalysisLayout<Controls: View, Result: View>: View {
    @ViewBuilder let controls: () -> Controls
    @ViewBuilder let result: () -> Result

    var body: some View {
        GeometryReader { geometry in
            if geometry.size.width >= 780 {
                HSplitView {
                    controls().frame(minWidth: 280, idealWidth: 340, maxWidth: 410)
                    result().frame(minWidth: 360, maxWidth: .infinity, maxHeight: .infinity)
                }
            } else {
                VSplitView {
                    controls().frame(minHeight: 190, idealHeight: geometry.size.height * 0.45)
                    result().frame(maxWidth: .infinity, minHeight: 220, maxHeight: .infinity)
                }
            }
        }
    }
}
