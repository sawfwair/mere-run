import ActivityKit
import SwiftUI
import WidgetKit

@main
struct MereRunWidgetsBundle: WidgetBundle {
    var body: some Widget {
        RunActivityWidget()
    }
}

/// Live Activity for a fleet run: lock-screen banner and Dynamic Island.
struct RunActivityWidget: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: RunActivityAttributes.self) { context in
            HStack(spacing: 10) {
                Image(systemName: "sparkles")
                    .foregroundStyle(RunActivityPalette.accent)
                VStack(alignment: .leading, spacing: 3) {
                    Text(context.attributes.title)
                        .font(.subheadline.weight(.semibold))
                    if let fraction = context.state.fraction {
                        ProgressView(value: fraction)
                            .tint(RunActivityPalette.accent)
                    }
                    Text(context.state.detail ?? context.state.stateLabel)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
            }
            .padding(12)
            .activityBackgroundTint(Color(red: 0xFA / 255, green: 0xF8 / 255, blue: 0xF3 / 255))
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    Image(systemName: "sparkles")
                        .foregroundStyle(RunActivityPalette.accent)
                }
                DynamicIslandExpandedRegion(.center) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(context.attributes.title)
                            .font(.subheadline.weight(.semibold))
                        if let fraction = context.state.fraction {
                            ProgressView(value: fraction)
                                .tint(RunActivityPalette.accent)
                        }
                    }
                }
                DynamicIslandExpandedRegion(.bottom) {
                    Text(context.state.detail ?? context.state.stateLabel)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            } compactLeading: {
                Image(systemName: "sparkles")
                    .foregroundStyle(RunActivityPalette.accent)
            } compactTrailing: {
                if let fraction = context.state.fraction {
                    Text("\(Int(fraction * 100))%")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(RunActivityPalette.accent)
                } else {
                    ProgressView()
                }
            } minimal: {
                Image(systemName: "sparkles")
                    .foregroundStyle(RunActivityPalette.accent)
            }
        }
    }
}
