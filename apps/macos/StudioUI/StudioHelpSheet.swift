import StudioKit
import SwiftUI

/// In-app help powered by the offline `guide` command: a topic list on the left and the selected
/// topic's Markdown rendered on the right.
struct StudioHelpSheet: View {
    @EnvironmentObject private var controller: MereRunController
    @Environment(\.dismiss) private var dismiss

    @State private var topics: [StudioGuideTopic] = []
    @State private var selected: StudioGuideTopic?
    @State private var content = ""
    @State private var isLoading = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.4))
            HStack(spacing: 0) {
                sidebar
                    .frame(width: 220)
                Divider().overlay(MereRunTheme.border.opacity(0.4))
                detail
            }
        }
        .background(MereRunTheme.background)
        .task { await loadTopics() }
    }

    private var header: some View {
        HStack {
            Text("Guide")
                .font(MereRunTheme.titleFont)
            Spacer()
            Button("Done") { dismiss() }
                .keyboardShortcut(.defaultAction)
        }
        .padding(20)
    }

    private var sidebar: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 2) {
                ForEach(topics) { topic in
                    StudioHelpTopicRow(
                        title: topic.title,
                        isSelected: selected?.id == topic.id
                    ) {
                        select(topic)
                    }
                }
            }
            .padding(8)
        }
    }

    private var detail: some View {
        ScrollView {
            if isLoading {
                ProgressView()
                    .frame(maxWidth: .infinity)
                    .padding(40)
            } else if content.isEmpty {
                Text("Select a topic to read its guide.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(24)
            } else {
                StudioMarkdownText(content: content)
                    .frame(maxWidth: 680, alignment: .leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(24)
            }
        }
    }

    private func loadTopics() async {
        topics = await controller.loadGuideTopics()
        if selected == nil, let first = topics.first {
            select(first)
        }
    }

    private func select(_ topic: StudioGuideTopic) {
        selected = topic
        isLoading = true
        Task {
            let text = await controller.loadGuideContent(commandPath: topic.commandPath)
            content = text
            isLoading = false
        }
    }
}

private struct StudioHelpTopicRow: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12.5, weight: isSelected ? .semibold : .regular))
                .foregroundStyle(isSelected ? MereRunTheme.textPrimary : MereRunTheme.textSecondary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 6)
                .padding(.horizontal, 10)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                        .fill(isSelected ? MereRunTheme.accentSoft : (hovering ? MereRunTheme.hoverFill : .clear))
                }
                .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}
