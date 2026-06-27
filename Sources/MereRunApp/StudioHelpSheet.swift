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
                    Button { select(topic) } label: {
                        Text(topic.title)
                            .font(MereRunTheme.bodyFont)
                            .foregroundStyle(selected?.id == topic.id ? MereRunTheme.accent : MereRunTheme.textSecondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.vertical, 6)
                            .padding(.horizontal, 14)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 8)
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
                Text(renderedMarkdown)
                    .font(MereRunTheme.bodyFont)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(24)
            }
        }
    }

    private var renderedMarkdown: AttributedString {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: content, options: options)) ?? AttributedString(content)
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
