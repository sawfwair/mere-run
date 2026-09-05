import StudioKit
import SwiftUI

/// Both handbook families and command cookbooks are read from the bundled CLI, without a network request.
struct StudioHelpSheet: View {
    @EnvironmentObject private var controller: MereRunController
    @Environment(\.dismiss) private var dismiss

    @State private var topics: [StudioGuideTopic] = []
    @State private var modelTopics: [StudioGuideTopic] = []
    @State private var selected: StudioGuideTopic?
    @State private var selectedModel = ""
    @State private var query = ""
    @State private var showModels = true
    @State private var content = ""
    @State private var isLoading = false
    @State private var loadingTopics = true

    private var visibleTopics: [StudioGuideTopic] {
        (showModels ? modelTopics : topics).filter { $0.matches(query) }
    }

    private var contentKey: String { (selected?.id ?? "") + ":" + selectedModel }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.4))
            HStack(spacing: 0) {
                sidebar
                    .frame(width: 280)
                Divider().overlay(MereRunTheme.border.opacity(0.4))
                detail
            }
        }
        .background(MereRunTheme.background)
        .task { await loadTopics() }
        .task(id: contentKey) { await loadSelection() }
        .onChange(of: showModels) { _, _ in select(visibleTopics.first) }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Offline handbook")
                    .font(MereRunTheme.titleFont)
                Text("Model recipes and command help. Source links are optional online reading.")
                    .font(.caption)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            Spacer()
            Button("Done") { dismiss() }
                .keyboardShortcut(.defaultAction)
        }
        .padding(20)
    }

    private var sidebar: some View {
        VStack(spacing: 8) {
            Picker("Guide collection", selection: $showModels) {
                Text("Models").tag(true)
                Text("Commands").tag(false)
            }
            .pickerStyle(.segmented)
            TextField("Search family, model ID, or command", text: $query)
                .textFieldStyle(.roundedBorder)
                .accessibilityLabel("Search offline guides")
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    if loadingTopics {
                        ProgressView().padding()
                    } else if visibleTopics.isEmpty {
                        Text(query.isEmpty
                             ? "No guides loaded. Check the CLI in Settings, then retry."
                             : "No matching guides.")
                            .font(.caption)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .padding()
                        if query.isEmpty {
                            Button("Retry") { Task { await loadTopics() } }
                        }
                    }
                    ForEach(visibleTopics) { topic in
                        StudioHelpTopicRow(
                            title: topic.title,
                            isSelected: selected?.id == topic.id
                        ) {
                            select(topic)
                        }
                    }
                }
            }
        }
        .padding(8)
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
                VStack(alignment: .leading, spacing: 16) {
                    if let selected, selected.isModelGuide {
                        Picker("Model", selection: $selectedModel) {
                            ForEach(selected.models, id: \.self) { model in
                                Text(model).tag(model)
                            }
                        }
                        .accessibilityLabel("Model covered by this guide")
                    }
                    StudioMarkdownText(content: content)
                }
                    .id(contentKey)
                    .frame(maxWidth: 680, alignment: .leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(24)
            }
        }
    }

    private func loadTopics() async {
        loadingTopics = true
        topics = await controller.loadGuideTopics()
        modelTopics = await controller.loadModelGuideTopics()
        loadingTopics = false
        if selected == nil { select(visibleTopics.first) }
    }

    private func select(_ topic: StudioGuideTopic?) {
        selected = topic
        selectedModel = topic?.isModelGuide == true ? (topic?.models.first ?? "") : ""
        content = ""
    }

    private func loadSelection() async {
        guard let selected else {
            isLoading = false
            return
        }
        let requestKey = contentKey
        isLoading = true
        let text = await controller.loadGuideContent(
            commandPath: selected.commandPath,
            model: selected.isModelGuide ? selectedModel : nil
        )
        guard !Task.isCancelled, contentKey == requestKey else { return }
        content = text
        isLoading = false
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
                .lineLimit(2)
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
