import SwiftUI
import MereRunRelayKit

/// Multi-turn chat over the fleet: bubbles on paper, the composer pinned to
/// the bottom, the model as a quiet chip. Each turn is a stateless
/// `text.generate` job; the thread lives on the phone.
struct ChatView: View {
    @EnvironmentObject private var relay: RelayStore
    @StateObject private var chat: ChatStore
    @ObservedObject private var local = LocalEngine.shared
    @State private var draft = ""
    @State private var showOnDeviceModels = false

    init(relay: RelayStore) {
        _chat = StateObject(wrappedValue: ChatStore(relay: relay))
    }

    private var chatModels: [String] {
        (relay.workerProbe?.installedModelIDs ?? []).filter {
            $0.hasPrefix("text-chat-") || $0.hasPrefix("text-agent-")
        }
    }

    private var fleetOffersChat: Bool {
        relay.workerProbe.map { $0.nodeKinds.contains("text.generate") } ?? true
    }

    private var localChatReady: Bool {
        local.state(of: local.selectedChatModelID) == .ready
    }

    private var canSend: Bool {
        chat.runLocally ? localChatReady : fleetOffersChat
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: MereTheme.Spacing.m) {
                            if chat.messages.isEmpty {
                                emptyState
                            }
                            ForEach(chat.messages) { message in
                                bubble(message)
                                    .id(message.id)
                            }
                            if chat.awaitingReply {
                                if let partial = chat.streamingReply {
                                    HStack {
                                        Text(partial)
                                            .font(.body)
                                            .foregroundStyle(MereTheme.textPrimary)
                                            .padding(MereTheme.Spacing.m)
                                            .background(
                                                RoundedRectangle(cornerRadius: MereTheme.Radius.panel)
                                                    .fill(MereTheme.surface)
                                            )
                                            .overlay(
                                                RoundedRectangle(cornerRadius: MereTheme.Radius.panel)
                                                    .stroke(MereTheme.accent.opacity(0.35), lineWidth: 1)
                                            )
                                        Spacer(minLength: 40)
                                    }
                                    .id("pending")
                                } else {
                                    HStack(spacing: MereTheme.Spacing.s) {
                                        ProgressView()
                                        Text(chat.runLocally
                                            ? (local.chatStatus ?? "Thinking on this iPhone…")
                                            : "Running on your fleet…")
                                            .font(.footnote)
                                            .foregroundStyle(MereTheme.textMuted)
                                    }
                                    .id("pending")
                                }
                            }
                        }
                        .padding(MereTheme.Spacing.l)
                    }
                    .defaultScrollAnchor(.bottom)
                    .onChange(of: chat.messages.count) {
                        if let last = chat.messages.last?.id {
                            withAnimation { proxy.scrollTo(last, anchor: .bottom) }
                        }
                    }
                    .onChange(of: chat.streamingReply) {
                        if chat.streamingReply != nil {
                            proxy.scrollTo("pending", anchor: .bottom)
                        }
                    }
                }

                if !chat.runLocally, !fleetOffersChat {
                    MereBannerView(
                        text: "Your fleet's nodes don't offer text generation yet. Update mere.run on your nodes to add it.",
                        color: MereTheme.caution
                    )
                    .padding(.horizontal, MereTheme.Spacing.l)
                    .padding(.bottom, MereTheme.Spacing.s)
                }


                composer
            }
            .background(MereTheme.background.ignoresSafeArea())
            .navigationTitle("Chat")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("New chat") { chat.newChat() }
                        .disabled(chat.messages.isEmpty || chat.awaitingReply)
                }
            }
            .task {
                await relay.refreshWorkerProbe()
                if chat.runLocally { local.warmChat() }
            }
            .sheet(isPresented: $showOnDeviceModels) { OnDeviceModelsView() }
        }
    }

    /// Fleet model IDs carry their kind as a prefix; the menu already says
    /// what kind this is, so display just the model's own name.
    static func displayName(for id: String) -> String {
        for prefix in ["text-chat-", "text-agent-"] where id.hasPrefix(prefix) {
            return String(id.dropFirst(prefix.count))
        }
        return id
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: MereTheme.Spacing.s) {
            Text("Think it through.")
                .font(.system(.largeTitle, design: .serif))
                .foregroundStyle(MereTheme.textPrimary)
            Text("Your words go to your machines and nowhere else.")
                .font(.callout)
                .foregroundStyle(MereTheme.textSecondary)
        }
        .padding(.top, MereTheme.Spacing.xxl)
    }

    private func bubble(_ message: ChatStore.Message) -> some View {
        HStack {
            if message.role == .user { Spacer(minLength: 40) }
            Text(message.content)
                .font(.body)
                .foregroundStyle(message.failed ? MereTheme.failure : MereTheme.textPrimary)
                .textSelection(.enabled)
                .padding(MereTheme.Spacing.m)
                .background(
                    RoundedRectangle(cornerRadius: MereTheme.Radius.panel)
                        .fill(message.role == .user ? MereTheme.accent.opacity(0.14) : MereTheme.surface)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: MereTheme.Radius.panel)
                        .stroke(
                            message.role == .user
                                ? MereTheme.accent.opacity(0.35)
                                : MereTheme.border.opacity(0.6),
                            lineWidth: 1
                        )
                )
            if message.role == .assistant { Spacer(minLength: 40) }
        }
    }

    private var composer: some View {
        VStack(spacing: MereTheme.Spacing.s) {
            HStack(alignment: .bottom, spacing: MereTheme.Spacing.s) {
                TextField(
                    "",
                    text: $draft,
                    prompt: Text(chat.runLocally ? "Ask this iPhone…" : "Ask your fleet…")
                        .foregroundColor(MereTheme.textMuted),
                    axis: .vertical
                )
                .lineLimit(1...5)
                .font(.body)
                .foregroundStyle(MereTheme.textPrimary)
                .padding(MereTheme.Spacing.m)
                .merePanel()
                .accessibilityIdentifier("chat.composer")
                Button {
                    let text = draft
                    draft = ""
                    Task { await chat.send(text) }
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 30))
                }
                .disabled(
                    draft.trimmingCharacters(in: .whitespaces).isEmpty
                        || chat.awaitingReply
                        || !canSend
                )
                .accessibilityLabel("Send")
            }
            HStack {
                Menu {
                    Section("Your fleet") {
                        Button("Fleet default") {
                            chat.runLocally = false
                            chat.model = ""
                        }
                        ForEach(chatModels, id: \.self) { id in
                            Button(Self.displayName(for: id)) {
                                chat.runLocally = false
                                chat.model = id
                            }
                        }
                    }
                    if LocalEngine.showsOnDeviceUI {
                        Section("This iPhone") {
                            ForEach(LocalEngine.chatModels.filter(\.isCompatible)) { candidate in
                                if local.state(of: candidate.id) == .ready {
                                    Button(candidate.title) {
                                        local.selectedChatModelID = candidate.id
                                        chat.runLocally = true
                                        local.warmChat()
                                    }
                                }
                            }
                            Button("Get on-device models…") {
                                showOnDeviceModels = true
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: chat.runLocally ? "iphone" : "cpu")
                        Text(chat.runLocally
                            ? "\(LocalEngine.model(withID: local.selectedChatModelID)?.title ?? "On-device") — this iPhone"
                            : (chat.model.isEmpty ? "Fleet default" : Self.displayName(for: chat.model)))
                            .lineLimit(1)
                    }
                    .font(.footnote)
                    .foregroundStyle(MereTheme.textSecondary)
                }
                Spacer()
            }
        }
        .padding(.horizontal, MereTheme.Spacing.l)
        .padding(.bottom, MereTheme.Spacing.s)
    }
}
