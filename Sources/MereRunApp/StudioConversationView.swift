import SwiftUI

/// The chat/code canvas: a scrolling transcript of message bubbles for the active conversation,
/// plus a live streaming bubble while a turn is in flight. The composer lives in the shared
/// prompt bar below; this view only renders the thread and a "New chat" affordance.
struct StudioConversationView: View {
    let item: StudioLibraryItem?
    let liveText: String?
    let isRunning: Bool
    let mode: StudioMode
    let onNewChat: () -> Void
    let onCopy: (String) -> Void
    let onRetry: () -> Void
    let onEdit: (UUID) -> Void

    private static let streamingBubbleID = "studio.conversation.streaming"

    private var messages: [StudioMessage] { item?.messages ?? [] }

    /// How many earlier turns the next prompt would drop to fit the budget — surfaced so the
    /// trimming is never silent.
    private var droppedFromContext: Int {
        guard !messages.isEmpty else { return 0 }
        return ConversationTranscript.render(messages: messages, systemPrompt: item?.systemPrompt).droppedCount
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.4))
            if droppedFromContext > 0 { contextTrimBanner }
            content
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var contextTrimBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "scissors")
            Text("Earlier messages are trimmed from the next prompt to fit the context window (\(droppedFromContext) omitted).")
            Spacer(minLength: 0)
        }
        .font(MereRunTheme.captionFont)
        .foregroundStyle(MereRunTheme.textMuted)
        .padding(.horizontal, 24)
        .padding(.vertical, 8)
        .background(MereRunTheme.surface.opacity(0.5))
        .accessibilityElement(children: .combine)
    }

    private var header: some View {
        HStack(spacing: 10) {
            Text(item?.displayTitle ?? "New chat")
                .font(MereRunTheme.sectionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer(minLength: 12)
            Button(action: onNewChat) {
                Label("New chat", systemImage: "square.and.pencil")
                    .font(MereRunTheme.captionFont)
            }
            .buttonStyle(.plain)
            .foregroundStyle(MereRunTheme.accent)
            .help("Start a new conversation")
            .accessibilityLabel("Start a new conversation")
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 14)
    }

    @ViewBuilder
    private var content: some View {
        if messages.isEmpty && !isRunning {
            StudioConversationEmptyState()
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 16) {
                        ForEach(messages) { message in
                            StudioMessageBubble(
                                role: message.role,
                                content: message.content,
                                failed: message.failed,
                                monospaced: mode == .code && message.role == .assistant,
                                onCopy: message.content.isEmpty ? nil : { onCopy(message.content) },
                                onRetry: retryAction(for: message),
                                onEdit: editAction(for: message)
                            )
                            .id(message.id)
                        }
                        if isRunning {
                            StudioMessageBubble(
                                role: .assistant,
                                content: liveText ?? "",
                                failed: false,
                                isStreaming: true,
                                monospaced: mode == .code
                            )
                            .id(Self.streamingBubbleID)
                        }
                    }
                    .padding(24)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .onChange(of: item?.id) { _, _ in scrollToEnd(proxy) }
                .onChange(of: messages.count) { _, _ in scrollToEnd(proxy) }
                .onChange(of: liveText) { _, _ in scrollToEnd(proxy) }
                .onChange(of: isRunning) { _, _ in scrollToEnd(proxy) }
                .onAppear { scrollToEnd(proxy) }
            }
        }
    }

    /// Retry is offered only on the final assistant turn, and only when idle.
    private func retryAction(for message: StudioMessage) -> (() -> Void)? {
        guard !isRunning, message.role == .assistant, message.id == messages.last?.id else { return nil }
        return onRetry
    }

    /// Edit is offered on user turns when idle (truncates the thread back to that turn).
    private func editAction(for message: StudioMessage) -> (() -> Void)? {
        guard !isRunning, message.role == .user else { return nil }
        return { onEdit(message.id) }
    }

    private func scrollToEnd(_ proxy: ScrollViewProxy) {
        withAnimation(.easeOut(duration: 0.15)) {
            if isRunning {
                proxy.scrollTo(Self.streamingBubbleID, anchor: .bottom)
            } else if let last = messages.last {
                proxy.scrollTo(last.id, anchor: .bottom)
            }
        }
    }
}

private struct StudioMessageBubble: View {
    let role: StudioMessageRole
    let content: String
    var failed: Bool = false
    var isStreaming: Bool = false
    var monospaced: Bool = false
    var onCopy: (() -> Void)?
    var onRetry: (() -> Void)?
    var onEdit: (() -> Void)?

    private var isUser: Bool { role == .user }

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if isUser { Spacer(minLength: 64) }
            VStack(alignment: .leading, spacing: 6) {
                Text(isUser ? "You" : "Assistant")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                bubbleBody
                if failed {
                    Label("This turn failed", systemImage: "exclamationmark.triangle")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.red)
                }
                if !isStreaming, onCopy != nil || onRetry != nil || onEdit != nil {
                    HStack(spacing: 14) {
                        if let onCopy {
                            Button(action: onCopy) { Label("Copy", systemImage: "doc.on.doc") }
                                .buttonStyle(.plain)
                                .help("Copy message")
                        }
                        if let onEdit {
                            Button(action: onEdit) { Label("Edit", systemImage: "pencil") }
                                .buttonStyle(.plain)
                                .help("Edit and re-run from this turn")
                        }
                        if let onRetry {
                            Button(action: onRetry) { Label("Retry", systemImage: "arrow.clockwise") }
                                .buttonStyle(.plain)
                                .help("Regenerate this reply")
                        }
                    }
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .padding(.top, 2)
                }
            }
            .padding(14)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.cornerRadius)
                    .fill(isUser ? MereRunTheme.surfaceRaised : MereRunTheme.surface)
                    .overlay {
                        RoundedRectangle(cornerRadius: MereRunTheme.cornerRadius)
                            .strokeBorder((failed ? MereRunTheme.red : MereRunTheme.border).opacity(0.6), lineWidth: 1)
                    }
            }
            .frame(maxWidth: 620, alignment: .leading)
            if !isUser { Spacer(minLength: 64) }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityText)
        .accessibilityAddTraits(isStreaming ? .updatesFrequently : [])
    }

    private var accessibilityText: String {
        let speaker = isUser ? "You" : "Assistant"
        if isStreaming && content.isEmpty { return "\(speaker) is generating a reply" }
        let suffix = failed ? " (this turn failed)" : ""
        return "\(speaker): \(content)\(suffix)"
    }

    @ViewBuilder
    private var bubbleBody: some View {
        if isStreaming && content.isEmpty {
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Thinking…")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        } else {
            Text(content)
                .font(monospaced ? MereRunTheme.monoFont : MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textPrimary)
                .textSelection(.enabled)
        }
    }
}

private struct StudioConversationEmptyState: View {
    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 40, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
                .frame(width: 84, height: 84)
                .background { Circle().fill(MereRunTheme.surfaceRaised) }
            Text("Start a conversation")
                .font(.system(size: 22, weight: .semibold))
            Text("Type a message below. Replies stay on this Mac, and the whole thread is remembered for follow-up turns.")
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
