import SwiftUI

/// The chat/code canvas: user turns as warm bubbles on the right, assistant turns as unboxed
/// Markdown on the left, a live streaming turn while a reply is in flight. The composer lives
/// in the shared prompt bar below; this view renders the thread and a "New chat" affordance.
struct StudioConversationView: View {
    let item: StudioLibraryItem?
    let liveText: String?
    let isRunning: Bool
    let mode: StudioMode
    let onNewChat: () -> Void
    let onCopy: (String) -> Void
    let onRetry: () -> Void
    let onEdit: (UUID) -> Void
    var onUseExample: ((String) -> Void)?

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
            // A blank new chat shows only its empty state — no header, so the
            // thread title and the "New chat" action never read as duplicates.
            if hasActiveConversation {
                header
                Divider().overlay(MereRunTheme.border.opacity(0.4))
            }
            if droppedFromContext > 0 { contextTrimBanner }
            content
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var hasActiveConversation: Bool {
        item != nil || !messages.isEmpty || isRunning
    }

    private var contextTrimBanner: some View {
        MereBanner(
            severity: .info,
            text: "Earlier messages are trimmed from the next prompt to fit the context window (\(droppedFromContext) omitted).",
            systemImage: "scissors"
        )
        .padding(.horizontal, 24)
        .padding(.vertical, 8)
    }

    private var header: some View {
        HStack(spacing: MereRunTheme.Spacing.sm) {
            Text(item?.displayTitle ?? "Untitled chat")
                .font(MereRunTheme.sectionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer(minLength: 12)
            Button(action: onNewChat) {
                Label("New chat", systemImage: "square.and.pencil")
                    .font(MereRunTheme.captionFont)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
            }
            .buttonStyle(.mereIcon(tint: MereRunTheme.accent))
            .help("Start a new conversation")
            .accessibilityLabel("Start a new conversation")
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 10)
    }

    @ViewBuilder
    private var content: some View {
        if messages.isEmpty && !isRunning {
            StudioConversationEmptyState(mode: mode, onUseExample: onUseExample)
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                        ForEach(messages) { message in
                            StudioTurnView(
                                role: message.role,
                                content: message.content,
                                failed: message.failed,
                                monospaced: mode == .code && message.role == .assistant,
                                modeIcon: mode.systemImage,
                                onCopy: message.content.isEmpty ? nil : { onCopy(message.content) },
                                onRetry: retryAction(for: message),
                                onEdit: editAction(for: message)
                            )
                            .id(message.id)
                        }
                        if isRunning {
                            StudioTurnView(
                                role: .assistant,
                                content: liveText ?? "",
                                failed: false,
                                isStreaming: true,
                                monospaced: mode == .code,
                                modeIcon: mode.systemImage
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
        withAnimation(MereRunTheme.Motion.quick) {
            if isRunning {
                proxy.scrollTo(Self.streamingBubbleID, anchor: .bottom)
            } else if let last = messages.last {
                proxy.scrollTo(last.id, anchor: .bottom)
            }
        }
    }
}

/// One conversation turn. User turns read as authored notes (warm bubble, right side);
/// assistant turns read as the document itself (unboxed Markdown behind a small mode glyph).
private struct StudioTurnView: View {
    let role: StudioMessageRole
    let content: String
    var failed: Bool = false
    var isStreaming: Bool = false
    var monospaced: Bool = false
    var modeIcon: String = "bubble.left.and.bubble.right"
    var onCopy: (() -> Void)?
    var onRetry: (() -> Void)?
    var onEdit: (() -> Void)?

    @State private var hovering = false

    private var isUser: Bool { role == .user }
    private var hasActions: Bool { onCopy != nil || onRetry != nil || onEdit != nil }

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if isUser {
                Spacer(minLength: 80)
                userBubble
            } else {
                assistantBlock
                Spacer(minLength: 80)
            }
        }
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .contextMenu { contextMenuItems }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityText)
        .accessibilityAddTraits(isStreaming ? .updatesFrequently : [])
    }

    private var userBubble: some View {
        VStack(alignment: .trailing, spacing: 4) {
            Text(content)
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textPrimary)
                .textSelection(.enabled)
                .padding(.horizontal, MereRunTheme.Spacing.md)
                .padding(.vertical, MereRunTheme.Spacing.sm)
                .background {
                    UnevenRoundedRectangle(
                        topLeadingRadius: 16,
                        bottomLeadingRadius: 16,
                        bottomTrailingRadius: 5,
                        topTrailingRadius: 16
                    )
                    .fill(MereRunTheme.accentSoft)
                    .overlay {
                        UnevenRoundedRectangle(
                            topLeadingRadius: 16,
                            bottomLeadingRadius: 16,
                            bottomTrailingRadius: 5,
                            topTrailingRadius: 16
                        )
                        .strokeBorder(
                            failed ? MereRunTheme.red.opacity(0.6) : MereRunTheme.accent.opacity(0.14),
                            lineWidth: 1
                        )
                    }
                }
                .frame(maxWidth: 520, alignment: .trailing)

            actionRow
        }
    }

    private var assistantBlock: some View {
        HStack(alignment: .top, spacing: MereRunTheme.Spacing.sm) {
            Image(systemName: modeIcon)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
                .frame(width: 26, height: 26)
                .background {
                    Circle().fill(MereRunTheme.accentSoft.opacity(0.7))
                }
                .padding(.top, 1)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 6) {
                assistantBody

                if failed {
                    Label("This turn failed", systemImage: "exclamationmark.triangle")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.red)
                }

                actionRow
            }
            .frame(maxWidth: 660, alignment: .leading)
        }
    }

    @ViewBuilder
    private var assistantBody: some View {
        if isStreaming && content.isEmpty {
            StudioThinkingIndicator()
        } else {
            VStack(alignment: .leading, spacing: 6) {
                StudioMarkdownText(
                    content: content,
                    bodyFont: monospaced ? MereRunTheme.monoFont : MereRunTheme.bodyFont
                )
                if isStreaming {
                    StudioStreamingCaret()
                }
            }
        }
    }

    /// Copy / edit / retry surface on hover; the row keeps its height so nothing jumps.
    @ViewBuilder
    private var actionRow: some View {
        if !isStreaming && hasActions {
            HStack(spacing: 2) {
                if let onCopy {
                    turnAction("Copy", systemImage: "doc.on.doc", help: "Copy message", action: onCopy)
                }
                if let onEdit {
                    turnAction("Edit", systemImage: "pencil", help: "Edit and re-run from this turn", action: onEdit)
                }
                if let onRetry {
                    turnAction("Retry", systemImage: "arrow.clockwise", help: "Regenerate this reply", action: onRetry)
                }
            }
            .opacity(hovering ? 1 : 0)
            .allowsHitTesting(hovering)
            .frame(height: 22)
        }
    }

    private func turnAction(
        _ title: String,
        systemImage: String,
        help: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(MereRunTheme.captionFont)
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
        }
        .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
        .help(help)
    }

    @ViewBuilder
    private var contextMenuItems: some View {
        if let onCopy { Button("Copy") { onCopy() } }
        if let onEdit { Button("Edit…") { onEdit() } }
        if let onRetry { Button("Retry") { onRetry() } }
    }

    private var accessibilityText: String {
        let speaker = isUser ? "You" : "Assistant"
        if isStreaming && content.isEmpty { return "\(speaker) is generating a reply" }
        let suffix = failed ? " (this turn failed)" : ""
        return "\(speaker): \(content)\(suffix)"
    }
}

/// The live-reply cursor: a small bronze bar breathing at the end of the streamed text.
private struct StudioStreamingCaret: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 1.5)
            .fill(MereRunTheme.accent)
            .frame(width: 8, height: 15)
            .phaseAnimator([0.25, 1.0]) { view, phase in
                view.opacity(phase)
            } animation: { _ in
                .easeInOut(duration: 0.55)
            }
            .accessibilityHidden(true)
    }
}

/// Three quiet dots taking turns while the model decides what to say.
private struct StudioThinkingIndicator: View {
    var body: some View {
        HStack(spacing: 8) {
            HStack(spacing: 4) {
                ForEach(0..<3, id: \.self) { index in
                    Circle()
                        .fill(MereRunTheme.accent)
                        .frame(width: 6, height: 6)
                        .phaseAnimator([0, 1, 2]) { view, phase in
                            view.opacity(phase == Double(index) ? 1 : 0.28)
                        } animation: { _ in
                            .easeInOut(duration: 0.38)
                        }
                }
            }
            Text("Thinking…")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
        .padding(.vertical, 4)
        .accessibilityLabel("Generating a reply")
    }
}

private struct StudioConversationEmptyState: View {
    let mode: StudioMode
    var onUseExample: ((String) -> Void)?

    var body: some View {
        StudioEmptyState(mode: mode, onUseExample: onUseExample, onAttach: nil)
    }
}
