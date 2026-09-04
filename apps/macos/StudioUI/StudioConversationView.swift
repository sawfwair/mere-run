import StudioKit
import SwiftUI

/// The Converse surface: the thread header (title, model, system prompt) over the transcript,
/// with the readiness state layered the way the media canvas layers it. Chat and Code share
/// it — Code is a preset (the `text code` command and its defaults), not a second surface.
struct StudioConverseView: View {
    let mode: StudioMode
    let item: StudioLibraryItem?
    let liveText: String?
    let isRunning: Bool
    let readiness: ModelReadinessState
    let error: String?
    /// The transcript budget the next turn is trimmed to; the trim banner reports against it.
    let budgetChars: Int
    let modelInventory: [StudioModelInventoryRow]
    /// The model and system prompt the NEXT turn runs with. Changing either here records on
    /// that turn; earlier turns keep what they ran with.
    @Binding var model: String
    @Binding var systemPrompt: String
    let onPullModel: () -> Void
    let onShowDetails: () -> Void
    let onShowModels: () -> Void
    let onCopy: (String) -> Void
    let onRetry: () -> Void
    let onEdit: (UUID) -> Void
    let onBranch: (UUID) -> Void
    let onUseExample: (String) -> Void

    private var hasTurns: Bool {
        !(item?.messages ?? []).isEmpty || isRunning
    }

    private var needsAttention: Bool {
        !isRunning && (readiness.blocksRun || error != nil)
    }

    var body: some View {
        ZStack {
            VStack(spacing: 0) {
                StudioThreadHeader(
                    title: item?.displayTitle ?? "New thread",
                    mode: mode,
                    model: $model,
                    systemPrompt: $systemPrompt,
                    modelInventory: modelInventory,
                    readiness: readiness,
                    onShowModels: onShowModels
                )
                if needsAttention && hasTurns {
                    readinessNotice
                }
                StudioConversationView(
                    item: item,
                    liveText: liveText,
                    isRunning: isRunning,
                    mode: mode,
                    onNewChat: {},
                    onCopy: onCopy,
                    onRetry: onRetry,
                    onEdit: onEdit,
                    onUseExample: onUseExample,
                    onBranch: onBranch,
                    budgetChars: budgetChars
                )
            }

            // A thread that has not started yet gets the full readiness card; one with turns
            // keeps its transcript visible and shows the compact notice above it instead.
            if needsAttention && !hasTurns {
                VStack(spacing: 0) {
                    Spacer(minLength: 0)
                    if error == nil {
                        StudioReadinessCard(
                            readiness: readiness,
                            pullJob: nil,
                            onPullModel: onPullModel,
                            onShowDetails: onShowDetails,
                            onCancelPull: { _ in }
                        )
                    } else {
                        readinessNotice
                    }
                    Spacer(minLength: 0)
                }
                .frame(maxWidth: StudioThreadHeader.maxWidth)
                .padding(.horizontal, 24)
                .transition(.opacity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var readinessNotice: some View {
        HStack(spacing: MereRunTheme.Spacing.sm) {
            Image(systemName: error == nil ? "arrow.down.circle" : "exclamationmark.triangle")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(error == nil ? MereRunTheme.accent : MereRunTheme.red)
            Text(error ?? readiness.message)
                .font(.system(size: 12))
                .foregroundStyle(MereRunTheme.textPrimary)
                .lineLimit(2)
            Spacer(minLength: 8)
            if error == nil, readiness.canPull {
                Button("Get the model", action: onPullModel)
                    .buttonStyle(.mereSecondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                .fill(MereRunTheme.surfaceRaised)
        }
        .frame(maxWidth: 760 - 48)
        .padding(.top, 12)
        .accessibilityElement(children: .combine)
    }
}

/// The thread header: title, the model chip (the same filtered picker as the composer), and the
/// system prompt chip. Both chips edit what the next turn runs with.
struct StudioThreadHeader: View {
    let title: String
    let mode: StudioMode
    @Binding var model: String
    @Binding var systemPrompt: String
    let modelInventory: [StudioModelInventoryRow]
    let readiness: ModelReadinessState
    let onShowModels: () -> Void

    @State private var editingSystemPrompt = false

    static let maxWidth: CGFloat = 760

    private var systemLabel: String {
        systemPrompt.isBlank ? "System: default" : "System: custom"
    }

    var body: some View {
        HStack(spacing: 10) {
            Text(title)
                .font(.system(size: 13.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.textPrimary)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityAddTraits(.isHeader)
            StudioModelChip(
                mode: mode,
                model: $model,
                modelInventory: modelInventory,
                readiness: readiness,
                onShowModels: onShowModels
            )
            systemChip
        }
        .padding(.top, 14)
        .padding(.horizontal, 24)
        .padding(.bottom, 10)
        .frame(maxWidth: Self.maxWidth)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(MereRunTheme.border.opacity(0.4))
                .frame(height: 1)
        }
        .frame(maxWidth: .infinity)
    }

    private var systemChip: some View {
        Button {
            editingSystemPrompt = true
        } label: {
            StudioComposerChipLabel(title: systemLabel)
        }
        .buttonStyle(.plain)
        .help(systemPrompt.isBlank ? "System prompt: the command's default" : "System prompt: \(systemPrompt)")
        .accessibilityLabel("System prompt")
        .accessibilityValue(systemPrompt.isBlank ? "Default" : systemPrompt)
        .popover(isPresented: $editingSystemPrompt, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                Text("System prompt")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textPrimary)
                TextEditor(text: $systemPrompt)
                    .font(.system(size: 13))
                    .scrollContentBackground(.hidden)
                    .padding(6)
                    .frame(width: 340, height: 140)
                    .merePanel(cornerRadius: MereRunTheme.Radius.sm)
                    .accessibilityLabel("System prompt")
                HStack {
                    Button("Use default") { systemPrompt = "" }
                        .buttonStyle(.mereSecondary)
                        .disabled(systemPrompt.isBlank)
                    Spacer(minLength: 8)
                    Text("Applies from the next turn")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            .padding(MereRunTheme.Spacing.md)
            .background(MereRunTheme.background)
            .foregroundStyle(MereRunTheme.textPrimary)
        }
    }
}

/// The transcript: user turns as warm bubbles on the right, assistant turns as unboxed Markdown
/// behind a chat glyph on the left, a live streaming turn while a reply is in flight, all
/// bottom-aligned in a 760pt column. The composer lives in the shared prompt bar below.
struct StudioConversationView: View {
    let item: StudioLibraryItem?
    let liveText: String?
    let isRunning: Bool
    let mode: StudioMode
    /// Unused by the Converse surface (the thread list owns "new thread"); kept for the
    /// canvas call site until the Main board drops its conversation branch.
    let onNewChat: () -> Void
    let onCopy: (String) -> Void
    let onRetry: () -> Void
    let onEdit: (UUID) -> Void
    var onUseExample: ((String) -> Void)?
    /// Branches a new thread at a turn: before a user turn (with that turn loaded for editing),
    /// after an assistant turn.
    var onBranch: ((UUID) -> Void)?
    var budgetChars: Int = ConversationTranscript.defaultBudgetChars

    private static let streamingBubbleID = "studio.conversation.streaming"
    static let columnWidth: CGFloat = 760

    private var messages: [StudioMessage] { item?.messages ?? [] }

    /// How many earlier turns the next prompt would drop to fit the budget — surfaced so the
    /// trimming is never silent.
    private var droppedFromContext: Int {
        guard !messages.isEmpty else { return 0 }
        return ConversationTranscript.render(
            messages: messages,
            systemPrompt: item?.systemPrompt,
            budgetChars: budgetChars
        ).droppedCount
    }

    var body: some View {
        VStack(spacing: 0) {
            if droppedFromContext > 0 { contextTrimBanner }
            content
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var contextTrimBanner: some View {
        MereBanner(
            severity: .info,
            text: "Earlier messages are trimmed from the next prompt to fit the context window (\(droppedFromContext) omitted).",
            systemImage: "scissors"
        )
        .frame(maxWidth: Self.columnWidth - 48)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private var content: some View {
        if messages.isEmpty && !isRunning {
            StudioEmptyState(mode: mode, onUseExample: onUseExample, onAttach: nil)
        } else {
            ScrollViewReader { proxy in
                GeometryReader { geometry in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 20) {
                            ForEach(messages) { message in
                                turn(for: message)
                                    .id(message.id)
                            }
                            if isRunning {
                                StudioTurnView(
                                    role: .assistant,
                                    content: liveText ?? "",
                                    isStreaming: true
                                )
                                .id(Self.streamingBubbleID)
                            }
                        }
                        .padding(EdgeInsets(top: 22, leading: 24, bottom: 8, trailing: 24))
                        .frame(maxWidth: Self.columnWidth)
                        .frame(maxWidth: .infinity)
                        .frame(minHeight: geometry.size.height, alignment: .bottom)
                    }
                }
                .onChange(of: item?.id) { _, _ in scrollToEnd(proxy) }
                .onChange(of: messages.count) { _, _ in scrollToEnd(proxy) }
                .onChange(of: liveText) { _, _ in scrollToEnd(proxy) }
                .onChange(of: isRunning) { _, _ in scrollToEnd(proxy) }
                .onAppear { scrollToEnd(proxy) }
            }
        }
    }

    private func turn(for message: StudioMessage) -> some View {
        StudioTurnView(
            role: message.role,
            content: message.content,
            failed: message.failed,
            meta: message.role == .assistant ? meta(for: message) : nil,
            onCopy: message.content.isEmpty ? nil : { onCopy(message.content) },
            onRetry: retryAction(for: message),
            onEdit: editAction(for: message),
            onBranch: branchAction(for: message),
            actionsEnabled: !isRunning
        )
    }

    /// "Qwen3.6 4B · 41 tok/s · 1:20 PM": the model the turn ran on (the thread's when the turn
    /// predates per-turn recording), its decode speed when the run reported one, and its time.
    private func meta(for message: StudioMessage) -> String {
        var parts: [String] = []
        let modelID = message.model ?? item?.model ?? ""
        if !modelID.isBlank {
            parts.append(StudioModelNaming.displayName(modelID))
        }
        if let tokensPerSecond = message.tokensPerSecond, tokensPerSecond > 0 {
            parts.append("\(Int(tokensPerSecond.rounded())) tok/s")
        }
        parts.append(Self.timeFormatter.string(from: message.createdAt))
        return parts.joined(separator: " · ")
    }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        return formatter
    }()

    /// Retry is offered on the latest assistant turn (it re-runs the thread's last user turn);
    /// the row disables it while a reply is in flight.
    private func retryAction(for message: StudioMessage) -> (() -> Void)? {
        guard message.role == .assistant,
              message.id == messages.last(where: { $0.role == .assistant })?.id else { return nil }
        return onRetry
    }

    /// Edit is offered on user turns when idle (truncates the thread back to that turn).
    private func editAction(for message: StudioMessage) -> (() -> Void)? {
        guard !isRunning, message.role == .user else { return nil }
        return { onEdit(message.id) }
    }

    /// Branch is always in the row (disabled while a reply streams) so a turn's actions never
    /// shift as the thread runs.
    private func branchAction(for message: StudioMessage) -> (() -> Void)? {
        guard let onBranch else { return nil }
        return { onBranch(message.id) }
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
/// assistant turns read as the document itself (unboxed Markdown behind the chat glyph) with a
/// row of actions and the turn's provenance underneath.
private struct StudioTurnView: View {
    let role: StudioMessageRole
    let content: String
    var failed = false
    var isStreaming = false
    /// "Model · speed · time" under an assistant turn.
    var meta: String?
    var onCopy: (() -> Void)?
    var onRetry: (() -> Void)?
    var onEdit: (() -> Void)?
    var onBranch: (() -> Void)?
    /// False while a reply streams: Retry and Branch stay in the row but cannot fire.
    var actionsEnabled = true

    @State private var hovering = false

    private var isUser: Bool { role == .user }

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if isUser {
                Spacer(minLength: 0)
                userBubble
            } else {
                assistantBlock
                Spacer(minLength: 0)
            }
        }
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .contextMenu { contextMenuItems }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityText)
        .accessibilityAddTraits(isStreaming ? .updatesFrequently : [])
    }

    // MARK: User

    private var userBubble: some View {
        Text(content)
            .font(.system(size: 14))
            .lineSpacing(3)
            .foregroundStyle(MereRunTheme.textPrimary)
            .textSelection(.enabled)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background {
                bubbleShape.fill(MereRunTheme.accentSoft)
            }
            .overlay {
                if failed {
                    bubbleShape.strokeBorder(MereRunTheme.red.opacity(0.6), lineWidth: 1)
                }
            }
            .frame(maxWidth: 520, alignment: .trailing)
            // The hover actions sit in the 20pt gap under the bubble rather than reserving
            // their own row, so the turn rhythm stays exactly as designed.
            .overlay(alignment: .bottomTrailing) {
                userActions.offset(y: 20)
            }
    }

    private var bubbleShape: UnevenRoundedRectangle {
        UnevenRoundedRectangle(
            topLeadingRadius: 14,
            bottomLeadingRadius: 14,
            bottomTrailingRadius: 4,
            topTrailingRadius: 14
        )
    }

    /// Edit and Branch surface on hover; the row keeps its height so nothing jumps.
    @ViewBuilder
    private var userActions: some View {
        if onEdit != nil || onBranch != nil {
            HStack(spacing: 2) {
                if let onEdit {
                    labeledAction("Edit", systemImage: "pencil", help: "Edit and re-run from this turn", action: onEdit)
                }
                if let onBranch {
                    labeledAction(
                        "Branch",
                        systemImage: "arrow.triangle.branch",
                        help: "Start a new thread from this turn",
                        action: onBranch
                    )
                    .disabled(!actionsEnabled)
                }
            }
            .opacity(hovering ? 1 : 0)
            .allowsHitTesting(hovering)
            .frame(height: 18)
        }
    }

    private func labeledAction(
        _ title: String,
        systemImage: String,
        help: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(MereRunTheme.captionFont)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
        }
        .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
        .help(help)
    }

    // MARK: Assistant

    private var assistantBlock: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "bubble.left")
                .font(.system(size: 14, weight: .regular))
                .foregroundStyle(MereRunTheme.accent)
                .frame(width: 16, height: 16)
                .padding(.top, 3)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 10) {
                assistantBody

                if failed {
                    Label("This turn failed", systemImage: "exclamationmark.triangle")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.red)
                }

                if !isStreaming { assistantActions }
            }
        }
        .frame(maxWidth: 640, alignment: .leading)
    }

    @ViewBuilder
    private var assistantBody: some View {
        if isStreaming && content.isEmpty {
            StudioThinkingIndicator()
        } else {
            StudioMarkdownText(
                content: content,
                bodyFont: .system(size: 14),
                lineSpacing: 4.5,
                streamingCaret: isStreaming
            )
        }
    }

    /// Copy, Retry, Branch as 28pt icon buttons, then the turn's provenance.
    private var assistantActions: some View {
        HStack(spacing: 2) {
            if let onCopy {
                iconAction("Copy message", systemImage: "doc.on.doc", action: onCopy)
            }
            if let onRetry {
                iconAction("Regenerate this reply", systemImage: "arrow.clockwise", action: onRetry)
                    .disabled(!actionsEnabled)
            }
            if let onBranch {
                iconAction("Branch a new thread from here", systemImage: "arrow.triangle.branch", action: onBranch)
                    .disabled(!actionsEnabled)
            }
            if let meta {
                Text(meta)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                    .padding(.leading, 6)
            }
        }
    }

    private func iconAction(_ label: String, systemImage: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 13, weight: .medium))
                .frame(width: 28, height: 28)
        }
        .buttonStyle(.mereIcon)
        .help(label)
        .accessibilityLabel(label)
    }

    @ViewBuilder
    private var contextMenuItems: some View {
        if let onCopy { Button("Copy") { onCopy() } }
        if let onEdit { Button("Edit…") { onEdit() } }
        if let onRetry, actionsEnabled { Button("Retry") { onRetry() } }
        if let onBranch, actionsEnabled { Button("Branch from here") { onBranch() } }
    }

    private var accessibilityText: String {
        let speaker = isUser ? "You" : "Assistant"
        if isStreaming && content.isEmpty { return "\(speaker) is generating a reply" }
        let suffix = failed ? " (this turn failed)" : ""
        let provenance = meta.map { ", \($0)" } ?? ""
        return "\(speaker): \(content)\(suffix)\(provenance)"
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
