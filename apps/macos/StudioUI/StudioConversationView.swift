import StudioKit
import SwiftUI

/// The Converse surface: the thread header (title, model, system prompt) over the transcript,
/// with readiness above the transcript so setup never covers the conversation. Chat and Code share
/// it — Code is a preset (the `text code` command and its defaults), not a second surface.
struct StudioConverseView: View {
    let mode: StudioMode
    let item: StudioLibraryItem?
    /// The reply streaming in, split into its answer and reasoning; nil while no turn runs.
    let liveReply: ConversationTranscript.Reply?
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
    let readinessActions: StudioReadinessActions
    let onShowModels: () -> Void
    let onCopy: (String) -> Void
    @Environment(\.studioModelTitles) private var titles
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
            if needsAttention {
                if hasTurns || error != nil {
                    readinessNotice
                } else {
                    StudioReadinessCard(
                        readiness: readiness,
                        pullJob: nil,
                        actions: readinessActions,
                        onCancelPull: { _ in }
                    )
                    .frame(maxWidth: StudioThreadHeader.maxWidth)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 12)
                }
            }
            StudioConversationView(
                item: item,
                liveReply: liveReply,
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
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var readinessNotice: some View {
        HStack(spacing: MereRunTheme.Spacing.sm) {
            Image(systemName: error == nil ? "arrow.down.circle" : "exclamationmark.triangle")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(error == nil ? MereRunTheme.accent : MereRunTheme.red)
            Text(error ?? readiness.message(titles: titles))
                .font(.system(size: 12))
                .foregroundStyle(MereRunTheme.textPrimary)
                .lineLimit(2)
            Spacer(minLength: 8)
            if error == nil, readiness.canPull {
                Button("Get the model", action: readinessActions.pullModel)
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
    @Environment(\.studioScopeSource) private var scopeSource
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
                scope: StudioModelScope(mode: mode, source: scopeSource),
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
///
/// New output is followed only while the reader is at the bottom. Scrolling up stops the
/// following and shows a "Jump to latest" pill; scrolling back down, or the pill, resumes it.
struct StudioConversationView: View {
    let item: StudioLibraryItem?
    /// The reply streaming in, split into its answer and reasoning; nil while no turn runs.
    let liveReply: ConversationTranscript.Reply?
    let isRunning: Bool
    let mode: StudioMode
    @Environment(\.studioModelTitles) private var titles
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

    /// Whether the transcript scrolls to keep new output in view. True until the reader scrolls
    /// away from the bottom; true again once they are back there.
    @State private var followsLatest = true
    /// Bookkeeping that never drives a render, so writing it never re-evaluates the transcript.
    @State private var scratch = TranscriptScratch()

    private static let streamingBubbleID = "studio.conversation.streaming"
    private static let interruptedRowID = "studio.conversation.interrupted"
    static let columnWidth: CGFloat = 760
    /// How far above the end the transcript can rest and still count as at the bottom.
    private static let bottomTolerance: CGFloat = 32

    private var messages: [StudioMessage] { item?.messages ?? [] }

    /// A thread whose reply never arrived because Studio closed mid-turn: the last turn is the
    /// person's, nothing runs, and the row was reconciled to interrupted on launch.
    private var awaitsInterruptedReply: Bool {
        !isRunning && item?.status == .interrupted && messages.last?.role == .user
    }

    /// How many earlier turns the next prompt would drop to fit the budget — surfaced so the
    /// trimming is never silent. Rendering the whole thread is not free, so the count is kept
    /// until the thread, its length, or the budget changes.
    private var droppedFromContext: Int {
        guard !messages.isEmpty else { return 0 }
        let key = TranscriptScratch.TrimKey(
            item: item?.id, count: messages.count, last: messages.last?.id,
            systemPromptLength: item?.systemPrompt?.count ?? 0, budget: budgetChars
        )
        if scratch.trimKey != key {
            scratch.trimKey = key
            scratch.trimDropped = ConversationTranscript.render(
                messages: messages,
                systemPrompt: item?.systemPrompt,
                budgetChars: budgetChars
            ).droppedCount
        }
        return scratch.trimDropped
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
                                    content: liveReply?.answer ?? "",
                                    reasoning: liveReply?.reasoning,
                                    isThinking: liveReply?.isThinking == true,
                                    isStreaming: true
                                )
                                .id(Self.streamingBubbleID)
                            } else if awaitsInterruptedReply {
                                StudioTurnView(
                                    role: .assistant,
                                    content: "",
                                    failed: true,
                                    failureReason: "Interrupted when Studio closed.",
                                    onRetry: onRetry
                                )
                                .id(Self.interruptedRowID)
                            }
                        }
                        .padding(EdgeInsets(top: 22, leading: 24, bottom: 8, trailing: 24))
                        .frame(maxWidth: Self.columnWidth)
                        .frame(maxWidth: .infinity)
                        .frame(minHeight: geometry.size.height, alignment: .bottom)
                    }
                    .onScrollPhaseChange { previous, phase in
                        // A scroll the reader made has come to rest: follow again only if it
                        // ended at the bottom. A programmatic scroll settling never changes the
                        // decision, and it clears its mark only once it has settled.
                        let readerScrollEnded = phase == .idle && isReaderDriven(previous)
                        if phase == .idle { scratch.programmaticScroll = false }
                        scratch.readerScrolling = isReaderDriven(phase)
                        if readerScrollEnded {
                            setFollowsLatest(scratch.distanceFromBottom <= Self.bottomTolerance)
                        }
                    }
                    .onScrollGeometryChange(for: CGFloat.self) { scroll in
                        scroll.contentSize.height - scroll.visibleRect.maxY
                    } action: { _, distance in
                        scratch.distanceFromBottom = distance
                        if scratch.readerScrolling { setFollowsLatest(distance <= Self.bottomTolerance) }
                    }
                    .overlay(alignment: .bottom) {
                        if !followsLatest { jumpToLatest(proxy) }
                    }
                    .animation(MereRunTheme.Motion.quick, value: followsLatest)
                }
                .onChange(of: item?.id) { _, _ in
                    followsLatest = true
                    scrollToEnd(proxy)
                }
                .onChange(of: messages.count) { _, _ in
                    // A turn the reader just sent always comes into view; a reply landing while
                    // they read earlier turns does not pull them away from it.
                    if messages.last?.role == .user { followsLatest = true }
                    scrollToEndIfFollowing(proxy)
                }
                .onChange(of: liveReply) { _, _ in scrollToEndIfFollowing(proxy) }
                .onChange(of: isRunning) { _, _ in scrollToEndIfFollowing(proxy) }
                .onAppear {
                    followsLatest = true
                    scrollToEnd(proxy)
                }
            }
        }
    }

    private func turn(for message: StudioMessage) -> some View {
        StudioTurnView(
            role: message.role,
            content: message.content,
            reasoning: message.reasoning,
            failed: message.failed,
            cancelled: message.cancelled == true,
            failureReason: message.failureReason,
            logTail: message.logTail ?? [],
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
            parts.append(StudioModelNaming.displayName(modelID, titles: titles))
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

    /// The pill that takes a reader who scrolled up back to the newest output and resumes
    /// following it.
    private func jumpToLatest(_ proxy: ScrollViewProxy) -> some View {
        Button {
            followsLatest = true
            scrollToEnd(proxy)
        } label: {
            Label("Jump to latest", systemImage: "arrow.down")
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(MereRunTheme.textPrimary)
                .padding(.horizontal, 12)
                .frame(height: 28)
                .background {
                    Capsule()
                        .fill(MereRunTheme.surface)
                        .overlay {
                            Capsule().strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                        }
                }
                .mereShadow(radius: 8, y: 2)
        }
        .buttonStyle(.plain)
        .help("Scroll to the newest message")
        .accessibilityLabel("Jump to latest")
        .padding(.bottom, 12)
        .transition(.opacity.combined(with: .move(edge: .bottom)))
    }

    /// The phases in which the reader, not the transcript, is moving the scroll position. The
    /// keyboard (Page Down, arrows, End) scrolls in `.animating`, the same phase as
    /// `scrollToEnd`, so that phase counts as the reader's unless a `scrollToEnd` is in flight.
    private func isReaderDriven(_ phase: ScrollPhase) -> Bool {
        phase == .tracking || phase == .interacting || phase == .decelerating
            || (phase == .animating && !scratch.programmaticScroll)
    }

    private func setFollowsLatest(_ follows: Bool) {
        if followsLatest != follows { followsLatest = follows }
    }

    private func scrollToEndIfFollowing(_ proxy: ScrollViewProxy) {
        guard followsLatest else { return }
        scrollToEnd(proxy)
    }

    private func scrollToEnd(_ proxy: ScrollViewProxy) {
        // Only a scroll with somewhere to go marks itself: one that has nothing to move never
        // reaches `.idle` to clear the mark.
        if scratch.distanceFromBottom > 1 { scratch.programmaticScroll = true }
        withAnimation(MereRunTheme.Motion.quick) {
            if isRunning {
                proxy.scrollTo(Self.streamingBubbleID, anchor: .bottom)
            } else if awaitsInterruptedReply {
                proxy.scrollTo(Self.interruptedRowID, anchor: .bottom)
            } else if let last = messages.last {
                proxy.scrollTo(last.id, anchor: .bottom)
            }
        }
    }
}

/// The transcript's per-frame scroll figures and its context-trim memo. A class held in
/// `@State` so writing it, which happens on every scrolled frame, never invalidates the view.
private final class TranscriptScratch {
    struct TrimKey: Equatable {
        let item: UUID?
        let count: Int
        let last: UUID?
        let systemPromptLength: Int
        let budget: Int
    }

    var distanceFromBottom: CGFloat = 0
    /// True from the moment the reader starts scrolling until that scroll comes to rest, so an
    /// offset change caused by content landing never counts as the reader leaving the bottom.
    var readerScrolling = false
    /// True from a `scrollToEnd` until the scroll it started settles.
    var programmaticScroll = false
    var trimKey: TrimKey?
    var trimDropped = 0
}

/// One conversation turn. User turns read as authored notes (warm bubble, right side);
/// assistant turns read as the document itself (unboxed Markdown behind the chat glyph) with a
/// row of actions and the turn's provenance underneath. An assistant turn that thought first
/// carries its reasoning in a collapsed "Thinking" disclosure above the answer; one that failed
/// says why on one line, with the run's log behind "Show log".
private struct StudioTurnView: View {
    let role: StudioMessageRole
    let content: String
    /// The model's reasoning, shown collapsed above the answer; nil when the turn had none.
    var reasoning: String?
    /// True while a streaming reply is still inside its reasoning block.
    var isThinking = false
    var failed = false
    var cancelled = false
    var isStreaming = false
    /// Why a failed turn failed, in one line.
    var failureReason: String?
    /// The last lines a failed run wrote, behind "Show log".
    var logTail: [String] = []
    /// "Model · speed · time" under an assistant turn.
    var meta: String?
    var onCopy: (() -> Void)?
    var onRetry: (() -> Void)?
    var onEdit: (() -> Void)?
    var onBranch: (() -> Void)?
    /// False while a reply streams: Retry and Branch stay in the row but cannot fire.
    var actionsEnabled = true

    @State private var hovering = false
    @State private var showsReasoning = false
    @State private var showsLog = false

    private var isUser: Bool { role == .user }
    private var hasReasoning: Bool { isThinking || !(reasoning ?? "").isEmpty }
    /// A stopped reply is not a failure: it keeps its "Reply stopped" note and its regenerate
    /// icon, and never the reason row.
    private var showsFailureRow: Bool { failed && !cancelled }

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
                if hasReasoning { reasoningDisclosure }
                assistantBody

                if cancelled {
                    Label("Reply stopped", systemImage: "stop.circle")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                } else if showsFailureRow {
                    failure
                }

                if !isStreaming { assistantActions }
            }
        }
        .frame(maxWidth: 640, alignment: .leading)
    }

    @ViewBuilder
    private var assistantBody: some View {
        if isStreaming && content.isEmpty {
            // Before the first word: the dots, unless the reasoning header already says the
            // model is thinking.
            if !isThinking { StudioThinkingIndicator() }
        } else if !content.isEmpty {
            StudioMarkdownText(
                content: content,
                bodyFont: .system(size: 14),
                lineSpacing: 4.5,
                streamingCaret: isStreaming
            )
        }
    }

    /// "Thinking" above the answer: collapsed by default, live (with the dots) while the model
    /// is still inside its reasoning block. The text is plain, not Markdown, and reads quieter
    /// than the answer.
    private var reasoningDisclosure: some View {
        VStack(alignment: .leading, spacing: 8) {
            disclosureToggle(
                isThinking ? "Thinking…" : "Thinking",
                isExpanded: $showsReasoning,
                accessibilityLabel: isThinking ? "Thinking, in progress" : "Thinking",
                hint: "the model's reasoning",
                pulsing: isThinking
            )
            if showsReasoning, let reasoning, !reasoning.isEmpty {
                Text(reasoning)
                    .font(.system(size: 12.5))
                    .lineSpacing(3.5)
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, 12)
                    .overlay(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 1)
                            .fill(MereRunTheme.border)
                            .frame(width: 2)
                    }
            }
        }
    }

    /// Why the turn failed, Retry when this is the turn Retry acts on, and the run's log behind
    /// "Show log".
    private var failure: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Label {
                    Text(failureReason ?? "This turn failed")
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.red)
                .accessibilityLabel("Failed: \(failureReason ?? "this turn failed")")
                if let onRetry {
                    Button("Retry", action: onRetry)
                        .buttonStyle(.mereSecondary)
                        .disabled(!actionsEnabled)
                        .help("Run this turn again")
                }
            }
            if !logTail.isEmpty {
                disclosureToggle(
                    showsLog ? "Hide log" : "Show log",
                    isExpanded: $showsLog,
                    accessibilityLabel: showsLog ? "Hide log" : "Show log",
                    hint: "what the run wrote"
                )
                if showsLog {
                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(Array(logTail.enumerated()), id: \.offset) { _, line in
                            Text(line)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(MereRunTheme.textMuted)
                                .textSelection(.enabled)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    .merePanel(cornerRadius: MereRunTheme.Radius.sm)
                }
            }
        }
    }

    /// The chevron-and-caption toggle the failure card uses for its log, shared by the turn's
    /// two disclosures. `pulsing` adds the dots while the model is still thinking.
    private func disclosureToggle(
        _ title: String,
        isExpanded: Binding<Bool>,
        accessibilityLabel: String,
        hint: String,
        pulsing: Bool = false
    ) -> some View {
        Button {
            withAnimation(MereRunTheme.Motion.quick) { isExpanded.wrappedValue.toggle() }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: isExpanded.wrappedValue ? "chevron.down" : "chevron.right")
                    .font(.system(size: 9, weight: .semibold))
                    .frame(width: 9)
                if pulsing { StudioPulsingDots() }
                Text(title)
                    .font(MereRunTheme.captionFont)
            }
            .foregroundStyle(MereRunTheme.textMuted)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityValue(isExpanded.wrappedValue ? "Expanded" : "Collapsed")
        .accessibilityHint(isExpanded.wrappedValue ? "Hides \(hint)" : "Shows \(hint)")
    }

    /// Copy, Retry, Branch as 28pt icon buttons, then the turn's provenance. A failed turn's
    /// Retry sits beside its reason instead.
    private var assistantActions: some View {
        HStack(spacing: 2) {
            if let onCopy {
                iconAction("Copy message", systemImage: "doc.on.doc", action: onCopy)
            }
            if let onRetry, !showsFailureRow {
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

    /// The turn is one VoiceOver element, so what its disclosures show when open is read here.
    private var accessibilityText: String {
        let speaker = isUser ? "You" : "Assistant"
        if isStreaming && content.isEmpty && !(isThinking && showsReasoning) {
            return isThinking ? "\(speaker) is thinking" : "\(speaker) is generating a reply"
        }
        let suffix: String
        if cancelled {
            suffix = " (reply stopped)"
        } else if failed {
            suffix = " (failed: \(failureReason ?? "this turn failed"))"
        } else {
            suffix = ""
        }
        let provenance = meta.map { ", \($0)" } ?? ""
        var text = "\(speaker): \(content)\(suffix)\(provenance)"
        if showsReasoning, let reasoning, !reasoning.isEmpty {
            text += ". Reasoning: \(reasoning)"
        }
        if showsLog, showsFailureRow, !logTail.isEmpty {
            text += ". Run log: \(logTail.joined(separator: ". "))"
        }
        return text
    }
}

/// Three quiet dots taking turns while the model decides what to say.
private struct StudioThinkingIndicator: View {
    var body: some View {
        HStack(spacing: 8) {
            StudioPulsingDots()
            Text("Thinking…")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
        .padding(.vertical, 4)
        .accessibilityLabel("Generating a reply")
    }
}

/// The dots themselves, shared by the indicator and a live "Thinking…" disclosure.
private struct StudioPulsingDots: View {
    var body: some View {
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
        .accessibilityHidden(true)
    }
}
