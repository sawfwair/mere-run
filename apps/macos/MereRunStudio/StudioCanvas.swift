import AppKit
import SwiftUI

// The prompt workspace's shared canvas pieces: the empty state, the flow layout its example
// chips use, and the image and text previews the feed cards and Library rows draw with. The feed
// itself is `StudioFeedCanvas.swift`.

/// A blank canvas that teaches: serif headline, one line of guidance, and prompts you can
/// click straight into the composer.
struct StudioEmptyState: View {
    let mode: StudioMode
    var isCompact = false
    var onUseExample: ((String) -> Void)?
    var onAttach: (() -> Void)?

    var body: some View {
        VStack(spacing: isCompact ? MereRunTheme.Spacing.md : MereRunTheme.Spacing.xl) {
            Image(systemName: mode.systemImage)
                .font(.system(size: isCompact ? 28 : 42, weight: .medium))
                .foregroundStyle(MereRunTheme.accent)
                .frame(width: isCompact ? 64 : 96, height: isCompact ? 64 : 96)
                .background {
                    Circle().fill(MereRunTheme.accentSoft.opacity(0.75))
                }
                .overlay {
                    Circle().strokeBorder(MereRunTheme.accent.opacity(0.18), lineWidth: 1)
                }

            VStack(spacing: MereRunTheme.Spacing.xs) {
                Text(mode.emptyTitle)
                    .font(isCompact ? MereRunTheme.displaySmallFont : MereRunTheme.displayFont)
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .multilineTextAlignment(.center)
                Text(mode.emptyMessage)
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 460)
            }

            if mode.requiresAttachment, let onAttach {
                Button(action: onAttach) {
                    Label(attachLabel, systemImage: "paperclip")
                }
                .buttonStyle(.merePrimary)
            }

            if !mode.examplePrompts.isEmpty, let onUseExample {
                FlowLayout(spacing: MereRunTheme.Spacing.xs, lineSpacing: 8) {
                    examplePrompts(onUseExample: onUseExample)
                }
                .frame(maxWidth: isCompact ? 320 : 620)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func examplePrompts(onUseExample: @escaping (String) -> Void) -> some View {
        ForEach(mode.examplePrompts, id: \.self) { example in
            StudioExampleChip(text: example) { onUseExample(example) }
        }
    }

    private var attachLabel: String {
        switch mode {
        case .listen: return "Choose audio…"
        case .track: return "Choose video…"
        default: return "Choose image…"
        }
    }
}

/// One clickable example prompt. Filling, never auto-running, keeps the user in charge.
private struct StudioExampleChip: View {
    let text: String
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Text("“\(text)”")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(hovering ? MereRunTheme.textPrimary : MereRunTheme.textSecondary)
                .lineLimit(2)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 300, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .fill(hovering ? MereRunTheme.surfaceRaised : MereRunTheme.surface.opacity(0.7))
                        .overlay {
                            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                                .strokeBorder(MereRunTheme.border.opacity(0.65), lineWidth: 1)
                        }
                }
                .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .help("Use this prompt")
        .accessibilityLabel("Use example prompt: \(text)")
    }
}

/// Places example chips left-to-right and wraps to a new centered row when the next
/// chip would overflow the available width — so prompts size to their content and
/// never truncate mid-word regardless of window width.
struct FlowLayout: Layout {
    var spacing: CGFloat = 8
    var lineSpacing: CGFloat = 8

    private struct Row {
        var indices: [Int] = []
        var width: CGFloat = 0
        var height: CGFloat = 0
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        let rows = arrange(subviews: subviews, maxWidth: maxWidth)
        let height = rows.map(\.height).reduce(0, +)
            + CGFloat(max(0, rows.count - 1)) * lineSpacing
        let width = rows.map(\.width).max() ?? 0
        return CGSize(width: min(width, maxWidth), height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) {
        let rows = arrange(subviews: subviews, maxWidth: bounds.width)
        var y = bounds.minY
        for row in rows {
            var x = bounds.minX + (bounds.width - row.width) / 2
            for index in row.indices {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(
                    at: CGPoint(x: x, y: y),
                    anchor: .topLeading,
                    proposal: ProposedViewSize(size)
                )
                x += size.width + spacing
            }
            y += row.height + lineSpacing
        }
    }

    private func arrange(subviews: Subviews, maxWidth: CGFloat) -> [Row] {
        var rows: [Row] = []
        var current = Row()
        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            let projected = current.indices.isEmpty ? size.width : current.width + spacing + size.width
            if !current.indices.isEmpty, projected > maxWidth {
                rows.append(current)
                current = Row()
            }
            current.width = current.indices.isEmpty ? size.width : current.width + spacing + size.width
            current.height = max(current.height, size.height)
            current.indices.append(index)
        }
        if !current.indices.isEmpty { rows.append(current) }
        return rows
    }
}

enum StudioImageContentMode: Equatable {
    case fit
    case fill
}

enum StudioImageLoadState {
    case loading
    case loaded(NSImage)
    case unavailable
}

/// Downsampled, off-main image preview with a gentle entrance once decoded.
struct StudioAsyncImagePreview: View {
    let url: URL
    let maxPixelSize: CGFloat
    let contentMode: StudioImageContentMode
    let fallbackSystemImage: String

    @State private var loadState = StudioImageLoadState.loading

    private var stateKey: Int {
        switch loadState {
        case .loading: return 0
        case .loaded: return 1
        case .unavailable: return 2
        }
    }

    var body: some View {
        Group {
            switch loadState {
            case .loading:
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .loaded(let image):
                Group {
                    if contentMode == .fill {
                        Image(nsImage: image)
                            .resizable()
                            .scaledToFill()
                    } else {
                        Image(nsImage: image)
                            .resizable()
                            .scaledToFit()
                    }
                }
                .transition(.opacity.combined(with: .scale(scale: 0.985)))
            case .unavailable:
                Image(systemName: fallbackSystemImage)
                    .font(.system(size: 32, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .animation(MereRunTheme.Motion.gentleSpring, value: stateKey)
        .task(id: url) {
            loadState = .loading
            let loaded = await Task.detached(priority: .userInitiated) {
                StudioImagePreviewLoader.downsampledImage(from: url, maxPixelSize: maxPixelSize)
            }.value
            guard !Task.isCancelled else { return }
            if let image = loaded?.image {
                loadState = .loaded(image)
            } else {
                loadState = .unavailable
            }
        }
    }
}

struct StudioTextFilePreview: View {
    let url: URL

    @State private var text: String?
    @State private var didLoad = false

    var body: some View {
        ScrollView {
            if let text {
                Text(text)
                    .font(MereRunTheme.monoFont)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(MereRunTheme.Spacing.xl)
            } else if didLoad {
                Text("Text preview unavailable.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(MereRunTheme.Spacing.xl)
            } else {
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, minHeight: 180)
                    .padding(MereRunTheme.Spacing.xl)
            }
        }
        .task(id: url) {
            text = nil
            didLoad = false
            let preview = await Task.detached(priority: .userInitiated) {
                StudioTextPreviewReader.previewText(from: url)
            }.value
            guard !Task.isCancelled else { return }
            text = preview
            didLoad = true
        }
    }
}
