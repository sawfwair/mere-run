import AppKit
import StudioKit
import SwiftUI

/// A shared viewport makes a comparison about the output, with the changed settings beside it.
struct StudioResultWorkspaceView: View {
    let item: StudioLibraryItem
    let url: URL
    let items: [StudioLibraryItem]
    var initialComparison: StudioResultSelection? = nil
    let onClose: () -> Void
    let onVary: (StudioLibraryItem) -> Void
    let onSave: (URL) -> Void
    let onContinue: (StudioResultContinuation, StudioLibraryItem, URL) -> Void

    @State private var comparison: StudioResultSelection?
    @State private var zoom: CGFloat = 1
    @State private var pan: CGSize = .zero
    @State private var showsSettings = true

    private var candidates: [(StudioLibraryItem, URL)] {
        items.filter { $0.status == .completed }.flatMap { candidate in
            candidate.allArtifactURLs.filter {
                StudioOutputFileKind.classify($0) == .image && $0 != url
            }.map { (candidate, $0) }
        }.prefix(30).map { $0 }
    }

    private var comparedItem: StudioLibraryItem? {
        items.first { $0.id == comparison?.itemID }
    }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            GeometryReader { geometry in
                if geometry.size.width < 680, comparison != nil {
                    VStack(spacing: 1) { panes }
                } else {
                    HStack(spacing: 1) { panes }
                }
            }
            .background(MereRunTheme.border)
            if showsSettings, let comparedItem {
                differences(comparedItem)
            }
            footer
        }
        .background(MereRunTheme.background)
        .onAppear { comparison = initialComparison }
        .onChange(of: url) { _, _ in comparison = nil; resetViewport() }
        .onExitCommand(perform: onClose)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(comparison == nil ? "Focused result" : "Compare results")
    }

    private var toolbar: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) { back; title; Spacer(); tools }
            VStack(alignment: .leading, spacing: 10) {
                HStack { back; title; Spacer() }
                HStack { tools; Spacer() }
            }
        }
        .padding(14)
    }

    private var back: some View {
        Button(action: onClose) { Label("Results", systemImage: "chevron.left") }
            .buttonStyle(.mereSecondary)
            .help("Return to your results and draft (Escape)")
    }

    private var title: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(comparison == nil ? "Focus" : "Compare")
                .font(.headline)
            Text(url.lastPathComponent).font(.caption).foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1).truncationMode(.middle)
        }
    }

    private var tools: some View {
        HStack(spacing: 10) {
            Button { zoom = max(1, zoom / 1.5); if zoom == 1 { pan = .zero } } label: {
                Image(systemName: "minus.magnifyingglass")
            }.keyboardShortcut("-", modifiers: .command).help("Zoom out")
            Button { resetViewport() } label: { Text(zoom == 1 ? "Fit" : "\(Int(zoom * 100))%") }
                .keyboardShortcut("0", modifiers: .command).help("Fit both images")
            Button { zoom = min(8, zoom * 1.5) } label: { Image(systemName: "plus.magnifyingglass") }
                .keyboardShortcut("+", modifiers: .command).help("Zoom in")
            Menu {
                if comparison != nil { Button("Single image") { comparison = nil; resetViewport() }; Divider() }
                ForEach(candidates.indices, id: \.self) { index in
                    let candidate = candidates[index]
                    Button("\(candidate.0.displayTitle) · \(candidate.1.lastPathComponent)") {
                        comparison = StudioResultSelection(itemID: candidate.0.id, url: candidate.1)
                        resetViewport()
                    }
                }
            } label: { Label("Compare", systemImage: "rectangle.split.2x1") }
            .disabled(candidates.isEmpty)
            .help(candidates.isEmpty ? "Generate another image to compare results" : "Choose another image")
        }
        .buttonStyle(.borderless)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Linked image zoom and comparison")
    }

    @ViewBuilder private var panes: some View {
        StudioResultImagePane(url: url, label: comparison == nil ? "Result" : "A · Original", zoom: $zoom, pan: $pan)
        if let comparison {
            StudioResultImagePane(url: comparison.url, label: "B · Comparison", zoom: $zoom, pan: $pan)
        }
    }

    private func differences(_ other: StudioLibraryItem) -> some View {
        let differences = StudioResultComparison.differences(item, other)
        return ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                Text("Changed settings").font(.headline)
                if differences.isEmpty { Text(StudioResultComparison.hasSettings(item) && StudioResultComparison.hasSettings(other)
                    ? "The recorded settings match." : "Settings were not recorded for one of these results.").font(.callout) }
                ForEach(differences) { difference in
                    HStack(alignment: .top, spacing: 14) {
                        Text(difference.title).fontWeight(.medium).frame(width: 110, alignment: .leading)
                        Text(difference.first).frame(maxWidth: .infinity, alignment: .leading)
                        Text(difference.second).frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .font(.callout)
                    .textSelection(.enabled)
                    .accessibilityElement(children: .combine)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
        }
        .frame(height: min(180, CGFloat(max(1, differences.count)) * 34 + 52))
        .background(MereRunTheme.surface)
    }

    private var footer: some View {
        ViewThatFits(in: .horizontal) {
            HStack { provenance; Spacer(); nextSteps }
            VStack(alignment: .leading, spacing: 10) { provenance; HStack { nextSteps; Spacer() } }
        }
        .padding(14)
    }

    private var provenance: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(item.model ?? item.mode.title).font(.caption).lineLimit(1)
            if let parentID = item.parentID, let parent = items.first(where: { $0.id == parentID }) {
                Text("From \(parent.displayTitle)").font(.caption).foregroundStyle(MereRunTheme.textSecondary).lineLimit(1)
            }
            if comparison != nil {
                Button(showsSettings ? "Hide changed settings" : "Show changed settings") { showsSettings.toggle() }
                    .font(.caption).buttonStyle(.borderless)
            }
        }
    }

    private var nextSteps: some View {
        HStack(spacing: 10) {
            Button { onVary(item) } label: { Label("Vary", systemImage: "shuffle") }
                .disabled(item.commandDraft == nil)
            Menu {
                ForEach(StudioResultContinuation.allCases) { action in
                    Button { onContinue(action, item, url) } label: { Label(action.title, systemImage: action.symbol) }
                }
            } label: { Text("Continue with…") }
            Button("Save copy…") { onSave(url) }
        }
        .buttonStyle(.mereSecondary)
    }

    private func resetViewport() { zoom = 1; pan = .zero }
}

private struct StudioResultImagePane: View {
    let url: URL
    let label: String
    @Binding var zoom: CGFloat
    @Binding var pan: CGSize
    @State private var dragOrigin: CGSize?
    @State private var zoomOrigin: CGFloat?

    var body: some View {
        GeometryReader { geometry in
            StudioAsyncImagePreview(url: url, maxPixelSize: 4096, contentMode: .fit, fallbackSystemImage: "photo")
                .frame(width: geometry.size.width, height: geometry.size.height)
                .scaleEffect(zoom)
                .offset(pan)
                .frame(width: geometry.size.width, height: geometry.size.height)
                .background(MereRunTheme.surfaceRaised)
                .clipped()
                .contentShape(Rectangle())
                .gesture(DragGesture().onChanged { value in
                    guard zoom > 1 else { return }
                    let origin = dragOrigin ?? pan
                    dragOrigin = origin
                    pan = CGSize(width: origin.width + value.translation.width, height: origin.height + value.translation.height)
                }.onEnded { _ in dragOrigin = nil })
                .simultaneousGesture(MagnifyGesture().onChanged { value in
                    let origin = zoomOrigin ?? zoom
                    zoomOrigin = origin
                    zoom = min(8, max(1, origin * value.magnification))
                    if zoom == 1 { pan = .zero }
                }.onEnded { _ in zoomOrigin = nil })
                .overlay(alignment: .topLeading) {
                    Text(label).font(.caption.weight(.semibold)).padding(8)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6)).padding(10)
                }
                .accessibilityLabel("\(label): \(url.lastPathComponent)")
                .accessibilityHint("Use the zoom buttons to inspect both images at the same scale")
        }
    }
}
