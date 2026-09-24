import AppKit
import StudioKit
import SwiftUI

/// The stems a separation wrote, one row each with its own player, inside the Analyze result
/// panel. The rows come from the manifest `music separate` writes; a run whose manifest is
/// missing still lists the audio files in its output folder, so an older row plays too.
struct StudioStemsList: View {
    let item: StudioLibraryItem
    let manifest: StudioSeparationManifest?

    private var stems: [StudioSeparationManifest.Stem] {
        Self.stems(for: item, manifest: manifest)
    }

    var body: some View {
        if stems.isEmpty {
            Text("No stems were written.")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
            hairline
        } else {
            ForEach(stems) { stem in
                StudioStemRow(stem: stem)
                hairline
            }
        }
    }

    private var hairline: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.27))
            .frame(height: 1)
    }

    /// The manifest's stems that are still on disk, else the audio files in the output folder
    /// by name, else the row's audio artifacts.
    static func stems(for item: StudioLibraryItem, manifest: StudioSeparationManifest?) -> [StudioSeparationManifest.Stem] {
        let fileManager = FileManager.default
        if let manifest {
            let present = manifest.stems.filter { fileManager.fileExists(atPath: $0.path) }
            if !present.isEmpty { return present }
        }
        if let folder = item.outputURL,
           let names = try? fileManager.contentsOfDirectory(atPath: folder.path) {
            let files = names.sorted().map { folder.appendingPathComponent($0, isDirectory: false) }
                .filter { StudioOutputFileKind.classify($0) == .audio }
            if !files.isEmpty {
                return files.map { StudioSeparationManifest.Stem(name: $0.deletingPathExtension().lastPathComponent, path: $0.path) }
            }
        }
        return item.allArtifactURLs
            .filter { StudioOutputFileKind.classify($0) == .audio && fileManager.fileExists(atPath: $0.path) }
            .map { StudioSeparationManifest.Stem(name: $0.deletingPathExtension().lastPathComponent, path: $0.path) }
    }
}

/// One stem: play or pause, its name over a thin progress track that seeks on click, and the
/// clock. Every row has its own player, so two stems can play together the way they were mixed.
struct StudioStemRow: View {
    let stem: StudioSeparationManifest.Stem

    @StateObject private var player = StudioAudioPlayer()
    private let ticker = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    private var fraction: Double {
        guard player.duration > 0 else { return 0 }
        return min(1, max(0, player.currentTime / player.duration))
    }

    var body: some View {
        HStack(spacing: 12) {
            Button {
                player.togglePlay()
            } label: {
                Image(systemName: player.isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(MereRunTheme.onAccent)
                    .frame(width: 26, height: 26)
                    .background { Circle().fill(player.isReady ? MereRunTheme.accent : MereRunTheme.textMuted) }
            }
            .buttonStyle(.plain)
            .disabled(!player.isReady)
            .accessibilityLabel(player.isPlaying ? "Pause \(stem.title)" : "Play \(stem.title)")

            VStack(alignment: .leading, spacing: 6) {
                Text(stem.title)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                track
            }

            Text(clock)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(MereRunTheme.textMuted)
                .fixedSize()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .contextMenu {
            Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([stem.url]) }
        }
        .task(id: stem.path) { player.load(url: stem.url) }
        .onReceive(ticker) { _ in player.refresh() }
        .onDisappear { player.stop() }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(stem.title) stem, \(clock)")
    }

    private var clock: String {
        player.isReady ? "\(StudioTimeFormat.string(player.currentTime)) / \(StudioTimeFormat.string(player.duration))" : "–:––"
    }

    private var track: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(MereRunTheme.surfaceRaised)
                Capsule().fill(MereRunTheme.accent)
                    .frame(width: max(0, geometry.size.width * fraction))
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                guard player.isReady, geometry.size.width > 0 else { return }
                player.seek(to: min(max(0, value.location.x / geometry.size.width), 1) * player.duration)
            })
        }
        .frame(height: 4)
        .accessibilityHidden(true)
    }
}
