import AppKit
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// Music ▸ Train's dataset: the clips to train on, each with a caption and optional lyrics. Studio
/// writes the manifest the trainer reads from these rows; a manifest made elsewhere still imports.
struct StudioMusicManifestEditor: View {
    @Binding var manifest: StudioMusicTrainingManifest
    /// Set when an import or export fails, in the page's words.
    @Binding var message: String?
    @State private var isDropTargeted = false

    private var readyCount: Int { manifest.readyClipCount() }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            if manifest.clips.isEmpty {
                emptyState
            } else {
                VStack(spacing: 8) {
                    ForEach($manifest.clips) { $clip in
                        StudioMusicClipRow(
                            clip: $clip,
                            number: (manifest.clips.firstIndex { $0.id == clip.id } ?? 0) + 1,
                            problems: manifest.clipProblems(clip),
                            onRemove: { manifest.clips.removeAll { $0.id == clip.id } }
                        )
                    }
                }
            }
            let problems = manifest.problems()
            if !problems.isEmpty, !manifest.clips.isEmpty {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(problems, id: \.self) { problem in
                        Label(problem, systemImage: "exclamationmark.circle")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textSecondary)
                    }
                }
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            add(urls)
            return true
        } isTargeted: { isDropTargeted = $0 }
        .overlay {
            if isDropTargeted {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .strokeBorder(MereRunTheme.accent, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                    .padding(-6)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Clips")
                .font(MereRunTheme.sectionFont)
            if !manifest.clips.isEmpty {
                Text("\(manifest.clips.count) · \(readyCount) ready")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(readyCount == manifest.clips.count ? MereRunTheme.green : MereRunTheme.textMuted)
                    .monospacedDigit()
            }
            Spacer()
            Button {
                add(StudioSpecialistFiles.chooseFile(title: "Add audio clips", allowedContentTypes: [.audio], allowsMultipleSelection: true))
            } label: {
                Label("Add audio…", systemImage: "plus")
            }
            .buttonStyle(.mereSecondary)
            .controlSize(.small)
            Menu {
                Button("Add a Folder of Clips…", action: addFolder)
                Divider()
                Button("Import Manifest…", action: importManifest)
                Button("Export Manifest…", action: exportManifest)
                    .disabled(manifest.clips.isEmpty)
                Divider()
                Button("Clear Clips", role: .destructive) { manifest.clips.removeAll() }
                    .disabled(manifest.clips.isEmpty)
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("Add a folder, import, and export")
            .accessibilityLabel("More clip actions")
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Add the audio clips to learn from, then describe each one.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
            Text("Drop audio files here, or add a folder whose clips have matching .txt captions.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .strokeBorder(MereRunTheme.border.opacity(0.7), style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
        }
    }

    /// Adds audio files, and every audio file at the top of a dropped folder; skips files already listed.
    private func add(_ urls: [URL]) {
        let listed = Set(manifest.clips.map { $0.audioURL.path })
        var added: [StudioMusicTrainingClip] = []
        for url in urls {
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
                added += StudioMusicTrainingManifest.clips(scanning: url)
            } else if StudioMusicTrainingManifest.isAudioFile(url) {
                added.append(StudioMusicTrainingClip(audioPath: url.path))
            }
        }
        manifest.clips += added.filter { !listed.contains($0.audioURL.path) }
    }

    private func addFolder() {
        guard let folder = StudioSpecialistFiles.chooseDirectory(title: "Add a folder of clips") else { return }
        let before = manifest.clips.count
        add([folder])
        if manifest.clips.count == before {
            message = "No audio files were found at the top of \(folder.lastPathComponent)."
        }
    }

    private func importManifest() {
        guard let url = StudioSpecialistFiles.chooseFile(title: "Import a training manifest", allowedContentTypes: [.json, .plainText]).first else {
            return
        }
        do {
            manifest = try StudioMusicTrainingManifest.importing(Data(contentsOf: url), from: url)
            message = nil
        } catch {
            message = "That file is not a training manifest: \(error.localizedDescription)"
        }
    }

    private func exportManifest() {
        guard let url = StudioSpecialistFiles.saveFile(
            title: "Export the training manifest",
            suggestedName: "dataset.jsonl",
            allowedContentTypes: [.json, .plainText]
        ) else { return }
        do {
            try manifest.jsonl().write(to: url, options: .atomic)
        } catch {
            message = "Studio could not save the manifest: \(error.localizedDescription)"
        }
    }
}

/// One clip: the file, a play button, its caption, and lyrics behind a disclosure until they exist.
private struct StudioMusicClipRow: View {
    @Binding var clip: StudioMusicTrainingClip
    let number: Int
    let problems: [String]
    let onRemove: () -> Void
    @State private var showsLyrics = false

    private var hasAudio: Bool {
        !clip.audioPath.isBlank && FileManager.default.fileExists(atPath: clip.audioURL.path)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 8) {
                Image(systemName: hasAudio ? "waveform" : "waveform.slash")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(hasAudio ? MereRunTheme.accent : MereRunTheme.yellow)
                    .frame(width: 18)
                VStack(alignment: .leading, spacing: 1) {
                    Text(clip.fileName.isEmpty ? "No audio file" : clip.fileName)
                        .font(.system(size: 12.5, weight: .semibold))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(StudioOutputLocation.abbreviate(clip.audioURL.deletingLastPathComponent()))
                        .font(.system(size: 10.5))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 6)
                if hasAudio {
                    Button {
                        QuickLookCoordinator.shared.preview(clip.audioURL)
                    } label: {
                        Image(systemName: "play.circle")
                            .frame(width: 22, height: 22)
                    }
                    .buttonStyle(.mereIcon)
                    .help("Play \(clip.fileName)")
                    .accessibilityLabel("Play clip \(number)")
                }
                Button(action: onRemove) {
                    Image(systemName: "minus.circle")
                        .frame(width: 22, height: 22)
                }
                .buttonStyle(.mereIcon)
                .help("Remove")
                .accessibilityLabel("Remove clip \(number)")
            }
            TextField("What it sounds like: genre, instruments, mood, tempo", text: $clip.caption, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...3)
                .padding(7)
                .merePanel()
                .accessibilityLabel("Caption for clip \(number)")
            if showsLyrics || !clip.lyrics.isBlank {
                TextField("Lyrics, one line per sung line (optional)", text: $clip.lyrics, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.captionFont)
                    .lineLimit(2...8)
                    .padding(7)
                    .merePanel()
                    .accessibilityLabel("Lyrics for clip \(number)")
            } else {
                Button {
                    showsLyrics = true
                } label: {
                    Label("Add lyrics", systemImage: "text.quote")
                        .font(.caption.weight(.medium))
                }
                .buttonStyle(.plain)
                .foregroundStyle(MereRunTheme.accent)
            }
        }
        .padding(10)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface.opacity(0.72))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .strokeBorder(
                            problems.isEmpty ? MereRunTheme.border.opacity(0.55) : MereRunTheme.yellow.opacity(0.6),
                            lineWidth: 1
                        )
                }
        }
    }
}
