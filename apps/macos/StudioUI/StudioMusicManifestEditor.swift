import AppKit
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// Music ▸ Train's dataset: the clips to train on, each with a caption and optional lyrics. Studio
/// writes the manifest the trainer reads from these rows; a manifest made elsewhere still imports.
///
/// The editor works on its own copy of the manifest and writes it back to the page's stored value
/// a moment after typing stops: the stored value decodes its JSON on every read, which a list of
/// hundreds of clips cannot afford on every keystroke. Whether each clip's audio file exists is
/// checked once per change of the file list, not once per render.
struct StudioMusicManifestEditor: View {
    @Binding var manifest: StudioMusicTrainingManifest
    /// Set when an import or export fails, in the page's words.
    @Binding var message: String?
    @State private var draft = StudioMusicTrainingManifest()
    @State private var existingFiles: Set<String> = []
    @State private var isDropTargeted = false

    private var audioPaths: [String] { draft.clips.map { $0.audioURL.path } }

    var body: some View {
        let fileExists: StudioMusicTrainingManifest.FileCheck = { existingFiles.contains($0.path) }
        let rowProblems = draft.clips.map { draft.clipProblems($0, fileExists: fileExists) }
        let readyCount = rowProblems.filter(\.isEmpty).count
        let problems = draft.problems(fileExists: fileExists)
        let numbers = Dictionary(draft.clips.enumerated().map { ($1.id, $0 + 1) }, uniquingKeysWith: { first, _ in first })
        VStack(alignment: .leading, spacing: 10) {
            header(readyCount: readyCount)
            if draft.clips.isEmpty {
                emptyState
            } else {
                LazyVStack(spacing: 8) {
                    ForEach($draft.clips) { $clip in
                        let number = numbers[clip.id] ?? 0
                        StudioMusicClipRow(
                            clip: $clip,
                            number: number,
                            hasAudio: !clip.audioPath.isBlank && existingFiles.contains(clip.audioURL.path),
                            problems: rowProblems.indices.contains(number - 1) ? rowProblems[number - 1] : [],
                            onRemove: { draft.clips.removeAll { $0.id == clip.id } }
                        )
                    }
                }
            }
            if !problems.isEmpty, !draft.clips.isEmpty {
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
        .onAppear {
            draft = manifest
            refreshExistingFiles()
        }
        .onChange(of: manifest) { _, manifest in
            // The page adopted a manifest from elsewhere; a write-back arrives equal to the draft.
            if manifest != draft { draft = manifest }
        }
        .onChange(of: audioPaths) { _, _ in refreshExistingFiles() }
        .task(id: draft) {
            guard draft != manifest else { return }
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            manifest = draft
        }
        .onDisappear {
            // Leaving the page inside the debounce window must not lose the last edits.
            if draft != manifest { manifest = draft }
        }
    }

    private func header(readyCount: Int) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Clips")
                .font(MereRunTheme.sectionFont)
            if !draft.clips.isEmpty {
                Text("\(draft.clips.count) · \(readyCount) ready")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(readyCount == draft.clips.count ? MereRunTheme.green : MereRunTheme.textMuted)
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
                Button("Add a folder of clips…", action: addFolder)
                Divider()
                Button("Import manifest…", action: importManifest)
                Button("Export manifest…", action: exportManifest)
                    .disabled(draft.clips.isEmpty)
                Divider()
                Button("Clear clips", role: .destructive) { draft.clips.removeAll() }
                    .disabled(draft.clips.isEmpty)
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

    private func refreshExistingFiles() {
        existingFiles = Set(audioPaths.filter { !$0.isEmpty && FileManager.default.fileExists(atPath: $0) })
    }

    /// Adds audio files, and every audio file at the top of a dropped folder; skips files already listed.
    private func add(_ urls: [URL]) {
        let listed = Set(audioPaths)
        var added: [StudioMusicTrainingClip] = []
        for url in urls {
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
                added += StudioMusicTrainingManifest.clips(scanning: url)
            } else if StudioMusicTrainingManifest.isAudioFile(url) {
                added.append(StudioMusicTrainingClip(audioPath: url.path))
            }
        }
        draft.clips += added.filter { !listed.contains($0.audioURL.path) }
    }

    private func addFolder() {
        guard let folder = StudioSpecialistFiles.chooseDirectory(title: "Add a folder of clips") else { return }
        let before = draft.clips.count
        add([folder])
        if draft.clips.count == before {
            message = "No audio files were found at the top of \(folder.lastPathComponent)."
        }
    }

    /// Any file type: `.jsonl` has no declared type, so a type filter would grey the files out.
    private func importManifest() {
        guard let url = StudioSpecialistFiles.chooseFile(title: "Import a training manifest").first else { return }
        do {
            draft = try StudioMusicTrainingManifest.importing(Data(contentsOf: url), from: url)
            message = nil
        } catch {
            message = "That file is not a training manifest: \(error.localizedDescription)"
        }
    }

    private func exportManifest() {
        guard let url = StudioSpecialistFiles.saveFile(title: "Export the training manifest", suggestedName: "dataset.jsonl") else { return }
        do {
            try draft.jsonl().write(to: url, options: .atomic)
        } catch {
            message = "Studio could not save the manifest: \(error.localizedDescription)"
        }
    }
}

/// One clip: the file, a play button, its caption, and lyrics behind a disclosure until they exist.
private struct StudioMusicClipRow: View {
    @Binding var clip: StudioMusicTrainingClip
    let number: Int
    let hasAudio: Bool
    let problems: [String]
    let onRemove: () -> Void
    @State private var showsLyrics = false

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
