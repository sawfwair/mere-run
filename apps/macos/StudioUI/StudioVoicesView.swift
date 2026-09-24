import AppKit
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// Voice ▸ Voices: the Manage surface for saved voices. A list of the profiles the CLI keeps, a
/// detail that plays the reference and shows its transcript, Delete behind a confirmation, and
/// New voice — a name, a reference recording in an attachment well (chosen, dropped, or
/// recorded), an optional transcript, and a language — run through the task runner as
/// `speech profile create`, so it lands in the Library under Voice like any run. Speak's clone
/// mode lists the same profiles.
struct StudioVoicesView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @Environment(\.studioTaskRunner) private var runner
    @Environment(\.studioTaskSessions) private var sessions
    @Environment(\.studioVoiceProfileSeed) private var profileSeed
    @StudioStoredValue("Voice.selectedProfileID") private var selectedProfileID: UUID? = nil
    @StateObject private var jobMonitor = StudioJobMonitor()
    @State private var profiles: [StudioVoiceProfileRecord] = []
    @State private var isCreating = false
    @State private var pendingDelete: StudioVoiceProfileRecord?
    @State private var error: String?

    private static let task = StudioTask.voiceVoices
    private static let listWidth: CGFloat = 300

    // MARK: Draft

    /// The New voice form: the `speech profile create` task draft the Command view edits too.
    private var draft: StudioTaskDraft {
        get { sessions?.taskDraft(for: Self.task) ?? StudioTaskDraft(templateID: .speechProfileCreate) }
        nonmutating set { sessions?.setTaskDraft(newValue, for: Self.task) }
    }

    private var draftBinding: Binding<StudioTaskDraft> {
        Binding(get: { draft }, set: { draft = $0 })
    }

    private func field(_ flag: String) -> Binding<String> {
        Binding(
            get: { draft.text(flag) },
            set: { text in
                var next = draft
                next.form[flag] = text.isEmpty ? .unset : .text(text)
                draft = next
            }
        )
    }

    private var audioSlot: StudioAttachmentSlot? {
        StudioTaskSchema.primarySlot(for: .speechProfileCreate)
    }

    private var selectedProfile: StudioVoiceProfileRecord? {
        profiles.first { $0.id == selectedProfileID }
    }

    /// The create or delete run in flight, if any.
    private var activeJob: Job? {
        _ = jobMonitor.generation
        return runner?.currentJob(for: Self.task)
    }

    private var isCreatingProfile: Bool {
        activeJob?.request.templateID == .speechProfileCreate
    }

    private var deletingProfileID: UUID? {
        guard let job = activeJob, job.request.templateID == .speechProfileDelete,
              let form = job.request.execution?.form else { return nil }
        return UUID(uuidString: form.text("--id"))
    }

    // MARK: Body

    var body: some View {
        VStack(spacing: 0) {
            header
            hairline
            HStack(spacing: 0) {
                list
                    .frame(width: Self.listWidth)
                Rectangle()
                    .fill(MereRunTheme.border.opacity(0.53))
                    .frame(width: 1)
                detail
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            if let error {
                MereBanner(severity: .error, text: error, onDismiss: { self.error = nil })
                    .padding(.horizontal, 24)
                    .padding(.bottom, 16)
            }
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onAppear {
            jobMonitor.attach(controller.jobs)
            controller.checkReadiness(for: Self.task, modelID: StudioTaskSchema.modelID(for: draft))
            refreshProfiles()
        }
        .onReceive(controller.runCompletions) { result in
            guard [CommandTemplateID.speechProfileCreate, .speechProfileDelete].contains(result.templateID) else { return }
            let before = Set(profiles.map(\.id))
            refreshProfiles()
            if result.templateID == .speechProfileCreate, result.exitCode == 0 {
                if let created = profiles.first(where: { !before.contains($0.id) }) {
                    selectedProfileID = created.id
                }
                isCreating = false
            } else if result.exitCode != 0 {
                error = result.templateID == .speechProfileCreate
                    ? "The voice could not be created. The Library row has the CLI's output."
                    : "The voice could not be deleted. The Library row has the CLI's output."
            }
        }
        .confirmationDialog(
            "Delete \u{201C}\(pendingDelete?.name ?? "")\u{201D}?",
            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
            titleVisibility: .visible,
            presenting: pendingDelete
        ) { profile in
            Button("Delete voice", role: .destructive) { delete(profile) }
            Button("Cancel", role: .cancel) {}
        } message: { _ in
            Text("Speak's clone mode will no longer offer it. The reference recording is removed with it.")
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Saved voices")
    }

    private var hairline: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.4))
            .frame(height: 1)
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: MereRunTheme.Spacing.md) {
            Text(profiles.isEmpty ? "No saved voices yet" : (profiles.count == 1 ? "1 saved voice" : "\(profiles.count) saved voices"))
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Spacer()
            Button {
                isCreating = true
            } label: {
                Label("New voice", systemImage: "plus")
            }
            .buttonStyle(.merePrimary)
            .disabled(isCreating)
            .help("Save a reference recording as a named voice")
        }
        .padding(.horizontal, 24)
        .frame(height: 52)
    }

    // MARK: List

    @ViewBuilder
    private var list: some View {
        if profiles.isEmpty {
            VStack(spacing: MereRunTheme.Spacing.sm) {
                Image(systemName: "person.wave.2")
                    .font(.system(size: 24, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                Text("Voices you save appear here.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollView {
                LazyVStack(spacing: 4) {
                    ForEach(profiles) { profile in
                        profileRow(profile)
                    }
                }
                .padding(12)
            }
        }
    }

    private func profileRow(_ profile: StudioVoiceProfileRecord) -> some View {
        let isSelected = !isCreating && selectedProfileID == profile.id
        return Button {
            selectedProfileID = profile.id
            isCreating = false
        } label: {
            HStack(spacing: 10) {
                Image(systemName: "person.wave.2.fill")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(MereRunTheme.accent)
                    .frame(width: 18)
                VStack(alignment: .leading, spacing: 2) {
                    Text(profile.name)
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                    Text(profile.language ?? "Automatic language")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
                if deletingProfileID == profile.id {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel("Deleting")
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(isSelected ? MereRunTheme.accentSoft : Color.clear)
            }
            .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(profile.name)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    // MARK: Detail

    @ViewBuilder
    private var detail: some View {
        if isCreating || (profiles.isEmpty && selectedProfile == nil) {
            ScrollView {
                createForm
                    .frame(maxWidth: 560, alignment: .leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(24)
            }
        } else if let profile = selectedProfile {
            ScrollView {
                profileDetail(profile)
                    .frame(maxWidth: 720, alignment: .leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(24)
            }
        } else {
            StudioEmptyState(presentation: Self.task.presentation, isCompact: true)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func profileDetail(_ profile: StudioVoiceProfileRecord) -> some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(profile.name)
                        .font(MereRunTheme.titleFont)
                    Text(profile.id.uuidString)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .textSelection(.enabled)
                }
                Spacer()
                Button("Reveal recording") {
                    NSWorkspace.shared.activateFileViewerSelecting([profile.referenceAudioURL])
                }
                .buttonStyle(.mereSecondary)
                .help("Show the reference recording in Finder")
                Button(role: .destructive) {
                    pendingDelete = profile
                } label: {
                    Label("Delete…", systemImage: "trash")
                }
                .buttonStyle(.mereSecondary)
                .disabled(deletingProfileID == profile.id)
                .help("Remove this voice and its reference recording")
            }
            StudioAudioPlayerView(url: profile.referenceAudioURL)
                .frame(height: 220)
                .merePanel()
            VStack(alignment: .leading, spacing: 7) {
                Text("Reference transcript")
                    .font(MereRunTheme.sectionFont)
                Text(profile.transcript)
                    .font(MereRunTheme.bodyFont)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 14) {
                    Label(profile.language ?? "Automatic language", systemImage: "globe")
                    Label(profile.createdAt.formatted(date: .abbreviated, time: .shortened), systemImage: "calendar")
                }
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .merePanel()
        }
    }

    // MARK: New voice

    private var createForm: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            VStack(alignment: .leading, spacing: 4) {
                Text("New voice")
                    .font(MereRunTheme.titleFont)
                Text("Save a reference recording as a named voice. Speak's clone mode lists it.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
            }
            formRow("Name") {
                StudioInspectorTextField(placeholder: "Narrator", text: field("--name"))
            }
            if let audioSlot {
                formRow("Reference audio") {
                    HStack(spacing: 10) {
                        StudioAttachmentSlotView(slot: audioSlot, draft: draftBinding) {
                            var next = draft
                            StudioAttachmentPicker.pick(for: audioSlot, into: &next)
                            draft = next
                        }
                        VStack(alignment: .leading, spacing: 2) {
                            Text(audioSlot.caption(in: draft))
                                .font(.caption.weight(.medium))
                                .foregroundStyle(MereRunTheme.textPrimary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Text("A clean sample of a few seconds. WAV, MP3, M4A…")
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                        }
                        Spacer(minLength: 8)
                        StudioAudioRecordButton(slot: audioSlot, draft: draftBinding, domain: .voice)
                    }
                }
            }
            formRow("Transcript") {
                StudioInspectorTextField(
                    placeholder: "Optional. Left empty, mere.run transcribes the recording.",
                    text: field("--text"),
                    lines: 2...6
                )
            }
            formRow("Language") {
                StudioInspectorTextField(placeholder: "auto", text: field("--language"))
                    .frame(maxWidth: 160)
            }
            HStack(spacing: 10) {
                if isCreatingProfile {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityHidden(true)
                    Text("Creating…")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .accessibilityLabel("Creating the voice")
                }
                Spacer()
                if !profiles.isEmpty {
                    Button("Cancel") {
                        isCreating = false
                    }
                    .buttonStyle(.mereSecondary)
                }
                Button {
                    create()
                } label: {
                    Label("Create voice", systemImage: "person.badge.plus")
                }
                .buttonStyle(.merePrimary)
                .disabled(isCreatingProfile)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
    }

    private func formRow<Content: View>(_ label: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.callout.weight(.medium))
                .foregroundStyle(MereRunTheme.textSecondary)
            content()
        }
    }

    // MARK: Actions

    private func create() {
        error = nil
        guard let runner else { return }
        if draft.text("--name").isBlank {
            error = "Give the voice a name."
            return
        }
        let audio = draft.text("--audio")
        guard !audio.isBlank, FileManager.default.fileExists(atPath: audio) else {
            error = "Choose or record reference audio."
            return
        }
        do {
            _ = try runner.run(draft, task: Self.task)
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func delete(_ profile: StudioVoiceProfileRecord) {
        error = nil
        guard let runner else { return }
        do {
            _ = try runner.run(.deletingVoiceProfile(profile.id), task: Self.task)
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func refreshProfiles() {
        profiles = profileSeed ?? StudioVoiceProfileStore.load()
        if selectedProfileID == nil || !profiles.contains(where: { $0.id == selectedProfileID }) {
            selectedProfileID = profiles.first?.id
        }
    }
}

private struct StudioVoiceProfileSeedKey: EnvironmentKey {
    static let defaultValue: [StudioVoiceProfileRecord]? = nil
}

extension EnvironmentValues {
    /// Test seam: the voices the page lists instead of reading the CLI's manifest, so the
    /// snapshot harness renders a list and a detail without touching Application Support.
    var studioVoiceProfileSeed: [StudioVoiceProfileRecord]? {
        get { self[StudioVoiceProfileSeedKey.self] }
        set { self[StudioVoiceProfileSeedKey.self] = newValue }
    }
}
