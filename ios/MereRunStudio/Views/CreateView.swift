import SwiftUI
import MereRunRelayKit

/// Prompt-first creation in the Studio language: pick a mode, describe the
/// result, run it on your fleet. Modes, fields, and options all come from the
/// shared node catalog; availability comes from what the paired fleet
/// actually reports, and an out-of-date fleet is said out loud rather than
/// hidden. Asset inputs (photos, audio files) are the next step.
struct CreateView: View {
    struct Mode: Identifiable {
        let kind: String
        let symbol: String
        let headline: String
        let promptPlaceholder: String
        let promptField: String
        let modelPrefix: String

        var id: String { kind }
        var entry: WorkflowNodeCatalogEntry? { WorkflowNodeRegistry.entry(for: kind) }
    }

    static let modes: [Mode] = [
        Mode(
            kind: "image.generate",
            symbol: "photo",
            headline: "Make something\nvisible.",
            promptPlaceholder: "Describe the image…",
            promptField: "prompt",
            modelPrefix: "image-"
        ),
        Mode(
            kind: "video.generate",
            symbol: "film",
            headline: "Set it in motion.",
            promptPlaceholder: "Describe the shot…",
            promptField: "prompt",
            modelPrefix: "video-"
        ),
        Mode(
            kind: "music.generate",
            symbol: "music.note",
            headline: "Score the moment.",
            promptPlaceholder: "Describe the music…",
            promptField: "prompt",
            modelPrefix: "music-"
        ),
        Mode(
            kind: "sfx.generate",
            symbol: "waveform",
            headline: "Shape a sound.",
            promptPlaceholder: "Describe the sound effect…",
            promptField: "prompt",
            modelPrefix: "sfx-"
        ),
        Mode(
            kind: "speech.synthesize",
            symbol: "speaker.wave.2",
            headline: "Give it a voice.",
            promptPlaceholder: "Write what should be spoken…",
            promptField: "text",
            modelPrefix: "speech-tts-"
        ),
    ]

    @EnvironmentObject private var relay: RelayStore
    @State private var selected = Self.modes[0]
    @State private var prompt = ""
    @State private var model = ""
    @State private var textFields: [String: String] = [:]
    @State private var numberFields: [String: String] = [:]
    @State private var enumFields: [String: String] = [:]
    @State private var showOptions = false
    @State private var submitting = false
    @State private var errorMessage: String?
    @State private var submittedJobID: String?

    private var fleetKinds: Set<String>? {
        relay.workerProbe.map { Set($0.nodeKinds) }
    }

    private func isAvailable(_ mode: Mode) -> Bool {
        fleetKinds?.contains(mode.kind) ?? true
    }

    private var installedModels: [String] {
        (relay.workerProbe?.installedModelIDs ?? []).filter { $0.hasPrefix(selected.modelPrefix) }
    }

    private var optionalStringInputs: [WorkflowNodeField] {
        selected.entry?.inputs.filter {
            $0.type == .string && !$0.required && $0.name != "model" && $0.name != selected.promptField
        } ?? []
    }

    private var numericInputs: [WorkflowNodeField] {
        selected.entry?.inputs.filter { ($0.type == .integer || $0.type == .number) && !$0.required } ?? []
    }

    private var enumInputs: [WorkflowNodeField] {
        selected.entry?.inputs.filter { $0.type == .enumeration && !$0.required } ?? []
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: MereTheme.Spacing.xl) {
                    Text(selected.headline)
                        .font(.system(.largeTitle, design: .serif))
                        .foregroundStyle(MereTheme.textPrimary)
                        .padding(.top, MereTheme.Spacing.l)
                        .animation(nil, value: selected.kind)

                    modeRail
                    composer

                    if !isAvailable(selected) {
                        MereBannerView(
                            text: "Your fleet's nodes don't offer \(selected.entry?.title.lowercased() ?? selected.kind) yet. Update mere.run on your nodes to add it.",
                            color: MereTheme.caution
                        )
                    }

                    if let errorMessage {
                        MereBannerView(text: errorMessage, color: MereTheme.failure)
                    }

                    runButton

                    Text("Runs on your machines. Prompts and outputs stay between your devices.")
                        .font(.footnote)
                        .foregroundStyle(MereTheme.textMuted)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
                .padding(.horizontal, MereTheme.Spacing.xl)
                .padding(.bottom, MereTheme.Spacing.xxl)
            }
            .background(MereTheme.background.ignoresSafeArea())
            .scrollDismissesKeyboard(.interactively)
            .navigationDestination(item: $submittedJobID) { jobID in
                RunDetailView(jobID: jobID)
            }
            .sheet(isPresented: $showOptions) { optionsSheet }
            .task {
                await relay.refreshWorkerProbe()
            }
        }
    }

    private var modeRail: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: MereTheme.Spacing.s) {
                ForEach(Self.modes) { mode in
                    let isSelected = mode.kind == selected.kind
                    Button {
                        selected = mode
                        model = ""
                        textFields = [:]
                        numberFields = [:]
                        enumFields = [:]
                        errorMessage = nil
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: mode.symbol)
                            Text(mode.entry?.title ?? mode.kind)
                        }
                        .font(.subheadline.weight(isSelected ? .semibold : .regular))
                        .padding(.horizontal, MereTheme.Spacing.m)
                        .padding(.vertical, MereTheme.Spacing.s)
                        .background(
                            Capsule().fill(isSelected ? MereTheme.surfaceRaised : MereTheme.surface)
                        )
                        .overlay(
                            Capsule().stroke(
                                isSelected ? MereTheme.accent : MereTheme.border.opacity(0.6),
                                lineWidth: 1
                            )
                        )
                        .foregroundStyle(isAvailable(mode) ? MereTheme.textPrimary : MereTheme.textMuted)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 2)
        }
        .defaultScrollAnchor(.leading)
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: MereTheme.Spacing.m) {
            TextField(
                "",
                text: $prompt,
                prompt: Text(selected.promptPlaceholder).foregroundColor(MereTheme.textMuted),
                axis: .vertical
            )
            .font(.body)
            .foregroundStyle(MereTheme.textPrimary)
            .lineLimit(4...9)

            Divider().overlay(MereTheme.border.opacity(0.5))

            HStack {
                Menu {
                    Button("Fleet default") { model = "" }
                    ForEach(installedModels, id: \.self) { id in
                        Button(id) { model = id }
                    }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "cpu")
                        Text(model.isEmpty ? "Fleet default" : model)
                            .lineLimit(1)
                    }
                    .font(.footnote)
                    .foregroundStyle(MereTheme.textSecondary)
                }
                Spacer()
                Button {
                    showOptions = true
                } label: {
                    Label("Options", systemImage: "slider.horizontal.3")
                        .font(.footnote)
                        .foregroundStyle(MereTheme.textSecondary)
                }
            }
        }
        .padding(MereTheme.Spacing.l)
        .merePanel()
    }

    private var runButton: some View {
        Button {
            Task { await submit() }
        } label: {
            Group {
                if submitting {
                    ProgressView().tint(.white)
                } else {
                    Text("Run on your fleet")
                        .font(.body.weight(.semibold))
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .disabled(prompt.isEmpty || submitting || !isAvailable(selected))
    }

    private var optionsSheet: some View {
        NavigationStack {
            List {
                if !optionalStringInputs.isEmpty {
                    Section {
                        ForEach(optionalStringInputs, id: \.name) { field in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(fieldTitle(field.name))
                                    .font(.footnote)
                                    .foregroundStyle(MereTheme.textMuted)
                                TextField(
                                    "",
                                    text: binding(for: field.name, in: $textFields),
                                    axis: field.multiline == true ? .vertical : .horizontal
                                )
                                .lineLimit(field.multiline == true ? 2...6 : 1...1)
                            }
                        }
                    }
                }
                if !numericInputs.isEmpty {
                    Section {
                        ForEach(numericInputs, id: \.name) { field in
                            LabeledContent(fieldTitle(field.name)) {
                                TextField("auto", text: binding(for: field.name, in: $numberFields))
                                    .keyboardType(field.type == .integer ? .numberPad : .decimalPad)
                                    .multilineTextAlignment(.trailing)
                                    .frame(maxWidth: 120)
                            }
                        }
                    }
                }
                if !enumInputs.isEmpty {
                    Section {
                        ForEach(enumInputs, id: \.name) { field in
                            Picker(fieldTitle(field.name), selection: binding(for: field.name, in: $enumFields)) {
                                Text("auto").tag("")
                                ForEach(field.values ?? [], id: \.self) { value in
                                    Text(value).tag(value)
                                }
                            }
                        }
                    }
                }
            }
            .scrollContentBackground(.hidden)
            .background(MereTheme.background)
            .navigationTitle("Options")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { showOptions = false }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private func fieldTitle(_ name: String) -> String {
        name.replacingOccurrences(of: "_", with: " ")
    }

    private func binding(for name: String, in store: Binding<[String: String]>) -> Binding<String> {
        Binding(
            get: { store.wrappedValue[name] ?? "" },
            set: { store.wrappedValue[name] = $0 }
        )
    }

    private func submit() async {
        submitting = true
        defer { submitting = false }
        var arguments: [String: WorkflowValue] = [selected.promptField: .string(prompt)]
        if !model.isEmpty {
            arguments["model"] = .string(model)
        }
        for (name, raw) in textFields where !raw.isEmpty {
            arguments[name] = .string(raw)
        }
        for (name, raw) in enumFields where !raw.isEmpty {
            arguments[name] = .string(raw)
        }
        for field in numericInputs {
            let raw = (numberFields[field.name] ?? "").trimmingCharacters(in: .whitespaces)
            guard !raw.isEmpty else { continue }
            if field.type == .integer, let value = Int64(raw) {
                arguments[field.name] = .integer(value)
            } else if field.type == .number, let value = Double(raw) {
                arguments[field.name] = .number(value)
            }
        }
        do {
            let job = try await relay.submit(kind: selected.kind, arguments: arguments)
            errorMessage = nil
            submittedJobID = job.jobID
        } catch let error as RelayClientError {
            errorMessage = error.message
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

/// The single inline notice component, in the Studio's one voice.
struct MereBannerView: View {
    let text: String
    let color: Color

    var body: some View {
        HStack(alignment: .top, spacing: MereTheme.Spacing.s) {
            Circle().fill(color).frame(width: 8, height: 8).padding(.top, 5)
            Text(text)
                .font(.footnote)
                .foregroundStyle(MereTheme.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(MereTheme.Spacing.m)
        .merePanel()
    }
}
