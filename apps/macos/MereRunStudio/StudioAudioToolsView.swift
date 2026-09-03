import SwiftUI
import UniformTypeIdentifiers

enum StudioAudioTool: String, CaseIterable, Identifiable {
    case enhance
    case separate

    var id: String { rawValue }

    var title: String {
        switch self {
        case .enhance: "Enhance"
        case .separate: "Separate / Restore"
        }
    }

    var symbol: String {
        switch self {
        case .enhance: "waveform.badge.plus"
        case .separate: "slider.horizontal.3"
        }
    }

    var templateID: CommandTemplateID {
        switch self {
        case .enhance: .audioEnhance
        case .separate: .musicSeparate
        }
    }

    var mode: StudioMode {
        switch self {
        case .enhance: .listen
        case .separate: .music
        }
    }
}

struct StudioAudioToolsView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    /// Owned by the shell's task control; Enhance and Separate are toolbar tasks now.
    @Binding var tool: StudioAudioTool
    @State private var enhanceDraft: CommandDraft
    @State private var separationDraft: CommandDraft
    @State private var requestID: UUID?
    @State private var statusMessage: String?

    init(tool: Binding<StudioAudioTool>) {
        _tool = tool
        _enhanceDraft = State(
            initialValue: CommandCatalog.template(id: .audioEnhance)?.defaultDraft() ?? CommandDraft()
        )
        _separationDraft = State(
            initialValue: CommandCatalog.template(id: .musicSeparate)?.defaultDraft() ?? CommandDraft()
        )
    }

    private var activeDraft: CommandDraft {
        tool == .enhance ? enhanceDraft : separationDraft
    }

    var body: some View {
        HStack(spacing: 0) {
            controls
                .frame(width: 440)
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            resultPanel
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onChange(of: tool) { _, _ in
            requestID = nil
            statusMessage = nil
        }
    }

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                StudioPathField(
                    label: "Source audio",
                    placeholder: "WAV, MP3, M4A, FLAC, or supported audio",
                    path: inputBinding,
                    allowedContentTypes: [.audio]
                )
                if tool == .enhance {
                    enhancementControls
                } else {
                    separationControls
                }
                Divider().overlay(MereRunTheme.border.opacity(0.5))
                StudioPathField(
                    label: tool == .enhance ? "Enhanced WAV" : "Stem output directory",
                    placeholder: tool == .enhance ? "48 kHz mono WAV" : "Directory for stems and manifest",
                    path: outputBinding,
                    picksDirectory: tool == .separate
                )
                Button {
                    run()
                } label: {
                    Label("Run \(tool.title)", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
                if let statusMessage {
                    Text(statusMessage)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            .padding(18)
        }
    }

    @ViewBuilder
    private var enhancementControls: some View {
        Picker("Workflow", selection: $enhanceDraft.model) {
            Text("Speech · AP-BWE").tag("audio-enhance-ap-bwe-16kto48k")
            Text("General audio · UniverSR").tag("audio-enhance-universr-audio")
        }
        .pickerStyle(.segmented)

        Text(enhanceDraft.model.localizedCaseInsensitiveContains("universr")
            ? "Flow-matching super-resolution for speech, music, or effects with 8–24 kHz effective input bandwidth."
            : "Deterministic speech bandwidth extension from 16 kHz to 48 kHz.")
            .font(MereRunTheme.captionFont)
            .foregroundStyle(MereRunTheme.textMuted)

        if enhanceDraft.model.localizedCaseInsensitiveContains("universr") {
            Picker(
                "Input bandwidth",
                selection: Binding(
                    get: { enhanceDraft.audioInputRate ?? 0 },
                    set: { enhanceDraft.audioInputRate = $0 == 0 ? nil : $0 }
                )
            ) {
                Text("Auto").tag(0)
                Text("8 kHz").tag(8_000)
                Text("12 kHz").tag(12_000)
                Text("16 kHz").tag(16_000)
                Text("24 kHz").tag(24_000)
            }
            Picker(
                "ODE method",
                selection: Binding(
                    get: { enhanceDraft.audioODEMethod ?? "midpoint" },
                    set: { enhanceDraft.audioODEMethod = $0 }
                )
            ) {
                Text("Euler").tag("euler")
                Text("Midpoint").tag("midpoint")
                Text("RK4").tag("rk4")
            }
            .pickerStyle(.segmented)
            HStack {
                Stepper(
                    "ODE steps \(enhanceDraft.audioODESteps ?? 4)",
                    value: Binding(
                        get: { enhanceDraft.audioODESteps ?? 4 },
                        set: { enhanceDraft.audioODESteps = $0 }
                    ),
                    in: 1...100
                )
                Stepper(
                    "Chunk \(enhanceDraft.audioChunkSeconds ?? 10)s",
                    value: Binding(
                        get: { enhanceDraft.audioChunkSeconds ?? 10 },
                        set: { enhanceDraft.audioChunkSeconds = $0 }
                    ),
                    in: 3...600
                )
            }
            numberField(
                "Guidance",
                value: Binding(
                    get: { enhanceDraft.audioGuidanceScale ?? 1.5 },
                    set: { enhanceDraft.audioGuidanceScale = $0 }
                )
            )
            labeledTextField("Seed", placeholder: "42", text: $enhanceDraft.seed)
        } else {
            overlapStepper(draft: $enhanceDraft)
        }
        computePicker(draft: $enhanceDraft, fallback: "float32")
    }

    @ViewBuilder
    private var separationControls: some View {
        Picker("Workflow", selection: $separationDraft.model) {
            Text("Vocals + instrumental").tag("music-separate-bs-roformer-viperx-1297")
            Text("Four stems").tag("music-separate-bs-roformer-4stem")
            Text("Dereverb").tag("music-separate-mel-roformer-dereverb")
            Text("Denoise").tag("music-separate-mel-roformer-denoise")
        }
        Text(separationDescription)
            .font(MereRunTheme.captionFont)
            .foregroundStyle(MereRunTheme.textMuted)
        overlapStepper(draft: $separationDraft)
        computePicker(draft: $separationDraft, fallback: "float16")
    }

    private var resultPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Result")
                .font(MereRunTheme.sectionFont)
            if let requestID {
                StudioSpecialistResultView(
                    requestID: requestID,
                    preferredKinds: [.audio, .text]
                )
            } else {
                ContentUnavailableView(
                    "No audio run yet",
                    systemImage: tool.symbol,
                    description: Text("Choose a source and workflow. Outputs and provenance stay in Library.")
                )
            }
        }
        .padding(18)
    }

    private var inputBinding: Binding<String> {
        tool == .enhance ? $enhanceDraft.inputPath : $separationDraft.inputPath
    }

    private var outputBinding: Binding<String> {
        tool == .enhance ? $enhanceDraft.outputPath : $separationDraft.outputPath
    }

    private var separationDescription: String {
        if separationDraft.model.contains("4stem") {
            return "Produces drums, bass, other, and vocals as ordinary 44.1 kHz stereo WAV stems."
        }
        if separationDraft.model.contains("dereverb") {
            return "Removes room reverberation with native MelBand RoFormer restoration."
        }
        if separationDraft.model.contains("denoise") {
            return "Removes broadband noise with native MelBand RoFormer restoration."
        }
        return "Produces vocals and instrumental as ordinary 44.1 kHz stereo WAV stems."
    }

    private func overlapStepper(draft: Binding<CommandDraft>) -> some View {
        Stepper(
            "Chunk overlap \(draft.wrappedValue.audioOverlap ?? 2)",
            value: Binding(
                get: { draft.wrappedValue.audioOverlap ?? 2 },
                set: { draft.wrappedValue.audioOverlap = $0 }
            ),
            in: 1...64
        )
    }

    private func computePicker(draft: Binding<CommandDraft>, fallback: String) -> some View {
        Picker(
            "Compute",
            selection: Binding(
                get: { draft.wrappedValue.audioDType ?? fallback },
                set: { draft.wrappedValue.audioDType = $0 }
            )
        ) {
            Text("Float 16").tag("float16")
            Text("Float 32").tag("float32")
        }
        .pickerStyle(.segmented)
    }

    private func numberField(_ label: String, value: Binding<Double>) -> some View {
        HStack {
            Text(label)
                .font(MereRunTheme.captionFont)
            Spacer()
            TextField(label, value: value, format: .number)
                .frame(width: 100)
                .mereField(cornerRadius: MereRunTheme.Radius.sm)
        }
    }

    private func labeledTextField(_ label: String, placeholder: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            TextField(placeholder, text: text)
                .mereField()
        }
    }

    private func run() {
        guard let template = CommandCatalog.template(id: tool.templateID) else {
            statusMessage = "The selected audio workflow is unavailable."
            return
        }
        let draft = activeDraft
        if let validation = template.validationMessage(for: draft) {
            statusMessage = validation
            return
        }
        requestID = StudioSpecialistRunner.submit(
            templateID: tool.templateID,
            mode: tool.mode,
            draft: draft,
            controller: controller,
            library: library
        )
        statusMessage = "\(tool.title) submitted."
        advanceOutput()
    }

    private func advanceOutput() {
        guard let template = CommandCatalog.template(id: tool.templateID) else { return }
        let next = template.defaultDraft().outputPath
        if tool == .enhance {
            enhanceDraft.outputPath = next
        } else {
            separationDraft.outputPath = next
        }
    }
}
