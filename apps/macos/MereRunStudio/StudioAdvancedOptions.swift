import AppKit
import MereRunContract
import SwiftUI
import UniformTypeIdentifiers

/// The inspector's Advanced section: every control of the mode's command that has no section
/// of its own. The essentials moved into the inspector sections (negative prompt and system
/// prompt to Prompt; size, length, and quality to Output; model, LoRA, adapters, and voice mode
/// to Model & adapters; steps, guidance, seed, temperature, top-p, thinking, and response format
/// to Sampling); everything below stayed as it was in the former options popover.
struct StudioAdvancedOptions: View {
    let mode: StudioMode
    @Binding var draft: StudioDraft
    let onShowAdapters: () -> Void
    let onShowRealtimeMusic: () -> Void
    @EnvironmentObject private var controller: MereRunController
    @State private var voiceProfiles: [StudioVoiceProfile] = []
    @State private var showImageEditor = false

    var body: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            if mode == .speak {
                speakOptions
            }

            if mode == .video {
                videoOptions
            }

            if mode == .chat {
                chatOptions
            }

            if mode == .createImage {
                imageOptions
            }

            if mode == .music {
                musicOptions
            }

            if !visibleOptionFields.isEmpty {
                ForEach(visibleOptionFields) { field in
                    optionRow(field)
                }
            }
        }
        .font(MereRunTheme.captionFont)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task {
            if mode == .speak { voiceProfiles = await controller.loadVoiceProfiles() }
        }
        .onAppear(perform: normalizeMiniMaxH3Draft)
        .onChange(of: draft.model) { _, _ in normalizeMiniMaxH3Draft() }
        .sheet(isPresented: $showImageEditor) {
            if !draft.inputPath.isBlank {
                StudioImageEditor(
                    inputURL: URL(fileURLWithPath: draft.inputPath),
                    outputWidth: draft.width,
                    outputHeight: draft.height,
                    maskPath: $draft.imageMaskPath,
                    outpaintTop: $draft.imageOutpaintTop,
                    outpaintRight: $draft.imageOutpaintRight,
                    outpaintBottom: $draft.imageOutpaintBottom,
                    outpaintLeft: $draft.imageOutpaintLeft,
                    feather: $draft.imageMaskFeather
                )
            }
        }
    }

    /// The schema fields without a section of their own: guidance (`cfg`) lives in Sampling,
    /// the frame count in Output, and temperature, top-p, and max tokens in Sampling.
    private var visibleOptionFields: [StudioOptionField] {
        var owned: Set<String> = ["cfg", "numFrames", "temperature", "topP", "maxTokens"]
        if mode == .video, videoFamily.isMiniMaxH3 { owned.insert("fps") }
        return StudioOptionSchema.fields(for: mode).filter { !owned.contains($0.id) }
    }

    private var videoFamily: StudioVideoModelFamily {
        StudioVideoModelFamily(model: draft.model)
    }

    @ViewBuilder
    private var speakOptions: some View {
        if draft.voiceMode == "clone" {
            Picker("Profile", selection: $draft.voiceProfile) {
                Text("None").tag("")
                ForEach(voiceProfiles) { profile in
                    Text(profile.name).tag(profile.id)
                }
            }
            HStack(spacing: MereRunTheme.Spacing.sm) {
                Text(draft.refAudioPath.isEmpty
                    ? "No reference audio"
                    : URL(fileURLWithPath: draft.refAudioPath).lastPathComponent)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                Spacer()
                Button("Reference audio…") { chooseReferenceAudio() }
                    .controlSize(.small)
            }
            TextField("Save as profile (optional)", text: $draft.saveProfileName)
                .mereField()
        }
    }

    /// Renders one schema field as the appropriate control, bound through the draft key path.
    @ViewBuilder
    private func optionRow(_ field: StudioOptionField) -> some View {
        switch field.control {
        case let .int(keyPath, range, step):
            Stepper("\(field.label): \(draft[keyPath: keyPath])", value: binding(keyPath), in: range, step: step)
                .font(MereRunTheme.captionFont)
        case let .double(keyPath):
            HStack {
                Text(field.label)
                    .font(MereRunTheme.captionFont)
                Spacer()
                TextField(field.label, value: binding(keyPath), format: .number)
                    .frame(width: 80)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
            }
        case let .bool(keyPath):
            Toggle(field.label, isOn: binding(keyPath))
                .font(MereRunTheme.captionFont)
        case let .text(keyPath, placeholder):
            HStack {
                Text(field.label)
                    .font(MereRunTheme.captionFont)
                Spacer()
                TextField(placeholder, text: binding(keyPath))
                    .frame(width: 150)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
            }
        }
    }

    private func binding<Value>(_ keyPath: WritableKeyPath<StudioDraft, Value>) -> Binding<Value> {
        Binding(get: { draft[keyPath: keyPath] }, set: { draft[keyPath: keyPath] = $0 })
    }

    private func chooseReferenceAudio() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            draft.refAudioPath = url.path
        }
    }

    @ViewBuilder
    private var chatOptions: some View {
        if draft.model.localizedCaseInsensitiveContains("inkling") {
            HStack {
                Text("Reasoning effort")
                    .font(MereRunTheme.captionFont)
                Spacer()
                TextField(
                    "0.9",
                    value: Binding(
                        get: { draft.reasoningEffort ?? 0.9 },
                        set: { draft.reasoningEffort = min(max($0, 0), 0.99) }
                    ),
                    format: .number
                )
                .frame(width: 80)
                .mereField(cornerRadius: MereRunTheme.Radius.sm)
            }
        }

        Stepper(
            "Context: \(draft.contextSize == 0 ? "model default" : String(draft.contextSize))",
            value: $draft.contextSize,
            in: 0...262_144,
            step: 1_024
        )
        .font(MereRunTheme.captionFont)
        Stepper(
            "Top-k: \(draft.topK == 0 ? "model default" : String(draft.topK))",
            value: $draft.topK,
            in: 0...512
        )
        .font(MereRunTheme.captionFont)
        TextField("Min-p (0 = model default)", value: $draft.minP, format: .number)
            .textFieldStyle(.roundedBorder)

        Picker("KV bits", selection: $draft.kvBits) {
            Text("Auto").tag(0)
            Text("4-bit").tag(4)
            Text("8-bit").tag(8)
        }
        .pickerStyle(.segmented)
        Picker("KV scheme", selection: $draft.kvQuantScheme) {
            Text("Auto").tag("")
            Text("Uniform").tag("uniform")
            Text("Polar").tag("polar")
            Text("Turbo").tag("turboquant")
        }
        .pickerStyle(.segmented)
        Stepper("KV group: \(draft.kvGroupSize)", value: $draft.kvGroupSize, in: 0...1_024, step: 8)
            .font(MereRunTheme.captionFont)
        Stepper(
            "Quantize after: \(draft.quantizedKVStart)",
            value: $draft.quantizedKVStart,
            in: 0...262_144,
            step: 128
        )
        .font(MereRunTheme.captionFont)

        TextField("Tools: write_file, shell_exec", text: $draft.tools)
            .mereField()
        TextField("Tool sandbox directory", text: $draft.sandboxDir)
            .mereField()
        Toggle("Tool loop", isOn: $draft.toolLoop)
            .font(MereRunTheme.captionFont)
        Toggle("Allow shell execution", isOn: $draft.allowShellExec)
            .font(MereRunTheme.captionFont)
        Toggle("Allow absolute tool paths", isOn: $draft.allowAbsoluteToolPaths)
            .font(MereRunTheme.captionFont)
        Toggle("Auto-approve tool calls", isOn: $draft.autoApproveTools)
            .font(MereRunTheme.captionFont)
        Toggle("Generation stats", isOn: $draft.stats)
            .font(MereRunTheme.captionFont)
        Toggle("Preflight only", isOn: $draft.preflight)
            .font(MereRunTheme.captionFont)
        if draft.preflight {
            Toggle("JSON preflight report", isOn: $draft.preflightJSON)
                .font(MereRunTheme.captionFont)
        }
        Toggle("Require installed model", isOn: $draft.requireInstalled)
            .font(MereRunTheme.captionFont)
    }

    @ViewBuilder
    private var videoOptions: some View {
        if videoFamily.isMiniMaxH3 {
            Text("Synchronized 24 fps video + 32 kHz stereo audio")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Picker(
                "Weights",
                selection: Binding(
                    get: { draft.h3WeightMode ?? "auto" },
                    set: { draft.h3WeightMode = $0 }
                )
            ) {
                Text("Auto").tag("auto")
                Text("Quantized").tag("quantized")
                Text("BF16").tag("resident-bf16")
            }
            .pickerStyle(.segmented)
            Picker(
                "Denoise",
                selection: Binding(
                    get: { draft.h3AccelerationMode ?? "quality" },
                    set: { draft.h3AccelerationMode = $0 }
                )
            ) {
                Text("Exact").tag("quality")
                Text("Balanced").tag("balanced")
                Text("Maximum").tag("maximum")
            }
            .pickerStyle(.segmented)
            Text("Balanced and Maximum trade exact-seed trajectory fidelity for faster denoising.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Toggle(
                "Override adaptive schedule",
                isOn: Binding(
                    get: { draft.h3Steps != nil },
                    set: { draft.h3Steps = $0 ? 21 : nil }
                )
            )
            .font(MereRunTheme.captionFont)
            if draft.h3Steps != nil {
                Stepper(
                    "Schedule points \(draft.h3Steps ?? 21)",
                    value: Binding(
                        get: { draft.h3Steps ?? 21 },
                        set: { draft.h3Steps = $0 }
                    ),
                    in: 1...64
                )
                .font(MereRunTheme.captionFont)
            }
            Stepper("Frames \(draft.numFrames) · 17n+5", value: $draft.numFrames, in: 22...600, step: 17)
                .font(MereRunTheme.captionFont)
            Text("Frame rate is fixed at 24 fps.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            if videoFamily == .miniMaxH3Ref2VA {
                compactH3References
            } else {
                videoAssetRow(title: "End keyframe", path: $draft.endImagePath, type: .image)
            }
        } else {
            if videoFamily == .ltx {
                Picker("Quality", selection: $draft.videoQuality) {
                    Text("Draft").tag(LTXVideoQuality.draft)
                    Text("Final").tag(LTXVideoQuality.final)
                }
                .pickerStyle(.segmented)

                Picker("Output", selection: $draft.videoOutputMode) {
                    Text("Video").tag(LTXVideoOutputMode.videoOnly)
                    Text("Audio + Video").tag(LTXVideoOutputMode.audioVideo)
                }
                .pickerStyle(.segmented)

                videoAssetRow(title: "Source audio", path: $draft.audioPath, type: .audio)
            }
            videoAssetRow(title: "End keyframe", path: $draft.endImagePath, type: .image)
        }

        if videoFamily == .ltx, !draft.audioPath.isBlank {
            numberField("Audio start seconds", value: $draft.audioStartTime)
            numberField("Audio max seconds (0 = video)", value: $draft.audioMaxDuration)
            Stepper("A2V steps \(draft.a2vSteps)", value: $draft.a2vSteps, in: 1...100)
                .font(MereRunTheme.captionFont)
            numberField("A2V guidance", value: $draft.a2vGuidanceScale)
            numberField("Video CFG", value: $draft.videoCFGGuidanceScale)
            numberField("Audio CFG", value: $draft.audioCFGGuidanceScale)
            numberField("V2A guidance", value: $draft.v2aGuidanceScale)
        }
        if !draft.endImagePath.isBlank {
            numberField("End image strength", value: $draft.endImageStrength)
        }
        Toggle("Preflight only", isOn: $draft.preflight)
            .font(MereRunTheme.captionFont)
        if !videoFamily.isMiniMaxH3 {
            Toggle("Capture phase timings", isOn: $draft.timings)
                .font(MereRunTheme.captionFont)
        }
    }

    @ViewBuilder
    private var compactH3References: some View {
        let references = draft.h3ReferenceInputs ?? []
        VStack(alignment: .leading, spacing: 6) {
            Text("Ordered Ref2VA references")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            ForEach(Array(references.enumerated()), id: \.offset) { index, reference in
                HStack(spacing: 5) {
                    Text(reference)
                        .font(MereRunTheme.captionFont)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button { moveH3Reference(index, by: -1) } label: { Image(systemName: "arrow.up") }
                        .disabled(index == 0)
                    Button { moveH3Reference(index, by: 1) } label: { Image(systemName: "arrow.down") }
                        .disabled(index == references.count - 1)
                    Button(role: .destructive) { removeH3Reference(index) } label: { Image(systemName: "trash") }
                }
            }
            HStack {
                Button("Image") { chooseH3Reference(kind: "image", type: .image) }
                Button("Video") { chooseH3Reference(kind: "video", type: .movie) }
                Button("Audio") { chooseH3Reference(kind: "audio", type: .audio) }
            }
            .controlSize(.small)
        }
    }

    @ViewBuilder
    private var imageOptions: some View {
        if !draft.inputPath.isBlank {
            HStack(spacing: MereRunTheme.Spacing.sm) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(draft.imageMaskPath.isBlank && !hasImageOutpaint ? "Whole-image transform" : "Masked image edit")
                        .font(.system(size: 12.5, weight: .medium))
                    Text(draft.imageMaskPath.isBlank
                        ? (hasImageOutpaint ? "Outpaint edges are editable" : "Paint a mask or expand the canvas")
                        : URL(fileURLWithPath: draft.imageMaskPath).lastPathComponent)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
                Button("Edit canvas…") { showImageEditor = true }
                    .controlSize(.small)
            }
            .padding(MereRunTheme.Spacing.sm)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .fill(MereRunTheme.accentSoft.opacity(0.55))
            }
        }

        HStack(spacing: MereRunTheme.Spacing.sm) {
            Text(draft.referenceImagePaths.isBlank
                ? "No extra references"
                : "\(imageReferencePaths.count) reference image\(imageReferencePaths.count == 1 ? "" : "s")")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Spacer()
            Button("References…", action: chooseImageReferences)
                .controlSize(.small)
            if !draft.referenceImagePaths.isBlank {
                Button {
                    draft.referenceImagePaths = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.borderless)
                .accessibilityLabel("Remove reference images")
            }
        }

        Toggle("Keep original aspect for one HiDream reference", isOn: $draft.keepOriginalAspect)
            .font(MereRunTheme.captionFont)

        Toggle("Expand into a structured JSON prompt", isOn: $draft.structuredPrompt)
            .font(MereRunTheme.captionFont)
        if draft.structuredPrompt {
            TextField("Structured prompt model", text: $draft.structuredPromptModel)
                .mereField()
            Stepper(
                "Prompt max tokens \(draft.structuredPromptMaxTokens)",
                value: $draft.structuredPromptMaxTokens,
                in: 128...16_384,
                step: 128
            )
            .font(MereRunTheme.captionFont)
        }

        DisclosureGroup("Krea tuning") {
            VStack(spacing: MereRunTheme.Spacing.sm) {
                numberField(
                    "Conditioning multiplier",
                    value: $draft.kreaConditioningMultiplier
                )
                TextField("Layer weights, comma-separated", text: $draft.kreaConditioningLayerWeights)
                    .mereField()
                Picker("Base quantization", selection: $draft.kreaBaseQuantizationBits) {
                    Text("Automatic").tag("")
                    Text("4-bit").tag("4")
                    Text("8-bit").tag("8")
                }
            }
            .padding(.top, MereRunTheme.Spacing.xs)
        }

        Toggle("Preflight only", isOn: $draft.preflight)
            .font(MereRunTheme.captionFont)
        if draft.preflight {
            Toggle("JSON preflight report", isOn: $draft.preflightJSON)
                .font(MereRunTheme.captionFont)
        }
        Toggle("Machine-readable progress", isOn: $draft.progressJSON)
            .font(MereRunTheme.captionFont)
    }

    @ViewBuilder
    private var musicOptions: some View {
        Button {
            onShowRealtimeMusic()
        } label: {
            HStack {
                Image(systemName: "waveform.path.ecg")
                    .foregroundStyle(MereRunTheme.accent)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Open realtime performance")
                        .font(.system(size: 12.5, weight: .semibold))
                    Text("Live playback, recording, MIDI, and prompt steering")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(MereRunTheme.Spacing.sm)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .fill(MereRunTheme.accentSoft.opacity(0.55))
            }
        }
        .buttonStyle(.plain)

        Picker("Task", selection: $draft.musicTask) {
            Text("Create").tag("text2music")
            Text("Cover").tag("cover")
            Text("No-FSQ").tag("cover-nofsq")
            Text("Repaint").tag("repaint")
            Text("Extract").tag("extract")
            Text("Lego").tag("lego")
            Text("Complete").tag("complete")
        }

        if draft.musicTask != "text2music" || draft.musicFlowEdit {
            videoAssetRow(title: "Source audio", path: $draft.musicSourceAudio, type: .audio)
            numberField("Cover strength", value: $draft.musicCoverStrength)
            numberField("Cover noise", value: $draft.musicCoverNoiseStrength)
        }

        HStack(spacing: MereRunTheme.Spacing.sm) {
            Text(draft.musicReferenceAudioPaths.isBlank
                ? "No timbre references"
                : "\(musicReferencePaths.count) timbre reference\(musicReferencePaths.count == 1 ? "" : "s")")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
            Spacer()
            Button("References…", action: chooseMusicReferences)
                .controlSize(.small)
        }

        Picker("LM planning", selection: $draft.musicLMMode) {
            Text("Preset").tag("auto")
            Text("On").tag("use")
            Text("Off").tag("disable")
        }
        .pickerStyle(.segmented)
        Toggle("Analyze source metadata", isOn: $draft.musicAnalyzeSourceAudio)
            .font(MereRunTheme.captionFont)
        HStack(spacing: MereRunTheme.Spacing.sm) {
            numberField("LM temperature", value: $draft.musicLMTemperature)
            numberField(
                "LM repetition penalty",
                value: $draft.musicLMRepetitionPenalty
            )
            numberField("LM CFG scale", value: $draft.musicLMCFGScale)
        }
        TextField("LM negative prompt", text: $draft.musicLMNegativePrompt)
            .textFieldStyle(.roundedBorder)
        Toggle("Instrumental", isOn: $draft.musicInstrumental)
            .font(MereRunTheme.captionFont)

        Stepper(
            "Candidates \(draft.musicCandidates == 0 ? "preset" : String(draft.musicCandidates))",
            value: $draft.musicCandidates,
            in: 0...16
        )
        .font(MereRunTheme.captionFont)
        Toggle("Keep all ranked candidates", isOn: $draft.musicKeepCandidates)
            .font(MereRunTheme.captionFont)

        Toggle("Flow edit source toward prompt", isOn: $draft.musicFlowEdit)
            .font(MereRunTheme.captionFont)
        if draft.musicFlowEdit {
            TextField("Source caption", text: $draft.musicSourceCaption)
                .mereField()
            TextField("Source lyrics", text: $draft.musicSourceLyrics, axis: .vertical)
                .lineLimit(1...4)
                .mereField()
        }

        DisclosureGroup("Adapters & delivery") {
            VStack(spacing: MereRunTheme.Spacing.sm) {
                Picker("Adapter kind", selection: $draft.musicAdapterKind) {
                    Text("Auto").tag("auto")
                    Text("LoRA").tag("lora")
                    Text("LoKr").tag("lokr")
                }
                .pickerStyle(.segmented)
                TextField("Adapter scales, one per line", text: $draft.musicAdapterScales)
                    .mereField()
                videoAssetRow(title: "LRC lyrics", path: $draft.musicLRCFile, type: .plainText)
                TextField("Stems (Drums,Bass,Vocals)", text: $draft.musicStems)
                    .mereField()
                TextField("DAW bundle directory", text: $draft.musicDAWBundle)
                    .mereField()
                Picker("WAV format", selection: $draft.musicExportFormat) {
                    Text("PCM 16").tag("pcm16")
                    Text("PCM 24").tag("pcm24")
                    Text("Float 32").tag("float32")
                }
                .pickerStyle(.segmented)
                Toggle("Skip recipe sidecar", isOn: $draft.musicNoRecipe)
                    .font(MereRunTheme.captionFont)
            }
            .padding(.top, MereRunTheme.Spacing.xs)
        }
    }

    private func numberField(_ label: String, value: Binding<Double>) -> some View {
        HStack {
            Text(label)
                .font(MereRunTheme.captionFont)
            Spacer()
            TextField(label, value: value, format: .number)
                .frame(width: 90)
                .mereField(cornerRadius: MereRunTheme.Radius.sm)
        }
    }

    private func videoAssetRow(title: String, path: Binding<String>, type: UTType) -> some View {
        HStack(spacing: MereRunTheme.Spacing.sm) {
            Text(path.wrappedValue.isEmpty
                ? "No \(title.lowercased())"
                : URL(fileURLWithPath: path.wrappedValue).lastPathComponent)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
            Spacer()
            Button(title) {
                let panel = NSOpenPanel()
                panel.allowedContentTypes = [type]
                panel.allowsMultipleSelection = false
                panel.canChooseDirectories = false
                if panel.runModal() == .OK, let url = panel.url {
                    path.wrappedValue = url.path
                    if mode == .video && type == .audio {
                        draft.videoQuality = .final
                        draft.videoOutputMode = .audioVideo
                    }
                }
            }
            .controlSize(.small)
            if !path.wrappedValue.isEmpty {
                Button {
                    path.wrappedValue = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.borderless)
                .accessibilityLabel("Remove \(title.lowercased())")
            }
        }
    }

    private func normalizeMiniMaxH3Draft() {
        guard mode == .video, videoFamily.isMiniMaxH3 else { return }
        draft.fps = 24
        draft.width = max(32, (draft.width / 32) * 32)
        draft.height = max(32, (draft.height / 32) * 32)
        draft.numFrames = StudioVideoModelFamily.alignedMiniMaxH3FrameCount(draft.numFrames)
        draft.audioPath = ""
        draft.timings = false
        draft.timingsOutputPath = ""
        if videoFamily == .miniMaxH3Ref2VA {
            draft.inputPath = ""
            draft.endImagePath = ""
        } else {
            draft.h3ReferenceInputs = []
        }
    }

    private func chooseH3Reference(kind: String, type: UTType) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [type]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        draft.h3ReferenceInputs = (draft.h3ReferenceInputs ?? []) + ["\(kind):\(url.path)"]
    }

    private func moveH3Reference(_ index: Int, by offset: Int) {
        var references = draft.h3ReferenceInputs ?? []
        references.swapAt(index, index + offset)
        draft.h3ReferenceInputs = references
    }

    private func removeH3Reference(_ index: Int) {
        var references = draft.h3ReferenceInputs ?? []
        references.remove(at: index)
        draft.h3ReferenceInputs = references
    }

    private var imageReferencePaths: [String] {
        draft.referenceImagePaths.components(separatedBy: .newlines)
            .flatMap { $0.split(separator: ",", omittingEmptySubsequences: true) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private var musicReferencePaths: [String] {
        separatedPaths(draft.musicReferenceAudioPaths)
    }

    private var hasImageOutpaint: Bool {
        [
            draft.imageOutpaintTop,
            draft.imageOutpaintRight,
            draft.imageOutpaintBottom,
            draft.imageOutpaintLeft,
        ].contains(where: { $0 > 0 })
    }

    private func separatedPaths(_ raw: String) -> [String] {
        raw.components(separatedBy: .newlines)
            .flatMap { $0.split(separator: ",", omittingEmptySubsequences: true) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func chooseImageReferences() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        if panel.runModal() == .OK {
            draft.referenceImagePaths = panel.urls.map(\.path).joined(separator: "\n")
        }
    }

    private func chooseMusicReferences() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        if panel.runModal() == .OK {
            draft.musicReferenceAudioPaths = panel.urls.map(\.path).joined(separator: "\n")
        }
    }

}
