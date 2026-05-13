import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct MereRunRootView: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        StudioRootView()
            .background(MereRunTheme.background.ignoresSafeArea())
            .foregroundStyle(MereRunTheme.textPrimary)
            .onAppear {
                controller.refreshResolvedCLI()
            }
    }
}

struct AdvancedControlSurface: View {
    var body: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 0) {
                CommandSidebar()
                    .frame(width: 268)

                Divider()
                    .overlay(MereRunTheme.border.opacity(0.6))

                CommandEditor()
                    .frame(width: 500)

                Divider()
                    .overlay(MereRunTheme.border.opacity(0.6))

                RunConsole()
                    .frame(width: 440)
            }
            .frame(width: 1_210)
        }
        .background(MereRunTheme.background.ignoresSafeArea())
        .foregroundStyle(MereRunTheme.textPrimary)
    }
}

private struct CommandSidebar: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                Text("mere.run")
                    .font(.system(size: 24, weight: .semibold))
                Text("local control surface")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 18)
            .padding(.top, 22)
            .padding(.bottom, 18)

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(CommandCategory.allCases) { category in
                        let templates = CommandCatalog.templates(in: category)
                        if !templates.isEmpty {
                            VStack(alignment: .leading, spacing: 6) {
                                Text(category.rawValue.uppercased())
                                    .font(MereRunTheme.sectionFont)
                                    .foregroundStyle(MereRunTheme.textMuted)
                                    .padding(.horizontal, 18)

                                ForEach(templates) { template in
                                    CommandRow(
                                        template: template,
                                        isSelected: template.id == controller.selectedTemplate.id
                                    )
                                }
                            }
                        }
                    }
                }
                .padding(.bottom, 18)
            }

            Divider()
                .overlay(MereRunTheme.border.opacity(0.6))

            VStack(alignment: .leading, spacing: 8) {
                Label(controller.resolvedCLI, systemImage: "terminal")
                    .lineLimit(2)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)

                HStack {
                    Circle()
                        .fill(controller.isRunning ? MereRunTheme.yellow : statusColor)
                        .frame(width: 8, height: 8)
                    Text(controller.status)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    Spacer()
                }
            }
            .padding(18)
        }
        .background(MereRunTheme.background)
    }

    private var statusColor: Color {
        guard let exitCode = controller.lastExitCode else { return MereRunTheme.green }
        return exitCode == 0 ? MereRunTheme.green : MereRunTheme.red
    }
}

private struct CommandRow: View {
    @EnvironmentObject private var controller: MereRunController
    let template: CommandTemplate
    let isSelected: Bool

    var body: some View {
        Button {
            controller.select(template)
        } label: {
            HStack(spacing: 10) {
                Image(systemName: template.systemImage)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(isSelected ? MereRunTheme.background : MereRunTheme.accent)
                    .frame(width: 22)

                VStack(alignment: .leading, spacing: 2) {
                    Text(template.title)
                        .font(.system(size: 13, weight: .semibold))
                    Text(template.subtitle)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(isSelected ? MereRunTheme.background.opacity(0.72) : MereRunTheme.textMuted)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
            }
            .foregroundStyle(isSelected ? MereRunTheme.background : MereRunTheme.textPrimary)
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background {
                RoundedRectangle(cornerRadius: 7)
                    .fill(isSelected ? MereRunTheme.accent : Color.clear)
            }
            .padding(.horizontal, 10)
        }
        .buttonStyle(.plain)
    }
}

private struct CommandEditor: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                commandPreview
                templateFields
                runtimeFields
                actionRow
            }
            .padding(22)
        }
        .background(MereRunTheme.background)
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: controller.selectedTemplate.systemImage)
                .font(.system(size: 24, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
                .frame(width: 38, height: 38)
                .background {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(MereRunTheme.surfaceRaised)
                }

            VStack(alignment: .leading, spacing: 4) {
                Text(controller.selectedTemplate.title)
                    .font(MereRunTheme.titleFont)
                Text(controller.selectedTemplate.subtitle)
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
            }

            Spacer()
        }
    }

    private var commandPreview: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("COMMAND")
                .font(MereRunTheme.sectionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Text(controller.advancedCommandPreview)
                .font(MereRunTheme.monoFont)
                .textSelection(.enabled)
                .lineLimit(5)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .merePanel()
        }
    }

    @ViewBuilder
    private var templateFields: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let promptLabel = controller.selectedTemplate.promptLabel {
                EditorSection(promptLabel) {
                    TextEditor(text: $controller.draft.prompt)
                        .font(MereRunTheme.bodyFont)
                        .scrollContentBackground(.hidden)
                        .frame(minHeight: 88)
                        .padding(8)
                        .merePanel()
                }
            }

            if let secondaryLabel = controller.selectedTemplate.secondaryLabel {
                EditorSection(secondaryLabel) {
                    TextEditor(text: $controller.draft.secondaryText)
                        .font(MereRunTheme.bodyFont)
                        .scrollContentBackground(.hidden)
                        .frame(minHeight: secondaryLabel == "Lyrics" ? 88 : 58)
                        .padding(8)
                        .merePanel()
                }
            }

            if controller.selectedTemplate.inputKind != .none {
                EditorSection(controller.selectedTemplate.inputKind.title) {
                    PathField(
                        path: $controller.draft.inputPath,
                        placeholder: "Choose \(controller.selectedTemplate.inputKind.title.lowercased())",
                        mode: .openFile(controller.selectedTemplate.inputKind.allowedTypes)
                    )
                }
            }

            if controller.selectedTemplate.outputKind != .none {
                EditorSection(outputLabel) {
                    PathField(
                        path: $controller.draft.outputPath,
                        placeholder: outputLabel,
                        mode: controller.selectedTemplate.outputKind == .directory ? .openDirectory : .saveFile
                    )
                }
            }

            if showsModelField {
                EditorSection("Model") {
                    TextField("Managed model id or local path", text: $controller.draft.model)
                        .textFieldStyle(.plain)
                        .font(MereRunTheme.bodyFont)
                        .padding(10)
                        .merePanel()
                }
            }

            parameterFields
        }
    }

    @ViewBuilder
    private var parameterFields: some View {
        switch controller.selectedTemplate.id {
        case .imageGenerate:
            DimensionsGrid()
            GenerationOptions()
        case .imageValidate:
            ImageValidationOptions()
        case .textChat, .textCode, .textEmbed, .textAnonymize, .visionInspect, .visionCaption, .visionOCR:
            TextGenerationOptions()
        case .speechSynthesize:
            SpeechOptions()
        case .speechTranscribe:
            SpeechTranscribeOptions()
        case .speechProfileCreate:
            SpeechProfileCreateOptions()
        case .visionSegment, .visionTrack, .visionTrackLive:
            VisionTrackingOptions()
        case .musicGenerate:
            MusicOptions()
        case .videoGenerate:
            DimensionsGrid()
            VideoOptions()
        case .videoExportLatents:
            DimensionsGrid()
            VideoLatentsOptions()
        case .apiServe:
            APIOptions()
        case .setup:
            SetupOptions()
        case .agentOnboard:
            AgentOptions()
        case .agentInstallPi:
            AgentInstallOptions()
        case .agentStart:
            AgentStartOptions()
        case .modelPull:
            ModelPullOptions()
        case .modelRemove, .modelRepairManifests:
            ModelMaintenanceOptions()
        case .modelCapabilities, .modelInfo:
            ModelInspectionOptions()
        default:
            EmptyView()
        }

        if controller.selectedTemplate.id != .custom {
            EditorSection("Extra arguments") {
                TextField("--flag value", text: $controller.draft.extraArguments)
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.monoFont)
                    .padding(10)
                    .merePanel()
            }
        } else {
            EditorSection("Arguments") {
                TextEditor(text: $controller.draft.extraArguments)
                    .font(MereRunTheme.monoFont)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 150)
                    .padding(8)
                    .merePanel()
            }
        }
    }

    private var runtimeFields: some View {
        EditorSection("Runtime") {
            VStack(spacing: 10) {
                PathField(path: $controller.cliPath, placeholder: "Auto-detect mere.run", mode: .openFile([.unixExecutable, .item]))
                PathField(path: $controller.modelsRoot, placeholder: "Optional models root", mode: .openDirectory)
                PathField(path: $controller.workingDirectory, placeholder: "Working directory", mode: .openDirectory)
            }
        }
    }

    private var actionRow: some View {
        HStack(spacing: 10) {
            Button {
                controller.run()
            } label: {
                Label("Run", systemImage: "play.fill")
                    .frame(minWidth: 86)
            }
            .buttonStyle(.borderedProminent)
            .tint(MereRunTheme.accent)
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(controller.isRunning)

            Button {
                controller.cancel()
            } label: {
                Label("Stop", systemImage: "stop.fill")
                    .frame(minWidth: 76)
            }
            .buttonStyle(.bordered)
            .disabled(!controller.isRunning)

            Spacer()

            if controller.lastOutputURL != nil {
                Button {
                    controller.openLastOutput()
                } label: {
                    Label("Open", systemImage: "arrow.up.right.square")
                }
                .buttonStyle(.bordered)

                Button {
                    controller.revealLastOutput()
                } label: {
                    Image(systemName: "finder")
                }
                .buttonStyle(.bordered)
                .help("Reveal in Finder")
            }
        }
    }

    private var showsModelField: Bool {
        ![
            .agentInstallPi,
            .imageValidate,
            .modelList,
            .modelCapabilities,
            .modelRepairManifests,
            .setup,
            .custom,
            .speechProfileList,
            .speechProfileCreate,
            .speechProfileDelete,
        ].contains(controller.selectedTemplate.id)
    }

    private var outputLabel: String {
        controller.selectedTemplate.outputKind == .directory ? "Output directory" : "Output file"
    }
}

private struct RunConsole: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Run")
                        .font(.system(size: 17, weight: .semibold))
                    Text(controller.status)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
                if controller.isRunning {
                    ProgressView()
                        .controlSize(.small)
                }
            }
            .padding(18)

            Divider()
                .overlay(MereRunTheme.border.opacity(0.6))

            if controller.logs.isEmpty {
                VStack(spacing: 10) {
                    Image(systemName: "terminal")
                        .font(.system(size: 28, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                    Text("Run output will appear here.")
                        .font(MereRunTheme.bodyFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 6) {
                            ForEach(controller.logs) { line in
                                HStack(alignment: .firstTextBaseline, spacing: 8) {
                                    Text(line.stream.label)
                                        .font(MereRunTheme.monoFont)
                                        .foregroundStyle(color(for: line.stream))
                                        .frame(width: 34, alignment: .leading)
                                    Text(line.text)
                                        .font(MereRunTheme.monoFont)
                                        .foregroundStyle(line.stream == .stderr ? MereRunTheme.textSecondary : MereRunTheme.textPrimary)
                                        .textSelection(.enabled)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                .id(line.id)
                            }
                        }
                        .padding(14)
                    }
                    .onChange(of: controller.logs.count) {
                        if let last = controller.logs.last {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }

            if let output = controller.lastOutputURL {
                Divider()
                    .overlay(MereRunTheme.border.opacity(0.6))
                OutputPreview(url: output)
                    .padding(14)
            }
        }
        .background(MereRunTheme.surface.opacity(0.55))
    }

    private func color(for stream: LogStream) -> Color {
        switch stream {
        case .system: return MereRunTheme.accent
        case .stdout: return MereRunTheme.green
        case .stderr: return MereRunTheme.yellow
        }
    }
}

private struct OutputPreview: View {
    let url: URL

    var body: some View {
        HStack(spacing: 12) {
            preview
                .frame(width: 76, height: 58)
                .background(MereRunTheme.background)
                .clipShape(RoundedRectangle(cornerRadius: 6))

            VStack(alignment: .leading, spacing: 4) {
                Text(url.lastPathComponent)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                Text(url.deletingLastPathComponent().path)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }
            Spacer()
        }
        .padding(10)
        .merePanel()
    }

    @ViewBuilder
    private var preview: some View {
        if let image = NSImage(contentsOf: url) {
            Image(nsImage: image)
                .resizable()
                .scaledToFill()
        } else {
            Image(systemName: iconName)
                .font(.system(size: 24, weight: .medium))
                .foregroundStyle(MereRunTheme.accent)
        }
    }

    private var iconName: String {
        switch url.pathExtension.lowercased() {
        case "wav", "mp3", "m4a": return "waveform"
        case "mp4", "mov": return "film"
        case "json": return "curlybraces"
        default: return "doc"
        }
    }
}

private struct EditorSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(MereRunTheme.sectionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            content
        }
    }
}

private enum PathFieldMode: Equatable {
    case openFile([UTType])
    case openDirectory
    case saveFile
}

private struct PathField: View {
    @Binding var path: String
    let placeholder: String
    let mode: PathFieldMode

    var body: some View {
        HStack(spacing: 8) {
            TextField(placeholder, text: $path)
                .textFieldStyle(.plain)
                .font(MereRunTheme.bodyFont)
            Button {
                choosePath()
            } label: {
                Image(systemName: mode == .saveFile ? "square.and.arrow.down" : "folder")
            }
            .buttonStyle(.borderless)
            .help(mode == .saveFile ? "Choose output path" : "Choose path")
        }
        .padding(10)
        .merePanel()
    }

    private func choosePath() {
        switch mode {
        case .openFile(let types):
            let panel = NSOpenPanel()
            panel.canChooseFiles = true
            panel.canChooseDirectories = false
            panel.allowsMultipleSelection = false
            panel.allowedContentTypes = types
            if panel.runModal() == .OK, let url = panel.url {
                path = url.path
            }
        case .openDirectory:
            let panel = NSOpenPanel()
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.allowsMultipleSelection = false
            if panel.runModal() == .OK, let url = panel.url {
                path = url.path
            }
        case .saveFile:
            let panel = NSSavePanel()
            panel.nameFieldStringValue = URL(fileURLWithPath: path).lastPathComponent
            if panel.runModal() == .OK, let url = panel.url {
                path = url.path
            }
        }
    }
}

private struct DimensionsGrid: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Size") {
            HStack(spacing: 10) {
                NumberStepper(title: "Width", value: $controller.draft.width, range: 64...4096, step: 64)
                NumberStepper(title: "Height", value: $controller.draft.height, range: 64...4096, step: 64)
            }
        }
    }
}

private struct GenerationOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Generation") {
            VStack(spacing: 10) {
                HStack(spacing: 10) {
                    NumberStepper(title: "Steps", value: $controller.draft.steps, range: 1...80, step: 1)
                    NumberField(title: "CFG", value: $controller.draft.cfgScale)
                    NumberField(title: "Strength", value: $controller.draft.strength)
                }
                TextField("Seed", text: $controller.draft.seed)
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.bodyFont)
                    .padding(10)
                    .merePanel()
                Toggle("Quiet", isOn: $controller.draft.quiet)
            }
        }
    }
}

private struct ImageValidationOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Validation") {
            VStack(spacing: 10) {
                Picker("Suite", selection: $controller.draft.backend) {
                    Text("All").tag("all")
                    Text("VAE").tag("vae")
                    Text("Encoder").tag("encoder")
                    Text("Transformer").tag("transformer")
                    Text("Pipeline").tag("pipeline")
                }
                .pickerStyle(.segmented)
                Picker("Family", selection: $controller.draft.variant) {
                    Text("ZImage").tag("zimage")
                    Text("Klein").tag("klein")
                }
                .pickerStyle(.segmented)
                HStack {
                    Toggle("Save reference", isOn: $controller.draft.force)
                    Toggle("Compare", isOn: $controller.draft.all)
                    Spacer()
                }
            }
        }
    }
}

private struct TextGenerationOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Parameters") {
            VStack(spacing: 10) {
                HStack(spacing: 10) {
                    NumberStepper(title: "Max tokens", value: $controller.draft.maxTokens, range: 1...32768, step: 64)
                    NumberField(title: "Temp", value: $controller.draft.temperature)
                    NumberField(title: "Top-p", value: $controller.draft.topP)
                }
                HStack {
                    if [.textChat].contains(controller.selectedTemplate.id) {
                        Toggle("Thinking", isOn: $controller.draft.all)
                    }
                    if [.textChat, .textCode].contains(controller.selectedTemplate.id) {
                        Toggle("Stats", isOn: $controller.draft.force)
                    }
                    if controller.selectedTemplate.id == .textCode {
                        Toggle("Stream", isOn: $controller.draft.stream)
                    }
                    if [.textEmbed, .textAnonymize].contains(controller.selectedTemplate.id) {
                        Toggle("Pretty", isOn: $controller.draft.force)
                    }
                    if controller.selectedTemplate.id == .textAnonymize {
                        Toggle("JSON", isOn: $controller.draft.all)
                    }
                    Spacer()
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
            }
        }
    }
}

private struct SpeechOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Speech") {
            HStack(spacing: 10) {
                NumberField(title: "Temperature", value: $controller.draft.temperature)
                Toggle("Stream", isOn: $controller.draft.stream)
                Toggle("Quiet", isOn: $controller.draft.quiet)
            }
        }
    }
}

private struct SpeechTranscribeOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Transcription") {
            VStack(spacing: 10) {
                Picker("Backend", selection: $controller.draft.backend) {
                    Text("Auto").tag("auto")
                    Text("Parakeet").tag("parakeet")
                    Text("Qwen").tag("qwen")
                }
                .pickerStyle(.segmented)
                Picker("Task", selection: $controller.draft.task) {
                    Text("Transcribe").tag("transcribe")
                    Text("Translate").tag("translate")
                }
                .pickerStyle(.segmented)
                HStack {
                    NumberStepper(title: "Max tokens", value: $controller.draft.maxTokens, range: 1...4096, step: 64)
                    TextField("Language", text: $controller.draft.language)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    Toggle("Stream", isOn: $controller.draft.stream)
                    Toggle("Timestamps", isOn: $controller.draft.timestamps)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
            }
        }
    }
}

private struct SpeechProfileCreateOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Profile") {
            HStack(spacing: 10) {
                TextField("Language", text: $controller.draft.language)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                Toggle("Quiet", isOn: $controller.draft.quiet)
            }
        }
    }
}

private struct VisionTrackingOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Vision") {
            HStack {
                if controller.selectedTemplate.id == .visionTrackLive {
                    NumberField(title: "Seconds", value: $controller.draft.durationSeconds)
                }
                Toggle("Show boxes", isOn: $controller.draft.force)
                Spacer()
            }
        }
    }
}

private struct MusicOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Music") {
            VStack(spacing: 10) {
                HStack(spacing: 10) {
                    NumberField(title: "Seconds", value: $controller.draft.durationSeconds)
                    NumberStepper(title: "Steps", value: $controller.draft.steps, range: 1...80, step: 1)
                }
                TextField("Seed", text: $controller.draft.seed)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                Toggle("Quiet", isOn: $controller.draft.quiet)
            }
        }
    }
}

private struct VideoOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Video") {
            VStack(spacing: 10) {
                Picker("Variant", selection: $controller.draft.variant) {
                    Text("Distilled").tag("distilled")
                    Text("Unified AV").tag("unified-av")
                }
                .pickerStyle(.segmented)
                HStack(spacing: 10) {
                    NumberStepper(title: "Frames", value: $controller.draft.numFrames, range: 9...257, step: 8)
                    NumberStepper(title: "FPS", value: $controller.draft.fps, range: 1...60, step: 1)
                    NumberField(title: "Image strength", value: $controller.draft.strength)
                }
                TextField("Seed", text: $controller.draft.seed)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                Toggle("Quiet", isOn: $controller.draft.quiet)
            }
        }
    }
}

private struct VideoLatentsOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Latents") {
            VStack(spacing: 10) {
                NumberStepper(title: "Frames", value: $controller.draft.numFrames, range: 9...257, step: 8)
                TextField("Seed", text: $controller.draft.seed)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                Toggle("Quiet", isOn: $controller.draft.quiet)
            }
        }
    }
}

private struct APIOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Server") {
            VStack(spacing: 10) {
                Picker("Engine", selection: $controller.draft.engine) {
                    Text("Chat Gemma4").tag("text-chat-gemma4")
                    Text("Chat Q35").tag("text-chat-q35")
                    Text("Chat Klein").tag("text-chat-klein")
                    Text("Code").tag("text-code")
                }
                .pickerStyle(.segmented)
                HStack(spacing: 10) {
                    TextField("Host", text: $controller.draft.host)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    NumberStepper(title: "Port", value: $controller.draft.port, range: 1...65535, step: 1)
                }
                SecureField("API key", text: $controller.draft.apiKey)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
            }
        }
    }
}

private struct AgentInstallOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Install") {
            Toggle("Force reinstall", isOn: $controller.draft.force)
        }
    }
}

private struct AgentStartOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Agent") {
            VStack(spacing: 10) {
                HStack(spacing: 10) {
                    TextField("Host", text: $controller.draft.host)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    NumberStepper(title: "Port", value: $controller.draft.port, range: 1...65535, step: 1)
                }
                HStack {
                    Toggle("Skip server", isOn: $controller.draft.stream)
                    Toggle("Allow unsupported", isOn: $controller.draft.force)
                    Spacer()
                }
            }
        }
    }
}

private struct SetupOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Setup") {
            VStack(spacing: 10) {
                Picker("Mode", selection: $controller.draft.setupMode) {
                    Text("Agent").tag("agent")
                    Text("BYOA").tag("byoa")
                    Text("Manual").tag("manual")
                }
                .pickerStyle(.segmented)
                Picker("Agent", selection: $controller.draft.agentModel) {
                    Text("Small").tag("small")
                    Text("Tier").tag("tier")
                    Text("Premier").tag("premier")
                }
                .pickerStyle(.segmented)
                HStack {
                    Toggle("Install", isOn: $controller.draft.force)
                    Toggle("Start", isOn: $controller.draft.stream)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
            }
        }
    }
}

private struct AgentOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Agent") {
            VStack(spacing: 10) {
                HStack {
                    Toggle("Pull recommended", isOn: $controller.draft.force)
                    Toggle("Install Pi", isOn: $controller.draft.all)
                    Toggle("Configure Pi", isOn: $controller.draft.stream)
                }
                HStack(spacing: 10) {
                    TextField("Host", text: $controller.draft.host)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    NumberStepper(title: "Port", value: $controller.draft.port, range: 1...65535, step: 1)
                }
            }
        }
    }
}

private struct ModelPullOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Download") {
            HStack {
                Toggle("All", isOn: $controller.draft.all)
                Toggle("Force", isOn: $controller.draft.force)
                Toggle("Allow unsupported", isOn: $controller.draft.stream)
                Toggle("Quiet", isOn: $controller.draft.quiet)
            }
        }
    }
}

private struct ModelMaintenanceOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Maintenance") {
            if controller.selectedTemplate.id == .modelRemove {
                Toggle("Force", isOn: $controller.draft.force)
            } else {
                Toggle("Dry run", isOn: $controller.draft.force)
            }
        }
    }
}

private struct ModelInspectionOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Output") {
            HStack {
                if controller.selectedTemplate.id == .modelCapabilities {
                    Toggle("All", isOn: $controller.draft.all)
                    Toggle("Recommended", isOn: $controller.draft.force)
                } else {
                    Toggle("JSON", isOn: $controller.draft.all)
                    Toggle("Components", isOn: $controller.draft.force)
                }
                Spacer()
            }
        }
    }
}

private struct NumberStepper: View {
    let title: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    let step: Int

    var body: some View {
        HStack {
            Text(title)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Spacer()
            Stepper(value: $value, in: range, step: step) {
                Text("\(value)")
                    .font(MereRunTheme.monoFont)
            }
        }
        .padding(10)
        .merePanel()
    }
}

private struct NumberField: View {
    let title: String
    @Binding var value: Double

    var body: some View {
        HStack {
            Text(title)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Spacer()
            TextField(title, value: $value, format: .number)
                .multilineTextAlignment(.trailing)
                .textFieldStyle(.plain)
                .font(MereRunTheme.monoFont)
                .frame(width: 76)
        }
        .padding(10)
        .merePanel()
    }
}

struct MereRunSettingsView: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("mere.run")
                .font(MereRunTheme.titleFont)
            VStack(spacing: 12) {
                PathField(path: $controller.cliPath, placeholder: "Auto-detect executable", mode: .openFile([.unixExecutable, .item]))
                PathField(path: $controller.modelsRoot, placeholder: "Optional models root", mode: .openDirectory)
                PathField(path: $controller.hubCache, placeholder: "Optional Hugging Face hub cache", mode: .openDirectory)
                PathField(path: $controller.workingDirectory, placeholder: "Working directory", mode: .openDirectory)
            }
            Text("The app uses a bundled `mere.run` first, then nearby SwiftPM build products, common install locations, and the current package checkout.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            EditorSection("Install") {
                HStack(spacing: 10) {
                    Button {
                        controller.installTerminalCLI()
                    } label: {
                        Label("Install CLI", systemImage: "terminal")
                    }
                    .buttonStyle(.borderedProminent)

                    Button {
                        controller.installCodexSkills()
                    } label: {
                        Label("Install Skill", systemImage: "sparkles")
                    }
                    .buttonStyle(.bordered)
                }
                Text("CLI installs from the app bundle without sudo. Skill install copies the bundled `use-mere-run` Codex skill to `~/.codex/skills`.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            Spacer()
        }
        .padding(22)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
    }
}
