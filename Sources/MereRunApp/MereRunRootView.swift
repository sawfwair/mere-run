import AppKit
import MereRunContract
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
                controller.refreshCLIVersion()
            }
    }
}

struct AdvancedControlSurface: View {
    /// Docked beside the Studio canvas shows a single resizable column (template picker + editor);
    /// detached shows the full three-pane (sidebar · editor · console).
    var docked = false
    var onDetach: (() -> Void)?
    var onClose: (() -> Void)?

    var body: some View {
        Group {
            if docked {
                DockedAdvancedEditor(onDetach: onDetach, onClose: onClose)
            } else {
                fullSurface
            }
        }
        .background(MereRunTheme.background.ignoresSafeArea())
        .foregroundStyle(MereRunTheme.textPrimary)
    }

    private var fullSurface: some View {
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
    }
}

/// The docked Advanced column: a compact template picker (the sidebar's job, condensed) above the
/// scrollable editor, so the 560-ish rail never scrolls horizontally. A detach button promotes it
/// to the full three-pane.
private struct DockedAdvancedEditor: View {
    @EnvironmentObject private var controller: MereRunController
    var onDetach: (() -> Void)?
    var onClose: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.5))
            CommandEditor()
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Menu {
                ForEach(CommandCategory.allCases) { category in
                    let templates = CommandCatalog.templates(in: category)
                    if !templates.isEmpty {
                        Section(category.rawValue) {
                            ForEach(templates) { template in
                                Button(template.title) { controller.select(template) }
                            }
                        }
                    }
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: controller.selectedTemplate.systemImage)
                    Text(controller.selectedTemplate.title)
                        .font(MereRunTheme.sectionFont)
                        .lineLimit(1)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            .menuStyle(.borderlessButton)
            .fixedSize()

            Spacer(minLength: 0)

            if let onClose {
                Button(action: onClose) {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .semibold))
                        .frame(width: 24, height: 24)
                }
                .buttonStyle(.mereIcon)
                .help("Hide Advanced (⌃⌘E)")
                .accessibilityLabel("Hide Advanced")
            }

            if let onDetach {
                Button(action: onDetach) {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                }
                .buttonStyle(.mereIcon)
                .help("Detach to the full control surface")
                .accessibilityLabel("Detach to the full control surface")
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
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
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(controller.isRunning ? "Running: \(controller.status)" : "Status: \(controller.status)")
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
            .frame(maxWidth: .infinity, alignment: .topLeading)
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
                        mode: controller.selectedTemplate.inputKind == .directory
                            ? .openDirectory
                            : .openFile(controller.selectedTemplate.inputKind.allowedTypes)
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
        case .imageTrainLoRA:
            DimensionsGrid()
            LoRATrainingOptions()
        case .imageValidate:
            ImageValidationOptions()
        case .textChat, .textCode, .textEmbed, .textAnonymize, .visionInspect, .visionCaption, .visionOCR:
            TextGenerationOptions()
        case .textTrainLoRA:
            TextLoRATrainingOptions()
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
        case .videoAnimate:
            DimensionsGrid()
            VideoAnimateOptions()
        case .videoCosmos3:
            DimensionsGrid()
            VideoCosmos3Options()
        case .videoPrepareMasks:
            VideoPrepareMasksOptions()
        case .videoExportLatents:
            DimensionsGrid()
            VideoLatentsOptions()
        case .videoSession:
            VideoSessionOptions()
        case .sfxGenerate, .sfxVideo:
            SFXOptions()
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
                PathField(path: $controller.modelsRoot, placeholder: "Optional model links/local-files root", mode: .openDirectory)
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
                    Image(systemName: "magnifyingglass")
                }
                .buttonStyle(.bordered)
                .help("Reveal in Finder")
                .accessibilityLabel("Reveal output in Finder")
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
            .accessibilityLabel(mode == .saveFile ? "Choose output path" : "Choose path")
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

private struct AdaptiveControlRow<Content: View>: View {
    var spacing: CGFloat = 10
    @ViewBuilder var content: () -> Content

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: spacing, content: content)
            VStack(alignment: .leading, spacing: spacing, content: content)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct DimensionsGrid: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Size") {
            AdaptiveControlRow {
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
                AdaptiveControlRow {
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

private struct LoRATrainingOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Training") {
            VStack(spacing: 10) {
                NumberStepper(title: "Steps", value: $controller.draft.steps, range: 1...100_000, step: 100)
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
                AdaptiveControlRow {
                    Toggle("Save reference", isOn: $controller.draft.force)
                    Toggle("Compare", isOn: $controller.draft.all)
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
                if [.textChat, .textCode].contains(controller.selectedTemplate.id) {
                    AdaptiveControlRow {
                        NumberStepper(
                            title: "Max tokens",
                            value: $controller.draft.maxTokens,
                            range: 1...262_144,
                            step: 64
                        )
                        NumberField(title: "Temp", value: $controller.draft.temperature)
                        NumberField(title: "Top-p", value: $controller.draft.topP)
                    }
                } else {
                    NumberStepper(
                        title: "Max tokens",
                        value: $controller.draft.maxTokens,
                        range: 1...262_144,
                        step: 64
                    )
                }

                AdaptiveControlRow {
                    if [.textChat, .textCode].contains(controller.selectedTemplate.id) {
                        Toggle("Stats", isOn: $controller.draft.force)
                        Toggle("Stream", isOn: $controller.draft.stream)
                        Toggle("Quiet", isOn: $controller.draft.quiet)
                    }
                    if [.textEmbed, .textAnonymize].contains(controller.selectedTemplate.id) {
                        Toggle("Pretty", isOn: $controller.draft.force)
                    }
                    if controller.selectedTemplate.id == .textAnonymize {
                        Toggle("JSON", isOn: $controller.draft.all)
                    }
                }

                if controller.selectedTemplate.id == .textChat {
                    Divider().overlay(MereRunTheme.border.opacity(0.4))
                    Picker("Response", selection: $controller.draft.responseFormat) {
                        Text("Text").tag(TextResponseFormat.text)
                        Text("JSON object").tag(TextResponseFormat.jsonObject)
                    }
                    .pickerStyle(.segmented)
                    Picker("Reasoning", selection: $controller.draft.thinkingMode) {
                        Text("Model default").tag(TextThinkingMode.automatic)
                        Text("Show").tag(TextThinkingMode.show)
                        Text("Disable").tag(TextThinkingMode.hide)
                    }
                    .pickerStyle(.segmented)
                    AdaptiveControlRow {
                        NumberStepper(
                            title: "Context",
                            value: $controller.draft.contextSize,
                            range: 0...262_144,
                            step: 1_024
                        )
                        NumberStepper(
                            title: "Top-k",
                            value: $controller.draft.topK,
                            range: 0...512,
                            step: 1
                        )
                    }
                    PathField(
                        path: $controller.draft.modelRoot,
                        placeholder: "Local model root (optional)",
                        mode: .openDirectory
                    )
                    PathField(
                        path: $controller.draft.loraPath,
                        placeholder: "Catalog adapter id or LoRA file (optional)",
                        mode: .openFile([.data])
                    )
                    if !controller.draft.loraPath.isBlank {
                        NumberField(title: "LoRA scale", value: $controller.draft.loraScale)
                    }
                    Picker("KV bits", selection: $controller.draft.kvBits) {
                        Text("Automatic").tag(0)
                        Text("4-bit").tag(4)
                        Text("8-bit").tag(8)
                    }
                    .pickerStyle(.segmented)
                    Picker("KV scheme", selection: $controller.draft.kvQuantScheme) {
                        Text("Automatic").tag("")
                        Text("Uniform").tag("uniform")
                        Text("Polar").tag("polar")
                        Text("TurboQuant").tag("turboquant")
                    }
                    .pickerStyle(.segmented)
                    AdaptiveControlRow {
                        NumberStepper(
                            title: "KV group",
                            value: $controller.draft.kvGroupSize,
                            range: 0...1_024,
                            step: 8
                        )
                        NumberStepper(
                            title: "KV starts",
                            value: $controller.draft.quantizedKVStart,
                            range: 0...262_144,
                            step: 128
                        )
                    }
                    AdaptiveControlRow {
                        Toggle("Preflight", isOn: $controller.draft.preflight)
                        Toggle("JSON report", isOn: $controller.draft.json)
                            .disabled(!controller.draft.preflight)
                        Toggle("Require installed", isOn: $controller.draft.requireInstalled)
                    }

                    Divider().overlay(MereRunTheme.border.opacity(0.4))
                    TextField("Tools (comma-separated: write_file, shell_exec)", text: $controller.draft.tools)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    AdaptiveControlRow {
                        Toggle("Tool loop", isOn: $controller.draft.toolLoop)
                        Toggle("Allow shell exec", isOn: $controller.draft.allowShellExec)
                        Toggle("Absolute paths", isOn: $controller.draft.allowAbsoluteToolPaths)
                        Toggle("Auto-approve", isOn: $controller.draft.autoApproveTools)
                    }
                    PathField(
                        path: $controller.draft.sandboxDir,
                        placeholder: "Tool sandbox directory (optional)",
                        mode: .openDirectory
                    )
                    PathField(
                        path: $controller.draft.imagePath,
                        placeholder: "Image for vision chat (optional)",
                        mode: .openFile([.image])
                    )
                }

                if controller.selectedTemplate.id == .textAnonymize {
                    TextField("Replacement template", text: $controller.draft.replacement)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
            }
        }
    }
}

private struct TextLoRATrainingOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Training") {
            VStack(spacing: 10) {
                PathField(
                    path: $controller.draft.modelRoot,
                    placeholder: "Explicit base model directory (optional)",
                    mode: .openDirectory
                )
                PathField(
                    path: $controller.draft.evalPath,
                    placeholder: "Eval prompts JSONL (optional)",
                    mode: .openFile([.json, .plainText])
                )
                TextField("Adapter name", text: $controller.draft.adapterName)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                AdaptiveControlRow {
                    NumberStepper(
                        title: "Steps",
                        value: $controller.draft.steps,
                        range: 1...100_000,
                        step: 100
                    )
                    NumberStepper(
                        title: "Batch",
                        value: $controller.draft.batchSize,
                        range: 1...128,
                        step: 1
                    )
                }
                AdaptiveControlRow {
                    NumberField(title: "Learning rate", value: $controller.draft.learningRate)
                    NumberStepper(
                        title: "Rank",
                        value: $controller.draft.rank,
                        range: 1...512,
                        step: 1
                    )
                    NumberField(title: "Alpha", value: $controller.draft.alpha)
                }
                NumberStepper(
                    title: "Max sequence",
                    value: $controller.draft.maxSequenceLength,
                    range: 128...262_144,
                    step: 128
                )
                TextField("Seed", text: $controller.draft.seed)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                TextField("Target modules", text: $controller.draft.targetModules)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                AdaptiveControlRow {
                    Toggle("Dry run", isOn: $controller.draft.dryRun)
                    Toggle("Visualize", isOn: $controller.draft.visualize)
                    Toggle("JSON summary", isOn: $controller.draft.json)
                }
                if controller.draft.visualize {
                    NumberStepper(
                        title: "Dashboard port",
                        value: $controller.draft.visualizePort,
                        range: 1...65_535,
                        step: 1
                    )
                }
            }
        }
    }
}

private struct SpeechOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Speech") {
            VStack(spacing: 10) {
                Picker("Mode", selection: $controller.draft.voiceMode) {
                    Text("Style").tag("style")
                    Text("Clone").tag("clone")
                }
                .pickerStyle(.segmented)

                if controller.draft.voiceMode == "clone" {
                    PathField(
                        path: $controller.draft.refAudioPath,
                        placeholder: "Reference audio (clone)",
                        mode: .openFile([.audio])
                    )
                    TextField("Profile id or name", text: $controller.draft.voiceProfile)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("Reference transcript (optional)", text: $controller.draft.refText)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("Save as profile (optional)", text: $controller.draft.saveProfileName)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }

                AdaptiveControlRow {
                    NumberField(title: "Temperature", value: $controller.draft.temperature)
                    Toggle("Stream", isOn: $controller.draft.stream)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
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
                AdaptiveControlRow {
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
            AdaptiveControlRow {
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
            AdaptiveControlRow {
                if controller.selectedTemplate.id == .visionTrackLive {
                    NumberField(title: "Seconds", value: $controller.draft.durationSeconds)
                }
                Toggle("Show boxes", isOn: $controller.draft.force)
            }
        }
    }
}

private struct MusicOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Music") {
            VStack(spacing: 10) {
                AdaptiveControlRow {
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

private struct SFXOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Sound FX") {
            VStack(spacing: 10) {
                AdaptiveControlRow {
                    NumberField(title: "Seconds", value: $controller.draft.durationSeconds)
                    NumberStepper(title: "Steps", value: $controller.draft.steps, range: 1...64, step: 1)
                    NumberField(title: "CFG", value: $controller.draft.cfgScale)
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
                Picker("Quality", selection: $controller.draft.videoQuality) {
                    Text("Draft").tag(LTXVideoQuality.draft)
                    Text("Final").tag(LTXVideoQuality.final)
                }
                .pickerStyle(.segmented)
                Picker("Output", selection: $controller.draft.videoOutputMode) {
                    Text("Video").tag(LTXVideoOutputMode.videoOnly)
                    Text("Audio + Video").tag(LTXVideoOutputMode.audioVideo)
                }
                .pickerStyle(.segmented)
                PathField(
                    path: $controller.draft.audioPath,
                    placeholder: "Source audio for A2V (optional)",
                    mode: .openFile([.audio])
                )
                PathField(
                    path: $controller.draft.endImagePath,
                    placeholder: "End keyframe (optional)",
                    mode: .openFile([.image])
                )
                AdaptiveControlRow {
                    NumberStepper(title: "Frames", value: $controller.draft.numFrames, range: 9...257, step: 8)
                    NumberStepper(title: "FPS", value: $controller.draft.fps, range: 1...60, step: 1)
                    NumberField(title: "Image strength", value: $controller.draft.strength)
                }
                Toggle("Use duration instead of frame count", isOn: $controller.draft.useDuration)
                if controller.draft.useDuration {
                    NumberField(title: "Duration", value: $controller.draft.durationSeconds)
                }
                if !controller.draft.audioPath.isBlank {
                    AdaptiveControlRow {
                        NumberField(title: "Audio start", value: $controller.draft.audioStartTime)
                        NumberStepper(title: "A2V steps", value: $controller.draft.a2vSteps, range: 1...100, step: 1)
                    }
                    AdaptiveControlRow {
                        NumberField(title: "A2V", value: $controller.draft.a2vGuidanceScale)
                        NumberField(title: "Video CFG", value: $controller.draft.videoCFGGuidanceScale)
                        NumberField(title: "Audio CFG", value: $controller.draft.audioCFGGuidanceScale)
                        NumberField(title: "V2A", value: $controller.draft.v2aGuidanceScale)
                    }
                }
                if !controller.draft.endImagePath.isBlank {
                    NumberField(title: "End image strength", value: $controller.draft.endImageStrength)
                }
                TextField("Seed", text: $controller.draft.seed)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                AdaptiveControlRow {
                    Toggle("Preflight", isOn: $controller.draft.preflight)
                    Toggle("JSON", isOn: $controller.draft.json)
                        .disabled(!controller.draft.preflight)
                    Toggle("Timings", isOn: $controller.draft.timings)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
                if controller.draft.timings {
                    PathField(
                        path: $controller.draft.timingsOutputPath,
                        placeholder: "Timing report JSON (optional)",
                        mode: .saveFile
                    )
                }
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

private struct VideoAnimateOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("SCAIL-2 inputs") {
            VStack(spacing: 10) {
                PathField(
                    path: $controller.draft.referenceMaskPath,
                    placeholder: "Seven-color reference mask",
                    mode: .openFile([.image])
                )
                PathField(
                    path: $controller.draft.drivingVideoPath,
                    placeholder: "Driving video",
                    mode: .openFile([.movie, .video])
                )
                PathField(
                    path: $controller.draft.drivingMaskPath,
                    placeholder: "Seven-color driving mask video",
                    mode: .openFile([.movie, .video])
                )
                Picker("Task", selection: $controller.draft.videoTaskMode) {
                    Text("Animation").tag("animation")
                    Text("Replacement").tag("replacement")
                }
                .pickerStyle(.segmented)
                Picker("Profile", selection: $controller.draft.renderProfile) {
                    Text("Fast").tag("fast")
                    Text("Quality").tag("quality")
                }
                .pickerStyle(.segmented)
                if controller.draft.renderProfile == "quality" {
                    AdaptiveControlRow {
                        NumberStepper(
                            title: "Steps",
                            value: $controller.draft.steps,
                            range: 1...100,
                            step: 1
                        )
                        NumberField(title: "CFG", value: $controller.draft.cfgScale)
                        NumberField(title: "Shift", value: $controller.draft.scheduleShift)
                    }
                    Picker("Sampler", selection: $controller.draft.sampler) {
                        Text("UniPC").tag("unipc")
                        Text("Euler").tag("euler")
                    }
                    .pickerStyle(.segmented)
                }
                AdaptiveControlRow {
                    NumberStepper(
                        title: "FPS",
                        value: $controller.draft.fps,
                        range: 1...60,
                        step: 1
                    )
                    NumberStepper(
                        title: "Segment",
                        value: $controller.draft.segmentLength,
                        range: 5...401,
                        step: 4
                    )
                    NumberStepper(
                        title: "Overlap",
                        value: $controller.draft.segmentOverlap,
                        range: 1...81,
                        step: 4
                    )
                }
                Picker("Tail", selection: $controller.draft.tailPolicy) {
                    Text("Drop").tag("drop")
                    Text("Pad + trim").tag("pad-trim")
                }
                .pickerStyle(.segmented)
                Picker("Audio", selection: $controller.draft.audioSource) {
                    Text("None").tag("none")
                    Text("Driving video").tag("driving")
                }
                .pickerStyle(.segmented)
                AdaptiveControlRow {
                    Toggle("Preflight", isOn: $controller.draft.preflight)
                    Toggle("JSON", isOn: $controller.draft.json)
                        .disabled(!controller.draft.preflight)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
            }
        }
    }
}

private struct VideoCosmos3Options: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Cosmos3") {
            VStack(spacing: 10) {
                Picker("Mode", selection: $controller.draft.cosmosMode) {
                    ForEach(
                        MereRunCapabilityCatalog.videoCosmos3.options
                            .first { $0.flag == "--mode" }?.choices ?? [],
                        id: \.self
                    ) { mode in
                        Text(mode.replacingOccurrences(of: "-", with: " ").capitalized)
                            .tag(mode)
                    }
                }
                PathField(
                    path: $controller.draft.cosmosImagePath,
                    placeholder: "Conditioning image (optional)",
                    mode: .openFile([.image])
                )
                PathField(
                    path: $controller.draft.cosmosVideoPath,
                    placeholder: "Conditioning video (optional)",
                    mode: .openFile([.movie, .video])
                )
                PathField(
                    path: $controller.draft.actionsOutputPath,
                    placeholder: "Predicted actions JSON (optional)",
                    mode: .saveFile
                )
                AdaptiveControlRow {
                    NumberStepper(
                        title: "Frames",
                        value: $controller.draft.numFrames,
                        range: 1...601,
                        step: 4
                    )
                    NumberStepper(
                        title: "Steps · 0 auto",
                        value: $controller.draft.steps,
                        range: 0...100,
                        step: 1
                    )
                    NumberStepper(
                        title: "FPS · 0 auto",
                        value: $controller.draft.fps,
                        range: 0...60,
                        step: 1
                    )
                }
                Picker("Schedule", selection: $controller.draft.schedule) {
                    Text("NVIDIA").tag("nvidia")
                    Text("Published Karras").tag("published-karras")
                }
                .pickerStyle(.segmented)
                AdaptiveControlRow {
                    NumberField(title: "CFG · 0 auto", value: $controller.draft.cfgScale)
                    NumberField(title: "Shift · 0 auto", value: $controller.draft.scheduleShift)
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

private struct VideoPrepareMasksOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Mask preparation") {
            VStack(spacing: 10) {
                TextField("Preview frame (optional)", text: $controller.draft.previewFrame)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                AdaptiveControlRow {
                    Toggle("Preflight", isOn: $controller.draft.preflight)
                    Toggle("JSON", isOn: $controller.draft.json)
                        .disabled(!controller.draft.preflight)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
            }
        }
    }
}

private struct VideoSessionOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Resident session") {
            VStack(spacing: 10) {
                PathField(
                    path: $controller.draft.modelRoot,
                    placeholder: "Local LTX model root (optional)",
                    mode: .openDirectory
                )
                Toggle("Quiet", isOn: $controller.draft.quiet)
                Text("Run loads LTX once. When the status says ready, submit as many renders as you need.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }

        EditorSection("Resident render") {
            VStack(spacing: 10) {
                TextEditor(text: $controller.draft.prompt)
                    .font(MereRunTheme.bodyFont)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 72)
                    .padding(8)
                    .merePanel()
                    .overlay(alignment: .topLeading) {
                        if controller.draft.prompt.isBlank {
                            Text("Prompt")
                                .font(MereRunTheme.bodyFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                                .padding(12)
                                .allowsHitTesting(false)
                        }
                    }
                PathField(
                    path: $controller.draft.outputPath,
                    placeholder: "Output MP4",
                    mode: .saveFile
                )
                PathField(
                    path: $controller.draft.imagePath,
                    placeholder: "Start image (optional)",
                    mode: .openFile([.image])
                )
                PathField(
                    path: $controller.draft.endImagePath,
                    placeholder: "End keyframe (optional)",
                    mode: .openFile([.image])
                )
                AdaptiveControlRow {
                    NumberStepper(title: "Width", value: $controller.draft.width, range: 64...4096, step: 64)
                    NumberStepper(title: "Height", value: $controller.draft.height, range: 64...4096, step: 64)
                }
                AdaptiveControlRow {
                    NumberStepper(title: "Frames", value: $controller.draft.numFrames, range: 9...257, step: 8)
                    NumberStepper(title: "FPS", value: $controller.draft.fps, range: 1...60, step: 1)
                }
                AdaptiveControlRow {
                    NumberField(title: "Start strength", value: $controller.draft.strength)
                    NumberField(title: "End strength", value: $controller.draft.endImageStrength)
                }
                TextField("Seed (optional)", text: $controller.draft.seed)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                Button {
                    controller.submitVideoSessionRequest()
                } label: {
                    Label("Submit to resident session", systemImage: "bolt.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
                .disabled(
                    !controller.canSubmitVideoSessionRequest
                        || controller.draft.prompt.isBlank
                        || controller.draft.outputPath.isBlank
                )
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
                    Text("Chat Qwen 3.6").tag("text-chat-q36")
                    Text("Chat Klein").tag("text-chat-klein")
                    Text("Code").tag("text-code")
                }
                .pickerStyle(.segmented)
                AdaptiveControlRow {
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
                AdaptiveControlRow {
                    TextField("Host", text: $controller.draft.host)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    NumberStepper(title: "Port", value: $controller.draft.port, range: 1...65535, step: 1)
                }
                AdaptiveControlRow {
                    Toggle("Skip server", isOn: $controller.draft.stream)
                    Toggle("Allow unsupported", isOn: $controller.draft.force)
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
                AdaptiveControlRow {
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
                AdaptiveControlRow {
                    Toggle("Pull recommended", isOn: $controller.draft.force)
                    Toggle("Install Pi", isOn: $controller.draft.all)
                    Toggle("Configure Pi", isOn: $controller.draft.stream)
                }
                Toggle("Acknowledge third-party model terms", isOn: $controller.draft.acceptModelLicense)
                    .toggleStyle(.checkbox)
                AdaptiveControlRow {
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
            VStack(alignment: .leading, spacing: 10) {
                AdaptiveControlRow {
                    Toggle("All", isOn: $controller.draft.all)
                    Toggle("Force", isOn: $controller.draft.force)
                    Toggle("Allow unsupported", isOn: $controller.draft.stream)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
                Toggle("Acknowledge third-party model terms", isOn: $controller.draft.acceptModelLicense)
                    .toggleStyle(.checkbox)
                Text("Required for new downloads whose owners publish non-commercial, research-only, gated, revenue-limited, or custom acceptable-use terms. The command output lists the exact model/component terms. Mere does not determine whether your intended use is permitted; you are responsible for compliance.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
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
            AdaptiveControlRow {
                if controller.selectedTemplate.id == .modelCapabilities {
                    Toggle("All", isOn: $controller.draft.all)
                    Toggle("Recommended", isOn: $controller.draft.force)
                    Toggle("JSON", isOn: $controller.draft.json)
                } else {
                    Toggle("JSON", isOn: $controller.draft.all)
                    Toggle("Components", isOn: $controller.draft.force)
                }
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
    @State private var hfToken = ""
    @State private var hfStatus: String?
    @State private var hfEndpoint = ""
    @State private var hfEndpointStatus: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .firstTextBaseline) {
                Text("mere.run")
                    .font(MereRunTheme.titleFont)
                Spacer()
                Text("App \(controller.appVersion) · CLI \(controller.cliVersion ?? "—")")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            VStack(spacing: 12) {
                PathField(path: $controller.cliPath, placeholder: "Auto-detect executable", mode: .openFile([.unixExecutable, .item]))
                PathField(path: $controller.modelsRoot, placeholder: "Optional model links/local-files root", mode: .openDirectory)
                PathField(path: $controller.hubCache, placeholder: "Optional model payload storage", mode: .openDirectory)
                PathField(path: $controller.workingDirectory, placeholder: "Working directory", mode: .openDirectory)
            }
            Text("The app uses a bundled `mere.run` first, then nearby SwiftPM build products, common install locations, and the current package checkout.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            EditorSection("Hugging Face token") {
                HStack(spacing: 10) {
                    SecureField("hf_… (for gated/private model pulls)", text: $hfToken)
                        .textFieldStyle(.plain)
                        .font(MereRunTheme.bodyFont)
                        .padding(10)
                        .merePanel()
                    Button("Save") {
                        Task {
                            let ok = await controller.saveHuggingFaceToken(hfToken)
                            hfStatus = ok ? "Saved" : "Could not save token"
                            if ok { hfToken = "" }
                        }
                    }
                    .buttonStyle(.merePrimary)
                }
                if let hfStatus {
                    Text(hfStatus)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            EditorSection("Hugging Face endpoint") {
                HStack(spacing: 10) {
                    TextField("https://huggingface.co (override mirror)", text: $hfEndpoint)
                        .textFieldStyle(.plain)
                        .font(MereRunTheme.bodyFont)
                        .padding(10)
                        .merePanel()
                    Button("Save") {
                        Task {
                            let ok = await controller.saveHuggingFaceEndpoint(hfEndpoint)
                            hfEndpointStatus = ok ? "Saved" : "Could not save endpoint"
                        }
                    }
                    .buttonStyle(.merePrimary)
                }
                if let hfEndpointStatus {
                    Text(hfEndpointStatus)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            .task {
                hfEndpoint = await controller.loadHuggingFaceEndpoint()
            }
            EditorSection("Runtime server") {
                HStack(spacing: 10) {
                    TextField("Host", text: $controller.runtimeHost)
                        .textFieldStyle(.plain)
                        .font(MereRunTheme.bodyFont)
                        .padding(10)
                        .merePanel()
                    TextField("Port", value: $controller.runtimePort, format: .number.grouping(.never))
                        .textFieldStyle(.plain)
                        .font(MereRunTheme.bodyFont)
                        .frame(width: 90)
                        .padding(10)
                        .merePanel()
                    SecureField("API key (optional)", text: $controller.runtimeAPIKey)
                        .textFieldStyle(.plain)
                        .font(MereRunTheme.bodyFont)
                        .padding(10)
                        .merePanel()
                }
                Text("Where the Models panel sends load/unload requests for the running runtime (`mere.run api serve`).")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
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
