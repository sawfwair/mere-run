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
    @EnvironmentObject private var library: StudioLibraryStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                commandPreview
                runStatus
                templateFields
                if controller.selectedTemplate.externalURL == nil {
                    runtimeFields
                }
                actionRow
            }
            .padding(22)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .background(MereRunTheme.background)
    }

    @ViewBuilder
    private var runStatus: some View {
        if controller.isRunning {
            VStack(alignment: .leading, spacing: 7) {
                HStack {
                    Label(controller.status, systemImage: "bolt.horizontal.circle.fill")
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(MereRunTheme.accent)
                    Spacer()
                    if controller.queuedRunCount > 0 {
                        Text("\(controller.queuedRunCount) queued")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                }
                if let progress = controller.currentProgress {
                    if let fraction = progress.fractionCompleted {
                        ProgressView(value: fraction)
                            .tint(MereRunTheme.accent)
                    } else {
                        ProgressView()
                            .controlSize(.small)
                    }
                    HStack {
                        Text(progress.label)
                        Spacer()
                        if let detail = progress.detail { Text(detail) }
                    }
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                } else {
                    ProgressView()
                        .controlSize(.small)
                }
            }
            .padding(12)
            .merePanel()
        }
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
            Text(controller.selectedTemplate.externalURL == nil ? "COMMAND" : "DESTINATION")
                .font(MereRunTheme.sectionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Text(controller.selectedTemplate.externalURL?.absoluteString ?? controller.advancedCommandPreview)
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
            if let url = controller.selectedTemplate.externalURL {
                ExternalProductOptions(template: controller.selectedTemplate, url: url)
            } else {
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
    }

    @ViewBuilder
    private var parameterFields: some View {
        switch controller.selectedTemplate.id {
        case .imageGenerate:
            DimensionsGrid()
            ImageGenerationOptions()
        case .imageTrainLoRA:
            DimensionsGrid()
            ImageLoRATrainingOptions()
        case .imageValidate:
            ImageValidationOptions()
        case .imageDatasetDiscover:
            ImageDatasetDiscoveryOptions()
        case .imageRunPlan:
            ImageRunPlanOptions()
        case .imageVisualizeRun:
            ImageRunViewerOptions()
        case .imageReconstruct3D:
            ImageReconstructionOptions(kind: .triposr)
        case .imageReconstruct3DTrellis2:
            ImageReconstructionOptions(kind: .trellis2)
        case .imageReconstruct3DMultiview:
            ImageReconstructionOptions(kind: .multiview)
        case .textChat, .textCode, .textEmbed, .textAnonymize:
            TextGenerationOptions()
        case .visionInspect:
            VisionLanguageOptions()
        case .visionCaption:
            VisionCaptionOptions()
        case .visionOCR:
            VisionOCROptions()
        case .visionGround:
            VisionGroundingOptions()
        case .textTrainLoRA:
            TextLoRATrainingOptions()
        case .speechSynthesize:
            SpeechOptions()
        case .speechTranscribe:
            SpeechTranscribeOptions()
        case .speechDiarize:
            SpeechDiarizationOptions()
        case .speechProfileCreate:
            SpeechProfileCreateOptions()
        case .visionSegment, .visionTrack, .visionTrackLive:
            VisionTrackingOptions()
        case .visionFaceDetect, .visionFaceEmbed, .visionFaceCompare, .visionFaceBatch:
            VisionFaceOptions()
        case .visionPose:
            VisionPoseOptions()
        case .visionFlow:
            VisionFlowOptions()
        case .visionDepthVideo:
            VisionDepthOptions()
        case .visionGeometry:
            VisionGeometryOptions()
        case .visionGeometryMultiview:
            VisionMultiviewOptions()
        case .audioEnhance:
            AudioEnhancementOptions()
        case .musicGenerate:
            MusicGenerationOptions()
        case .musicAnalyze:
            MusicAnalysisOptions()
        case .musicTranscribe:
            MusicTranscriptionOptions()
        case .musicSeparate:
            MusicSeparationOptions()
        case .musicRealtime:
            MusicRealtimeOptions()
        case .musicTrainAdapter:
            MusicTrainingOptions()
        case .musicServe:
            MusicServeOptions()
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
        case .adapterList, .adapterPull:
            AdapterOperationsOptions()
        case .runList, .runInspect, .runWatch, .runFetch, .runCancel, .runRetry:
            DurableRunOptions()
        case .worldServe:
            WorldServeOptions()
        case .statusSnapshot:
            StatusSnapshotOptions()
        case .qualityGate:
            QualityGateOptions()
        case .modelStorage, .modelGarbageCollect:
            ModelStorageOptions()
        case .modelOptimize:
            ModelOptimizeOptions()
        case .modelRuntimeGet, .modelRuntimeSet:
            ModelRuntimePolicyOptions()
        case .sfxGenerate, .sfxVideo:
            SFXOptions()
        case .modelBenchmark:
            ModelBenchmarkOptions()
        case .modelBenchmarkLagunaDFlash:
            LagunaDFlashBenchmarkOptions()
        case .pluginList, .pluginInstall, .pluginDoctor:
            PluginOptions()
        case .openWebui:
            OpenWebUIOptions()
        case .apiServe:
            APIOptions()
        case .setup:
            SetupOptions()
        case .agentOnboard:
            AgentOptions()
        case .agentStatus:
            AgentStatusOptions()
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

        if controller.selectedTemplate.externalURL != nil {
            EmptyView()
        } else if controller.selectedTemplate.id != .custom {
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
            if let url = controller.selectedTemplate.externalURL {
                Button {
                    NSWorkspace.shared.open(url)
                } label: {
                    Label("Open", systemImage: "arrow.up.right.square")
                        .frame(minWidth: 100)
                }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
                .keyboardShortcut(.return, modifiers: .command)
            } else {
                Button {
                    runAdvancedCommand()
                } label: {
                    Label(controller.isRunning ? "Queue" : "Run", systemImage: "play.fill")
                        .frame(minWidth: 86)
                }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
                .keyboardShortcut(.return, modifiers: .command)

                Button {
                    controller.cancel()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(minWidth: 76)
                }
                .buttonStyle(.bordered)
                .disabled(!controller.isRunning)
            }

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

    private func runAdvancedCommand() {
        let template = controller.selectedTemplate
        let draft = controller.draft
        let request = StudioRunRequest(
            mode: template.libraryMode,
            templateID: template.id,
            template: template,
            draft: draft
        )
        let preview = controller.commandPreview(template: template, draft: draft, masksSecrets: true)
        let status: StudioLibraryStatus = controller.isRunning || controller.queuedRunCount > 0
            ? .queued
            : .running
        library.start(request: request, commandPreview: preview, status: status)
        _ = controller.run(studio: request)
    }

    private var showsModelField: Bool {
        ![
            .agentStatus,
            .agentInstallPi,
            .imageDatasetDiscover,
            .imageRunPlan,
            .imageValidate,
            .imageVisualizeRun,
            .modelList,
            .modelCapabilities,
            .modelRepairManifests,
            .adapterList,
            .adapterPull,
            .runList,
            .runInspect,
            .runWatch,
            .runFetch,
            .runCancel,
            .runRetry,
            .statusSnapshot,
            .qualityGate,
            .modelStorage,
            .modelGarbageCollect,
            .graphStudio,
            .nodeConsole,
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
                    if let progress = controller.currentProgress,
                       let fraction = progress.fractionCompleted {
                        ProgressView(value: fraction)
                            .frame(width: 110)
                            .tint(MereRunTheme.accent)
                            .help(progress.detail ?? progress.label)
                    } else {
                        ProgressView()
                            .controlSize(.small)
                    }
                }
                if !controller.logs.isEmpty {
                    Button {
                        copyConsole()
                    } label: {
                        Image(systemName: "doc.on.doc")
                    }
                    .buttonStyle(.bordered)
                    .help("Copy run output")
                    .accessibilityLabel("Copy run output")

                    Button {
                        saveConsole()
                    } label: {
                        Image(systemName: "square.and.arrow.down")
                    }
                    .buttonStyle(.bordered)
                    .help("Save run receipt")
                    .accessibilityLabel("Save run receipt")
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

            if let catalog = adapterCatalog {
                Divider()
                    .overlay(MereRunTheme.border.opacity(0.6))
                AdapterCatalogPreview(catalog: catalog)
                    .padding(14)
            } else if let json = prettyJSON {
                Divider()
                    .overlay(MereRunTheme.border.opacity(0.6))
                StructuredReceiptPreview(json: json)
                    .padding(14)
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

    private var consoleText: String {
        controller.logs.map { "[\($0.stream.label)] \($0.text)" }.joined(separator: "\n")
    }

    private var resultText: String? {
        controller.lastRunResult?.outputText?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var prettyJSON: String? {
        guard let resultText, let data = resultText.data(using: .utf8),
              let value = try? JSONDecoder().decode(StudioJSONValue.self, from: data) else {
            return nil
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        guard let pretty = try? encoder.encode(value) else { return nil }
        return String(decoding: pretty, as: UTF8.self)
    }

    private var adapterCatalog: StudioAdapterCatalog? {
        guard controller.lastRunResult?.templateID == .adapterList,
              let resultText,
              let data = resultText.data(using: .utf8) else {
            return nil
        }
        return try? JSONDecoder().decode(StudioAdapterCatalog.self, from: data)
    }

    private func copyConsole() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(prettyJSON ?? resultText ?? consoleText, forType: .string)
    }

    private func saveConsole() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = prettyJSON == nil ? "mere-run-receipt.txt" : "mere-run-receipt.json"
        panel.allowedContentTypes = prettyJSON == nil ? [.plainText] : [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? (prettyJSON ?? resultText ?? consoleText).write(to: url, atomically: true, encoding: .utf8)
    }
}

private enum StudioJSONValue: Codable {
    case object([String: StudioJSONValue])
    case array([StudioJSONValue])
    case string(String)
    case number(Double)
    case boolean(Bool)
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .boolean(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: StudioJSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([StudioJSONValue].self) {
            self = .array(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported JSON value."
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .boolean(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}

private struct StudioAdapterCatalog: Decodable {
    let adapterStore: String
    let adapters: [StudioAdapterCatalogItem]
}

private struct StudioAdapterCatalogItem: Decodable, Identifiable {
    let id: String
    let title: String
    let version: String
    let summary: String
    let baseModelID: String
    let license: String
    let byteCount: Int64
    let installed: Bool
    let path: String?
}

private struct AdapterCatalogPreview: View {
    let catalog: StudioAdapterCatalog

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Verified adapters", systemImage: "slider.horizontal.3")
                    .font(.system(size: 13, weight: .semibold))
                Spacer()
                Text("\(catalog.adapters.count)")
                    .font(MereRunTheme.monoFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(catalog.adapters) { adapter in
                        VStack(alignment: .leading, spacing: 5) {
                            HStack {
                                Text(adapter.title)
                                    .font(.system(size: 13, weight: .semibold))
                                Text(adapter.version)
                                    .font(MereRunTheme.captionFont)
                                    .foregroundStyle(MereRunTheme.textMuted)
                                Spacer()
                                Label(
                                    adapter.installed ? "Installed" : "Available",
                                    systemImage: adapter.installed
                                        ? "checkmark.circle.fill"
                                        : "arrow.down.circle"
                                )
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(adapter.installed ? MereRunTheme.green : MereRunTheme.accent)
                            }
                            Text(adapter.id)
                                .font(MereRunTheme.monoFont)
                                .textSelection(.enabled)
                            Text(adapter.summary)
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textSecondary)
                            Text("\(adapter.baseModelID) · \(adapter.license) · \(ByteCountFormatter.string(fromByteCount: adapter.byteCount, countStyle: .file))")
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                            if let path = adapter.path {
                                Text(path)
                                    .font(MereRunTheme.monoFont)
                                    .foregroundStyle(MereRunTheme.textMuted)
                                    .textSelection(.enabled)
                            }
                        }
                        .padding(10)
                        .merePanel()
                    }
                }
            }
            .frame(maxHeight: 260)
            Text(catalog.adapterStore)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .textSelection(.enabled)
        }
    }
}

private struct StructuredReceiptPreview: View {
    let json: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Structured receipt", systemImage: "curlybraces")
                .font(.system(size: 13, weight: .semibold))
            ScrollView([.horizontal, .vertical]) {
                Text(json)
                    .font(MereRunTheme.monoFont)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 260)
        }
        .padding(10)
        .merePanel()
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

private struct MultiPathField: View {
    @Binding var paths: String
    let title: String
    let allowedTypes: [UTType]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                Spacer()
                Button {
                    choosePaths()
                } label: {
                    Label("Choose", systemImage: "photo.on.rectangle.angled")
                }
                .buttonStyle(.borderless)
            }
            TextEditor(text: $paths)
                .font(MereRunTheme.monoFont)
                .scrollContentBackground(.hidden)
                .frame(minHeight: 72)
                .padding(8)
                .merePanel()
                .overlay(alignment: .topLeading) {
                    if paths.isBlank {
                        Text("One path per line")
                            .font(MereRunTheme.bodyFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .padding(18)
                            .allowsHitTesting(false)
                    }
                }
        }
    }

    private func choosePaths() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = allowedTypes
        if panel.runModal() == .OK {
            paths = panel.urls.map(\.path).joined(separator: "\n")
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

private struct ImageGenerationOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Generation") {
            VStack(spacing: 10) {
                AdaptiveControlRow {
                    NumberStepper(title: "Steps", value: $controller.draft.steps, range: 1...80, step: 1)
                    NumberField(title: "CFG", value: $controller.draft.cfgScale)
                    NumberField(title: "Strength", value: $controller.draft.strength)
                }
                AdaptiveControlRow {
                    NumberField(title: "Sigma shift", value: $controller.draft.sigmaShift)
                    NumberStepper(
                        title: "Max sequence",
                        value: $controller.draft.maxSequenceLength,
                        range: 64...8_192,
                        step: 64
                    )
                }
                TextField("Seed", text: $controller.draft.seed)
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.bodyFont)
                    .padding(10)
                    .merePanel()
                MultiPathField(
                    paths: $controller.draft.referenceImagePaths,
                    title: "Reference images",
                    allowedTypes: [.image]
                )
                Toggle("Keep original aspect for one HiDream reference", isOn: $controller.draft.keepOriginalAspect)
                PathField(
                    path: $controller.draft.loraPath,
                    placeholder: "LoRA catalog id or .safetensors path",
                    mode: .openFile([.data])
                )
                NumberField(title: "LoRA scale", value: $controller.draft.loraScale)
                DisclosureGroup("Structured prompt") {
                    VStack(spacing: 10) {
                        Toggle("Expand prompt with a local text model", isOn: $controller.draft.structuredPrompt)
                        if controller.draft.structuredPrompt {
                            TextField("Prompt model id", text: $controller.draft.structuredPromptModel)
                                .textFieldStyle(.plain)
                                .padding(10)
                                .merePanel()
                            PathField(
                                path: $controller.draft.structuredPromptModelRoot,
                                placeholder: "Prompt model root (optional)",
                                mode: .openDirectory
                            )
                            NumberStepper(
                                title: "Prompt max tokens",
                                value: $controller.draft.structuredPromptMaxTokens,
                                range: 128...16_384,
                                step: 128
                            )
                            PathField(
                                path: $controller.draft.structuredPromptOutputPath,
                                placeholder: "Save structured caption JSON (optional)",
                                mode: .saveFile
                            )
                        }
                    }
                    .padding(.top, 8)
                }
                DisclosureGroup("Krea tuning") {
                    VStack(spacing: 10) {
                        NumberField(
                            title: "Conditioning multiplier",
                            value: $controller.draft.kreaConditioningMultiplier
                        )
                        TextField(
                            "Layer weights, comma-separated",
                            text: $controller.draft.kreaConditioningLayerWeights
                        )
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                        Picker("Base quantization", selection: $controller.draft.kreaBaseQuantizationBits) {
                            Text("Automatic").tag("")
                            Text("4-bit").tag("4")
                            Text("8-bit").tag("8")
                        }
                    }
                    .padding(.top, 8)
                }
                AdaptiveControlRow {
                    Toggle("Preflight", isOn: $controller.draft.preflight)
                    Toggle("JSON report", isOn: $controller.draft.json)
                        .disabled(!controller.draft.preflight)
                    Toggle("Progress JSON", isOn: $controller.draft.progressJSON)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
            }
        }
    }
}

private struct ImageLoRATrainingOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Training") {
            VStack(spacing: 10) {
                Picker("Recipe", selection: $controller.draft.trainingRecipe) {
                    Text("Custom").tag("")
                    Text("Krea fast style").tag("krea-fast-style")
                    Text("Krea cinematic").tag("krea-cinematic-style")
                    Text("Klein fast style").tag("klein-fast-style")
                }
                if !controller.draft.trainingRecipe.isBlank {
                    Toggle(
                        "Override recipe core parameters",
                        isOn: $controller.draft.overrideTrainingRecipe
                    )
                }
                AdaptiveControlRow {
                    NumberStepper(title: "Steps", value: $controller.draft.steps, range: 1...100_000, step: 100)
                    NumberStepper(title: "Batch", value: $controller.draft.batchSize, range: 1...128, step: 1)
                }
                AdaptiveControlRow {
                    NumberField(title: "Learning rate", value: $controller.draft.learningRate)
                    NumberStepper(title: "Rank", value: $controller.draft.rank, range: 1...512, step: 1)
                    NumberField(title: "Alpha", value: $controller.draft.alpha)
                }
                AdaptiveControlRow {
                    NumberStepper(
                        title: "Max text",
                        value: $controller.draft.maxSequenceLength,
                        range: 64...8_192,
                        step: 64
                    )
                    NumberStepper(
                        title: "Scheduler steps",
                        value: $controller.draft.schedulerSteps,
                        range: 1...10_000,
                        step: 100
                    )
                    NumberField(title: "Caption dropout", value: $controller.draft.captionDropout)
                }
                TextField("Seed", text: $controller.draft.seed)
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.bodyFont)
                    .padding(10)
                    .merePanel()
                DisclosureGroup("Memory and checkpoints") {
                    VStack(spacing: 10) {
                        Picker("Base quantization", selection: $controller.draft.baseQuantizationBits) {
                            Text("Full precision").tag("")
                            Text("4-bit").tag("4")
                            Text("8-bit").tag("8")
                        }
                        AdaptiveControlRow {
                            Toggle("Lite targets", isOn: $controller.draft.trainingLite)
                            Toggle("Low RAM", isOn: $controller.draft.lowRAM)
                            Toggle("Progressive", isOn: $controller.draft.progressive)
                        }
                        AdaptiveControlRow {
                            Toggle("Disable compile", isOn: $controller.draft.disableCompile)
                            Toggle(
                                "Gradient checkpointing",
                                isOn: $controller.draft.gradientCheckpointing
                            )
                            Toggle(
                                "Exclude preview images",
                                isOn: $controller.draft.excludePreviewImages
                            )
                        }
                        AdaptiveControlRow {
                            NumberStepper(
                                title: "Checkpoint interval",
                                value: $controller.draft.checkpointInterval,
                                range: 0...100_000,
                                step: 100
                            )
                            NumberStepper(
                                title: "Max resolution",
                                value: $controller.draft.maxResolution,
                                range: 0...4_096,
                                step: 64
                            )
                        }
                    }
                    .padding(.top, 8)
                }
                DisclosureGroup("Preview and benchmark") {
                    VStack(spacing: 10) {
                        AdaptiveControlRow {
                            NumberStepper(
                                title: "Sample interval",
                                value: $controller.draft.sampleInterval,
                                range: 0...100_000,
                                step: 100
                            )
                            NumberStepper(
                                title: "Benchmark steps",
                                value: $controller.draft.benchmarkSteps,
                                range: 0...10_000,
                                step: 1
                            )
                            NumberStepper(
                                title: "Warmup",
                                value: $controller.draft.benchmarkWarmupSteps,
                                range: 0...1_000,
                                step: 1
                            )
                        }
                        if controller.draft.sampleInterval > 0 {
                            TextField("Preview prompt", text: $controller.draft.samplePrompt)
                                .textFieldStyle(.plain)
                                .padding(10)
                                .merePanel()
                            TextField("Preview model id", text: $controller.draft.sampleModel)
                                .textFieldStyle(.plain)
                                .padding(10)
                                .merePanel()
                            AdaptiveControlRow {
                                NumberStepper(
                                    title: "Preview steps",
                                    value: $controller.draft.sampleSteps,
                                    range: 1...100,
                                    step: 1
                                )
                                NumberField(title: "Preview CFG", value: $controller.draft.sampleCFG)
                                NumberField(
                                    title: "Preview LoRA",
                                    value: $controller.draft.sampleLoRAScale
                                )
                            }
                            TextField("Preview seed (optional)", text: $controller.draft.sampleSeed)
                                .textFieldStyle(.plain)
                                .padding(10)
                                .merePanel()
                        }
                    }
                    .padding(.top, 8)
                }
                DisclosureGroup("Klein target and schedule") {
                    VStack(spacing: 10) {
                        TextField("Target ranks map", text: $controller.draft.loraTargetRanks)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                        AdaptiveControlRow {
                            Picker("Rank preset", selection: $controller.draft.loraRankPreset) {
                                Text("None").tag("")
                                Text("FLUX.2 style 128").tag("flux2-style-128")
                            }
                            Picker("Target preset", selection: $controller.draft.loraTargetPreset) {
                                Text("None").tag("")
                                Text("fal Klein fast").tag("fal-klein-fast")
                            }
                        }
                        Picker("Target mode", selection: $controller.draft.loraTargetMode) {
                            Text("Default").tag("")
                            Text("Suffix").tag("suffix")
                            Text("Transformer linear walk").tag("transformer-linear-walk")
                        }
                        Picker("Timestep sampling", selection: $controller.draft.timestepSampling) {
                            Text("Default").tag("")
                            ForEach(
                                ["uniform", "bellCurve", "contentFocused", "styleFocused", "logitNormal", "shift"],
                                id: \.self
                            ) { Text($0).tag($0) }
                        }
                        AdaptiveControlRow {
                            Picker(
                                "Timestep weighting",
                                selection: $controller.draft.timestepLossWeighting
                            ) {
                                Text("Default").tag("")
                                Text("None").tag("none")
                                Text("Weighted").tag("weighted")
                            }
                            Picker("Loss weighting", selection: $controller.draft.lossWeighting) {
                                Text("Default").tag("")
                                Text("None").tag("none")
                                Text("SNR").tag("snr")
                                Text("minSNR").tag("minSNR")
                            }
                        }
                        AdaptiveControlRow {
                            NumberStepper(
                                title: "Timestep low",
                                value: $controller.draft.timestepLow,
                                range: 0...10_000,
                                step: 1
                            )
                            NumberStepper(
                                title: "Timestep high",
                                value: $controller.draft.timestepHigh,
                                range: 0...10_000,
                                step: 1
                            )
                        }
                        AdaptiveControlRow {
                            NumberStepper(
                                title: "LR warmup",
                                value: $controller.draft.lrWarmupSteps,
                                range: 0...100_000,
                                step: 10
                            )
                            NumberField(title: "LR floor", value: $controller.draft.lrMinFactor)
                            NumberField(
                                title: "Weight decay",
                                value: $controller.draft.adamWeightDecay
                            )
                        }
                        AdaptiveControlRow {
                            Toggle(
                                "Disable cosine scheduler",
                                isOn: $controller.draft.disableCosineScheduler
                            )
                            NumberStepper(
                                title: "Synthetic samples",
                                value: $controller.draft.syntheticSamples,
                                range: 0...10_000,
                                step: 1
                            )
                        }
                    }
                    .padding(.top, 8)
                }
                AdaptiveControlRow {
                    Toggle("Dashboard", isOn: $controller.draft.visualize)
                    Toggle("Preflight", isOn: $controller.draft.preflight)
                    Toggle("JSON report", isOn: $controller.draft.json)
                        .disabled(!controller.draft.preflight)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
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
                if controller.draft.all {
                    PathField(
                        path: $controller.draft.referenceDirectoryPath,
                        placeholder: "Reference artifacts directory",
                        mode: .openDirectory
                    )
                }
            }
        }
    }
}

private struct ImageDatasetDiscoveryOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Discovery") {
            VStack(spacing: 10) {
                AdaptiveControlRow {
                    NumberStepper(
                        title: "Max depth",
                        value: $controller.draft.maxDepth,
                        range: 0...32,
                        step: 1
                    )
                    NumberStepper(
                        title: "Minimum pairs",
                        value: $controller.draft.minUsablePairs,
                        range: 1...100_000,
                        step: 1
                    )
                }
                PathField(
                    path: $controller.draft.trainingOutputRoot,
                    placeholder: "Training output root (optional)",
                    mode: .openDirectory
                )
                TextField("Training model id (optional)", text: $controller.draft.trainingModel)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                Picker("Training recipe", selection: $controller.draft.trainingRecipe) {
                    Text("None").tag("")
                    Text("Krea fast style").tag("krea-fast-style")
                    Text("Krea cinematic").tag("krea-cinematic-style")
                    Text("Klein fast style").tag("klein-fast-style")
                }
                AdaptiveControlRow {
                    Toggle(
                        "Exclude preview images",
                        isOn: $controller.draft.excludePreviewImages
                    )
                    Toggle("JSON report", isOn: $controller.draft.json)
                }
            }
        }
    }
}

private struct ImageRunPlanOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Plan") {
            VStack(spacing: 10) {
                Picker("Action", selection: planAction) {
                    Text("Run").tag("run")
                    Text("Preflight").tag("preflight")
                    Text("Materialize").tag("materialize")
                }
                .pickerStyle(.segmented)
                if !controller.draft.materializePath.isBlank {
                    PathField(
                        path: $controller.draft.materializePath,
                        placeholder: "Durable run directory",
                        mode: .openDirectory
                    )
                }
                Toggle("JSON receipt", isOn: $controller.draft.json)
            }
        }
    }

    private var planAction: Binding<String> {
        Binding(
            get: {
                if !controller.draft.materializePath.isBlank { return "materialize" }
                return controller.draft.preflight ? "preflight" : "run"
            },
            set: { value in
                switch value {
                case "preflight":
                    controller.draft.preflight = true
                    controller.draft.materializePath = ""
                case "materialize":
                    controller.draft.preflight = false
                    if controller.draft.materializePath.isBlank {
                        controller.draft.materializePath = FileManager.default.temporaryDirectory
                            .appendingPathComponent("mere-image-run", isDirectory: true)
                            .path
                    }
                default:
                    controller.draft.preflight = false
                    controller.draft.materializePath = ""
                }
            }
        )
    }
}

private struct ImageRunViewerOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Dashboard") {
            NumberStepper(
                title: "Loopback port",
                value: $controller.draft.port,
                range: 1...65_535,
                step: 1
            )
        }
    }
}

private enum ImageReconstructionKind {
    case triposr
    case trellis2
    case multiview
}

private struct ImageReconstructionOptions: View {
    @EnvironmentObject private var controller: MereRunController
    let kind: ImageReconstructionKind

    var body: some View {
        EditorSection("Reconstruction") {
            VStack(spacing: 10) {
                if kind == .multiview {
                    MultiPathField(
                        paths: $controller.draft.referenceImagePaths,
                        title: "Ordered views (exactly 4 or 6)",
                        allowedTypes: [.image]
                    )
                    PathField(
                        path: $controller.draft.camerasPath,
                        placeholder: "Camera JSON (optional)",
                        mode: .openFile([.json])
                    )
                }
                if kind != .trellis2 {
                    NumberStepper(
                        title: "Grid resolution",
                        value: $controller.draft.reconstructionResolution,
                        range: 2...(kind == .triposr ? 512 : 256),
                        step: 2
                    )
                }
                if kind == .triposr {
                    AdaptiveControlRow {
                        NumberField(
                            title: "Density threshold",
                            value: $controller.draft.densityThreshold
                        )
                        NumberField(
                            title: "Foreground ratio",
                            value: $controller.draft.foregroundRatio
                        )
                    }
                }
                if kind == .trellis2 {
                    AdaptiveControlRow {
                        TextField("Seed", text: $controller.draft.seed)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                        NumberStepper(
                            title: "Maximum sparse tokens",
                            value: $controller.draft.maxTokens,
                            range: 1...8_388_608,
                            step: 65_536
                        )
                    }
                }
                AdaptiveControlRow {
                    if kind != .multiview {
                        Toggle("Already framed", isOn: $controller.draft.alreadyFramed)
                    }
                    if kind != .trellis2 {
                        Toggle("Geometry only", isOn: $controller.draft.noVertexColors)
                    }
                    Toggle("Dry run", isOn: $controller.draft.dryRun)
                    Toggle("JSON receipt", isOn: $controller.draft.json)
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
                        NumberField(title: "Min-p", value: $controller.draft.minP)
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
                    if controller.draft.model.localizedCaseInsensitiveContains("inkling") {
                        NumberField(
                            title: "Inkling reasoning effort",
                            value: Binding(
                                get: { controller.draft.reasoningEffort ?? 0.9 },
                                set: { controller.draft.reasoningEffort = min(max($0, 0), 0.99) }
                            )
                        )
                    }
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
                Picker("Base family", selection: $controller.draft.model) {
                    Text("Gemma 4 12B").tag("text-chat-gemma4-12b-4bit")
                    Text("Laguna XS 2.1").tag("text-chat-laguna-xs-2-1")
                    Text("Inkling-Small").tag("text-chat-inkling-small")
                }
                .pickerStyle(.segmented)
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
                Text("Leave target modules empty for the model-family defaults. Inkling includes attention, MLP, routed/shared experts, and unembedding.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                if controller.draft.model.localizedCaseInsensitiveContains("inkling") {
                    NumberField(
                        title: "Reasoning effort",
                        value: Binding(
                            get: { controller.draft.reasoningEffort ?? 0.9 },
                            set: { controller.draft.reasoningEffort = min(max($0, 0), 0.99) }
                        )
                    )
                }
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
                if controller.draft.stream {
                    NumberStepper(
                        title: "Chunk tokens",
                        value: $controller.draft.speechStreamChunkTokens,
                        range: 1...4_096,
                        step: 1
                    )
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
                if controller.draft.stream {
                    AdaptiveControlRow {
                        NumberStepper(
                            title: "Feed ms",
                            value: $controller.draft.speechStreamChunkMS,
                            range: 1...60_000,
                            step: 50
                        )
                        NumberStepper(
                            title: "Decode ms",
                            value: $controller.draft.speechStreamDecodeMS,
                            range: 1...60_000,
                            step: 100
                        )
                        Toggle("JSON Lines events", isOn: $controller.draft.speechJSONL)
                    }
                    AdaptiveControlRow {
                        TextField("Raw stdin format (optional)", text: $controller.draft.speechInputFormat)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                        NumberStepper(
                            title: "Sample rate",
                            value: $controller.draft.speechSampleRate,
                            range: 1...192_000,
                            step: 1_000
                        )
                    }
                }
            }
        }
    }
}

private struct SpeechDiarizationOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Speaker diarization") {
            VStack(spacing: 10) {
                Picker(
                    "Format",
                    selection: Binding(
                        get: { controller.draft.speechDiarizationFormat ?? "json" },
                        set: { controller.draft.speechDiarizationFormat = $0 }
                    )
                ) {
                    Text("JSON").tag("json")
                    Text("RTTM").tag("rttm")
                }
                .pickerStyle(.segmented)
                AdaptiveControlRow {
                    NumberField(
                        title: "Activity threshold",
                        value: Binding(
                            get: { controller.draft.speechDiarizationThreshold ?? 0.5 },
                            set: { controller.draft.speechDiarizationThreshold = $0 }
                        )
                    )
                    NumberField(
                        title: "Minimum seconds",
                        value: Binding(
                            get: { controller.draft.speechDiarizationMinDuration ?? 0.25 },
                            set: { controller.draft.speechDiarizationMinDuration = $0 }
                        )
                    )
                    NumberField(
                        title: "Merge gap",
                        value: Binding(
                            get: { controller.draft.speechDiarizationMergeGap ?? 0.25 },
                            set: { controller.draft.speechDiarizationMergeGap = $0 }
                        )
                    )
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

private struct VisionLanguageOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Vision language model") {
            AdaptiveControlRow {
                NumberStepper(title: "Max tokens", value: $controller.draft.maxTokens, range: 1...32_768, step: 64)
                NumberField(title: "Temperature", value: $controller.draft.temperature)
                NumberField(title: "Top-p", value: $controller.draft.topP)
            }
        }
    }
}

private struct VisionCaptionOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Captioning") {
            VStack(spacing: 10) {
                MultiPathField(
                    paths: $controller.draft.visionAdditionalInputs,
                    title: "Additional images",
                    allowedTypes: [.image]
                )
                PathField(
                    path: $controller.draft.visionPromptFile,
                    placeholder: "Prompt file",
                    mode: .openFile([.plainText])
                )
                TextField("Focus details, one per line", text: $controller.draft.visionFocus, axis: .vertical)
                    .lineLimit(2...8)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                TextField("LoRA trigger token", text: $controller.draft.visionTriggerToken)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                AdaptiveControlRow {
                    NumberStepper(title: "Max tokens", value: $controller.draft.maxTokens, range: 1...8_192, step: 32)
                    NumberField(title: "Temperature", value: $controller.draft.temperature)
                    NumberField(title: "Top-p", value: $controller.draft.topP)
                }
            }
        }
    }
}

private struct VisionOCROptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("OCR") {
            VStack(spacing: 10) {
                MultiPathField(
                    paths: $controller.draft.visionAdditionalInputs,
                    title: "Additional images",
                    allowedTypes: [.image]
                )
                Picker("Backend", selection: $controller.draft.backend) {
                    Text("LightOn").tag("lighton")
                    Text("GLM").tag("glm")
                    Text("Infinity").tag("infinity")
                }
                .pickerStyle(.segmented)
                AdaptiveControlRow {
                    Toggle("Compare backends", isOn: $controller.draft.all)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                    NumberStepper(title: "Max tokens", value: $controller.draft.maxTokens, range: 1...32_768, step: 64)
                    NumberField(title: "Temperature", value: $controller.draft.temperature)
                }
                if controller.draft.backend == "glm" || controller.draft.all {
                    TextField("glmocr executable", text: $controller.draft.visionGLMOCRCLI)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    PathField(
                        path: $controller.draft.visionGLMConfig,
                        placeholder: "GLM config YAML",
                        mode: .openFile([.yaml])
                    )
                }
                if controller.draft.backend == "infinity" || controller.draft.all {
                    Picker("Infinity runtime", selection: $controller.draft.visionInfinityRuntime) {
                        Text("Native").tag("native")
                        Text("External").tag("external")
                    }
                    .pickerStyle(.segmented)
                    TextField("Infinity model", text: $controller.draft.visionInfinityModel)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    Picker("Task", selection: $controller.draft.visionInfinityTask) {
                        Text("Document JSON").tag("doc2json")
                        Text("Document Markdown").tag("doc2md")
                        Text("Custom").tag("custom")
                    }
                    .pickerStyle(.segmented)
                    Picker("Output format", selection: $controller.draft.visionInfinityOutputFormat) {
                        Text("Markdown").tag("md")
                        Text("JSON").tag("json")
                    }
                    .pickerStyle(.segmented)
                    if controller.draft.visionInfinityTask == "custom" {
                        TextField("Custom Infinity prompt", text: $controller.draft.visionInfinityPrompt, axis: .vertical)
                            .lineLimit(2...6)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                    }
                    if controller.draft.visionInfinityRuntime == "external" {
                        TextField("Parser executable", text: $controller.draft.visionInfinityParserCLI)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                        Picker("External backend", selection: $controller.draft.visionInfinityBackend) {
                            Text("Transformers").tag("transformers")
                            Text("vLLM engine").tag("vllm-engine")
                            Text("vLLM server").tag("vllm-server")
                        }
                        .pickerStyle(.segmented)
                        TextField("Completions URL", text: $controller.draft.visionInfinityAPIURL)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                        SecureField("Infinity API key", text: $controller.draft.visionInfinityAPIKey)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                        PathField(
                            path: $controller.draft.visionInfinityModelCacheDirectory,
                            placeholder: "External model cache",
                            mode: .openDirectory
                        )
                    }
                    AdaptiveControlRow {
                        NumberStepper(
                            title: "Batch",
                            value: $controller.draft.visionInfinityBatchSize,
                            range: 1...256,
                            step: 1
                        )
                        NumberStepper(
                            title: "Min pixels",
                            value: $controller.draft.visionInfinityMinPixels,
                            range: 1...16_777_216,
                            step: 1_024
                        )
                        NumberStepper(
                            title: "Max pixels",
                            value: $controller.draft.visionInfinityMaxPixels,
                            range: 2_048...134_217_728,
                            step: 1_024
                        )
                    }
                }
            }
        }
    }
}

private struct VisionGroundingOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Grounding exports") {
            VStack(spacing: 10) {
                PathField(
                    path: $controller.draft.visionJSONOutputPath,
                    placeholder: "Detection JSON",
                    mode: .saveFile
                )
                PathField(
                    path: $controller.draft.visionMaskOutputDirectory,
                    placeholder: "Per-detection mask directory",
                    mode: .openDirectory
                )
            }
        }
    }
}

private struct VisionTrackingOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Vision") {
            VStack(spacing: 10) {
                if controller.selectedTemplate.id != .visionTrackLive {
                    TextField("Box prompts, one x1,y1,x2,y2,label per line", text: $controller.draft.visionBoxPrompts, axis: .vertical)
                        .lineLimit(1...6)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("Point prompts, one x,y,positive|negative,label per line", text: $controller.draft.visionPointPrompts, axis: .vertical)
                        .lineLimit(1...6)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
                PathField(
                    path: $controller.draft.visionJSONOutputPath,
                    placeholder: "Structured JSON output",
                    mode: .saveFile
                )
                if controller.selectedTemplate.id != .visionTrackLive {
                    PathField(
                        path: $controller.draft.visionMaskOutputDirectory,
                        placeholder: "Mask output directory",
                        mode: .openDirectory
                    )
                }
                AdaptiveControlRow {
                    NumberField(title: "Threshold", value: $controller.draft.visionThreshold)
                    NumberStepper(
                        title: "Resolution",
                        value: $controller.draft.visionResolution,
                        range: 64...4_096,
                        step: 16
                    )
                    NumberStepper(
                        title: "Init frame",
                        value: $controller.draft.visionInitFrame,
                        range: 0...100_000,
                        step: 1
                    )
                }
                if controller.selectedTemplate.id == .visionTrackLive {
                    NumberField(title: "Seconds", value: $controller.draft.durationSeconds)
                    NumberStepper(title: "Camera", value: $controller.draft.visionCamera, range: 0...32, step: 1)
                    NumberStepper(
                        title: "Seed search",
                        value: $controller.draft.visionSeedSearchFrames,
                        range: 0...1_000,
                        step: 1
                    )
                } else if controller.selectedTemplate.id == .visionTrack {
                    TextField("End frame (optional)", text: $controller.draft.visionEndFrame)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
                AdaptiveControlRow {
                    Toggle("Show boxes", isOn: $controller.draft.force)
                    if controller.selectedTemplate.id == .visionSegment {
                        Toggle("Multiple masks", isOn: $controller.draft.visionMultimask)
                    } else {
                        Toggle("Show labels", isOn: $controller.draft.visionShowLabels)
                    }
                    if controller.selectedTemplate.id == .visionTrack {
                        Toggle("Preflight only", isOn: $controller.draft.preflight)
                        if controller.draft.preflight {
                            Toggle("JSON preflight", isOn: $controller.draft.json)
                        }
                    }
                }
            }
        }
    }
}

private struct VisionFaceOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Face analysis") {
            VStack(spacing: 10) {
                if controller.selectedTemplate.id == .visionFaceCompare {
                    PathField(
                        path: $controller.draft.visionSecondInputPath,
                        placeholder: "Candidate image",
                        mode: .openFile([.image])
                    )
                }
                if controller.selectedTemplate.id == .visionFaceBatch {
                    MultiPathField(
                        paths: $controller.draft.visionAdditionalInputs,
                        title: "Additional images",
                        allowedTypes: [.image]
                    )
                    PathField(
                        path: $controller.draft.visionInputList,
                        placeholder: "Input-list file",
                        mode: .openFile([.plainText])
                    )
                    PathField(
                        path: $controller.draft.visionJSONLOutput,
                        placeholder: "JSONL output",
                        mode: .saveFile
                    )
                } else {
                    PathField(
                        path: $controller.draft.visionJSONOutputPath,
                        placeholder: "JSON output",
                        mode: .saveFile
                    )
                }
                Picker("Provider", selection: $controller.draft.visionExecutionProvider) {
                    Text("Auto").tag("auto")
                    Text("Core ML").tag("coreml")
                    Text("CPU").tag("cpu")
                }
                .pickerStyle(.segmented)
                AdaptiveControlRow {
                    NumberField(title: "Score threshold", value: $controller.draft.visionFaceScoreThreshold)
                    if [.visionFaceDetect, .visionFaceBatch].contains(controller.selectedTemplate.id) {
                        NumberStepper(
                            title: "Max faces (0 = all)",
                            value: $controller.draft.visionMaxFaces,
                            range: 0...1_000,
                            step: 1
                        )
                        Toggle("Embeddings", isOn: $controller.draft.visionIncludeEmbeddings)
                    }
                    Toggle("Print JSON", isOn: $controller.draft.json)
                }
                if controller.selectedTemplate.id == .visionFaceEmbed {
                    TextField("Face index (largest by default)", text: $controller.draft.visionFaceIndex)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
                if controller.selectedTemplate.id == .visionFaceCompare {
                    AdaptiveControlRow {
                        TextField("Reference face index", text: $controller.draft.visionReferenceFaceIndex)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                        TextField("Candidate face index", text: $controller.draft.visionCandidateFaceIndex)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                    }
                }
                if controller.selectedTemplate.id == .visionFaceBatch {
                    Toggle("Fail fast", isOn: $controller.draft.visionFailFast)
                }
            }
        }
    }
}

private struct VisionPoseOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Pose") {
            VStack(spacing: 10) {
                PathField(
                    path: $controller.draft.visionJSONOutputPath,
                    placeholder: "Landmark JSON output",
                    mode: .saveFile
                )
                AdaptiveControlRow {
                    Toggle("Body", isOn: $controller.draft.visionPoseBody)
                    Toggle("Hands", isOn: $controller.draft.visionPoseHands)
                    Toggle("Face", isOn: $controller.draft.visionPoseFace)
                    Toggle("Print JSON", isOn: $controller.draft.json)
                }
                AdaptiveControlRow {
                    NumberStepper(title: "Max hands", value: $controller.draft.visionMaxHands, range: 0...16, step: 1)
                    NumberField(title: "Minimum confidence", value: $controller.draft.visionMinimumConfidence)
                }
            }
        }
    }
}

private struct VisionFlowOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Optical flow") {
            VStack(spacing: 10) {
                PathField(
                    path: $controller.draft.visionSecondInputPath,
                    placeholder: "Target image",
                    mode: .openFile([.image])
                )
                PathField(
                    path: $controller.draft.visionJSONOutputPath,
                    placeholder: "Metadata JSON",
                    mode: .saveFile
                )
                Picker("Accuracy", selection: $controller.draft.visionFlowAccuracy) {
                    Text("Low").tag("low")
                    Text("Medium").tag("medium")
                    Text("High").tag("high")
                    Text("Very high").tag("very-high")
                }
                .pickerStyle(.segmented)
                Toggle("Print JSON", isOn: $controller.draft.json)
            }
        }
    }
}

private struct VisionDepthOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Video depth") {
            AdaptiveControlRow {
                NumberStepper(
                    title: "Input edge",
                    value: $controller.draft.visionInputSize,
                    range: 14...1_008,
                    step: 14
                )
                NumberStepper(
                    title: "Max frames",
                    value: $controller.draft.visionMaxFrames,
                    range: 1...2_400,
                    step: 1
                )
                Toggle("Dry run", isOn: $controller.draft.dryRun)
                Toggle("Print JSON", isOn: $controller.draft.json)
            }
        }
    }
}

private struct VisionGeometryOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Metric geometry") {
            AdaptiveControlRow {
                NumberStepper(
                    title: "Quality",
                    value: $controller.draft.visionResolutionLevel,
                    range: 0...9,
                    step: 1
                )
                NumberStepper(
                    title: "Token count",
                    value: $controller.draft.visionTokenCount,
                    range: 0...3_600,
                    step: 1
                )
                NumberStepper(
                    title: "Max points",
                    value: $controller.draft.visionMaxPoints,
                    range: 0...10_000_000,
                    step: 10_000
                )
                Toggle("Dry run", isOn: $controller.draft.dryRun)
                Toggle("Print JSON", isOn: $controller.draft.json)
            }
        }
    }
}

private struct VisionMultiviewOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Multi-view geometry") {
            VStack(spacing: 10) {
                MultiPathField(
                    paths: $controller.draft.visionAdditionalInputs,
                    title: "Additional ordered views",
                    allowedTypes: [.image]
                )
                PathField(
                    path: $controller.draft.camerasPath,
                    placeholder: "Optional W2C camera JSON",
                    mode: .openFile([.json])
                )
                Picker("Reference view", selection: $controller.draft.visionReferenceView) {
                    Text("First").tag("first")
                    Text("Middle").tag("middle")
                    Text("Saddle balanced").tag("saddle-balanced")
                    Text("Similarity range").tag("saddle-similarity-range")
                }
                AdaptiveControlRow {
                    NumberStepper(
                        title: "Process edge",
                        value: $controller.draft.visionProcessResolution,
                        range: 64...4_096,
                        step: 8
                    )
                    NumberField(
                        title: "Confidence percentile",
                        value: $controller.draft.visionConfidencePercentile
                    )
                    NumberStepper(
                        title: "Max points",
                        value: $controller.draft.visionMaxPoints,
                        range: 0...10_000_000,
                        step: 10_000
                    )
                    Toggle("Dry run", isOn: $controller.draft.dryRun)
                    Toggle("Print JSON", isOn: $controller.draft.json)
                }
            }
        }
    }
}

private struct MusicGenerationOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Production") {
            VStack(spacing: 10) {
                Picker("Quality", selection: $controller.draft.musicQuality) {
                    Text("Draft").tag("draft")
                    Text("Song").tag("song")
                    Text("Final").tag("final")
                    Text("Edit").tag("edit")
                }
                .pickerStyle(.segmented)
                Picker("Task", selection: $controller.draft.musicTask) {
                    Text("Create").tag("text2music")
                    Text("Cover").tag("cover")
                    Text("No-FSQ cover").tag("cover-nofsq")
                    Text("Repaint").tag("repaint")
                    Text("Extract").tag("extract")
                    Text("Lego").tag("lego")
                    Text("Complete").tag("complete")
                }
                AdaptiveControlRow {
                    Toggle("Set duration", isOn: $controller.draft.useDuration)
                    if controller.draft.useDuration {
                        NumberField(title: "Seconds", value: $controller.draft.durationSeconds)
                    }
                    NumberStepper(
                        title: "Candidates",
                        value: $controller.draft.musicCandidates,
                        range: 0...16,
                        step: 1
                    )
                }
                AdaptiveControlRow {
                    Toggle("Override steps", isOn: $controller.draft.musicOverrideSteps)
                    if controller.draft.musicOverrideSteps {
                        NumberStepper(title: "Steps", value: $controller.draft.steps, range: 1...200, step: 1)
                    }
                    Toggle("Keep candidates", isOn: $controller.draft.musicKeepCandidates)
                }
                TextField("Seed", text: $controller.draft.seed)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                Toggle("Quiet", isOn: $controller.draft.quiet)
            }
        }

        EditorSection("Conditioning & editing") {
            VStack(spacing: 10) {
                PathField(
                    path: $controller.draft.musicSourceAudio,
                    placeholder: "Source audio for cover/edit",
                    mode: .openFile([.audio])
                )
                MultiPathField(
                    paths: $controller.draft.musicReferenceAudioPaths,
                    title: "Timbre reference audio",
                    allowedTypes: [.audio]
                )
                AdaptiveControlRow {
                    NumberField(title: "Cover strength", value: $controller.draft.musicCoverStrength)
                    NumberField(title: "Cover noise", value: $controller.draft.musicCoverNoiseStrength)
                    NumberField(title: "Retake variance", value: $controller.draft.musicRetakeVariance)
                }
                TextField("Retake seed", text: $controller.draft.musicRetakeSeed)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                if ["extract", "lego"].contains(controller.draft.musicTask) {
                    TextField("Track name", text: $controller.draft.musicTrackName)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
                if controller.draft.musicTask == "complete" {
                    TextField("Track classes (Drums,Bass,…)", text: $controller.draft.musicCompleteTrackClasses)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
                if ["repaint", "lego"].contains(controller.draft.musicTask) {
                    AdaptiveControlRow {
                        NumberField(title: "Repaint start", value: $controller.draft.musicRepaintStart)
                        NumberField(title: "Repaint end", value: $controller.draft.musicRepaintEnd)
                        NumberField(title: "Strength", value: $controller.draft.musicRepaintStrength)
                    }
                    Picker("Preservation", selection: $controller.draft.musicRepaintMode) {
                        Text("Conservative").tag("conservative")
                        Text("Balanced").tag("balanced")
                        Text("Aggressive").tag("aggressive")
                    }
                    .pickerStyle(.segmented)
                    Picker("Chunk mask", selection: $controller.draft.musicChunkMaskMode) {
                        Text("Auto").tag("auto")
                        Text("Explicit").tag("explicit")
                    }
                    .pickerStyle(.segmented)
                }
                Toggle("Flow edit toward this prompt", isOn: $controller.draft.musicFlowEdit)
                if controller.draft.musicFlowEdit {
                    TextField("Source caption", text: $controller.draft.musicSourceCaption)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("Source lyrics", text: $controller.draft.musicSourceLyrics, axis: .vertical)
                        .lineLimit(2...6)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    AdaptiveControlRow {
                        NumberField(title: "Flow start", value: $controller.draft.musicFlowEditNMin)
                        NumberField(title: "Flow end", value: $controller.draft.musicFlowEditNMax)
                        NumberStepper(
                            title: "Noise draws",
                            value: $controller.draft.musicFlowEditNAverage,
                            range: 1...64,
                            step: 1
                        )
                    }
                }
            }
        }

        EditorSection("Lyrics & planning") {
            VStack(spacing: 10) {
                PathField(
                    path: $controller.draft.musicLyricsFile,
                    placeholder: "Plain lyrics file",
                    mode: .openFile([.plainText])
                )
                PathField(
                    path: $controller.draft.musicLRCFile,
                    placeholder: "Synchronized LRC file",
                    mode: .openFile([.plainText])
                )
                Picker("LM planning", selection: $controller.draft.musicLMMode) {
                    Text("Quality preset").tag("auto")
                    Text("Force on").tag("use")
                    Text("Force off").tag("disable")
                }
                .pickerStyle(.segmented)
                Toggle("Analyze source to fill metadata", isOn: $controller.draft.musicAnalyzeSourceAudio)
                AdaptiveControlRow {
                    TextField("BPM", text: $controller.draft.musicBPM)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("Key (C major)", text: $controller.draft.musicKey)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("Time signature", text: $controller.draft.musicTimeSignature)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
                AdaptiveControlRow {
                    NumberStepper(title: "LM top-k", value: $controller.draft.musicLMTopK, range: 0...2_048, step: 1)
                    NumberField(title: "LM top-p", value: $controller.draft.musicLMTopP)
                }
                TextField("Vocal language", text: $controller.draft.musicVocalLanguage)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                TextField("ACE-Step instruction", text: $controller.draft.musicInstruction, axis: .vertical)
                    .lineLimit(2...5)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
            }
        }

        EditorSection("Diffusion controls") {
            VStack(spacing: 10) {
                AdaptiveControlRow {
                    TextField("Shift (preset)", text: $controller.draft.musicShift)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("Guidance (preset)", text: $controller.draft.musicGuidanceScale)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
                Picker("Inference", selection: $controller.draft.musicInferMethod) {
                    Text("Preset").tag("")
                    Text("ODE").tag("ode")
                    Text("SDE").tag("sde")
                }
                .pickerStyle(.segmented)
                Picker("Sampler", selection: $controller.draft.musicSampler) {
                    Text("Preset").tag("")
                    Text("Euler").tag("euler")
                    Text("Heun").tag("heun")
                }
                .pickerStyle(.segmented)
                Picker("Guidance mode", selection: $controller.draft.musicGuidanceMode) {
                    Text("Preset").tag("")
                    Text("APG").tag("apg")
                    Text("ADG").tag("adg")
                    Text("CFG").tag("cfg")
                }
                .pickerStyle(.segmented)
                AdaptiveControlRow {
                    TextField("CFG start", text: $controller.draft.musicCFGIntervalStart)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("CFG end", text: $controller.draft.musicCFGIntervalEnd)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("Velocity clamp", text: $controller.draft.musicVelocityNormThreshold)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("Velocity EMA", text: $controller.draft.musicVelocityEMAFactor)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
                AdaptiveControlRow {
                    Toggle("Disable tiled VAE", isOn: $controller.draft.musicNoTiledVAE)
                    NumberStepper(
                        title: "VAE chunk",
                        value: $controller.draft.musicVAEChunkSize,
                        range: 64...4_096,
                        step: 64
                    )
                    NumberStepper(
                        title: "Overlap",
                        value: $controller.draft.musicVAEOverlap,
                        range: 0...1_024,
                        step: 16
                    )
                }
            }
        }

        MusicAdapterAndLayoutOptions()
        MusicDeliveryOptions()
        if controller.draft.model.localizedCaseInsensitiveContains("magenta") {
            MusicMagentaOptions()
        }
    }
}

private struct MusicAnalysisOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Audio understanding") {
            VStack(spacing: 10) {
                Toggle("Limit analysis duration", isOn: $controller.draft.useDuration)
                if controller.draft.useDuration {
                    NumberField(title: "Seconds", value: $controller.draft.durationSeconds)
                }
                AdaptiveControlRow {
                    NumberStepper(
                        title: "Max tokens",
                        value: $controller.draft.musicAnalysisMaxTokens,
                        range: 1...32_768,
                        step: 128
                    )
                    NumberField(title: "Temperature", value: $controller.draft.musicAnalysisTemperature)
                    NumberStepper(title: "Top-k", value: $controller.draft.musicLMTopK, range: 0...2_048, step: 1)
                    NumberField(title: "Top-p", value: $controller.draft.musicLMTopP)
                }
                AdaptiveControlRow {
                    Toggle("Include raw LM", isOn: $controller.draft.musicIncludeRawLM)
                    Toggle("Include audio codes", isOn: $controller.draft.musicIncludeAudioCodes)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
            }
        }
        MusicModelLayoutOptions(includesLM: true, includesText: false, includesAdapters: false)
    }
}

private struct MusicTranscriptionOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("MuScriptor") {
            VStack(spacing: 10) {
                PathField(
                    path: $controller.draft.musicTranscribeModelPath,
                    placeholder: "Explicit model directory",
                    mode: .openDirectory
                )
                Picker("Architecture", selection: $controller.draft.musicTranscribeVariant) {
                    Text("Auto").tag("")
                    Text("Small").tag("small")
                    Text("Medium").tag("medium")
                    Text("Large").tag("large")
                }
                .pickerStyle(.segmented)
                Picker("Output", selection: $controller.draft.musicTranscribeFormat) {
                    Text("MIDI").tag("midi")
                    Text("JSON").tag("json")
                    Text("JSONL").tag("jsonl")
                }
                .pickerStyle(.segmented)
                TextField("Instrument groups", text: $controller.draft.musicInstruments)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                AdaptiveControlRow {
                    Toggle("List instruments", isOn: $controller.draft.musicListInstruments)
                    Toggle("Sample tokens", isOn: $controller.draft.musicSampling)
                    Toggle("Strict EOS", isOn: $controller.draft.musicStrictEOS)
                }
                AdaptiveControlRow {
                    NumberField(title: "Temperature", value: $controller.draft.temperature)
                    NumberStepper(
                        title: "Tokens/chunk",
                        value: $controller.draft.musicMaxTokensPerChunk,
                        range: 1...32_768,
                        step: 100
                    )
                    NumberStepper(title: "Beam", value: $controller.draft.musicBeamSize, range: 1...32, step: 1)
                    NumberStepper(
                        title: "Chunk batch",
                        value: $controller.draft.musicChunkBatchSize,
                        range: 1...64,
                        step: 1
                    )
                }
                Picker("Compute", selection: $controller.draft.musicDType) {
                    Text("bfloat16").tag("bfloat16")
                    Text("float16").tag("float16")
                    Text("float32").tag("float32")
                }
                .pickerStyle(.segmented)
                AdaptiveControlRow {
                    Toggle("Legacy fixed tempo", isOn: $controller.draft.musicNoMusicalContext)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
                PathField(
                    path: $controller.draft.musicContextOutput,
                    placeholder: "Musical context JSON output",
                    mode: .saveFile
                )
            }
        }
    }
}

private struct MusicRealtimeOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Realtime session") {
            VStack(spacing: 10) {
                AdaptiveControlRow {
                    NumberField(title: "Seconds", value: $controller.draft.durationSeconds)
                    Toggle("Play audio", isOn: $controller.draft.musicPlay)
                    Toggle("Interactive stdin", isOn: $controller.draft.musicInteractive)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
                MusicMagentaOptions()
            }
        }
        EditorSection("MIDI steering") {
            VStack(spacing: 10) {
                AdaptiveControlRow {
                    Toggle("List inputs", isOn: $controller.draft.musicListMIDIInputs)
                    Toggle("Monitor only", isOn: $controller.draft.musicMIDIMonitor)
                    Toggle("Log events", isOn: $controller.draft.musicMIDILogEvents)
                    Toggle("Log raw bytes", isOn: $controller.draft.musicMIDILogRaw)
                }
                AdaptiveControlRow {
                    TextField("MIDI source", text: $controller.draft.musicMIDIInput)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("Channel (all or 1–16)", text: $controller.draft.musicMIDIChannel)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    NumberStepper(
                        title: "Transpose",
                        value: $controller.draft.musicMIDINoteOffset,
                        range: -127...127,
                        step: 1
                    )
                }
                TextField(
                    "MIDI CC mappings, one per line (1=temp:0.2:1.4)",
                    text: $controller.draft.musicMIDICCMappings,
                    axis: .vertical
                )
                .lineLimit(2...8)
                .textFieldStyle(.plain)
                .padding(10)
                .merePanel()
            }
        }
    }
}

private struct MusicTrainingOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Adapter training") {
            VStack(spacing: 10) {
                Picker("Kind", selection: $controller.draft.musicTrainingKind) {
                    Text("LoRA").tag("lora")
                    Text("LoKr").tag("lokr")
                }
                .pickerStyle(.segmented)
                AdaptiveControlRow {
                    NumberStepper(title: "Steps", value: $controller.draft.steps, range: 1...1_000_000, step: 100)
                    NumberStepper(title: "Rank", value: $controller.draft.rank, range: 1...1_024, step: 1)
                    NumberField(title: "Alpha", value: $controller.draft.alpha)
                    NumberStepper(
                        title: "LoKr factor",
                        value: $controller.draft.musicTrainingFactor,
                        range: -1...1_024,
                        step: 1
                    )
                }
                AdaptiveControlRow {
                    NumberField(title: "Learning rate", value: $controller.draft.learningRate)
                    NumberField(title: "Weight decay", value: $controller.draft.musicTrainingWeightDecay)
                    NumberField(title: "Max seconds", value: $controller.draft.musicTrainingMaxDuration)
                    NumberStepper(
                        title: "Log every",
                        value: $controller.draft.musicTrainingLogEvery,
                        range: 1...100_000,
                        step: 1
                    )
                }
                TextField("Seed", text: $controller.draft.seed)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
            }
        }
        MusicModelLayoutOptions(includesLM: false, includesText: true, includesAdapters: false)
    }
}

private struct MusicServeOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Resident API") {
            VStack(spacing: 10) {
                AdaptiveControlRow {
                    TextField("Host", text: $controller.draft.host)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    NumberStepper(title: "Port", value: $controller.draft.port, range: 1...65_535, step: 1)
                }
                SecureField("API key (required off localhost)", text: $controller.draft.apiKey)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
            }
        }
        MusicModelLayoutOptions(includesLM: true, includesText: true, includesAdapters: true)
    }
}

private struct MusicAdapterAndLayoutOptions: View {
    var body: some View {
        MusicModelLayoutOptions(includesLM: true, includesText: true, includesAdapters: true)
    }
}

private struct MusicModelLayoutOptions: View {
    @EnvironmentObject private var controller: MereRunController
    let includesLM: Bool
    let includesText: Bool
    let includesAdapters: Bool

    var body: some View {
        EditorSection("Model layout & adapters") {
            VStack(spacing: 10) {
                PathField(
                    path: $controller.draft.musicCheckpointsRoot,
                    placeholder: "Checkpoints root (auto)",
                    mode: .openDirectory
                )
                AdaptiveControlRow {
                    TextField("Decoder subdirectory", text: $controller.draft.musicDecoderSubdirectory)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("VAE subdirectory", text: $controller.draft.musicVAESubdirectory)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
                if includesLM {
                    TextField("5Hz LM model (default: 1.7B)", text: $controller.draft.musicLMModel)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    TextField("5Hz LM subdirectory (auto)", text: $controller.draft.musicLMSubdirectory)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
                if includesText {
                    TextField("Text encoder subdirectory (auto)", text: $controller.draft.musicTextSubdirectory)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
                if includesAdapters {
                    MultiPathField(
                        paths: $controller.draft.musicAdapterPaths,
                        title: "ACE-Step adapters",
                        allowedTypes: [.data]
                    )
                    Picker("Adapter kind", selection: $controller.draft.musicAdapterKind) {
                        Text("Auto").tag("auto")
                        Text("LoRA").tag("lora")
                        Text("LoKr").tag("lokr")
                    }
                    .pickerStyle(.segmented)
                    TextField(
                        "Adapter scales, one per line",
                        text: $controller.draft.musicAdapterScales,
                        axis: .vertical
                    )
                    .lineLimit(1...6)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                }
            }
        }
    }
}

private struct MusicDeliveryOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Delivery") {
            VStack(spacing: 10) {
                AdaptiveControlRow {
                    Picker("Encoding", selection: $controller.draft.musicExportFormat) {
                        Text("PCM 16").tag("pcm16")
                        Text("PCM 24").tag("pcm24")
                        Text("Float 32").tag("float32")
                    }
                    Picker("Normalize", selection: $controller.draft.musicNormalization) {
                        Text("Peak").tag("peak")
                        Text("None").tag("none")
                    }
                }
                AdaptiveControlRow {
                    NumberField(title: "Peak dBFS", value: $controller.draft.musicTargetPeakDB)
                    NumberField(title: "Fade in ms", value: $controller.draft.musicFadeInMS)
                    NumberField(title: "Fade out ms", value: $controller.draft.musicFadeOutMS)
                }
                AdaptiveControlRow {
                    Toggle("Disable dither", isOn: $controller.draft.musicNoDither)
                    Toggle("Disable recipe", isOn: $controller.draft.musicNoRecipe)
                }
                PathField(
                    path: $controller.draft.musicLRCOutput,
                    placeholder: "Synchronized LRC output",
                    mode: .saveFile
                )
                PathField(
                    path: $controller.draft.musicRecipeOutput,
                    placeholder: "Recipe JSON output",
                    mode: .saveFile
                )
                TextField("DAW bundle directory", text: $controller.draft.musicDAWBundle)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                TextField("Stems (Drums,Bass,Vocals,…)", text: $controller.draft.musicStems)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
            }
        }
    }
}

private struct MusicMagentaOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        VStack(spacing: 10) {
            Picker("Style conditioning", selection: $controller.draft.musicStyleConditioning) {
                Text("Streaming").tag("streaming")
                Text("Full").tag("full")
            }
            .pickerStyle(.segmented)
            AdaptiveControlRow {
                NumberField(title: "Temperature", value: $controller.draft.musicTemperature)
                NumberStepper(title: "Top-k", value: $controller.draft.musicTopK, range: 0...2_048, step: 1)
                NumberField(title: "MusicCoCa CFG", value: $controller.draft.musicCFGMusicCoCa)
                NumberField(title: "Notes CFG", value: $controller.draft.musicCFGNotes)
                NumberField(title: "Drums CFG", value: $controller.draft.musicCFGDrums)
            }
            AdaptiveControlRow {
                Toggle("Drumless", isOn: $controller.draft.musicDrumless)
                Toggle("Prefill silence", isOn: $controller.draft.musicPrefillSilence)
                NumberStepper(
                    title: "Unmask width",
                    value: $controller.draft.musicUnmaskWidth,
                    range: 0...8_192,
                    step: 1
                )
                NumberStepper(
                    title: "Seed rotation",
                    value: $controller.draft.musicSeedRotation,
                    range: 0...8_192,
                    step: 1
                )
                NumberField(title: "Prefill seconds", value: $controller.draft.musicPrefillDuration)
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
                TextField("Renoise amount or schedule (optional)", text: $controller.draft.sfxRenoise)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                if controller.selectedTemplate.id == .sfxVideo {
                    TextField(
                        "Synchformer model id or path",
                        text: $controller.draft.sfxSynchformerModel
                    )
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                    AdaptiveControlRow {
                        NumberStepper(
                            title: "Sync batch",
                            value: $controller.draft.sfxSyncBatchSize,
                            range: 1...256,
                            step: 1
                        )
                        NumberStepper(
                            title: "CLIP batch",
                            value: $controller.draft.sfxClipBatchSize,
                            range: 1...256,
                            step: 1
                        )
                    }
                    AdaptiveControlRow {
                        Toggle("Preflight", isOn: $controller.draft.preflight)
                        Toggle("JSON", isOn: $controller.draft.json)
                            .disabled(!controller.draft.preflight)
                    }
                }
                Toggle("Quiet", isOn: $controller.draft.quiet)
            }
        }
    }
}

private struct AudioEnhancementOptions: View {
    @EnvironmentObject private var controller: MereRunController

    private var isUniverSR: Bool {
        controller.draft.model.localizedCaseInsensitiveContains("universr")
    }

    var body: some View {
        EditorSection("Audio enhancement") {
            VStack(spacing: 10) {
                Picker("Workflow", selection: $controller.draft.model) {
                    Text("Speech · AP-BWE").tag("audio-enhance-ap-bwe-16kto48k")
                    Text("General audio · UniverSR").tag("audio-enhance-universr-audio")
                }
                .pickerStyle(.segmented)
                PathField(
                    path: $controller.draft.modelRoot,
                    placeholder: "Explicit model directory (optional)",
                    mode: .openDirectory
                )
                Picker(
                    "Compute",
                    selection: Binding(
                        get: { controller.draft.audioDType ?? "float32" },
                        set: { controller.draft.audioDType = $0 }
                    )
                ) {
                    Text("Float 16").tag("float16")
                    Text("Float 32").tag("float32")
                }
                .pickerStyle(.segmented)
                if isUniverSR {
                    Picker(
                        "Input bandwidth",
                        selection: Binding(
                            get: { controller.draft.audioInputRate ?? 0 },
                            set: { controller.draft.audioInputRate = $0 == 0 ? nil : $0 }
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
                            get: { controller.draft.audioODEMethod ?? "midpoint" },
                            set: { controller.draft.audioODEMethod = $0 }
                        )
                    ) {
                        Text("Euler").tag("euler")
                        Text("Midpoint").tag("midpoint")
                        Text("RK4").tag("rk4")
                    }
                    .pickerStyle(.segmented)
                    AdaptiveControlRow {
                        NumberStepper(
                            title: "ODE steps",
                            value: Binding(
                                get: { controller.draft.audioODESteps ?? 4 },
                                set: { controller.draft.audioODESteps = $0 }
                            ),
                            range: 1...100,
                            step: 1
                        )
                        NumberStepper(
                            title: "Chunk seconds",
                            value: Binding(
                                get: { controller.draft.audioChunkSeconds ?? 10 },
                                set: { controller.draft.audioChunkSeconds = $0 }
                            ),
                            range: 3...600,
                            step: 1
                        )
                        NumberField(
                            title: "Guidance",
                            value: Binding(
                                get: { controller.draft.audioGuidanceScale ?? 1.5 },
                                set: { controller.draft.audioGuidanceScale = $0 }
                            )
                        )
                    }
                    TextField("Seed", text: $controller.draft.seed)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                } else {
                    NumberStepper(
                        title: "Chunk overlap",
                        value: Binding(
                            get: { controller.draft.audioOverlap ?? 2 },
                            set: { controller.draft.audioOverlap = $0 }
                        ),
                        range: 1...64,
                        step: 1
                    )
                }
                Toggle("Quiet", isOn: $controller.draft.quiet)
            }
        }
    }
}

private struct MusicSeparationOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Separation and restoration") {
            VStack(spacing: 10) {
                Picker("Workflow", selection: $controller.draft.model) {
                    Text("Vocals + instrumental").tag("music-separate-bs-roformer-viperx-1297")
                    Text("Four stems").tag("music-separate-bs-roformer-4stem")
                    Text("Dereverb").tag("music-separate-mel-roformer-dereverb")
                    Text("Denoise").tag("music-separate-mel-roformer-denoise")
                }
                PathField(
                    path: $controller.draft.modelRoot,
                    placeholder: "Explicit model directory (optional)",
                    mode: .openDirectory
                )
                AdaptiveControlRow {
                    NumberStepper(
                        title: "Chunk overlap",
                        value: Binding(
                            get: { controller.draft.audioOverlap ?? 2 },
                            set: { controller.draft.audioOverlap = $0 }
                        ),
                        range: 1...64,
                        step: 1
                    )
                    Picker(
                        "Compute",
                        selection: Binding(
                            get: { controller.draft.audioDType ?? "float16" },
                            set: { controller.draft.audioDType = $0 }
                        )
                    ) {
                        Text("Float 16").tag("float16")
                        Text("Float 32").tag("float32")
                    }
                }
                Toggle("Quiet", isOn: $controller.draft.quiet)
            }
        }
    }
}

private struct ModelOptimizeOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Optimization") {
            VStack(alignment: .leading, spacing: 10) {
                Text("Build the MiniMax-H3 inference-only AdaLN cache. The original model remains intact.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                AdaptiveControlRow {
                    Toggle("Replace compatible cache", isOn: $controller.draft.force)
                    Toggle("JSON result", isOn: $controller.draft.json)
                }
            }
        }
    }
}

private struct VideoOptions: View {
    @EnvironmentObject private var controller: MereRunController

    private var family: StudioVideoModelFamily {
        StudioVideoModelFamily(
            model: controller.draft.modelRoot.isBlank
                ? controller.draft.model
                : controller.draft.modelRoot
        )
    }

    var body: some View {
        EditorSection("Video") {
            VStack(spacing: 10) {
                if family.isMiniMaxH3 {
                    miniMaxH3Options
                } else {
                    ltxAndWanOptions
                }
                Toggle("Use duration instead of frame count", isOn: $controller.draft.useDuration)
                if controller.draft.useDuration {
                    NumberField(title: "Duration", value: $controller.draft.durationSeconds)
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
                    if !family.isMiniMaxH3 {
                        Toggle("Timings", isOn: $controller.draft.timings)
                    }
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
                if !family.isMiniMaxH3, controller.draft.timings {
                    PathField(
                        path: $controller.draft.timingsOutputPath,
                        placeholder: "Timing report JSON (optional)",
                        mode: .saveFile
                    )
                }
            }
        }
        .onAppear(perform: normalizeMiniMaxH3Geometry)
        .onChange(of: controller.draft.model) { _, _ in normalizeMiniMaxH3Geometry() }
        .onChange(of: controller.draft.modelRoot) { _, _ in normalizeMiniMaxH3Geometry() }
    }

    @ViewBuilder
    private var ltxAndWanOptions: some View {
        if family == .ltx {
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
        }
        PathField(
            path: $controller.draft.endImagePath,
            placeholder: "End keyframe (optional)",
            mode: .openFile([.image])
        )
        AdaptiveControlRow {
            NumberStepper(title: "Frames", value: $controller.draft.numFrames, range: 5...601, step: family == .wan ? 4 : 8)
            NumberStepper(title: "FPS", value: $controller.draft.fps, range: 1...60, step: 1)
            NumberField(title: "Image strength", value: $controller.draft.strength)
        }
        if family == .wan {
            AdaptiveControlRow {
                NumberStepper(title: "Steps", value: $controller.draft.steps, range: 1...100, step: 1)
                NumberField(title: "Guidance", value: $controller.draft.cfgScale)
                NumberField(title: "Shift", value: $controller.draft.scheduleShift)
            }
        }
        if family == .ltx, !controller.draft.audioPath.isBlank {
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
    }

    @ViewBuilder
    private var miniMaxH3Options: some View {
        Text("MiniMax-H3 renders synchronized 24 fps video and 32 kHz stereo audio in one pass.")
            .font(MereRunTheme.captionFont)
            .foregroundStyle(MereRunTheme.textMuted)
        Picker(
            "Transformer weights",
            selection: Binding(
                get: { controller.draft.h3WeightMode ?? "auto" },
                set: { controller.draft.h3WeightMode = $0 }
            )
        ) {
            Text("Automatic").tag("auto")
            Text("Quantized").tag("quantized")
            Text("Resident BF16").tag("resident-bf16")
        }
        .pickerStyle(.segmented)
        Toggle(
            "Override adaptive 9 / 16 / 31-point schedule",
            isOn: Binding(
                get: { controller.draft.h3Steps != nil },
                set: { controller.draft.h3Steps = $0 ? 31 : nil }
            )
        )
        if controller.draft.h3Steps != nil {
            NumberStepper(
                title: "Schedule points",
                value: Binding(
                    get: { controller.draft.h3Steps ?? 31 },
                    set: { controller.draft.h3Steps = $0 }
                ),
                range: 1...64,
                step: 1
            )
        }
        AdaptiveControlRow {
            NumberStepper(
                title: "Frames (17n+5)",
                value: $controller.draft.numFrames,
                range: 22...600,
                step: 17
            )
            Text("24 fps fixed")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            NumberField(title: "Image strength", value: $controller.draft.strength)
        }
        if family == .miniMaxH3Ref2VA {
            MiniMaxH3ReferencesEditor(references: $controller.draft.h3ReferenceInputs)
        } else {
            PathField(
                path: $controller.draft.endImagePath,
                placeholder: "Optional final keyframe",
                mode: .openFile([.image])
            )
        }
    }

    private func normalizeMiniMaxH3Geometry() {
        guard family.isMiniMaxH3 else { return }
        controller.draft.fps = 24
        controller.draft.width = max(32, (controller.draft.width / 32) * 32)
        controller.draft.height = max(32, (controller.draft.height / 32) * 32)
        controller.draft.numFrames = StudioVideoModelFamily.alignedMiniMaxH3FrameCount(
            controller.draft.numFrames
        )
        controller.draft.audioPath = ""
        controller.draft.timings = false
        controller.draft.timingsOutputPath = ""
        if family == .miniMaxH3Ref2VA {
            controller.draft.inputPath = ""
            controller.draft.endImagePath = ""
        } else {
            controller.draft.h3ReferenceInputs = []
        }
    }
}

private struct MiniMaxH3ReferencesEditor: View {
    @Binding var references: [String]?

    private var values: [String] { references ?? [] }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Ordered Ref2VA conditioning")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            ForEach(Array(values.enumerated()), id: \.offset) { index, value in
                HStack(spacing: 6) {
                    Text(referenceLabel(value))
                        .font(MereRunTheme.captionFont)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button { move(index, by: -1) } label: { Image(systemName: "arrow.up") }
                        .disabled(index == 0)
                    Button { move(index, by: 1) } label: { Image(systemName: "arrow.down") }
                        .disabled(index == values.count - 1)
                    Button(role: .destructive) { remove(index) } label: { Image(systemName: "trash") }
                }
                .padding(8)
                .merePanel()
            }
            HStack {
                Button("Add image") { choose(kind: "image", types: [.image]) }
                Button("Add video") { choose(kind: "video", types: [.movie, .video]) }
                Button("Add audio") { choose(kind: "audio", types: [.audio]) }
            }
            .buttonStyle(.bordered)
            Text("Order is semantic and is preserved exactly in repeated --reference arguments.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    private func choose(kind: String, types: [UTType]) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        references = values + ["\(kind):\(url.path)"]
    }

    private func move(_ index: Int, by offset: Int) {
        var next = values
        next.swapAt(index, index + offset)
        references = next
    }

    private func remove(_ index: Int) {
        var next = values
        next.remove(at: index)
        references = next
    }

    private func referenceLabel(_ specification: String) -> String {
        guard let split = specification.firstIndex(of: ":") else { return specification }
        let kind = String(specification[..<split]).capitalized
        let path = String(specification[specification.index(after: split)...])
        return "\(kind) · \(URL(fileURLWithPath: path).lastPathComponent)"
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

private struct ExternalProductOptions: View {
    let template: CommandTemplate
    let url: URL

    var body: some View {
        EditorSection("Product boundary") {
            VStack(alignment: .leading, spacing: 12) {
                Label(message, systemImage: template.systemImage)
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                Link(destination: url) {
                    Label(url.host ?? url.absoluteString, systemImage: "arrow.up.right.square")
                }
                .font(MereRunTheme.bodyFont)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .merePanel()
        }
    }

    private var message: String {
        switch template.id {
        case .graphStudio:
            return "Graph Studio owns visual Graph v2 authoring, preflight, submission, and project files. It executes the same catalog and immutable bundles as this CLI."
        case .nodeConsole:
            return "Node and Relay own device pairing, fleet eligibility, scheduling policy, leases, and remote execution. Durable run controls remain available here."
        default:
            return template.subtitle
        }
    }
}

private struct AdapterOperationsOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Adapter catalog") {
            VStack(alignment: .leading, spacing: 10) {
                if controller.selectedTemplate.id == .adapterList {
                    Toggle("Machine-readable catalog with install paths", isOn: $controller.draft.json)
                    Text("The result includes title, summary, compatible base model, format, license, immutable revision, size, checksum, and install state.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                } else {
                    Toggle("Replace an existing verified install", isOn: $controller.draft.force)
                    Toggle("Suppress download progress", isOn: $controller.draft.quiet)
                }
            }
        }
    }
}

private struct DurableRunOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Durable run") {
            VStack(alignment: .leading, spacing: 10) {
                switch controller.selectedTemplate.id {
                case .runList:
                    PathField(
                        path: $controller.draft.operationsRoot,
                        placeholder: "Local root to scan",
                        mode: .openDirectory
                    )
                    TextField("Or executor, e.g. relay:fleet", text: $controller.draft.operationsExecutor)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    if controller.draft.operationsExecutor.isBlank {
                        NumberStepper(
                            title: "Scan depth",
                            value: $controller.draft.maxDepth,
                            range: 0...32,
                            step: 1
                        )
                    } else {
                        NumberStepper(
                            title: "Remote limit",
                            value: $controller.draft.operationsLimit,
                            range: 1...500,
                            step: 1
                        )
                    }
                    Toggle("Structured JSON", isOn: $controller.draft.json)
                case .runInspect:
                    runReferenceField("Run directory, report, plan, ssh://, or relay://")
                    Toggle("Structured JSON", isOn: $controller.draft.json)
                case .runWatch:
                    runReferenceField("ssh://profile/job or relay://profile/job")
                    NumberField(
                        title: "Poll seconds",
                        value: $controller.draft.operationsPollInterval
                    )
                    Toggle("Stream NDJSON events", isOn: $controller.draft.operationsJSONStream)
                        .onChange(of: controller.draft.operationsJSONStream) { _, enabled in
                            if enabled { controller.draft.json = false }
                        }
                    Toggle("Emit final job JSON", isOn: $controller.draft.json)
                        .onChange(of: controller.draft.json) { _, enabled in
                            if enabled { controller.draft.operationsJSONStream = false }
                        }
                case .runFetch:
                    runReferenceField("ssh://profile/job or relay://profile/job")
                    Toggle("Fetch every intermediate artifact", isOn: $controller.draft.operationsAllArtifacts)
                    if !controller.draft.operationsAllArtifacts {
                        TextEditor(text: $controller.draft.operationsArtifacts)
                            .font(MereRunTheme.monoFont)
                            .scrollContentBackground(.hidden)
                            .frame(minHeight: 72)
                            .padding(8)
                            .merePanel()
                            .overlay(alignment: .topLeading) {
                                if controller.draft.operationsArtifacts.isBlank {
                                    Text("Named artifacts, one per line (optional)")
                                        .font(MereRunTheme.bodyFont)
                                        .foregroundStyle(MereRunTheme.textMuted)
                                        .padding(18)
                                        .allowsHitTesting(false)
                                }
                            }
                    }
                    Toggle("Structured JSON receipt", isOn: $controller.draft.json)
                case .runCancel:
                    runReferenceField("Local run directory, ssh://, or relay://")
                    Toggle("Structured JSON receipt", isOn: $controller.draft.json)
                case .runRetry:
                    runReferenceField("relay://profile/job")
                    Toggle("Structured JSON receipt", isOn: $controller.draft.json)
                default:
                    EmptyView()
                }
            }
        }
    }

    private func runReferenceField(_ placeholder: String) -> some View {
        TextField(placeholder, text: $controller.draft.operationsReference)
            .textFieldStyle(.plain)
            .font(MereRunTheme.monoFont)
            .padding(10)
            .merePanel()
    }
}

private struct WorldServeOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        Group {
            EditorSection("Diorama") {
                VStack(alignment: .leading, spacing: 10) {
                    Text(
                        "Diorama is the first-class Worlds app. mere.run Studio configures the local DreamX or Cosmos3 runtime endpoint; Diorama owns world projects, exploration, saved routes, and review."
                    )
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                    Link(destination: StudioProductBoundary.dioramaURL) {
                        Label("Open Diorama", systemImage: "arrow.up.right.square")
                    }
                }
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .merePanel()
            }
            EditorSection("World runtime") {
                VStack(spacing: 10) {
                    AdaptiveControlRow {
                        TextField("Host", text: $controller.draft.host)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                        NumberStepper(
                            title: "Port",
                            value: $controller.draft.port,
                            range: 1...65_535,
                            step: 1
                        )
                    }
                    Picker("Backend", selection: $controller.draft.operationsWorldBackend) {
                        Text("DreamX").tag("dreamx")
                        Text("Cosmos3").tag("cosmos3")
                    }
                    .pickerStyle(.segmented)
                    TextField("Wan TI2V base model id or path", text: $controller.draft.operationsBaseModel)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    PathField(
                        path: $controller.draft.operationsStateDirectory,
                        placeholder: "State directory (optional)",
                        mode: .openDirectory
                    )
                    Toggle("Warm every model before accepting requests", isOn: $controller.draft.operationsPrepare)
                }
            }
            EditorSection("Authentication") {
                SecureField("API key (required off loopback)", text: $controller.draft.apiKey)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
            }
        }
    }
}

private struct StatusSnapshotOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Probe") {
            VStack(spacing: 10) {
                AdaptiveControlRow {
                    TextField("Host", text: $controller.draft.host)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    NumberStepper(
                        title: "Port",
                        value: $controller.draft.port,
                        range: 1...65_535,
                        step: 1
                    )
                }
                NumberField(
                    title: "Timeout seconds",
                    value: $controller.draft.operationsTimeoutSeconds
                )
                SecureField("API key (optional)", text: $controller.draft.apiKey)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                Toggle("Structured JSON", isOn: $controller.draft.json)
            }
        }
    }
}

private struct QualityGateOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Installed-model gate") {
            VStack(alignment: .leading, spacing: 10) {
                TextField(
                    "Suites: all or text,speech,vision,image,embed",
                    text: $controller.draft.operationsGateSuite
                )
                .textFieldStyle(.plain)
                .padding(10)
                .merePanel()
                Toggle("List checks without running", isOn: $controller.draft.operationsListOnly)
                Toggle("Record new local baselines", isOn: $controller.draft.operationsUpdateBaselines)
                Toggle(
                    "Fail performance regressions",
                    isOn: $controller.draft.operationsStrictPerformance
                )
                Text("Baseline updates intentionally change machine-local quality evidence.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
    }
}

private struct ModelStorageOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Storage") {
            VStack(alignment: .leading, spacing: 10) {
                Toggle("Structured JSON", isOn: $controller.draft.json)
                if controller.selectedTemplate.id == .modelGarbageCollect {
                    Toggle("Delete the freshly recomputed safe plan", isOn: $controller.draft.force)
                        .tint(MereRunTheme.red)
                    Text(
                        controller.draft.force
                            ? "Execution deletes only unreferenced payloads and partial downloads in a freshly computed plan."
                            : "Dry run only. No files will be removed."
                    )
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(controller.draft.force ? MereRunTheme.yellow : MereRunTheme.textMuted)
                }
            }
        }
    }
}

private struct ModelRuntimePolicyOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        if controller.selectedTemplate.id == .modelRuntimeGet {
            EditorSection("Output") {
                Toggle("Structured JSON", isOn: $controller.draft.json)
            }
        } else {
            Group {
                EditorSection("Residency") {
                    VStack(spacing: 10) {
                        TextField("Alias (leave empty to preserve)", text: $controller.draft.operationsRuntimeAlias)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                        Toggle("Clear alias", isOn: $controller.draft.operationsClearAlias)
                        Picker("Pin policy", selection: pinPolicy) {
                            Text("Preserve").tag("preserve")
                            Text("Pinned").tag("pinned")
                            Text("Unpinned").tag("unpinned")
                        }
                        TextField("TTL seconds", text: $controller.draft.operationsRuntimeTTL)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                        Toggle("Clear TTL", isOn: $controller.draft.operationsClearTTL)
                    }
                }
                EditorSection("Generation defaults") {
                    VStack(spacing: 10) {
                        optionalPolicyField(
                            "Max context tokens",
                            value: $controller.draft.operationsRuntimeContext,
                            clear: $controller.draft.operationsClearContext
                        )
                        optionalPolicyField(
                            "Max output tokens",
                            value: $controller.draft.operationsRuntimeMaxTokens,
                            clear: $controller.draft.operationsClearMaxTokens
                        )
                        optionalPolicyField(
                            "Temperature",
                            value: $controller.draft.operationsRuntimeTemperature,
                            clear: $controller.draft.operationsClearTemperature
                        )
                        optionalPolicyField(
                            "Top-p",
                            value: $controller.draft.operationsRuntimeTopP,
                            clear: $controller.draft.operationsClearTopP
                        )
                        optionalPolicyField(
                            "Min-p",
                            value: $controller.draft.operationsRuntimeMinP,
                            clear: $controller.draft.operationsClearMinP
                        )
                    }
                }
                EditorSection("Engine & KV") {
                    VStack(spacing: 10) {
                        TextField(
                            "Engine override (optional)",
                            text: $controller.draft.operationsRuntimeEngine
                        )
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                        Toggle("Clear engine override", isOn: $controller.draft.operationsClearEngine)
                        Picker("KV cache", selection: $controller.draft.operationsRuntimeKVCacheMode) {
                            Text("Preserve").tag("")
                            ForEach(["default", "affine4", "affine8", "polar2", "auto"], id: \.self) {
                                Text($0).tag($0)
                            }
                        }
                        Toggle(
                            "Clear KV cache override",
                            isOn: $controller.draft.operationsClearKVCacheMode
                        )
                        Toggle("Structured JSON receipt", isOn: $controller.draft.json)
                    }
                }
            }
        }
    }

    private var pinPolicy: Binding<String> {
        Binding(
            get: {
                if controller.draft.operationsPinned { return "pinned" }
                if controller.draft.operationsUnpinned { return "unpinned" }
                return "preserve"
            },
            set: { value in
                controller.draft.operationsPinned = value == "pinned"
                controller.draft.operationsUnpinned = value == "unpinned"
            }
        )
    }

    private func optionalPolicyField(
        _ title: String,
        value: Binding<String>,
        clear: Binding<Bool>
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            TextField(title, text: value)
                .textFieldStyle(.plain)
                .padding(10)
                .merePanel()
            Toggle("Clear \(title.lowercased())", isOn: clear)
        }
    }
}

private struct ModelBenchmarkOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        VStack(spacing: 14) {
            EditorSection("Prompt matrix") {
                VStack(spacing: 10) {
                    TextEditor(text: $controller.draft.prompt)
                        .font(MereRunTheme.bodyFont)
                        .scrollContentBackground(.hidden)
                        .frame(minHeight: 72)
                        .padding(8)
                        .merePanel()
                    PathField(
                        path: $controller.draft.modelRoot,
                        placeholder: "Model root override (optional)",
                        mode: .openDirectory
                    )
                    PathField(
                        path: $controller.draft.benchmarkPromptFile,
                        placeholder: "Prompt file (optional)",
                        mode: .openFile([.plainText])
                    )
                    AdaptiveControlRow {
                        NumberStepper(
                            title: "Prompt repeats",
                            value: $controller.draft.benchmarkPromptRepeat,
                            range: 1...100_000,
                            step: 1
                        )
                        TextField(
                            "Repeat matrix (comma-separated)",
                            text: $controller.draft.benchmarkPromptRepeatValues
                        )
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    }
                }
            }
            EditorSection("Decode matrix") {
                VStack(spacing: 10) {
                    AdaptiveControlRow {
                        NumberStepper(
                            title: "Decode tokens",
                            value: $controller.draft.benchmarkDecodeTokens,
                            range: 1...1_048_576,
                            step: 1
                        )
                        TextField(
                            "Decode matrix (comma-separated)",
                            text: $controller.draft.benchmarkDecodeTokenValues
                        )
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    }
                    AdaptiveControlRow {
                        NumberField(title: "Temperature", value: $controller.draft.temperature)
                        TextField(
                            "Temperature matrix (comma-separated)",
                            text: $controller.draft.benchmarkTemperatureValues
                        )
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                        NumberField(title: "Top-p", value: $controller.draft.topP)
                    }
                    AdaptiveControlRow {
                        NumberStepper(
                            title: "Context",
                            value: $controller.draft.contextSize,
                            range: 1_024...1_048_576,
                            step: 1_024
                        )
                        TextField(
                            "MTP block size override",
                            text: $controller.draft.benchmarkMTPBlockSize
                        )
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                        NumberStepper(
                            title: "Forced MTP min prompt",
                            value: $controller.draft.benchmarkForcedMTPMinPromptTokens,
                            range: 1...1_048_576,
                            step: 1
                        )
                    }
                    Toggle("Structured JSON", isOn: $controller.draft.json)
                }
            }
        }
    }
}

private struct LagunaDFlashBenchmarkOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        VStack(spacing: 14) {
            EditorSection("Checkpoints") {
                VStack(spacing: 10) {
                    PathField(
                        path: $controller.draft.modelRoot,
                        placeholder: "Laguna S 2.1 checkpoint (required)",
                        mode: .openDirectory
                    )
                    PathField(
                        path: $controller.draft.secondaryText,
                        placeholder: "Laguna DFlash checkpoint (required)",
                        mode: .openDirectory
                    )
                }
            }
            EditorSection("Workload") {
                VStack(spacing: 10) {
                    TextEditor(text: $controller.draft.prompt)
                        .font(MereRunTheme.bodyFont)
                        .scrollContentBackground(.hidden)
                        .frame(minHeight: 72)
                        .padding(8)
                        .merePanel()
                    PathField(
                        path: $controller.draft.benchmarkPromptFile,
                        placeholder: "Prompt file (optional)",
                        mode: .openFile([.plainText])
                    )
                    Picker("Fixture", selection: $controller.draft.benchmarkFixture) {
                        Text("Deterministic prose").tag("deterministic-prose")
                        Text("Grounded email").tag("grounded-email")
                        Text("Code completion").tag("code-completion")
                    }
                    .pickerStyle(.segmented)
                    AdaptiveControlRow {
                        TextField(
                            "Decode lengths (comma-separated)",
                            text: $controller.draft.benchmarkDecodeTokenValues
                        )
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                        TextField(
                            "Concurrency levels (optional)",
                            text: $controller.draft.benchmarkConcurrencyValues
                        )
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    }
                }
            }
            EditorSection("Decode") {
                VStack(spacing: 10) {
                    AdaptiveControlRow {
                        NumberStepper(
                            title: "Repetitions",
                            value: $controller.draft.benchmarkRepetitions,
                            range: 1...1_000,
                            step: 1
                        )
                        NumberStepper(
                            title: "DFlash tokens",
                            value: $controller.draft.benchmarkLagunaDFlashTokens,
                            range: 1...15,
                            step: 1
                        )
                        NumberStepper(
                            title: "Warmups",
                            value: $controller.draft.benchmarkWarmupRepetitions,
                            range: 0...100,
                            step: 1
                        )
                    }
                    AdaptiveControlRow {
                        NumberField(title: "Temperature", value: $controller.draft.temperature)
                        NumberField(title: "Top-p", value: $controller.draft.topP)
                        NumberStepper(
                            title: "Top-k",
                            value: $controller.draft.topK,
                            range: 0...512,
                            step: 1
                        )
                        NumberField(title: "Min-p", value: $controller.draft.minP)
                    }
                    NumberStepper(
                        title: "Context",
                        value: $controller.draft.contextSize,
                        range: 1_024...1_048_576,
                        step: 1_024
                    )
                    AdaptiveControlRow {
                        Toggle("Mixed fixtures", isOn: $controller.draft.benchmarkMixedFixtures)
                        Toggle("Adaptive policy", isOn: $controller.draft.benchmarkIncludeAutomatic)
                        Toggle("Log responses", isOn: $controller.draft.benchmarkLogResponses)
                        Toggle("Structured JSON", isOn: $controller.draft.json)
                    }
                }
            }
        }
    }
}

private struct PluginOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Plugin catalog") {
            VStack(spacing: 10) {
                TextField("Catalog URL or local JSON (optional)", text: $controller.draft.pluginCatalogURL)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                if controller.selectedTemplate.id == .pluginInstall {
                    TextField("Install channel (optional)", text: $controller.draft.pluginChannel)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    AdaptiveControlRow {
                        Toggle("Execute install", isOn: $controller.draft.all)
                        Toggle("Force pipx install", isOn: $controller.draft.force)
                    }
                } else if controller.selectedTemplate.id == .pluginList {
                    Toggle("Structured JSON", isOn: $controller.draft.json)
                }
            }
        }
    }
}

private struct OpenWebUIOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        VStack(spacing: 14) {
            EditorSection("Services") {
                VStack(spacing: 10) {
                    Picker("API engine", selection: $controller.draft.engine) {
                        Text("Automatic").tag("")
                        Text("Gemma4").tag("text-chat-gemma4")
                        Text("Qwen 3.6").tag("text-chat-q36")
                        Text("Klein").tag("text-chat-klein")
                        Text("Laguna S 2.1").tag("text-chat-laguna")
                        Text("LFM2").tag("text-chat-lfm2")
                        Text("DeepSeek V4").tag("text-chat-deepseek-v4-flash")
                    }
                    AdaptiveControlRow {
                        TextField("API host", text: $controller.draft.host)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                        NumberStepper(
                            title: "API port",
                            value: $controller.draft.port,
                            range: 1...65_535,
                            step: 1
                        )
                    }
                    AdaptiveControlRow {
                        TextField("WebUI host", text: $controller.draft.openWebUIHost)
                            .textFieldStyle(.plain)
                            .padding(10)
                            .merePanel()
                        NumberStepper(
                            title: "WebUI port",
                            value: $controller.draft.openWebUIPort,
                            range: 1...65_535,
                            step: 1
                        )
                    }
                    SecureField("API key", text: $controller.draft.apiKey)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                }
            }
            EditorSection("Model suite") {
                VStack(spacing: 10) {
                    TextField("Vision model", text: $controller.draft.openWebUIVisionModel)
                    TextField("Embedding model", text: $controller.draft.openWebUIEmbeddingModel)
                    TextField("Image model", text: $controller.draft.openWebUIImageModel)
                    TextField("TTS model", text: $controller.draft.openWebUITTSModel)
                    TextField("STT model", text: $controller.draft.openWebUISTTModel)
                    Picker("TTS format", selection: $controller.draft.openWebUITTSFormat) {
                        ForEach(["wav", "mp3", "flac", "opus", "aac", "pcm"], id: \.self) {
                            Text($0.uppercased()).tag($0)
                        }
                    }
                }
                .textFieldStyle(.plain)
                .padding(10)
                .merePanel()
            }
            EditorSection("Container") {
                VStack(spacing: 10) {
                    TextField("Container name", text: $controller.draft.openWebUIContainerName)
                    TextField("Volume name", text: $controller.draft.openWebUIVolumeName)
                    TextField("Docker image", text: $controller.draft.openWebUIImage)
                    TextField("Admin email", text: $controller.draft.openWebUIAdminEmail)
                    SecureField("Admin password", text: $controller.draft.openWebUIAdminPassword)
                }
                .textFieldStyle(.plain)
                .padding(10)
                .merePanel()
                NumberStepper(
                    title: "Health wait seconds",
                    value: $controller.draft.openWebUIWaitSeconds,
                    range: 1...3_600,
                    step: 5
                )
            }
            EditorSection("Launch policy") {
                VStack(alignment: .leading, spacing: 10) {
                    AdaptiveControlRow {
                        Toggle("Pull models", isOn: $controller.draft.openWebUIPull)
                        Toggle("Skip API server", isOn: $controller.draft.openWebUISkipServer)
                        Toggle("Skip Docker", isOn: $controller.draft.openWebUISkipDocker)
                        Toggle("Skip configure", isOn: $controller.draft.openWebUISkipConfigure)
                    }
                    AdaptiveControlRow {
                        Toggle("Reset container & volume", isOn: $controller.draft.openWebUIReset)
                        Toggle("Dry run", isOn: $controller.draft.dryRun)
                        Toggle("Quiet", isOn: $controller.draft.quiet)
                    }
                    Toggle(
                        "Accept third-party model terms",
                        isOn: $controller.draft.acceptModelLicense
                    )
                    .toggleStyle(.checkbox)
                }
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
                    Text("Laguna S 2.1").tag("text-chat-laguna")
                    Text("LFM2").tag("text-chat-lfm2")
                    Text("DeepSeek V4").tag("text-chat-deepseek-v4-flash")
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
                TextField("Default adapter id or LoRA path", text: $controller.draft.apiLoRA)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                AdaptiveControlRow {
                    NumberStepper(
                        title: "Requests/min",
                        value: $controller.draft.apiRateLimitPerMinute,
                        range: 1...100_000,
                        step: 1
                    )
                    NumberStepper(
                        title: "Active requests",
                        value: $controller.draft.apiMaxActiveRequests,
                        range: 1...1_024,
                        step: 1
                    )
                    NumberStepper(
                        title: "Context",
                        value: $controller.draft.contextSize,
                        range: 1_024...1_048_576,
                        step: 1_024
                    )
                }
                Picker("Memory guard", selection: $controller.draft.apiMemoryGuard) {
                    ForEach(["off", "safe", "balanced", "aggressive", "custom"], id: \.self) {
                        Text($0.capitalized).tag($0)
                    }
                }
                if controller.draft.apiMemoryGuard == "custom" {
                    TextField(
                        "Custom memory ceiling (GiB)",
                        text: $controller.draft.apiMemoryGuardCustomCeilingGB
                    )
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
                }
                AdaptiveControlRow {
                    NumberStepper(title: "KV bits", value: $controller.draft.kvBits, range: 0...16, step: 1)
                    TextField("KV scheme", text: $controller.draft.kvQuantScheme)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    NumberStepper(
                        title: "KV group",
                        value: $controller.draft.kvGroupSize,
                        range: 0...4_096,
                        step: 8
                    )
                    NumberStepper(
                        title: "KV start",
                        value: $controller.draft.quantizedKVStart,
                        range: 0...1_048_576,
                        step: 128
                    )
                }
                AdaptiveControlRow {
                    Toggle("Preflight", isOn: $controller.draft.preflight)
                    Toggle("JSON", isOn: $controller.draft.json)
                        .disabled(!controller.draft.preflight)
                }
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

private struct AgentStatusOptions: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        EditorSection("Inspection") {
            PathField(
                path: $controller.draft.piPath,
                placeholder: "Pi executable override (optional)",
                mode: .openFile([.executable])
            )
            Toggle("JSON", isOn: $controller.draft.json)
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
                    Toggle("No bootstrap", isOn: $controller.draft.noBootstrap)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
                PathField(
                    path: $controller.draft.piPath,
                    placeholder: "Pi executable override (optional)",
                    mode: .openFile([.executable])
                )
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
                    Toggle("Dry run", isOn: $controller.draft.dryRun)
                    Toggle("Quiet", isOn: $controller.draft.quiet)
                }
                AdaptiveControlRow {
                    TextField("API host", text: $controller.draft.host)
                        .textFieldStyle(.plain)
                        .padding(10)
                        .merePanel()
                    NumberStepper(
                        title: "API port",
                        value: $controller.draft.port,
                        range: 1...65_535,
                        step: 1
                    )
                }
                PathField(
                    path: $controller.draft.piPath,
                    placeholder: "Pi executable override (optional)",
                    mode: .openFile([.executable])
                )
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
                Toggle("Accept third-party model terms", isOn: $controller.draft.acceptModelLicense)
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
                Toggle("Accept third-party model terms", isOn: $controller.draft.acceptModelLicense)
                    .toggleStyle(.checkbox)
                AdaptiveControlRow {
                    Toggle("Preflight", isOn: $controller.draft.preflight)
                    Toggle("JSON", isOn: $controller.draft.json)
                        .disabled(!controller.draft.preflight)
                }
                Text(
                    "Required for access-gated downloads and models with material non-commercial, research-only, "
                        + "or revenue-limited terms. Enabling this option confirms that you reviewed and accept the "
                        + "listed terms and agree to comply with them. A custom license alone does not trigger this "
                        + "requirement. Mere does not determine whether your intended use is permitted; you are "
                        + "responsible for compliance."
                )
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
                AdaptiveControlRow {
                    Toggle("Force", isOn: $controller.draft.force)
                    Toggle("Keep shared cache", isOn: $controller.draft.modelKeepCache)
                    Toggle("JSON receipt", isOn: $controller.draft.modelRemovalJSON)
                        .disabled(!controller.draft.force)
                }
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
