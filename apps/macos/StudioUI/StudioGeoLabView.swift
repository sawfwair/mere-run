import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// The four native geospatial workflows, each backed by a typed shared capability.
enum StudioGeoTool: String, CaseIterable, Identifiable, Codable {
    case flood
    case fire
    case tessera
    case olmoEarth

    var id: String { rawValue }

    var title: String {
        switch self {
        case .flood: "Flood"
        case .fire: "Fire"
        case .tessera: "TESSERA"
        case .olmoEarth: "OlmoEarth"
        }
    }

    var symbol: String {
        switch self {
        case .flood: "water.waves"
        case .fire: "flame"
        case .tessera: "square.stack.3d.down.right"
        case .olmoEarth: "globe.europe.africa"
        }
    }

    var templateID: CommandTemplateID {
        switch self {
        case .flood: .geoFlood
        case .fire: .geoFire
        case .tessera: .geoTessera
        case .olmoEarth: .geoOlmoEarth
        }
    }

    /// What the run writes, in the operator's words.
    var outputLabel: String {
        switch self {
        case .flood, .fire: "Logits safetensors"
        case .tessera, .olmoEarth: "Embedding safetensors"
        }
    }

    var summary: String {
        switch self {
        case .flood:
            return "TerraMind flood segmentation over a normalized S2L2A, S1RTC, and DEM tile batch."
        case .fire:
            return "TerraMind fire segmentation over a normalized S2L2A, S1RTC, and DEM tile batch."
        case .tessera:
            return "TESSERA v2 student encoder over raw Sentinel-1/2 observations and day-of-year tensors."
        case .olmoEarth:
            return "OlmoEarth v1.2 encoder over multisensor observations with a TIMESTAMPS tensor."
        }
    }

    /// The tensors the input safetensors file must carry for the run to succeed.
    var requiredTensors: [String] {
        switch self {
        case .flood, .fire: ["S2L2A", "S1RTC", "DEM"]
        case .tessera: ["S2", "S1", "DOY"]
        case .olmoEarth: ["TIMESTAMPS"]
        }
    }
}

/// The Earth domain's content: native Earth-observation inference, one form per workflow.
/// Runs stay durable Library jobs, and every produced safetensors file is previewable
/// and revealable from the result panel like any other Studio artifact.
struct StudioGeoLabView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    /// Owned by the shell's task control (Flood · Fire · TESSERA · OlmoEarth).
    @Binding var tool: StudioGeoTool
    @StudioStoredValue("GeoLab.drafts") private var drafts: [StudioGeoTool: CommandDraft] = [:]
    @StudioStoredValue("requestID") private var requestID: UUID? = nil
    @State private var statusMessage: String?

    init(tool: Binding<StudioGeoTool>) {
        _tool = tool
        var seeded: [StudioGeoTool: CommandDraft] = [:]
        for item in StudioGeoTool.allCases {
            seeded[item] = CommandCatalog.template(id: item.templateID)?.defaultDraft() ?? CommandDraft()
        }
        _drafts = StudioStoredValue(wrappedValue: seeded, "GeoLab.drafts")
    }

    private var draft: CommandDraft {
        drafts[tool] ?? CommandDraft()
    }

    private func binding<Value>(_ keyPath: WritableKeyPath<CommandDraft, Value>) -> Binding<Value> {
        Binding(
            get: { drafts[tool]?[keyPath: keyPath] ?? CommandDraft()[keyPath: keyPath] },
            set: { newValue in
                var updated = drafts[tool] ?? CommandDraft()
                updated[keyPath: keyPath] = newValue
                drafts[tool] = updated
            }
        )
    }

    var body: some View {
        StudioAnalysisLayout { controls } result: { resultPanel }
        .studioTaskCommand(tool.templateID, draft: draft)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onChange(of: tool) { _, _ in
            statusMessage = nil
        }
    }

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                Text(tool.summary)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)

                StudioPathField(
                    label: "Input safetensors",
                    placeholder: "Normalized tile batch written by your preprocessing step",
                    path: binding(\.inputPath),
                    allowedContentTypes: [.data]
                )

                requiredTensorsHint

                StudioPathField(
                    label: tool.outputLabel,
                    placeholder: "Destination .safetensors path",
                    path: binding(\.outputPath)
                )

                labeledTextField(
                    "Model",
                    placeholder: "Managed model id or converted model root (optional)",
                    text: binding(\.model)
                )

                toolControls

                Divider().overlay(MereRunTheme.border.opacity(0.5))

                Toggle("Preflight only", isOn: binding(\.preflight))
                    .help("Validate inputs and model availability without loading weights")
                Toggle("Structured JSON output", isOn: binding(\.json))

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
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(MereRunTheme.Spacing.lg)
        }
    }

    /// Names the tensors the CLI will look for, so a malformed bundle is caught by the
    /// operator before a run rather than by a ValidationError afterwards.
    private var requiredTensorsHint: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Required tensors")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            HStack(spacing: 6) {
                ForEach(tool.requiredTensors, id: \.self) { name in
                    Text(name)
                        .font(MereRunTheme.monoFont)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(MereRunTheme.surfaceRaised)
                        .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))
                }
            }
            // Read as one phrase rather than a run of unrelated mono tokens.
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(
                "Required tensors: \(tool.requiredTensors.joined(separator: ", "))"
            )
        }
    }

    @ViewBuilder
    private var toolControls: some View {
        switch tool {
        case .tessera:
            labeledTextField(
                "Output dimensions",
                placeholder: "Students: 16, 32, 64, or 128. Teacher: 1024. Blank uses the checkpoint default.",
                text: binding(\.geoDimensions)
            )
        case .olmoEarth:
            Picker("Spatial patch size", selection: binding(\.geoPatchSize)) {
                Text("1 px").tag(1)
                Text("2 px").tag(2)
                Text("4 px").tag(4)
                Text("8 px").tag(8)
            }
            .pickerStyle(.segmented)
            HStack {
                Text("Input resolution")
                    .font(MereRunTheme.captionFont)
                Spacer()
                TextField("Metres", value: binding(\.geoInputResolution), format: .number)
                    .frame(width: 100)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                Text("m GSD")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            Toggle("Also write full space-time tokens", isOn: binding(\.geoIncludeTokens))
                .help("Writes space-time tokens in addition to the time-pooled grids")
        case .flood, .fire:
            EmptyView()
        }
    }

    private var resultPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Result")
                .font(MereRunTheme.sectionFont)
            if let requestID {
                StudioSpecialistResultView(
                    requestID: requestID,
                    preferredKinds: [.image, .text]
                )
            } else {
                ContentUnavailableView(
                    "No geospatial run yet",
                    systemImage: tool.symbol,
                    description: Text(
                        "Choose an input bundle and workflow. Outputs and provenance stay in Library."
                    )
                )
            }
        }
        .padding(MereRunTheme.Spacing.lg)
    }

    private func labeledTextField(
        _ label: String,
        placeholder: String,
        text: Binding<String>
    ) -> some View {
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
            statusMessage = "The selected geospatial workflow is unavailable."
            return
        }
        let current = draft
        if current.inputPath.isBlank {
            statusMessage = "Choose an input safetensors bundle first."
            return
        }
        if current.outputPath.isBlank {
            statusMessage = "Choose where to write the \(tool.outputLabel.lowercased())."
            return
        }
        if let validation = template.validationMessage(for: current) {
            statusMessage = validation
            return
        }
        requestID = StudioSpecialistRunner.submit(
            templateID: tool.templateID,
            mode: .readImage,
            draft: current,
            controller: controller,
            library: library
        )
        statusMessage = current.preflight
            ? "\(tool.title) preflight submitted."
            : "\(tool.title) submitted."
    }
}
