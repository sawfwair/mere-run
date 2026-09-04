import AppKit
import StudioKit
import SwiftUI

/// The Command Console's third pane: the run's log, its structured receipt when the command
/// printed JSON, and the artifact it wrote. It mirrors the foreground job the same way the
/// Studio window's job bar does, so a run started here is visible wherever you are looking.
struct StudioConsoleLog: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Run")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(MereRunTheme.textPrimary)
                    Text(controller.status)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
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
                            .font(.system(size: 11, weight: .semibold))
                            .frame(width: 22, height: 22)
                    }
                    .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
                    .help("Copy run output")
                    .accessibilityLabel("Copy run output")

                    Button {
                        saveConsole()
                    } label: {
                        Image(systemName: "square.and.arrow.down")
                            .font(.system(size: 11, weight: .semibold))
                            .frame(width: 22, height: 22)
                    }
                    .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
                    .help("Save run receipt")
                    .accessibilityLabel("Save run receipt")
                }
            }
            .padding(.horizontal, 16)
            .frame(height: 52)
            .overlay(alignment: .bottom) {
                Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(height: 1)
            }

            if controller.logs.isEmpty {
                VStack(spacing: 10) {
                    Image(systemName: "terminal")
                        .font(.system(size: 28, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                    Text("Run output will appear here.")
                        .font(.system(size: 12, weight: .medium))
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

