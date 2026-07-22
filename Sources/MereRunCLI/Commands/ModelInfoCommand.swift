import ArgumentParser
import Foundation
import MereRunCore

struct ModelInfo: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Print a model's manifest, validation status, and resolved component paths."
    )

    @Argument(help: "Canonical model id (for example: text-chat-gemma4 or image-zimage-nano) or local model root path.")
    var target: String

    @Flag(name: [.long], help: "Print the raw `mererun_model.json` (if present) to stdout.")
    var json: Bool = false

    @Flag(name: [.long], help: "Resolve and print component directories (tokenizer/text_encoder/transformer/vae/scheduler).")
    var components: Bool = false

    func run() throws {
        let fm = FileManager.default

        let resolved: ModelResolver.Resolution?
        let rootURL: URL
        let expectedModelID: String?

        let asPath = URL(fileURLWithPath: target).standardizedFileURL
        if fm.fileExists(atPath: asPath.path) {
            resolved = nil
            rootURL = asPath
            expectedModelID = nil
        } else if let id = ModelResolver.ModelID(rawValue: target) {
            do {
                let r = try ModelResolver().resolve(id)
                resolved = r
                rootURL = r.rootURL
                expectedModelID = id.rawValue
            } catch {
                if id == .gemma4 {
                    throw ValidationError("Model \(id.rawValue) is not installed in the local model store. Gemma4 resolves through the native Hugging Face snapshot path on first use, or you can point info at a local model path.")
                }
                throw ValidationError("Model \(id.rawValue) not found. Pull it with `\(CLICommandDisplay.modelPullCommand(for: id.rawValue))` or point info at a local model path.")
            }
        } else {
            throw ValidationError("Not a path and not a known model id: \(target)")
        }

        let report = MereRunModelValidator.validate(modelRoot: rootURL, expectedModelID: expectedModelID)
        let manifest = report.manifest

        if json {
            guard let manifest else {
                throw ValidationError("No \(MereRunModelManifest.filename) present at: \(rootURL.path)")
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(manifest)
            if let text = String(data: data, encoding: .utf8) {
                print(text)
            } else {
                throw ValidationError("Failed to encode manifest as UTF-8")
            }
            return
        }

        print("Model Root: \(rootURL.path)")
        if let resolved {
            print("Model ID: \(resolved.modelID.rawValue)")
            print("Source: \(resolved.source.rawValue)")
        }

        let usage = FileSystemHelper.directoryUsage(at: rootURL)
        print("\nStorage")
        print("  layout: \(usage.layoutDescription)")
        print("  size: \(ByteCountFormatter.string(fromByteCount: usage.resolvedBytes, countStyle: .file))")
        if usage.localBytes != usage.resolvedBytes {
            print("  localWrapperSize: \(ByteCountFormatter.string(fromByteCount: usage.localBytes, countStyle: .file))")
        }
        let standardizedRoot = rootURL.standardizedFileURL.path
        let resolvedRoot = rootURL.resolvingSymlinksInPath().standardizedFileURL.path
        if resolvedRoot != standardizedRoot {
            print("  resolvedRoot: \(resolvedRoot)")
        }
        if usage.symlinkCount > 0 {
            print(
                "  symlinks: \(usage.symlinkCount) total, \(usage.symlinkedDirectoryCount) directories, \(usage.symlinkedFileCount) files"
            )
        }

        if let manifest {
            print("\nManifest (\(MereRunModelManifest.filename))")
            print("  schemaVersion: \(manifest.schemaVersion)")
            print("  id: \(manifest.id)")
            if let engine = manifest.engine?.rawValue { print("  engine: \(engine)") }
            if let family = manifest.family?.rawValue { print("  family: \(family)") }
            if let tier = manifest.tier?.rawValue { print("  tier: \(tier)") }
            if let variant = manifest.variant?.rawValue { print("  variant: \(variant)") }
            if let precision = manifest.precision?.rawValue { print("  precision: \(precision)") }

            if let q = manifest.quantization {
                let bits = q.bits.map(String.init) ?? "?"
                let groupSize = q.groupSize.map(String.init) ?? "?"
                print("  quantization: bits=\(bits) groupSize=\(groupSize) scheme=\(q.scheme ?? "unknown")")
                if let svdRank = q.svdResidualRank, svdRank > 0 {
                    print("    svdResidualRank: \(svdRank)")
                    if let targets = q.svdTargets, !targets.isEmpty {
                        print("    svdTargets: \(targets.joined(separator: ", "))")
                    }
                    if let maxLayers = q.svdMaxLayers, maxLayers > 0 {
                        print("    svdMaxLayers: \(maxLayers)")
                    }
                }
            }

            if let defaults = manifest.defaults {
                let steps = defaults.steps.map(String.init) ?? "?"
                let cfg = defaults.cfg.map { String(format: "%.2f", $0) } ?? "?"
                if let sigmaShift = defaults.sigmaShift {
                    print("  defaults: steps=\(steps) cfg=\(cfg) sigma_shift=\(String(format: "%.2f", sigmaShift))")
                } else {
                    print("  defaults: steps=\(steps) cfg=\(cfg)")
                }
            }

            if let supports = manifest.supports, !supports.isEmpty {
                print("  supports: \(supports.map(\.rawValue).joined(separator: ", "))")
            }

            if let upstream = manifest.upstreamRepoId {
                print("  upstreamRepoId: \(upstream)")
            }

            if let sources = manifest.sources, !sources.isEmpty {
                print("  sources:")
                for source in sources {
                    let destination = source.destinationPath.map { " -> \($0)" } ?? ""
                    print("    - \(source.role): \(source.repository)@\(source.revision)\(destination)")
                }
            }

            if let terms = manifest.usageTerms, !terms.isEmpty {
                print("  usageTermsAcknowledged: \(manifest.usageTermsAcknowledged == true)")
                print("  usageTerms:")
                for term in terms {
                    print("    - \(term.component): \(term.license)")
                    print("      source: \(term.sourceRepoId)@\(term.sourceRevision)")
                    print("      summary: \(term.summary)")
                    print("      url: \(term.licenseURL)")
                }
            }

            if let createdAt = manifest.createdAt {
                let formatter = ISO8601DateFormatter()
                print("  createdAt: \(formatter.string(from: createdAt))")
            }
        } else {
            print("\nManifest: (missing)")
        }

        print("\nValidation")
        print("  isValid: \(report.isValid)")
        if !report.warnings.isEmpty {
            print("  warnings:")
            for w in report.warnings {
                print("    - \(w)")
            }
        }
        if !report.errors.isEmpty {
            print("  errors:")
            for e in report.errors {
                print("    - \(e)")
            }
        }

        if components {
            print("\nComponents")
            if Self.usesLTX23FullLayout(manifest: manifest, expectedModelID: expectedModelID) {
                let companionRoot = ModelResolver()
                    .resolveIfPresent(.ltxGemma3TwelveB4Bit)?
                    .rootURL
                for line in Self.ltx23FullComponentLines(
                    rootURL: rootURL,
                    companionRootURL: companionRoot,
                    fileManager: fm
                ) {
                    print(line)
                }
            } else if Self.usesLTX23A2VidLayout(manifest: manifest, expectedModelID: expectedModelID) {
                let companionRoot = ModelResolver()
                    .resolveIfPresent(.ltxGemma3TwelveB4Bit)?
                    .rootURL
                for line in Self.ltx23A2VidComponentLines(
                    rootURL: rootURL,
                    companionRootURL: companionRoot,
                    fileManager: fm
                ) {
                    print(line)
                }
            } else if Self.usesLTX23SplitLayout(manifest: manifest, expectedModelID: expectedModelID) {
                let companionRoot = ModelResolver()
                    .resolveIfPresent(.ltxGemma3TwelveB4Bit)?
                    .rootURL
                for line in Self.ltx23SplitComponentLines(
                    rootURL: rootURL,
                    companionRootURL: companionRoot,
                    fileManager: fm
                ) {
                    print(line)
                }
            } else if Self.usesLTXMergedLayout(manifest: manifest, expectedModelID: expectedModelID) {
                for line in Self.ltxMergedComponentLines(rootURL: rootURL, fileManager: fm) {
                    print(line)
                }
            } else {
                printGenericComponents(rootURL: rootURL, manifest: manifest)
            }
        }
    }

    private func printGenericComponents(rootURL: URL, manifest: MereRunModelManifest?) {
        let resolver = ModelComponentResolver(modelRootURL: rootURL, manifest: manifest)
        for component in ModelComponentResolver.Component.allCases {
            let fallback = component.manifestKey
            do {
                let resolved = try resolver.resolveDirectory(for: component, fallbackLocalPath: fallback)
                let source = resolved.sourceModelRootURL.standardizedFileURL.path
                let path = resolved.directoryURL.standardizedFileURL.path
                if source == rootURL.standardizedFileURL.path {
                    print("  \(component.manifestKey): \(path)")
                } else {
                    print("  \(component.manifestKey): \(path)  (from \(source))")
                }
            } catch {
                print("  \(component.manifestKey): (unresolved) \(error.localizedDescription)")
            }
        }
    }

    static func usesLTX23SplitLayout(manifest: MereRunModelManifest?, expectedModelID: String?) -> Bool {
        let id = expectedModelID ?? manifest?.id
        guard let id else { return false }
        return ManagedModelCatalog.spec(for: id)?.validationKind == .ltxVideo23MLX
    }

    static func usesLTX23A2VidLayout(manifest: MereRunModelManifest?, expectedModelID: String?) -> Bool {
        let id = expectedModelID ?? manifest?.id
        guard let id else { return false }
        return ManagedModelCatalog.spec(for: id)?.validationKind == .ltxVideo23A2VMLX
    }

    static func usesLTX23FullLayout(manifest: MereRunModelManifest?, expectedModelID: String?) -> Bool {
        let id = expectedModelID ?? manifest?.id
        guard let id else { return false }
        return ManagedModelCatalog.spec(for: id)?.validationKind == .ltxVideo23FullMLX
    }

    static func usesLTXMergedLayout(manifest: MereRunModelManifest?, expectedModelID: String?) -> Bool {
        let id = expectedModelID ?? manifest?.id
        guard let id else { return false }
        return ManagedModelCatalog.spec(for: id)?.validationKind == .ltxVideo
    }

    static func ltxMergedComponentLines(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> [String] {
        let root = rootURL.standardizedFileURL
        let fileRoot = rootURL.resolvingSymlinksInPath().standardizedFileURL
        var lines = ["  layout: LTX distilled merged files"]
        lines.append(directoryLine(label: "tokenizer", rootURL: root, relativePath: "tokenizer", fileManager: fileManager))
        lines.append(
            directoryLine(label: "text_encoder", rootURL: root, relativePath: "text_encoder", fileManager: fileManager)
        )
        lines.append(
            matchedFileLine(
                label: "ltx_model",
                rootURL: fileRoot,
                missingDescription: "ltx-2-19*.safetensors",
                fileManager: fileManager
            ) { name in
                name.hasPrefix("ltx-2-19") && name.hasSuffix(".safetensors")
            }
        )
        lines.append(
            matchedFileLine(
                label: "spatial_upscaler",
                rootURL: fileRoot,
                missingDescription: "ltx-2-spatial-upscaler*.safetensors",
                fileManager: fileManager
            ) { name in
                name.hasPrefix("ltx-2-spatial-upscaler") && name.hasSuffix(".safetensors")
            }
        )
        return lines
    }

    static func ltx23SplitComponentLines(
        rootURL: URL,
        companionRootURL: URL?,
        fileManager: FileManager = .default
    ) -> [String] {
        var lines = ["  layout: LTX 2.3 distilled split MLX files"]
        if let companionRootURL {
            lines.append(
                "  text_encoder: \(companionRootURL.standardizedFileURL.path)  "
                    + "(companion \(ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue))"
            )
        } else {
            lines.append(
                "  text_encoder: \(ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue)  "
                    + "(companion model)"
            )
        }

        for file in ltx23SplitComponentFiles {
            let url = rootURL.appendingPathComponent(file.relativePath, isDirectory: false).standardizedFileURL
            let suffix = fileManager.fileExists(atPath: url.path) ? "" : "  (missing)"
            lines.append("  \(file.label): \(url.path)\(suffix)")
        }
        return lines
    }

    static func ltx23A2VidComponentLines(
        rootURL: URL,
        companionRootURL: URL?,
        fileManager: FileManager = .default
    ) -> [String] {
        var lines = ["  layout: LTX 2.3 A2Vid split MLX files"]
        if let companionRootURL {
            lines.append(
                "  text_encoder: \(companionRootURL.standardizedFileURL.path)  "
                    + "(companion \(ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue))"
            )
        } else {
            lines.append(
                "  text_encoder: \(ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue)  "
                    + "(companion model)"
            )
        }
        for file in ltx23A2VidComponentFiles {
            let url = rootURL.appendingPathComponent(file.relativePath, isDirectory: false).standardizedFileURL
            let suffix = fileManager.fileExists(atPath: url.path) ? "" : "  (missing)"
            lines.append("  \(file.label): \(url.path)\(suffix)")
        }
        return lines
    }

    static func ltx23FullComponentLines(
        rootURL: URL,
        companionRootURL: URL?,
        fileManager: FileManager = .default
    ) -> [String] {
        var lines = ["  layout: LTX 2.3 full split MLX files (unified AV + A2Vid)"]
        if let companionRootURL {
            lines.append(
                "  text_encoder: \(companionRootURL.standardizedFileURL.path)  "
                    + "(companion \(ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue))"
            )
        } else {
            lines.append(
                "  text_encoder: \(ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue)  "
                    + "(companion model)"
            )
        }
        for file in ltx23FullComponentFiles {
            let url = rootURL.appendingPathComponent(file.relativePath, isDirectory: false).standardizedFileURL
            let suffix = fileManager.fileExists(atPath: url.path) ? "" : "  (missing)"
            lines.append("  \(file.label): \(url.path)\(suffix)")
        }
        return lines
    }

    private static func directoryLine(
        label: String,
        rootURL: URL,
        relativePath: String,
        fileManager: FileManager
    ) -> String {
        let url = rootURL.appendingPathComponent(relativePath, isDirectory: true).standardizedFileURL
        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
        return "  \(label): \(url.path)\(exists ? "" : "  (missing)")"
    }

    private static func matchedFileLine(
        label: String,
        rootURL: URL,
        missingDescription: String,
        fileManager: FileManager,
        matches: (String) -> Bool
    ) -> String {
        let entries = (try? fileManager.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        if let url = entries.first(where: { matches($0.lastPathComponent) })?.standardizedFileURL {
            return "  \(label): \(url.path)"
        }
        return "  \(label): (missing \(missingDescription))"
    }

    private static let ltx23SplitComponentFiles: [(label: String, relativePath: String)] = [
        ("split_model", "split_model.json"),
        ("config", "config.json"),
        ("embedded_config", "embedded_config.json"),
        ("connector", "connector.safetensors"),
        ("transformer", "transformer-distilled.safetensors"),
        ("video_vae_decoder", "vae_decoder.safetensors"),
        ("video_vae_encoder", "vae_encoder.safetensors"),
        ("audio_vae", "audio_vae.safetensors"),
        ("vocoder_bwe", "vocoder.safetensors"),
        ("spatial_upscaler_x2", "spatial_upscaler_x2_v1_1.safetensors"),
        ("spatial_upscaler_x2_config", "spatial_upscaler_x2_v1_1_config.json"),
        ("spatial_upscaler_x1_5", "spatial_upscaler_x1_5_v1_0.safetensors"),
        ("spatial_upscaler_x1_5_config", "spatial_upscaler_x1_5_v1_0_config.json"),
        ("temporal_upscaler_x2", "temporal_upscaler_x2_v1_0.safetensors"),
        ("temporal_upscaler_x2_config", "temporal_upscaler_x2_v1_0_config.json"),
    ]

    private static let ltx23A2VidComponentFiles: [(label: String, relativePath: String)] = [
        ("split_model", "split_model.json"),
        ("config", "config.json"),
        ("embedded_config", "embedded_config.json"),
        ("connector", "connector.safetensors"),
        ("transformer_dev", "transformer-dev.safetensors"),
        ("distilled_lora", "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
        ("video_vae_decoder", "vae_decoder.safetensors"),
        ("video_vae_encoder", "vae_encoder.safetensors"),
        ("audio_vae", "audio_vae.safetensors"),
        ("spatial_upscaler_x2", "spatial_upscaler_x2_v1_1.safetensors"),
        ("spatial_upscaler_x2_config", "spatial_upscaler_x2_v1_1_config.json"),
    ]

    private static let ltx23FullComponentFiles = ltx23A2VidComponentFiles + [
        (label: "vocoder_bwe", relativePath: "vocoder.safetensors"),
    ]
}
