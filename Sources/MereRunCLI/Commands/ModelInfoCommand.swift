import ArgumentParser
import Foundation
import MereRunCore

struct ModelInfo: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Print a model's manifest, validation status, and resolved component paths."
    )

    @Argument(help: "Canonical model id (for example: text-chat-gemma4 or image-zimage-max) or local model root path.")
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
                throw ValidationError("Model \(id.rawValue) not found. Pull it with `mere.run model pull \(id.rawValue)` or point info at a local model path.")
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
                print("  defaults: steps=\(steps) cfg=\(cfg)")
            }

            if let supports = manifest.supports, !supports.isEmpty {
                print("  supports: \(supports.map(\.rawValue).joined(separator: ", "))")
            }

            if let upstream = manifest.upstreamRepoId {
                print("  upstreamRepoId: \(upstream)")
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
    }
}
