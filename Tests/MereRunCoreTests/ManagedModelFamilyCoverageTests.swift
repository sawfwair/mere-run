import Foundation
import MereRunContract
import Testing
import XCTest

@testable import MereRunCore

/// Capabilities that load managed models but do not pick one runtime family from their argv.
private let unroutedCapabilities: [String: String] = [
    "api.serve": "Serves several engines and routes per request, not per command line.",
    "agent.start": "Starts an agent session over a served model; the server owns the runtime.",
    "model.benchmark.chat": "Benchmarks a list of models, each on its own runtime.",
    "model.benchmark.code": "Benchmarks a list of models, each on its own runtime.",
    "model.benchmark.vlm": "Benchmarks a list of models, each on its own runtime."
]

/// `defaultCLICommands` entries that name no cataloged capability.
private let nonCapabilityCommands: [String: String] = [
    "vision image-to-3d": "CLI alias of `image reconstruct-3d`, exempt from the contract.",
    "vision image-to-3d-trellis2": "CLI alias of `image reconstruct-3d-trellis2`, exempt from the contract.",
    "vision image-to-3d-multiview": "CLI alias of `image reconstruct-3d-multiview`, exempt from the contract.",
]

/// A `defaultCLICommands` entry split into its command path and the flags it pins, so
/// `"video cosmos3 --mode reasoner"` reads as `video cosmos3` with `--mode reasoner`.
private struct ListedCommand {
    let path: [String]
    let flags: [String]

    init(_ entry: String) {
        let tokens = entry.split(separator: " ").map(String.init)
        path = Array(tokens.prefix { !$0.hasPrefix("-") })
        flags = Array(tokens.dropFirst(path.count))
    }

    var id: String { path.joined(separator: ".") }
}

private func resolve(
    _ model: String,
    in capability: MereRunCommandCapability,
    flags: [String] = []
) -> MereRunFamilyResolution? {
    guard let routing = capability.routing else { return nil }
    // An alias names its managed model's family, flag, and selectors, as the id itself would.
    let managed = ManagedModelCatalog.spec(for: model)?.id ?? model
    let owner = routing.families.first { $0.models.contains(managed) }
    guard let modelFlag = owner?.modelFlag ?? routing.modelFlags.last else { return nil }
    let selectors = owner?.selectors.flatMap(capability.arguments(satisfying:)) ?? []
    let invocation = MereRunCommandInvocation(capability: capability, arguments: flags + selectors + [modelFlag, model])
    return capability.resolveFamily(invocation) { identified in
        ModelFamilyIdentifier.identify(capabilityID: capability.id, model: identified, invocation: invocation)
    }
}

@Test func everyRoutedModelIsAManagedModelThatListsTheCommand() throws {
    for capability in MereRunCapabilityCatalog.document.commands {
        guard let routing = capability.routing else { continue }
        for family in routing.families {
            for model in family.models {
                let spec = try #require(
                    ManagedModelCatalog.allSpecs.first { $0.id == model },
                    "\(capability.id) \(family.id) lists \(model), which is not a managed model"
                )
                let commands = spec.defaultCLICommands.map { ListedCommand($0).id }
                #expect(commands.contains(capability.id), "\(model) runs \(capability.id) but does not list it")
            }
        }
        for model in routing.identifiedModels {
            #expect(ManagedModelCatalog.spec(for: model)?.id == model,
                    "\(capability.id) identifies \(model), which is not a managed model")
        }
        for excluded in routing.excludedModels {
            #expect(ManagedModelCatalog.spec(for: excluded.id)?.id == excluded.id,
                    "\(capability.id) excludes \(excluded.id), which is not a managed model")
        }
    }
}

/// A macOS default whose candidates span families has a machine chooser in Core that picks one
/// of them, so `catalog resolve` answers a blank model the way the command runs it.
@Test func everyMachineChosenDefaultHasAChooser() {
    for capability in MereRunCapabilityCatalog.document.commands {
        guard let routing = capability.routing else { continue }
        for rule in routing.defaultModels where rule.applies(on: "macos") && rule.family == nil {
            let families = Set(rule.models.flatMap { model in routing.families.filter { $0.models.contains(model) }.map(\.id) })
            guard families.count > 1 else { continue }
            let chosen = ModelFamilyIdentifier.machineDefault(capabilityID: capability.id, candidates: rule.models)
            #expect(chosen.map(rule.models.contains) == true, "\(capability.id) chose \(String(describing: chosen)) from \(rule.models)")
        }
    }
}

@Test func coverageExceptionsStayCurrent() {
    let catalog = MereRunCapabilityCatalog.document.commands
    for id in unroutedCapabilities.keys.sorted() {
        let capability = catalog.first { $0.id == id }
        #expect(capability != nil && capability?.routing == nil, "\(id) is routed or gone; remove it from unroutedCapabilities")
    }
    let listed = Set(ManagedModelCatalog.allSpecs.flatMap(\.defaultCLICommands).map { ListedCommand($0).path.joined(separator: " ") })
    for command in nonCapabilityCommands.keys.sorted() {
        #expect(listed.contains(command), "No managed model lists `\(command)`; remove it from nonCapabilityCommands")
    }
}

// MARK: - Resolution against an isolated model store

/// Resolving a managed id asks Core's identifier, which looks at what is installed, so these run
/// against an empty temporary model store with the model-location environment unset: the answer
/// is the one a fresh install gets, whatever this machine holds. Fixture stores then cover the
/// ids whose family depends on what is installed. The store is process state, so these are
/// serial XCTest cases that restore it.
final class ManagedModelFamilyResolutionTests: XCTestCase {
    /// Environment variables that point the commands at checkpoints outside the model store.
    private static let modelLocationKeys = [
        MereRunModelPaths.modelsDirEnvironmentKey, "MERERUN_MODEL_CACHE_HOME", "MERERUN_HUB_CACHE",
        "MERERUN_MUSIC_ACESTEP_ROOT", "MERERUN_VIDEO_LTX_MODEL_ROOT", "MERERUN_VIDEO_LTX_TEXT_ENCODER_ROOT"
    ]

    private var saved: [String: String] = [:]
    private var store: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        let environment = ProcessInfo.processInfo.environment
        saved = Dictionary(uniqueKeysWithValues: Self.modelLocationKeys.compactMap { key in environment[key].map { (key, $0) } })
        for key in Self.modelLocationKeys { unsetenv(key) }
        store = FileManager.default.temporaryDirectory.appendingPathComponent("family-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        MereRunModelPaths.setProcessModelsDirOverride(store, includeRegisteredLocations: false)
    }

    override func tearDownWithError() throws {
        MereRunModelPaths.setProcessModelsDirOverride(nil)
        for key in Self.modelLocationKeys { unsetenv(key) }
        for (key, value) in saved { setenv(key, value, 1) }
        try? FileManager.default.removeItem(at: store)
        try super.tearDownWithError()
    }

    func testEveryManagedModelResolvesToOneFamilyOfEachCommandItLists() {
        for spec in ManagedModelCatalog.allSpecs {
            for entry in spec.defaultCLICommands {
                let listed = ListedCommand(entry)
                guard let capability = MereRunCapabilityCatalog.command(id: listed.id) else {
                    XCTAssertNotNil(
                        nonCapabilityCommands[listed.path.joined(separator: " ")],
                        "\(spec.id) lists `\(entry)`, which is not a cataloged command path"
                    )
                    continue
                }
                guard capability.routing != nil else {
                    XCTAssertNotNil(unroutedCapabilities[capability.id], "\(capability.id) loads \(spec.id) but declares no routing")
                    continue
                }
                switch resolve(spec.id, in: capability, flags: listed.flags) {
                case .family(_, let model, _)?:
                    XCTAssertEqual(model, spec.id, "\(capability.id) resolved \(spec.id) as \(String(describing: model))")
                case .excluded?:
                    break
                case let other:
                    XCTFail("\(spec.id) lists `\(entry)` but resolves to \(String(describing: other))")
                }
            }
        }
    }

    /// An upstream repository id or other spelling the managed catalog knows resolves exactly like
    /// the managed id itself.
    func testAliasesResolveLikeTheManagedIdTheyName() throws {
        for capability in MereRunCapabilityCatalog.document.commands {
            guard let routing = capability.routing else { continue }
            for model in routing.families.flatMap(\.models) + routing.excludedModels.map(\.id) {
                let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: model))
                for alias in [spec.upstreamRepoId, model.uppercased()].compactMap({ $0 }) {
                    let canonical = try XCTUnwrap(ManagedModelCatalog.spec(for: alias)).id
                    XCTAssertEqual(resolve(alias, in: capability), resolve(canonical, in: capability), "\(capability.id) \(alias)")
                }
            }
        }
    }

    /// Every identified model resolves, with nothing installed, to the family of its own layout,
    /// or stays unidentified for a shell to ask `catalog resolve`; never to an error.
    func testIdentifiedModelsWithNothingInstalledNeverFail() {
        for capability in MereRunCapabilityCatalog.document.commands {
            for model in capability.routing?.identifiedModels ?? [] {
                switch resolve(model, in: capability) {
                case .family?, .unidentified?:
                    break
                case let other:
                    XCTFail("\(capability.id) \(model) resolves to \(String(describing: other))")
                }
            }
        }
    }

    /// For audio and video, `video-ltx-av` runs the LTX 2.3 Full checkpoint installed in the
    /// store, so it resolves to that family once the store holds one.
    func testAnInstalledCheckpointDecidesAnIdentifiedVideoModelsFamily() throws {
        let capability = MereRunCapabilityCatalog.videoGenerate
        let audioVideo = ["--output-mode", "audio-video"]
        let before = resolve("video-ltx-av", in: capability, flags: audioVideo)
        let full = store.appendingPathComponent("video-ltx23-full-mlx", isDirectory: true)
        let files = [
            "split_model.json", "config.json", "connector.safetensors", "transformer-dev.safetensors",
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
            "audio_vae.safetensors", "spatial_upscaler_x2_v1_1.safetensors", "vocoder.safetensors"
        ]
        for file in files {
            let url = full.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: url)
        }
        XCTAssertNotEqual(before, .family(id: "ltx23-full", model: "video-ltx-av", source: .identified))
        XCTAssertEqual(resolve("video-ltx-av", in: capability, flags: audioVideo), .family(id: "ltx23-full", model: "video-ltx-av", source: .identified),
                       "nothing installed: \(String(describing: before))")
    }

    /// An ACE-Step override root decides the family of every ACE-Step id, as the command loads it.
    func testAnInstalledOverrideRootDecidesAnIdentifiedModelsFamily() throws {
        let root = store.appendingPathComponent("ace-override", isDirectory: true)
        let decoder = root.appendingPathComponent("acestep-v15-xl-base", isDirectory: true)
        try FileManager.default.createDirectory(at: decoder, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: decoder.appendingPathComponent("config.json"))
        try Data().write(to: decoder.appendingPathComponent("model.safetensors"))
        try FileManager.default.createDirectory(at: root.appendingPathComponent("vae"), withIntermediateDirectories: true)
        let capability = MereRunCapabilityCatalog.musicGenerate
        XCTAssertEqual(resolve("music-acestep", in: capability), .family(id: "ace-step-turbo", model: "music-acestep", source: .model))

        setenv("MERERUN_MUSIC_ACESTEP_ROOT", root.path, 1)
        for model in capability.routing?.identifiedModels ?? [] {
            XCTAssertEqual(resolve(model, in: capability), .family(id: "ace-step-base", model: model, source: .identified), model)
        }
        XCTAssertEqual(resolve("music-yue2", in: capability), .family(id: "yue2", model: "music-yue2", source: .model))
    }
}
