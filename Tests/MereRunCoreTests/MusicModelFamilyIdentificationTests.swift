import Foundation
import MereRunContract
import Testing

@testable import MereRunCore

// Core's music routing and the contract's music families must agree: for every managed id, for
// every local layout the commands recognize, and for the RoFormer chunk sizes the contract copies.

private let generate = MereRunCapabilityCatalog.musicGenerate
private let serve = MereRunCapabilityCatalog.musicServe

private func runtime(ofGenerateFamily family: String) -> MusicModelRuntime {
    switch family {
    case "yue2": .yue2
    case "minimax-music3": .miniMaxMusic3
    case "magenta-rt2": .magentaRT2
    default: .aceStep
    }
}

private func temporaryDirectory() -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
}

private func write(_ contents: String, to relativePath: String, in root: URL) throws {
    let url = root.appendingPathComponent(relativePath)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(contents.utf8).write(to: url)
}

/// An ACE-Step checkpoints root with one decoder, laid out as the managed install lays it out.
private func aceStepRoot(decoder: String) throws -> URL {
    let root = temporaryDirectory()
    try write("{}", to: "\(decoder)/config.json", in: root)
    try write("", to: "\(decoder)/model.safetensors", in: root)
    try FileManager.default.createDirectory(at: root.appendingPathComponent("vae"), withIntermediateDirectories: true)
    return root
}

/// The decoder directory a managed ACE-Step model installs, read off its download layout.
private func managedDecoder(_ spec: ManagedModelSpec) -> String? {
    let paths = spec.mountedHubFallbacks.map(\.destinationPath)
        + (spec.hubFallback?.patterns ?? []).map { $0.replacingOccurrences(of: "/*", with: "") }
    return paths.first { $0.hasPrefix("acestep-v15") }
}

private func identify(
    _ capability: MereRunCommandCapability,
    _ arguments: [String]
) -> (MereRunCommandInvocation, MereRunFamilyResolution) {
    let invocation = MereRunCommandInvocation(capability: capability, arguments: arguments)
    let resolution = capability.resolveFamily(invocation) { model in
        ModelFamilyIdentifier.identify(capabilityID: capability.id, model: model, invocation: invocation)
    }
    return (invocation, resolution)
}

@Test func everyGenerateModelRoutesToItsContractFamilyInCore() throws {
    let families = try #require(generate.routing).families
    for family in families {
        for model in family.models {
            #expect(MusicModelRuntime.generation(model: model) == runtime(ofGenerateFamily: family.id), "\(model)")
            guard runtime(ofGenerateFamily: family.id) == .aceStep else { continue }
            let spec = try #require(ManagedModelCatalog.spec(for: model))
            let decoder = try #require(managedDecoder(spec), "\(model) installs no ACE-Step decoder")
            let root = try aceStepRoot(decoder: decoder)
            defer { try? FileManager.default.removeItem(at: root) }
            let (_, resolution) = identify(generate, ["--checkpoints-root", root.path, "--model", model])
            #expect(resolution == .family(id: family.id, model: root.path, source: .identified), "\(model): \(decoder)")
        }
    }
}

@Test func everyServeModelRoutesToItsContractFamilyInCore() throws {
    for family in try #require(serve.routing).families {
        for model in family.models {
            let expected: MusicModelRuntime = family.id == "minimax-music3" ? .miniMaxMusic3 : .aceStep
            #expect(MusicModelRuntime.serving(model: model) == expected, "\(model)")
        }
    }
}

@Test func localMusicLayoutsIdentifyAsTheFamilyTheCommandRuns() throws {
    let yue2 = temporaryDirectory()
    try write(#"{"model_type": "yue2"}"#, to: "config.json", in: yue2)
    let miniMax = temporaryDirectory()
    for path in ["modular_model_index.json", "language_model/config.json", "vocoder/config.json"] {
        try write("{}", to: path, in: miniMax)
    }
    let magenta = temporaryDirectory()
    try write("", to: "models/mrt2_small/mrt2_small.mlxfn", in: magenta)
    try FileManager.default.createDirectory(
        at: magenta.appendingPathComponent("resources/musiccoca"), withIntermediateDirectories: true
    )
    let sft = try aceStepRoot(decoder: "acestep-v15-xl-sft")
    let base = try aceStepRoot(decoder: "acestep-v15-base")
    defer {
        for root in [yue2, miniMax, magenta, sft, base] { try? FileManager.default.removeItem(at: root) }
    }

    let cases: [(MereRunCommandCapability, [String], String)] = [
        (generate, ["--model", yue2.path], "yue2"),
        (generate, ["--model", miniMax.path], "minimax-music3"),
        (generate, ["--model", magenta.path], "magenta-rt2"),
        (generate, ["--model", sft.path], "ace-step-sft"),
        (generate, ["--model", base.path], "ace-step-base"),
        (generate, ["--checkpoints-root", sft.path], "ace-step-sft"),
        // The model's own runtime wins over an ACE-Step root, as it does in the command.
        (generate, ["--checkpoints-root", sft.path, "--model", yue2.path], "yue2"),
        (serve, ["--model", miniMax.path], "minimax-music3"),
        (serve, ["--model", sft.path], "ace-step")
    ]
    for (capability, arguments, family) in cases {
        let (invocation, resolution) = identify(capability, arguments)
        guard case let .family(id, _, source) = resolution else {
            Issue.record("\(capability.id) \(arguments): \(resolution)")
            continue
        }
        #expect(id == family && source == .identified, "\(capability.id) \(arguments)")
        let model = try #require(arguments.last)
        #expect(ModelFamilyIdentifier.identify(capabilityID: capability.id, model: model, invocation: invocation)
            == .family(family))
    }

    // A folder with no ACE-Step checkpoint stays unidentified; the command reports it.
    let empty = temporaryDirectory()
    try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: empty) }
    #expect(identify(generate, ["--model", empty.path]).1 == .unidentified(model: empty.path))
}

@Test func separateOverlapsMatchTheBundledRoFormerConfigurations() throws {
    let overlap = try #require(MereRunCapabilityCatalog.musicSeparate.options.first { $0.flag == "--overlap" })
    for family in try #require(MereRunCapabilityCatalog.musicSeparate.routing).families {
        let model = try #require(family.models.first)
        let bundled: (chunkSize: Int, overlap: Int)
        if let profile = RoFormerModelProfile.allCases.first(where: { $0.modelID == model }) {
            let configuration = try RoFormerResources.loadBundledConfiguration(profile: profile)
            bundled = (configuration.chunkSize, configuration.overlap)
        } else {
            let configuration = try MelBandRoFormerResources.loadBundledConfiguration(
                profile: MelBandRoFormerProfile.resolve(modelID: model)
            )
            bundled = (configuration.chunkSize, configuration.overlap)
        }
        let (chunkSize, published) = bundled
        let rule = try #require(overlap.familyRules.first { $0.family == family.id })
        let divisors = (1...chunkSize).filter { chunkSize.isMultiple(of: $0) }.map(String.init)
        #expect(rule.values == divisors, "\(model)")
        #expect(rule.defaultValue == String(published), "\(model)")
    }
}
