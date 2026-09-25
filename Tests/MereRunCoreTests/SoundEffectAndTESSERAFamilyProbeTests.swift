import Foundation
import MereRunContract
import Testing

@testable import MereRunCore

/// The SFX and TESSERA probes wrap the detectors the commands route on, so a managed id and a
/// local folder of each layout land in the family the contract declares for them.

private func resolve(
    _ capability: MereRunCommandCapability,
    _ arguments: [String]
) -> MereRunFamilyResolution {
    let invocation = MereRunCommandInvocation(capability: capability, arguments: arguments)
    return capability.resolveFamily(invocation) { model in
        ModelFamilyIdentifier.identify(capabilityID: capability.id, model: model, invocation: invocation)
    }
}

private func probe(_ capability: MereRunCommandCapability, _ model: String) -> String? {
    let invocation = MereRunCommandInvocation(capability: capability, arguments: [])
    return ModelFamilyIdentifier.probes[capability.id]?(model, invocation)
}

private func folder(_ files: [String]) throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    for file in files {
        let url = root.appendingPathComponent(file)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data().write(to: url)
    }
    return root
}

// MARK: - Sound effects

@Test(arguments: [MereRunCapabilityCatalog.sfxGenerate, MereRunCapabilityCatalog.sfxVideoGenerate])
func soundEffectDetectorsAgreeWithTheContractForEveryManagedModel(capability: MereRunCommandCapability) throws {
    let routing = try #require(capability.routing)
    for family in routing.families {
        for model in family.models {
            #expect(probe(capability, model) == family.id, "\(capability.id) \(model)")
        }
    }
    // The Woosh variants a command refuses are excluded, and its detector finds no family.
    for excluded in routing.excludedModels {
        #expect(probe(capability, excluded.id) == nil, "\(capability.id) \(excluded.id)")
    }
}

@Test func soundEffectFoldersResolveByTheirLayout() throws {
    let mmaudio = try folder([MMAudioResources.networkFilename]).path
    let flow = try folder(["checkpoints/Woosh-AE/config.yaml", "checkpoints/Woosh-Flow/config.yaml"]).path
    let dvflow = try folder(["Woosh-AE/config.yaml", "Woosh-DVFlow-8s/config.yaml"]).path
    let text = MereRunCapabilityCatalog.sfxGenerate
    let video = MereRunCapabilityCatalog.sfxVideoGenerate

    #expect(resolve(text, ["--model", mmaudio]) == .family(id: "mmaudio", model: mmaudio, source: .identified))
    #expect(resolve(video, ["-m", mmaudio]) == .family(id: "mmaudio", model: mmaudio, source: .identified))
    #expect(resolve(text, ["--model", flow]) == .family(id: "woosh-flow", model: flow, source: .identified))
    #expect(resolve(video, ["--model", flow]) == .unidentified(model: flow))
    #expect(resolve(video, ["--model", dvflow]) == .family(id: "woosh-dvflow", model: dvflow, source: .identified))
    #expect(resolve(text, ["--model", dvflow]) == .unidentified(model: dvflow))

    let renoise = MereRunCommandInvocation(capability: text, arguments: ["--model", mmaudio, "--renoise", "0.5"])
    let report = text.resolutionReport(renoise) {
        ModelFamilyIdentifier.identify(capabilityID: text.id, model: $0, invocation: renoise)
    }
    #expect(report.violations == ["--renoise is not supported by MMAudio. It applies to Woosh DFlow."])
}

// MARK: - TESSERA

@Test func tesseraVariantsAgreeWithTheContractForEveryManagedModel() throws {
    let routing = try #require(MereRunCapabilityCatalog.geoTessera.routing)
    for family in routing.families {
        for model in family.models {
            let variant = try #require(TESSERAResources.spec(for: model)?.variant)
            #expect(ModelFamilyIdentifier.tesseraFamily(variant) == family.id, "\(model)")
        }
    }
    #expect(Set(routing.families.flatMap(\.models)) == Set(TESSERAResources.allSpecs.map(\.modelID)))
}

@Test func tesseraFoldersResolveByTheVariantTheyDeclare() throws {
    let capability = MereRunCapabilityCatalog.geoTessera
    for source in TESSERAResources.allSpecs {
        let root = try folder([TESSERAResources.weightsFilename])
        let configuration = TESSERAConversionConfiguration(
            format: TESSERAResources.conversionFormat, modelID: source.modelID, variant: source.variant,
            sourceRepository: source.sourceRepository, sourceRevision: source.sourceRevision,
            sourceCheckpoint: source.sourceCheckpointFilename, sourceCheckpointSHA256: source.sourceCheckpointSHA256,
            converter: "scripts/convert-tessera-v2-mlx.py@v1", dtype: "float32", tensorCount: source.tensorCount,
            scalarCount: source.scalarCount, architecture: source.architecture
        )
        try JSONEncoder().encode(configuration).write(to: root.appendingPathComponent(TESSERAResources.configurationFilename))
        let expected = source.variant == .teacher ? "tessera-teacher" : "tessera-student"
        #expect(resolve(capability, ["--model", root.path]) == .family(id: expected, model: root.path, source: .identified))
        let weights = root.appendingPathComponent(TESSERAResources.weightsFilename).path
        #expect(probe(capability, weights) == expected, "a weights file reads its folder's config")
    }
    #expect(probe(capability, try folder(["README.md"]).path) == nil)
}
