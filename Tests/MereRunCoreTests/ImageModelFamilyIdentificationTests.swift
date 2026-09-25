import Foundation
import MereRunContract
import Testing

@testable import MereRunCore

/// The image commands pick their runtime from the model manifest. For every model the contract
/// routes, the manifest it installs with must name the same family, and a local folder carrying
/// that manifest must be identified as that family too.

private let imageCapabilities = [MereRunCapabilityCatalog.imageGenerate, MereRunCapabilityCatalog.imageTrainLoRA]

private func family(of manifest: MereRunModelManifest, for capability: MereRunCommandCapability) -> String? {
    capability.id == "image.generate"
        ? ModelFamilyIdentifier.imageGenerateFamily(manifest)
        : ModelFamilyIdentifier.imageTrainLoRAFamily(manifest)
}

private func template(_ model: String) throws -> MereRunModelManifest {
    MereRunModelManifest.template(for: try #require(ModelResolver.ModelID(rawValue: model)))
}

@Test func coreReadsEveryRoutedImageModelAsTheContractFamily() throws {
    for capability in imageCapabilities {
        let routing = try #require(capability.routing)
        for runtime in routing.families {
            for model in runtime.models {
                #expect(family(of: try template(model), for: capability) == runtime.id, "\(capability.id) \(model)")
            }
        }
        for excluded in routing.excludedModels {
            #expect(family(of: try template(excluded.id), for: capability) == nil, "\(capability.id) \(excluded.id)")
        }
    }
}

@Test func aLocalFolderIsIdentifiedByItsManifest() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    for capability in imageCapabilities {
        for runtime in try #require(capability.routing).families {
            let folder = root.appendingPathComponent("\(capability.id)-\(runtime.id)", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            var manifest = try template(try #require(runtime.models.first))
            manifest.id = "local-copy"
            // Lightning is told apart by its manifest id, as the edit generator does.
            if runtime.id == "qwen-edit-lightning" { manifest.id = QwenImageEditRepository.lightning2511Id }
            try manifest.write(to: folder)
            let invocation = MereRunCommandInvocation(capability: capability, arguments: ["--model", folder.path])
            #expect(
                ModelFamilyIdentifier.identify(capabilityID: capability.id, model: folder.path, invocation: invocation)
                    == .family(runtime.id),
                "\(capability.id) \(runtime.id)"
            )
        }
    }
    let empty = root.appendingPathComponent("no-manifest", isDirectory: true)
    try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
    let invocation = MereRunCommandInvocation(capability: MereRunCapabilityCatalog.imageGenerate, arguments: [])
    #expect(ModelFamilyIdentifier.identify(capabilityID: "image.generate", model: empty.path, invocation: invocation) == nil)
}

/// The distilled Klein models are not listed for training, but the CLI trains them on the Klein
/// path, so the identifier gives them the Klein surface rather than none.
@Test func unlistedKleinModelsTrainOnTheKleinSurface() {
    let capability = MereRunCapabilityCatalog.imageTrainLoRA
    for model in ["image-klein-9b", "image-klein-nano", "image-flux2-dev"] {
        let invocation = MereRunCommandInvocation(capability: capability, arguments: ["--model", model])
        let resolution = capability.resolveFamily(invocation) { candidate in
            ModelFamilyIdentifier.identify(capabilityID: capability.id, model: candidate, invocation: invocation)
        }
        #expect(resolution == .family(id: "klein", model: model, source: .identified), "\(model)")
    }
}

/// The contract's recipe spellings are the ones Core's recipe table accepts: each alias, in any
/// case, trains the same base with the same settings as the recipe it names.
@Test func recipeSpellingsAgreeWithCoreRecipes() throws {
    let recipe = try #require(MereRunCapabilityCatalog.imageTrainLoRA.options.first { $0.flag == "--recipe" })
    let resolved = { (name: String) throws -> ImageLoRATrainingOptions.Resolved in
        var options = ImageLoRATrainingOptions(output: "/tmp/adapter.safetensors")
        options.recipe = name
        return try options.resolve()
    }
    for (alias, canonical) in try #require(recipe.choiceSpellings).aliases {
        let expected = try resolved(canonical)
        for written in [alias, alias.uppercased(), " \(alias) "] {
            let aliased = try resolved(written)
            #expect(
                aliased.model == expected.model && aliased.width == expected.width && aliased.height == expected.height
                    && aliased.trainingSteps == expected.trainingSteps && aliased.learningRate == expected.learningRate,
                "\(written) → \(canonical)"
            )
        }
    }
}
