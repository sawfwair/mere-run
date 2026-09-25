import Foundation
import MereRunContract
import Testing

@testable import MereRunCore

/// Multi-family capabilities whose routing a model-scope domain change declares. Each domain
/// removes its entries; the integration requires this to be empty.
private let pendingCapabilities: [String: String] = [
    "image.generate": "image domain",
    "image.train-lora": "image domain",
    "music.generate": "music domain",
    "music.separate": "music domain",
    "music.serve": "music domain",
    "text.chat": "text domain",
    "text.train-lora": "text domain",
    "speech.transcribe": "speech domain",
    "speech.diarize": "speech domain",
    "speech.synthesize": "speech domain",
    "audio.enhance": "audio, SFX, OCR, and TESSERA domain",
    "audio.edit": "audio, SFX, OCR, and TESSERA domain",
    "sfx.generate": "audio, SFX, OCR, and TESSERA domain",
    "sfx.video.generate": "audio, SFX, OCR, and TESSERA domain",
    "vision.ocr": "audio, SFX, OCR, and TESSERA domain",
    "geo.tessera": "audio, SFX, OCR, and TESSERA domain"
]

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
    "chat": "Mislabel of `text chat`; the text domain fixes the catalog data.",
    "agent": "Mislabel of `agent start`; the text domain fixes the catalog data.",
    "sfx clap": "Mislabel of `sfx clap score`; the SFX domain fixes the catalog data."
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
    let canonical = ManagedModelCatalog.spec(for: model)?.id ?? model
    let owner = routing.families.first { $0.models.contains(canonical) }
    guard let modelFlag = owner?.modelFlag ?? routing.modelFlags.last else { return nil }
    let selectors = owner?.selectors.flatMap { selector -> [String] in
        guard !selector.absent else { return [] }
        let option = capability.options.first { $0.flag == selector.flag }
        let value = selector.values?.first ?? (option?.kind == .boolean ? nil : option?.defaultValue ?? "value")
        return [selector.flag] + (value.map { [$0] } ?? [])
    } ?? []
    let invocation = MereRunCommandInvocation(capability: capability, arguments: flags + selectors + [modelFlag, model])
    return capability.resolveFamily(invocation) { identified in
        ModelFamilyIdentifier.identify(capabilityID: capability.id, model: identified, invocation: invocation)
    }
}

@Test func everyManagedModelResolvesToOneFamilyOfEachCommandItLists() {
    for spec in ManagedModelCatalog.allSpecs {
        for entry in spec.defaultCLICommands {
            let listed = ListedCommand(entry)
            guard let capability = MereRunCapabilityCatalog.command(id: listed.id) else {
                #expect(
                    nonCapabilityCommands[listed.path.joined(separator: " ")] != nil,
                    "\(spec.id) lists `\(entry)`, which is not a cataloged command path"
                )
                continue
            }
            guard pendingCapabilities[capability.id] == nil else { continue }
            guard capability.routing != nil else {
                #expect(unroutedCapabilities[capability.id] != nil, "\(capability.id) loads \(spec.id) but declares no routing")
                continue
            }
            switch resolve(spec.id, in: capability, flags: listed.flags) {
            case .family(_, let model, _)?:
                #expect(model == spec.id, "\(capability.id) resolved \(spec.id) as \(String(describing: model))")
            case .excluded?:
                break
            case let other:
                Issue.record("\(spec.id) lists `\(entry)` but resolves to \(String(describing: other))")
            }
        }
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
        for excluded in routing.excludedModels {
            #expect(ManagedModelCatalog.spec(for: excluded.id)?.id == excluded.id,
                    "\(capability.id) excludes \(excluded.id), which is not a managed model")
        }
    }
}

/// An upstream repository id or other spelling the managed catalog knows resolves exactly like
/// the managed id itself.
@Test func aliasesResolveLikeTheManagedIdTheyName() throws {
    for capability in MereRunCapabilityCatalog.document.commands {
        guard let routing = capability.routing else { continue }
        for model in routing.families.flatMap(\.models) + routing.excludedModels.map(\.id) {
            let spec = try #require(ManagedModelCatalog.spec(for: model))
            for alias in [spec.upstreamRepoId, model.uppercased()].compactMap({ $0 }) {
                let canonical = try #require(ManagedModelCatalog.spec(for: alias)).id
                #expect(resolve(alias, in: capability) == resolve(canonical, in: capability), "\(capability.id) \(alias)")
            }
        }
    }
}

@Test func coverageExceptionsStayCurrent() {
    let catalog = MereRunCapabilityCatalog.document.commands
    for id in pendingCapabilities.keys.sorted() {
        let capability = catalog.first { $0.id == id }
        #expect(capability != nil, "\(id) is not a cataloged capability")
        #expect(capability?.routing == nil, "\(id) declares routing now; remove it from pendingCapabilities")
    }
    for id in unroutedCapabilities.keys.sorted() {
        let capability = catalog.first { $0.id == id }
        #expect(capability != nil && capability?.routing == nil, "\(id) is routed or gone; remove it from unroutedCapabilities")
    }
    let listed = Set(ManagedModelCatalog.allSpecs.flatMap(\.defaultCLICommands).map { ListedCommand($0).path.joined(separator: " ") })
    for command in nonCapabilityCommands.keys.sorted() {
        #expect(listed.contains(command), "No managed model lists `\(command)`; remove it from nonCapabilityCommands")
    }
}
