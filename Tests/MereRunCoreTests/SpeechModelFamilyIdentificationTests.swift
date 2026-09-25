import Foundation
import MereRunContract
import Testing

@testable import MereRunCore

private let diarize = MereRunCapabilityCatalog.speechDiarize

private func invocation(_ arguments: [String]) -> MereRunCommandInvocation {
    MereRunCommandInvocation(capability: diarize, arguments: arguments)
}

private func family(of model: String) -> MereRunFamilyResolution {
    let given = invocation(["a.wav", "--model", model])
    return diarize.resolveFamily(given) { ModelFamilyIdentifier.identify(capabilityID: diarize.id, model: $0, invocation: given) }
}

/// `speech diarize` picks Nemotron 3 with `Nemotron3DiarizationResources.isNemotron3`; the
/// contract's families agree with it on every managed id, and the probe on local folders.
@Test func diarizeFamiliesAgreeWithTheCommandsDetector() throws {
    let empty = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: empty) }
    for runtime in try #require(diarize.routing).families {
        for model in runtime.models {
            let detected = Nemotron3DiarizationResources.isNemotron3(model: model, root: empty) ? "nemotron3" : "sortformer"
            #expect(runtime.id == detected, "\(model)")
            #expect(family(of: model) == .family(id: runtime.id, model: model, source: .model))
        }
    }

    #expect(family(of: empty.path) == .family(id: "sortformer", model: empty.path, source: .identified))
    try Data().write(to: empty.appendingPathComponent(Nemotron3DiarizationResources.archivePin.filename))
    #expect(family(of: empty.path) == .family(id: "nemotron3", model: empty.path, source: .identified))
    #expect(family(of: empty.appendingPathComponent("missing").path) == .unidentified(model: empty.appendingPathComponent("missing").path))
}
