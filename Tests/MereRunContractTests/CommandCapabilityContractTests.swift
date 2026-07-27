import Foundation
import MereRunContract
import Testing

@Test func videoCapabilityCatalogIsStableAndMachineReadable() throws {
    let document = MereRunCapabilityCatalog.document
    #expect(document.schemaVersion == 1)
    #expect(document.commands.map(\.id) == [
        "video.generate",
        "video.animate",
        "video.cosmos3",
        "video.prepare-masks",
        "video.export-latents",
        "video.session"
    ])
    #expect(document.commands.allSatisfy { !$0.options.isEmpty })

    let data = try JSONEncoder().encode(document)
    let decoded = try JSONDecoder().decode(MereRunCapabilityDocument.self, from: data)
    #expect(decoded == document)
}

@Test func videoProductChoicesComeFromTypedSharedEnums() {
    let generate = MereRunCapabilityCatalog.videoGenerate
    let quality = generate.options.first { $0.flag == "--quality" }
    let outputMode = generate.options.first { $0.flag == "--output-mode" }

    #expect(quality?.choices == LTXVideoQuality.allCases.map(\.rawValue))
    #expect(outputMode?.choices == LTXVideoOutputMode.allCases.map(\.rawValue))
    #expect(!generate.options.contains { $0.flag == "--variant" })
}
