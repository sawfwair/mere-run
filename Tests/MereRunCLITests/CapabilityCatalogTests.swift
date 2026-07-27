import MereRunContract
import Testing

@testable import MereRunCLI

@Test func videoCapabilityFlagsMatchArgumentParserHelp() {
    let helpByID: [String: String] = [
        "video.generate": VideoGenerate.helpMessage(),
        "video.animate": VideoAnimate.helpMessage(),
        "video.cosmos3": VideoCosmos3.helpMessage(),
        "video.prepare-masks": VideoPrepareMasks.helpMessage(),
        "video.export-latents": VideoExportLatents.helpMessage(),
        "video.session": VideoSession.helpMessage()
    ]

    for capability in MereRunCapabilityCatalog.document.commands {
        let help = helpByID[capability.id]
        #expect(help != nil, "Missing help fixture for \(capability.id)")
        for option in capability.options {
            #expect(
                help?.contains(option.flag) == true,
                "\(capability.id) advertises \(option.flag), but the CLI help does not"
            )
        }
    }
}

@Test func catalogCommandParsesASelectedCapability() throws {
    let command = try CatalogCommand.parse(["video.generate", "--json"])
    #expect(command.id == "video.generate")
    #expect(command.json)
    #expect(MereRunCapabilityCatalog.command(id: command.id ?? "")?.id == "video.generate")
}
