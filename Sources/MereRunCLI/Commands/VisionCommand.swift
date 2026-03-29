import ArgumentParser

struct Vision: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vision",
        abstract: "Caption, inspect, and OCR images.",
        subcommands: [
            VisionCaption.self,
            VisionInspect.self,
            VisionOCR.self,
        ]
    )
}
