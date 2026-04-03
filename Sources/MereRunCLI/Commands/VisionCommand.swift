import ArgumentParser

struct Vision: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vision",
        abstract: "Caption, inspect, segment, track, and OCR visual media.",
        subcommands: [
            VisionCaption.self,
            VisionInspect.self,
            VisionSegment.self,
            VisionTrack.self,
            VisionTrackLive.self,
            VisionOCR.self,
        ]
    )
}
