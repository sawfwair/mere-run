import ArgumentParser

struct Vision: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vision",
        abstract: "Caption, inspect, segment, track, pose, optical flow, and OCR visual media.",
        subcommands: [
            VisionCaption.self,
            VisionInspect.self,
            VisionGround.self,
            VisionSegment.self,
            VisionTrack.self,
            VisionTrackLive.self,
            VisionPose.self,
            VisionFlow.self,
            VisionOCR.self,
        ]
    )
}
