import ArgumentParser

struct Vision: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vision",
        abstract: "Caption, inspect, segment, track, pose, depth, geometry, optical flow, and OCR visual media.",
        subcommands: [
            VisionCaption.self,
            VisionInspect.self,
            VisionGround.self,
            VisionSegment.self,
            VisionTrack.self,
            VisionTrackLive.self,
            VisionPose.self,
            VisionFlow.self,
            VisionDepthVideo.self,
            VisionGeometry.self,
            VisionGeometryMultiView.self,
            VisionImageTo3D.self,
            VisionImageTo3DTrellis2.self,
            VisionImageTo3DMultiview.self,
            VisionOCR.self,
        ]
    )
}
