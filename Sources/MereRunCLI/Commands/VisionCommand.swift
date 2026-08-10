import ArgumentParser

struct Vision: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vision",
        abstract: "Caption, inspect, face-analyze, segment, track, pose, depth, geometry, optical flow, and OCR visual media.",
        subcommands: [
            VisionCaption.self,
            VisionInspect.self,
            VisionFace.self,
            VisionGround.self,
            VisionServe.self,
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
