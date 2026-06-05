import Foundation

public struct MagentaRT2RealtimeRequest: Hashable, Sendable {
    public let prompt: String
    public let resources: MagentaRT2Resources
    public let durationSeconds: Float
    public let controls: MagentaRT2Controls

    public init(
        prompt: String,
        resources: MagentaRT2Resources,
        durationSeconds: Float,
        controls: MagentaRT2Controls = MagentaRT2Controls()
    ) {
        self.prompt = prompt
        self.resources = resources
        self.durationSeconds = durationSeconds
        self.controls = controls
    }
}

public enum MagentaRT2RealtimeSession {
    public static func run(
        _ request: MagentaRT2RealtimeRequest,
        liveControls: (@Sendable (Int) throws -> MagentaRT2LiveControlSnapshot?)? = nil,
        onFrame: @escaping @Sendable (Int, MagentaRT2Frame) throws -> Void
    ) async throws {
        guard request.durationSeconds > 0 else {
            throw MagentaRT2Error.invalidDuration(request.durationSeconds)
        }
        let missing = request.resources.validate()
        guard missing.isEmpty else {
            throw MagentaRT2Error.missingAssets(missing)
        }

        let frameCount = max(1, Int((request.durationSeconds * Float(MagentaRT2Resources.frameRate)).rounded(.up)))
        try await MagentaRT2Renderer.renderFrameStream(
            MagentaRT2RenderRequest(
                prompt: request.prompt,
                resources: request.resources,
                durationSeconds: request.durationSeconds,
                controls: request.controls
            ),
            frameCount: frameCount,
            liveControls: liveControls,
            onFrame: onFrame
        )
    }
}
