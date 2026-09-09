import Foundation

public func ltx25FrameCount(
    predictedSeconds: Double,
    frameRate: Double,
    range: LTX25AutoDuration = LTX25AutoDuration()
) -> Int {
    precondition(frameRate > 0, "frameRate must be positive")
    let minimumFrames = Int((range.minimumSeconds * frameRate).rounded(.toNearestOrEven))
    let maximumFrames = Int((range.maximumSeconds * frameRate).rounded(.toNearestOrEven))
    let rawFrames = Int((predictedSeconds * frameRate).rounded(.toNearestOrEven))
    let clamped = min(max(rawFrames, minimumFrames), maximumFrames)
    let snappedDown = ((max(1, clamped) - 1) / 8) * 8 + 1
    if snappedDown >= minimumFrames {
        return snappedDown
    }
    return min((((minimumFrames - 1) + 7) / 8) * 8 + 1, maximumFrames)
}
