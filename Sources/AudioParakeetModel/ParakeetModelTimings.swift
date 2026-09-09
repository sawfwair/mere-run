import Dispatch
import Foundation

public struct ParakeetModelTimings {
    public var encoderSeconds: TimeInterval = 0
    public var decoderSeconds: TimeInterval = 0
    public var alignmentSeconds: TimeInterval = 0

    public init() {}
}

package enum ParakeetMonotonicClock {
    package static func now() -> UInt64 {
        DispatchTime.now().uptimeNanoseconds
    }

    package static func seconds(since start: UInt64) -> TimeInterval {
        TimeInterval(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
    }
}
