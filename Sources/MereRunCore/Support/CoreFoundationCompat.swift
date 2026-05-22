import Foundation

#if os(Linux)
typealias CFTimeInterval = TimeInterval

func CFAbsoluteTimeGetCurrent() -> CFTimeInterval {
    Date().timeIntervalSinceReferenceDate
}
#endif
