import Foundation

struct RunPollingPolicy: Equatable, Sendable {
    let regularDelayNanoseconds: UInt64
    let maximumDelayNanoseconds: UInt64
    let maximumConsecutiveFailures: Int

    init(
        regularDelayNanoseconds: UInt64 = 2_000_000_000,
        maximumDelayNanoseconds: UInt64 = 30_000_000_000,
        maximumConsecutiveFailures: Int = 6
    ) {
        self.regularDelayNanoseconds = regularDelayNanoseconds
        self.maximumDelayNanoseconds = maximumDelayNanoseconds
        self.maximumConsecutiveFailures = maximumConsecutiveFailures
    }

    func delayNanoseconds(afterConsecutiveFailures failures: Int) -> UInt64 {
        guard failures > 0 else { return regularDelayNanoseconds }
        let exponent = min(failures - 1, 62)
        let multiplier = UInt64(1) << UInt64(exponent)
        let (delay, overflow) = regularDelayNanoseconds.multipliedReportingOverflow(by: multiplier)
        return overflow ? maximumDelayNanoseconds : min(delay, maximumDelayNanoseconds)
    }

    func shouldStop(afterConsecutiveFailures failures: Int) -> Bool {
        failures >= maximumConsecutiveFailures
    }
}
