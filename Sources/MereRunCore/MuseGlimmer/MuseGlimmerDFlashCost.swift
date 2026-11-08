import Foundation

enum MuseGlimmerDFlashFallback: String, Decodable {
    case acceptance
    case throughput

    static var selected: Self {
        ProcessInfo.processInfo.environment["MERERUN_MUSE_GLIMMER_DFLASH_COST_FALLBACK"] == "1" ? .throughput : .acceptance
    }
}

struct MuseGlimmerDFlashCost {
    private(set) var serialSeconds: Double?
    private(set) var rounds = 0
    private(set) var confirmedTokens = 0
    private(set) var speculativeSeconds = 0.0

    init(serialSeconds: Double? = nil) { self.serialSeconds = serialSeconds }

    func needsCalibration(acceptanceRate: Double, minimumAcceptanceRate: Double) -> Bool {
        rounds >= 10 && serialSeconds == nil && acceptanceRate < minimumAcceptanceRate
    }

    mutating func calibrate(serialSeconds: Double) { self.serialSeconds = serialSeconds }

    mutating func observe(accepted: Int, seconds: Double) {
        rounds += 1
        // Serial calibration discards cold setup by taking warmed samples.
        // Give the different draft/verification graph the same opportunity.
        guard rounds > 2 else { return }
        confirmedTokens += accepted + 1
        speculativeSeconds += seconds
    }

    var shouldFallback: Bool {
        guard let serialSeconds else { return false }
        // Compare eight completed steady rounds after the two warmup blocks.
        return rounds >= 10 && Double(confirmedTokens) * serialSeconds < speculativeSeconds * 0.95
    }
}
