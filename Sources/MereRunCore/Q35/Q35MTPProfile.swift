import Foundation
import MLX

/// Synchronized diagnostic timings. Profiling changes scheduling and is excluded
/// from normal throughput receipts.
public struct ChatSpeculationProfile: Codable, Sendable, Hashable {
    public var rounds: [ChatSpeculationRoundProfile] = []
    public var samplingSeconds: Double = 0
    public var serialSeconds: Double = 0
    public var serialRounds: Int = 0
}

/// Wall time spent constructing graphs, waiting for their results, and accepting
/// or restoring one speculative round. Times include host scheduling delays.
public struct ChatSpeculationRoundProfile: Codable, Sendable, Hashable {
    public var draftDepth: Int
    public var acceptedDrafts: Int = 0
    public var draftGraphSeconds: Double = 0
    public var draftWaitSeconds: Double = 0
    public var verificationGraphSeconds: Double = 0
    public var verificationWaitSeconds: Double = 0
    public var acceptanceSeconds: Double = 0
    public var repairSeconds: Double = 0
}

final class Q35MTPProfile {
    private(set) var result = ChatSpeculationProfile()
    private var current = ChatSpeculationRoundProfile(draftDepth: 0)
    private var phaseStart: UInt64 = 0
    private var repairStart: UInt64?

    static func make(environment: [String: String] = ProcessInfo.processInfo.environment) -> Q35MTPProfile? {
        environment["MERERUN_Q35_MTP_PROFILE"] == "1" ? Q35MTPProfile() : nil
    }

    func clock() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    private func seconds(since start: UInt64) -> Double {
        Double(clock() - start) / 1_000_000_000
    }

    func recordSampling(since start: UInt64?) {
        if let start { result.samplingSeconds += seconds(since: start) }
    }

    func beginRound(depth: Int) {
        current = ChatSpeculationRoundProfile(draftDepth: depth)
        repairStart = nil
        phaseStart = clock()
    }

    func finishDraft(tokens: MLXArray) {
        current.draftGraphSeconds = seconds(since: phaseStart)
        phaseStart = clock()
        MLX.eval(tokens)
        Stream.gpu.synchronize()
        current.draftWaitSeconds = seconds(since: phaseStart)
        phaseStart = clock()
    }

    func verificationSubmitted() {
        current.verificationGraphSeconds = seconds(since: phaseStart)
        phaseStart = clock()
    }

    func verificationCompleted() {
        Stream.gpu.synchronize()
        current.verificationWaitSeconds = seconds(since: phaseStart)
        phaseStart = clock()
    }

    func accepted(_ count: Int) {
        current.acceptedDrafts = count
        current.acceptanceSeconds = seconds(since: phaseStart)
        repairStart = clock()
    }

    func finishRound() {
        Stream.gpu.synchronize()
        if let repairStart { current.repairSeconds = seconds(since: repairStart) }
        result.rounds.append(current)
    }

    func recordSerial(since start: UInt64?) {
        guard let start else { return }
        Stream.gpu.synchronize()
        result.serialSeconds += seconds(since: start)
        result.serialRounds += 1
    }
}
