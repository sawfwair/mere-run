import Foundation
import MLX
#if canImport(Darwin)
import Darwin
#endif

/// Synchronized diagnostic timings. Profiling changes scheduling and is excluded
/// from normal throughput receipts.
public struct ChatSpeculationProfile: Codable, Sendable, Hashable {
    public var rounds: [ChatSpeculationRoundProfile] = []
    public var samplingSeconds: Double = 0
    public var serialSeconds: Double = 0
    public var serialRounds: Int = 0
    public var taskPriority: UInt8 = Task.currentPriority.rawValue
    public var thermalState: Int?
}

/// Wall time spent submitting graphs, waiting for their results, and accepting
/// or restoring one speculative round. Submission can include internal GPU waits.
/// Times include host scheduling delays.
public struct ChatSpeculationRoundProfile: Codable, Sendable, Hashable {
    public var draftDepth: Int
    public var hostCPUNumber: Int?
    public var hostQoSClass: UInt32?
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
        guard environment["MERERUN_Q35_MTP_PROFILE"] == "1" else { return nil }
        let profile = Q35MTPProfile()
        #if canImport(Darwin)
        profile.result.thermalState = ProcessInfo.processInfo.thermalState.rawValue
        #endif
        return profile
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
        #if canImport(Darwin)
        var cpu = 0
        if pthread_cpu_number_np(&cpu) == 0 { current.hostCPUNumber = cpu }
        current.hostQoSClass = qos_class_self().rawValue
        #endif
        repairStart = nil
        phaseStart = clock()
    }

    func finishDraft(tokens: MLXArray) {
        current.draftGraphSeconds = seconds(since: phaseStart)
        phaseStart = clock()
        MLX.eval(tokens)
        StreamOrDevice.default.stream.synchronize()
        current.draftWaitSeconds = seconds(since: phaseStart)
        phaseStart = clock()
    }

    func verificationSubmitted() {
        current.verificationGraphSeconds = seconds(since: phaseStart)
        phaseStart = clock()
    }

    func verificationCompleted() {
        StreamOrDevice.default.stream.synchronize()
        current.verificationWaitSeconds = seconds(since: phaseStart)
        phaseStart = clock()
    }

    func accepted(_ count: Int) {
        current.acceptedDrafts = count
        current.acceptanceSeconds = seconds(since: phaseStart)
        repairStart = clock()
    }

    func finishRound() {
        StreamOrDevice.default.stream.synchronize()
        if let repairStart { current.repairSeconds = seconds(since: repairStart) }
        result.rounds.append(current)
    }

    func recordSerial(since start: UInt64?) {
        guard let start else { return }
        StreamOrDevice.default.stream.synchronize()
        result.serialSeconds += seconds(since: start)
        result.serialRounds += 1
    }

    func recordPipeline(seconds: Double, tokens: Int) {
        result.serialSeconds += seconds
        result.serialRounds += tokens
    }
}
