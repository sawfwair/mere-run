import Foundation
#if os(macOS)
import Metal
import os
#endif

/// Opt-in diagnostics. Captures include one complete native decode cycle;
/// no extra evaluation is inserted into the pipelined workload.
package final class SOVDecodeDiagnostics {
    private let captureURL: URL?
    private let captureRound: Int
    private var captured = false
    #if os(macOS)
    private let signposter = OSSignposter(subsystem: "run.mere.sov", category: "decode")
    private var generationInterval: OSSignpostIntervalState?
    private var roundInterval: OSSignpostIntervalState?
    private var stageInterval: OSSignpostIntervalState?
    private var stageName: StaticString = "SOVDraftGraph"
    private var capturing = false
    #endif

    package init(captureURL: URL? = nil, captureRound: Int = 3) {
        self.captureURL = captureURL
        self.captureRound = captureRound
    }

    package func beginGeneration() {
        #if os(macOS)
        generationInterval = signposter.beginInterval("SOVGeneration")
        #endif
    }

    package func beginRound(_ round: Int) throws {
        #if os(macOS)
        roundInterval = signposter.beginInterval("SOVDecodeRound", id: signposter.makeSignpostID(), "round=\(round)")
        if round == captureRound, let captureURL {
            guard !FileManager.default.fileExists(atPath: captureURL.path) else {
                throw SOVTranscribeError.invalid("Metal capture destination already exists; choose a new .gputrace path.")
            }
            try FileManager.default.createDirectory(at: captureURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard let device = MTLCreateSystemDefaultDevice() else {
                throw SOVTranscribeError.invalid("Metal capture requires an Apple GPU.")
            }
            let descriptor = MTLCaptureDescriptor()
            descriptor.captureObject = device
            descriptor.destination = .gpuTraceDocument
            descriptor.outputURL = captureURL
            try MTLCaptureManager.shared().startCapture(with: descriptor)
            capturing = true
        }
        #endif
    }

    package func stage(_ name: StaticString) {
        #if os(macOS)
        if let stageInterval { signposter.endInterval(stageName, stageInterval) }
        stageName = name
        stageInterval = signposter.beginInterval(name)
        #endif
    }

    package func endRound() {
        #if os(macOS)
        if let stageInterval { signposter.endInterval(stageName, stageInterval); self.stageInterval = nil }
        if let roundInterval { signposter.endInterval("SOVDecodeRound", roundInterval); self.roundInterval = nil }
        if capturing {
            MTLCaptureManager.shared().stopCapture()
            capturing = false
            captured = true
        }
        #endif
    }

    package func finish() throws {
        #if os(macOS)
        if let generationInterval { signposter.endInterval("SOVGeneration", generationInterval); self.generationInterval = nil }
        #endif
        if captureURL != nil && !captured {
            throw SOVTranscribeError.invalid("The requested Metal capture round was not reached.")
        }
    }
}
