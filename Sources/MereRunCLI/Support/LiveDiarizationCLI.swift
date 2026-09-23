import AudioSTT
import Foundation
import MereRunContract

actor LiveDiarizationSession {
    private let inference: Nemotron3DiarizationStreamingSession
    private let modelID: String
    private let latency: String
    private var sampleCount = 0
    private var speakers = Set<Int>()

    init(inference: Nemotron3DiarizationStreamingSession, modelID: String, latency: String) {
        self.inference = inference
        self.modelID = modelID
        self.latency = latency
    }

    func ready(inputFormat: String) -> DiarizationStreamEvent {
        DiarizationStreamEvent(
            type: .ready, model: modelID, latency: latency,
            sampleRate: 16_000, inputFormat: inputFormat
        )
    }

    func feed(samples: [Float]) throws -> [DiarizationStreamEvent] {
        sampleCount += samples.count
        return activityEvents(from: try inference.feed(samples: samples))
    }

    func finish(reason: String) throws -> [DiarizationStreamEvent] {
        let remaining = try inference.finish()
        return activityEvents(from: remaining) + [DiarizationStreamEvent(
            type: .final,
            audioSeconds: Double(sampleCount) / 16_000,
            speakerCount: speakers.count,
            reason: reason
        )]
    }

    private func activityEvents(
        from chunks: [Nemotron3DiarizationStreamChunk]
    ) -> [DiarizationStreamEvent] {
        chunks.map { chunk in
            let segments = chunk.segments.map { segment in
                speakers.insert(segment.speaker)
                return DiarizationStreamEvent.Segment(
                    speakerIndex: segment.speaker,
                    startSeconds: Double(segment.start),
                    endSeconds: Double(segment.end)
                )
            }
            return DiarizationStreamEvent(
                type: .activity,
                startSeconds: Double(chunk.startSeconds),
                endSeconds: min(Double(chunk.endSeconds), Double(sampleCount) / 16_000),
                audioSeconds: Double(sampleCount) / 16_000,
                speakerCount: speakers.count,
                segments: segments
            )
        }
    }
}

enum LiveDiarizationEventWriter {
    static func encode(_ event: DiarizationStreamEvent) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(event)
        data.append(0x0A)
        return data
    }

    static func write(_ event: DiarizationStreamEvent) throws {
        FileHandle.standardOutput.write(try encode(event))
    }
}
