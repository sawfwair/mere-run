import Foundation

public struct MuScriptorNote: Codable, Hashable, Sendable {
    public let instrument: String
    public let program: Int
    public let pitch: Int
    public let onset: Double
    public var offset: Double
    public let isDrum: Bool

    public init(
        instrument: String,
        program: Int,
        pitch: Int,
        onset: Double,
        offset: Double,
        isDrum: Bool
    ) {
        self.instrument = instrument
        self.program = program
        self.pitch = pitch
        self.onset = onset
        self.offset = offset
        self.isDrum = isDrum
    }
}

public struct MuScriptorTranscriptionEvent: Codable, Hashable, Sendable {
    public enum Kind: String, Codable, Sendable {
        case start
        case end
    }

    public let type: Kind
    public let pitch: Int?
    public let startTime: Double?
    public let endTime: Double?
    public let index: Int?
    public let startEventIndex: Int?
    public let instrument: String?

    enum CodingKeys: String, CodingKey {
        case type
        case pitch
        case startTime = "start_time"
        case endTime = "end_time"
        case index
        case startEventIndex = "start_event_index"
        case instrument
    }

    static func start(index: Int, pitch: Int, time: Double, instrument: String) -> Self {
        Self(
            type: .start,
            pitch: pitch,
            startTime: time,
            endTime: nil,
            index: index,
            startEventIndex: nil,
            instrument: instrument
        )
    }

    static func end(index: Int, time: Double) -> Self {
        Self(
            type: .end,
            pitch: nil,
            startTime: nil,
            endTime: time,
            index: nil,
            startEventIndex: index,
            instrument: nil
        )
    }
}

public struct MuScriptorTranscription: Codable, Hashable, Sendable {
    public let notes: [MuScriptorNote]
    public let events: [MuScriptorTranscriptionEvent]
    public let chunkCount: Int

    enum CodingKeys: String, CodingKey {
        case notes
        case events
        case chunkCount = "chunk_count"
    }
}

public enum MuScriptorInstruments {
    public static let groupIDs: [String: Int] = [
        "acoustic_piano": 0,
        "electric_piano": 1,
        "chromatic_percussion": 2,
        "organ": 3,
        "acoustic_guitar": 4,
        "clean_electric_guitar": 5,
        "distorted_electric_guitar": 6,
        "acoustic_bass": 7,
        "electric_bass": 8,
        "violin": 9,
        "viola": 10,
        "cello": 11,
        "contrabass": 12,
        "orchestral_harp": 13,
        "timpani": 14,
        "string_ensemble": 15,
        "synth_strings": 16,
        "voice": 17,
        "orchestra_hit": 18,
        "trumpet": 19,
        "trombone": 20,
        "tuba": 21,
        "french_horn": 22,
        "brass_section": 23,
        "soprano_and_alto_sax": 24,
        "tenor_sax": 25,
        "baritone_sax": 26,
        "oboe": 27,
        "english_horn": 28,
        "bassoon": 29,
        "clarinet": 30,
        "flutes": 31,
        "synth_lead": 32,
        "synth_pad": 33,
        "drums": 36,
    ]

    /// Representative General MIDI program emitted for each learned group.
    public static let programs: [String: Int] = [
        "acoustic_piano": 0, "electric_piano": 2, "chromatic_percussion": 8,
        "organ": 16, "acoustic_guitar": 24, "clean_electric_guitar": 26,
        "distorted_electric_guitar": 29, "acoustic_bass": 32, "electric_bass": 33,
        "violin": 40, "viola": 41, "cello": 42, "contrabass": 43,
        "orchestral_harp": 46, "timpani": 47, "string_ensemble": 48,
        "synth_strings": 50, "voice": 52, "orchestra_hit": 55, "trumpet": 56,
        "trombone": 57, "tuba": 58, "french_horn": 60, "brass_section": 61,
        "soprano_and_alto_sax": 64, "tenor_sax": 66, "baritone_sax": 67,
        "oboe": 68, "english_horn": 69, "bassoon": 70, "clarinet": 71,
        "flutes": 72, "synth_lead": 80, "synth_pad": 88, "drums": 128,
    ]

    private static let namesByProgram = Dictionary(
        uniqueKeysWithValues: programs.map { ($0.value, $0.key) }
    )

    public static var names: [String] { groupIDs.keys.sorted() }

    public static func resolve(_ values: [String]) throws -> [String] {
        try values.map { rawValue in
            let value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if groupIDs[value] != nil { return value }
            let matches = names.filter { $0.contains(value) }
            guard matches.count == 1, let match = matches.first else {
                throw MuScriptorError.invalidInstrument(rawValue)
            }
            return match
        }
    }

    static func name(for program: Int) -> String {
        namesByProgram[program] ?? "program_\(program)"
    }
}

public struct MuScriptorTokenDecoder {
    private enum Event {
        case special
        case shift(Int)
        case pitch(Int)
        case velocity(Int)
        case tie
        case program(Int)
        case drum(Int)
    }

    private struct NoteKey: Hashable {
        let program: Int
        let pitch: Int
    }

    private struct OpenNote {
        let index: Int
        let instrument: String
        let program: Int
        let pitch: Int
        let onset: Double
    }

    public init() {}

    public func decode(chunks: [[Int]]) throws -> MuScriptorTranscription {
        var openNotes: [NoteKey: OpenNote] = [:]
        var notes: [MuScriptorNote] = []
        var outputEvents: [MuScriptorTranscriptionEvent] = []
        var nextIndex = 0

        func close(_ key: NoteKey, at time: Double) {
            guard let open = openNotes.removeValue(forKey: key) else { return }
            let end = max(time, open.onset + 0.01)
            notes.append(MuScriptorNote(
                instrument: open.instrument,
                program: open.program,
                pitch: open.pitch,
                onset: open.onset,
                offset: end,
                isDrum: false
            ))
            outputEvents.append(.end(index: open.index, time: end))
        }

        var finalChunkLeftInPrologue = false
        for (chunkIndex, tokens) in chunks.enumerated() {
            let seekTime = Double(chunkIndex) * 5
            let nextSeekTime = chunkIndex + 1 < chunks.count ? seekTime + 5 : nil
            let startTick = chunkIndex * 500
            var tick = startTick
            var program: Int?
            var velocity: Int?
            var inPrologue = true
            var skipRest = false
            var tieSet: Set<NoteKey> = []

            for token in tokens {
                let decoded = try event(for: token)
                if inPrologue {
                    switch decoded {
                    case .tie:
                        inPrologue = false
                        velocity = nil
                        for key in Array(openNotes.keys) where !tieSet.contains(key) {
                            close(key, at: seekTime)
                        }
                    case .shift:
                        inPrologue = false
                        skipRest = true
                        for key in Array(openNotes.keys) {
                            close(key, at: seekTime)
                        }
                    case .program(let value):
                        program = value
                    case .pitch(let value):
                        if let program { tieSet.insert(NoteKey(program: program, pitch: value)) }
                    case .special, .velocity, .drum:
                        break
                    }
                    continue
                }
                if skipRest { continue }

                switch decoded {
                case .shift(let value):
                    if value > 0 { tick = startTick + value }
                case .program(let value):
                    program = value
                case .velocity(let value):
                    velocity = value
                case .drum(let pitch):
                    let time = Double(tick) / 100
                    if nextSeekTime == nil || time < nextSeekTime! {
                        let index = nextIndex
                        nextIndex += 1
                        outputEvents.append(.start(index: index, pitch: pitch, time: time, instrument: "drums"))
                        outputEvents.append(.end(index: index, time: time + 0.01))
                        notes.append(MuScriptorNote(
                            instrument: "drums",
                            program: 128,
                            pitch: pitch,
                            onset: time,
                            offset: time + 0.01,
                            isDrum: true
                        ))
                    }
                case .pitch(let pitch):
                    guard let program, let velocity else { continue }
                    let time = Double(tick) / 100
                    if let nextSeekTime, time >= nextSeekTime { continue }
                    let key = NoteKey(program: program, pitch: pitch)
                    if openNotes[key] != nil { close(key, at: time) }
                    if velocity > 0 {
                        let index = nextIndex
                        nextIndex += 1
                        let instrument = MuScriptorInstruments.name(for: program)
                        openNotes[key] = OpenNote(
                            index: index,
                            instrument: instrument,
                            program: program,
                            pitch: pitch,
                            onset: time
                        )
                        outputEvents.append(.start(
                            index: index,
                            pitch: pitch,
                            time: time,
                            instrument: instrument
                        ))
                    }
                case .tie, .special:
                    continue
                }
            }
            finalChunkLeftInPrologue = inPrologue
            if inPrologue {
                for key in Array(openNotes.keys) { close(key, at: seekTime) }
            }
        }

        if !finalChunkLeftInPrologue {
            for key in Array(openNotes.keys) {
                guard let open = openNotes[key] else { continue }
                close(key, at: open.onset + 0.01)
            }
        }

        return MuScriptorTranscription(
            notes: trimOverlaps(notes),
            events: outputEvents,
            chunkCount: chunks.count
        )
    }

    private func event(for token: Int) throws -> Event {
        switch token {
        case 0...2: .special
        case 3...1_003: .shift(token - 3)
        case 1_004...1_131: .pitch(token - 1_004)
        case 1_132...1_133: .velocity(token - 1_132)
        case 1_134: .tie
        case 1_135...1_264: .program(token - 1_135)
        case 1_265...1_392: .drum(token - 1_265)
        default: throw MuScriptorError.invalidToken(token)
        }
    }

    private func trimOverlaps(_ input: [MuScriptorNote]) -> [MuScriptorNote] {
        let groups = Dictionary(grouping: input) { note in
            "\(note.program):\(note.pitch):\(note.isDrum)"
        }
        var result: [MuScriptorNote] = []
        for var group in groups.values {
            group.sort { $0.onset < $1.onset }
            if group.count > 1 {
                for index in 1..<group.count where group[index - 1].offset > group[index].onset {
                    group[index - 1].offset = group[index].onset
                }
            }
            result.append(contentsOf: group.filter { $0.onset < $0.offset })
        }
        return result.sorted {
            ($0.onset, $0.isDrum ? 1 : 0, $0.program, $0.pitch, $0.offset)
                < ($1.onset, $1.isDrum ? 1 : 0, $1.program, $1.pitch, $1.offset)
        }
    }
}
