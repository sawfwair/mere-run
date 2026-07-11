import Foundation

public enum MuScriptorMIDI {
    private struct MIDIEvent {
        let tick: Int
        let order: Int
        let bytes: [UInt8]
    }

    /// Encodes a format-1 Standard MIDI File with one track per detected instrument.
    public static func encode(
        notes: [MuScriptorNote],
        tempoBPM: Int = 120,
        velocity: Int = 100
    ) throws -> Data {
        guard tempoBPM > 0 else { throw MuScriptorError.invalidMIDI("tempo must be positive") }
        guard (1...127).contains(velocity) else {
            throw MuScriptorError.invalidMIDI("velocity must be between 1 and 127")
        }

        let ticksPerBeat = 480
        let microsecondsPerBeat = 60_000_000 / tempoBPM
        let ticksPerSecond = Double(ticksPerBeat) * 1_000_000 / Double(microsecondsPerBeat)
        let grouped = Dictionary(grouping: notes) { $0.instrument }
        let trackNames = grouped.keys.sorted()

        var result = Data("MThd".utf8)
        result.appendBigEndian(UInt32(6))
        result.appendBigEndian(UInt16(1))
        result.appendBigEndian(UInt16(trackNames.count + 1))
        result.appendBigEndian(UInt16(ticksPerBeat))

        let tempoBytes: [UInt8] = [
            0x00, 0xFF, 0x51, 0x03,
            UInt8((microsecondsPerBeat >> 16) & 0xFF),
            UInt8((microsecondsPerBeat >> 8) & 0xFF),
            UInt8(microsecondsPerBeat & 0xFF),
            0x00, 0xFF, 0x2F, 0x00,
        ]
        result.appendChunk(type: "MTrk", payload: Data(tempoBytes))

        var nextMelodicChannel = 0
        for name in trackNames {
            let trackNotes = grouped[name] ?? []
            let isDrum = trackNotes.first?.isDrum == true
            let channel: Int
            if isDrum {
                channel = 9
            } else {
                if nextMelodicChannel == 9 { nextMelodicChannel += 1 }
                channel = min(nextMelodicChannel, 15)
                nextMelodicChannel += 1
            }

            var events: [MIDIEvent] = []
            if !isDrum, let program = trackNotes.first?.program {
                events.append(MIDIEvent(
                    tick: 0,
                    order: 0,
                    bytes: [UInt8(0xC0 | channel), UInt8(clamping: program)]
                ))
            }
            for note in trackNotes {
                let start = max(0, Int((note.onset * ticksPerSecond).rounded()))
                let end = max(start + 1, Int((note.offset * ticksPerSecond).rounded()))
                events.append(MIDIEvent(
                    tick: start,
                    order: 2,
                    bytes: [UInt8(0x90 | channel), UInt8(clamping: note.pitch), UInt8(velocity)]
                ))
                events.append(MIDIEvent(
                    tick: end,
                    order: 1,
                    bytes: [UInt8(0x80 | channel), UInt8(clamping: note.pitch), 0]
                ))
            }
            events.sort { ($0.tick, $0.order) < ($1.tick, $1.order) }

            var payload = Data()
            let nameBytes = Array(name.utf8)
            payload.append(0)
            payload.append(contentsOf: [0xFF, 0x03])
            payload.appendVariableLength(nameBytes.count)
            payload.append(contentsOf: nameBytes)

            var previousTick = 0
            for event in events {
                payload.appendVariableLength(event.tick - previousTick)
                payload.append(contentsOf: event.bytes)
                previousTick = event.tick
            }
            payload.append(contentsOf: [0x00, 0xFF, 0x2F, 0x00])
            result.appendChunk(type: "MTrk", payload: payload)
        }
        return result
    }
}

private extension Data {
    mutating func appendBigEndian(_ value: UInt16) {
        append(UInt8((value >> 8) & 0xFF))
        append(UInt8(value & 0xFF))
    }

    mutating func appendBigEndian(_ value: UInt32) {
        append(UInt8((value >> 24) & 0xFF))
        append(UInt8((value >> 16) & 0xFF))
        append(UInt8((value >> 8) & 0xFF))
        append(UInt8(value & 0xFF))
    }

    mutating func appendChunk(type: String, payload: Data) {
        append(Data(type.utf8))
        appendBigEndian(UInt32(payload.count))
        append(payload)
    }

    mutating func appendVariableLength(_ rawValue: Int) {
        var value = Swift.max(0, rawValue)
        var bytes = [UInt8(value & 0x7F)]
        value /= 128
        while value > 0 {
            bytes.append(UInt8((value & 0x7F) | 0x80))
            value /= 128
        }
        append(contentsOf: bytes.reversed())
    }
}
