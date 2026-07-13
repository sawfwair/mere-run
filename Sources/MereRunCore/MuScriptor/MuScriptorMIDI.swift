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
        try encode(
            notes: notes,
            tempoBPM: Double(tempoBPM),
            context: nil,
            velocity: velocity
        )
    }

    /// Encodes detected musical context into the conductor track while preserving
    /// MuScriptor's absolute note timing relative to the source audio.
    public static func encode(
        notes: [MuScriptorNote],
        context: MuScriptorMusicalContext,
        velocity: Int = 100
    ) throws -> Data {
        try encode(
            notes: notes,
            tempoBPM: context.tempo?.bpm ?? 120,
            context: context,
            velocity: velocity
        )
    }

    private static func encode(
        notes: [MuScriptorNote],
        tempoBPM: Double,
        context: MuScriptorMusicalContext?,
        velocity: Int
    ) throws -> Data {
        guard tempoBPM.isFinite, tempoBPM > 0 else {
            throw MuScriptorError.invalidMIDI("tempo must be positive")
        }
        guard (1...127).contains(velocity) else {
            throw MuScriptorError.invalidMIDI("velocity must be between 1 and 127")
        }

        let ticksPerBeat = 480
        let microsecondsPerBeat = Int((60_000_000 / tempoBPM).rounded())
        guard (1...0xFF_FFFF).contains(microsecondsPerBeat) else {
            throw MuScriptorError.invalidMIDI("tempo is outside the Standard MIDI File range")
        }
        let ticksPerSecond = Double(ticksPerBeat) * 1_000_000 / Double(microsecondsPerBeat)
        let grouped = Dictionary(grouping: notes) { $0.instrument }
        let trackNames = grouped.keys.sorted()

        var result = Data("MThd".utf8)
        result.appendBigEndian(UInt32(6))
        result.appendBigEndian(UInt16(1))
        result.appendBigEndian(UInt16(trackNames.count + 1))
        result.appendBigEndian(UInt16(ticksPerBeat))

        var conductorBytes: [UInt8] = [
            0x00, 0xFF, 0x51, 0x03,
            UInt8((microsecondsPerBeat >> 16) & 0xFF),
            UInt8((microsecondsPerBeat >> 8) & 0xFF),
            UInt8(microsecondsPerBeat & 0xFF),
        ]
        let timeSignatureMeta: [UInt8]?
        if let timeSignature = context?.timeSignature {
            guard (1...255).contains(timeSignature.numerator),
                  let denominatorPower = denominatorPower(timeSignature.denominator) else {
                throw MuScriptorError.invalidMIDI("invalid time signature \(timeSignature.name)")
            }
            let metronomeClocks = timeSignature.denominator == 8 && timeSignature.numerator.isMultiple(of: 3)
                ? 36
                : 24
            let meta: [UInt8] = [
                0xFF, 0x58, 0x04,
                UInt8(timeSignature.numerator), UInt8(denominatorPower), UInt8(metronomeClocks), 0x08,
            ]
            timeSignatureMeta = meta
            conductorBytes.append(0)
            conductorBytes += meta
        } else {
            timeSignatureMeta = nil
        }
        if let keySignature = context?.keySignature {
            guard (-7...7).contains(keySignature.sharpsFlats) else {
                throw MuScriptorError.invalidMIDI("key signature must be between seven flats and seven sharps")
            }
            conductorBytes += [
                0x00, 0xFF, 0x59, 0x02,
                UInt8(bitPattern: Int8(keySignature.sharpsFlats)), keySignature.isMinor ? 0x01 : 0x00,
            ]
        }
        if let downbeat = context?.timeSignature?.downbeatOffsetSeconds, downbeat > 0 {
            let marker = Array("Detected downbeat".utf8)
            conductorBytes.appendVariableLength(max(0, Int((downbeat * ticksPerSecond).rounded())))
            if let timeSignatureMeta {
                conductorBytes += timeSignatureMeta
            }
            conductorBytes.append(0)
            conductorBytes += [0xFF, 0x06]
            conductorBytes.appendVariableLength(marker.count)
            conductorBytes += marker
        }
        conductorBytes += [0x00, 0xFF, 0x2F, 0x00]
        result.appendChunk(type: "MTrk", payload: Data(conductorBytes))

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

    private static func denominatorPower(_ denominator: Int) -> Int? {
        guard denominator > 0, denominator.nonzeroBitCount == 1 else { return nil }
        return denominator.trailingZeroBitCount
    }
}

private extension Array where Element == UInt8 {
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
