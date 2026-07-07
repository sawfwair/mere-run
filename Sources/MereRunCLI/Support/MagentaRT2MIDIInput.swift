import Foundation
import MereRunCore

#if canImport(CoreMIDI) && os(macOS)
@preconcurrency import CoreMIDI
#endif

struct MagentaRT2MIDISource: Equatable {
    let name: String
    let manufacturer: String?
    let model: String?
    let uniqueID: Int32?

    var displayName: String {
        var parts = [name]
        if let manufacturer, !manufacturer.isEmpty {
            parts.append(manufacturer)
        }
        if let model, !model.isEmpty, model != name {
            parts.append(model)
        }
        if let uniqueID {
            parts.append("id \(uniqueID)")
        }
        return parts.joined(separator: " | ")
    }
}

struct MagentaRT2MIDIInputConfiguration: Equatable {
    let source: String
    let channel: Int?
    let noteOffset: Int
    let ccMappings: [MagentaRT2MIDICCMapping]

    static func parseChannel(_ value: String) throws -> Int? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed.lowercased() == "all" {
            return nil
        }
        guard let parsed = Int(trimmed), (1...16).contains(parsed) else {
            throw MagentaRT2MIDIInputError.invalidChannel(value)
        }
        return parsed
    }
}

struct MagentaRT2MIDICCMapping: Equatable {
    enum Target: Equatable {
        case temperature
        case topK
        case cfgMusicCoCa
        case cfgNotes
        case cfgDrums
        case drumless
        case unmaskWidth
        case seedRotation
        case onsetMode

        init(rawValue: String) throws {
            switch rawValue.lowercased() {
            case "temp", "temperature":
                self = .temperature
            case "topk", "top-k":
                self = .topK
            case "mc", "musiccoca", "cfg-musiccoca":
                self = .cfgMusicCoCa
            case "notes", "cfg-notes":
                self = .cfgNotes
            case "drums", "cfg-drums":
                self = .cfgDrums
            case "drumless":
                self = .drumless
            case "unmask", "unmask-width":
                self = .unmaskWidth
            case "seed", "seed-rotation":
                self = .seedRotation
            case "onset", "onset-mode":
                self = .onsetMode
            default:
                throw MagentaRT2MIDIInputError.invalidCCMapping(rawValue, "unsupported target")
            }
        }
    }

    let controller: UInt8
    let target: Target
    let minimum: Float
    let maximum: Float

    static func parse(_ rawValue: String) throws -> Self {
        let parts = rawValue.split(separator: "=", maxSplits: 1).map(String.init)
        guard parts.count == 2 else {
            throw MagentaRT2MIDIInputError.invalidCCMapping(rawValue, "expected cc=target:min:max")
        }
        guard let controller = UInt8(parts[0]), controller < 128 else {
            throw MagentaRT2MIDIInputError.invalidCCMapping(rawValue, "CC number must be 0-127")
        }
        let targetParts = parts[1].split(separator: ":", omittingEmptySubsequences: false).map(String.init)
        guard !targetParts.isEmpty else {
            throw MagentaRT2MIDIInputError.invalidCCMapping(rawValue, "missing target")
        }
        let target = try Target(rawValue: targetParts[0])
        let minimum: Float
        let maximum: Float
        if targetParts.count == 1 {
            minimum = 0
            maximum = 1
        } else if targetParts.count == 3,
                  let parsedMinimum = Float(targetParts[1]),
                  let parsedMaximum = Float(targetParts[2]) {
            minimum = parsedMinimum
            maximum = parsedMaximum
        } else {
            throw MagentaRT2MIDIInputError.invalidCCMapping(rawValue, "expected cc=target or cc=target:min:max")
        }
        return Self(controller: controller, target: target, minimum: minimum, maximum: maximum)
    }

    func apply(rawValue: UInt8, to queue: MagentaRT2LiveControlQueue) {
        let scaled = scaledValue(rawValue)
        switch target {
        case .temperature:
            queue.updateControls { $0.temperature = scaled }
        case .topK:
            queue.updateControls { $0.topK = max(0, Int32(scaled.rounded())) }
        case .cfgMusicCoCa:
            queue.updateControls { $0.cfgMusicCoCa = scaled }
        case .cfgNotes:
            queue.updateControls { $0.cfgNotes = scaled }
        case .cfgDrums:
            queue.updateControls { $0.cfgDrums = scaled }
        case .drumless:
            queue.updateControls { $0.drumless = rawValue >= 64 }
        case .unmaskWidth:
            queue.updateControls { $0.unmaskWidth = max(0, Int32(scaled.rounded())) }
        case .seedRotation:
            queue.updateControls { $0.seedRotation = Int32(scaled.rounded()) }
        case .onsetMode:
            queue.setOnsetMode(rawValue >= 64 ? 1 : 0)
        }
    }

    func scaledValue(_ rawValue: UInt8) -> Float {
        minimum + ((maximum - minimum) * Float(rawValue) / 127)
    }
}

/// Channel-voice events surfaced to the realtime control queue. Note-on with
/// velocity zero is normalized to note-off per the MIDI 1.0 specification.
enum MagentaRT2MIDIEvent: Equatable, Sendable {
    case noteOn(channel: Int, note: UInt8, velocity: UInt8)
    case noteOff(channel: Int, note: UInt8)
    case controlChange(channel: Int, controller: UInt8, value: UInt8)
}

/// Incremental MIDI 1.0 byte-stream parser.
///
/// CoreMIDI packets may carry several messages, use running status (a status
/// byte followed by multiple data-byte groups), and interleave system-realtime
/// bytes (clock 0xF8, active sensing 0xFE) anywhere — including inside a
/// message. The parser is stateful so running status survives packet
/// boundaries; feed each packet's bytes in arrival order.
struct MagentaRT2MIDIStreamParser {
    private var runningStatus: UInt8?
    private var pendingData: [UInt8] = []

    mutating func consume(_ bytes: some Sequence<UInt8>, onEvent: (MagentaRT2MIDIEvent) -> Void) {
        for byte in bytes {
            if byte >= 0xF8 {
                // System realtime: ignored, does not disturb running status.
                continue
            }
            if byte >= 0xF0 {
                // System common (incl. SysEx start/end) cancels running status.
                runningStatus = nil
                pendingData.removeAll(keepingCapacity: true)
                continue
            }
            if byte >= 0x80 {
                runningStatus = byte
                pendingData.removeAll(keepingCapacity: true)
                continue
            }
            guard let status = runningStatus else {
                // Data byte with no status context (e.g. inside SysEx): drop.
                continue
            }
            pendingData.append(byte)
            if pendingData.count == Self.dataByteCount(for: status) {
                emit(status: status, data: pendingData, onEvent: onEvent)
                // Keep runningStatus: subsequent data bytes reuse it.
                pendingData.removeAll(keepingCapacity: true)
            }
        }
    }

    private static func dataByteCount(for status: UInt8) -> Int {
        switch status & 0xF0 {
        case 0xC0, 0xD0:
            return 1
        default:
            return 2
        }
    }

    private func emit(status: UInt8, data: [UInt8], onEvent: (MagentaRT2MIDIEvent) -> Void) {
        let channel = Int(status & 0x0F) + 1
        switch status & 0xF0 {
        case 0x90 where data[1] > 0:
            onEvent(.noteOn(channel: channel, note: data[0], velocity: data[1]))
        case 0x80, 0x90:
            onEvent(.noteOff(channel: channel, note: data[0]))
        case 0xB0:
            onEvent(.controlChange(channel: channel, controller: data[0], value: data[1]))
        default:
            break
        }
    }
}

enum MagentaRT2MIDIInputError: Error, LocalizedError {
    case unsupportedPlatform
    case noSources
    case sourceNotFound(String, available: [String])
    case ambiguousSource(String, matches: [String])
    case coreMIDIStatus(String, Int32)
    case invalidChannel(String)
    case invalidCCMapping(String, String)

    var errorDescription: String? {
        switch self {
        case .unsupportedPlatform:
            return "MIDI input requires CoreMIDI on macOS."
        case .noSources:
            return "No MIDI input sources are available."
        case .sourceNotFound(let requested, let available):
            let suffix = available.isEmpty ? "" : " Available: \(available.joined(separator: ", "))"
            return "MIDI input source not found: \(requested).\(suffix)"
        case .ambiguousSource(let requested, let matches):
            return "MIDI input '\(requested)' matches several sources: \(matches.joined(separator: ", ")). Use the unique ID."
        case .coreMIDIStatus(let operation, let status):
            return "\(operation) failed with CoreMIDI status \(status)."
        case .invalidChannel(let value):
            return "--midi-channel must be all or an integer from 1 to 16. Received: \(value)"
        case .invalidCCMapping(let value, let reason):
            return "Invalid --midi-cc mapping '\(value)': \(reason)."
        }
    }
}

#if canImport(CoreMIDI) && os(macOS)
final class MagentaRT2MIDIInput: @unchecked Sendable {
    private let configuration: MagentaRT2MIDIInputConfiguration
    private let controls: MagentaRT2LiveControlQueue
    private let ccMappings: [UInt8: MagentaRT2MIDICCMapping]
    private let eventHandler: (@Sendable (MagentaRT2MIDIEvent) -> Void)?
    private let rawPacketHandler: (@Sendable ([UInt8]) -> Void)?
    private let sourceEndpoint: MIDIEndpointRef
    private var client = MIDIClientRef()
    private var inputPort = MIDIPortRef()
    /// Touched only on the CoreMIDI read thread.
    private var parser = MagentaRT2MIDIStreamParser()

    let sourceDisplayName: String

    init(
        configuration: MagentaRT2MIDIInputConfiguration,
        controls: MagentaRT2LiveControlQueue,
        eventHandler: (@Sendable (MagentaRT2MIDIEvent) -> Void)? = nil,
        rawPacketHandler: (@Sendable ([UInt8]) -> Void)? = nil
    ) throws {
        self.configuration = configuration
        self.controls = controls
        self.eventHandler = eventHandler
        self.rawPacketHandler = rawPacketHandler
        var mappings: [UInt8: MagentaRT2MIDICCMapping] = [:]
        for mapping in configuration.ccMappings {
            mappings[mapping.controller] = mapping
        }
        self.ccMappings = mappings
        let source = try Self.resolveSource(configuration.source)
        self.sourceEndpoint = source.endpoint
        self.sourceDisplayName = source.description.displayName
        try createClient()
    }

    deinit {
        stop()
    }

    static func availableSources() throws -> [MagentaRT2MIDISource] {
        (0..<MIDIGetNumberOfSources()).map { index in
            describeSource(MIDIGetSource(index)).description
        }
    }

    func stop() {
        if inputPort != 0 {
            MIDIPortDisconnectSource(inputPort, sourceEndpoint)
            MIDIPortDispose(inputPort)
            inputPort = 0
        }
        if client != 0 {
            MIDIClientDispose(client)
            client = 0
        }
    }

    private func createClient() throws {
        var status = MIDIClientCreate("mere.run Magenta RT2" as CFString, nil, nil, &client)
        guard status == noErr else {
            throw MagentaRT2MIDIInputError.coreMIDIStatus("MIDIClientCreate", status)
        }
        status = MIDIInputPortCreate(
            client,
            "Magenta RT2 MIDI In" as CFString,
            Self.readProc,
            Unmanaged.passUnretained(self).toOpaque(),
            &inputPort
        )
        guard status == noErr else {
            throw MagentaRT2MIDIInputError.coreMIDIStatus("MIDIInputPortCreate", status)
        }
        status = MIDIPortConnectSource(inputPort, sourceEndpoint, nil)
        guard status == noErr else {
            throw MagentaRT2MIDIInputError.coreMIDIStatus("MIDIPortConnectSource", status)
        }
    }

    private static func resolveSource(_ requested: String) throws -> (endpoint: MIDIEndpointRef, description: MagentaRT2MIDISource) {
        let sources = (0..<MIDIGetNumberOfSources()).map { describeSource(MIDIGetSource($0)) }
        guard !sources.isEmpty else {
            throw MagentaRT2MIDIInputError.noSources
        }
        let normalized = requested.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let requestedID = Int32(normalized),
           let exactID = sources.first(where: { $0.description.uniqueID == requestedID }) {
            return exactID
        }
        if let exactName = sources.first(where: { $0.description.name.lowercased() == normalized }) {
            return exactName
        }
        let contained = sources.filter { $0.description.displayName.lowercased().contains(normalized) }
        if contained.count == 1 {
            return contained[0]
        }
        if contained.count > 1 {
            throw MagentaRT2MIDIInputError.ambiguousSource(
                requested,
                matches: contained.map(\.description.displayName)
            )
        }
        throw MagentaRT2MIDIInputError.sourceNotFound(
            requested,
            available: sources.map(\.description.displayName)
        )
    }

    private static func describeSource(_ endpoint: MIDIEndpointRef) -> (
        endpoint: MIDIEndpointRef,
        description: MagentaRT2MIDISource
    ) {
        let name = stringProperty(endpoint, kMIDIPropertyDisplayName)
            ?? stringProperty(endpoint, kMIDIPropertyName)
            ?? "MIDI source \(endpoint)"
        return (
            endpoint,
            MagentaRT2MIDISource(
                name: name,
                manufacturer: stringProperty(endpoint, kMIDIPropertyManufacturer),
                model: stringProperty(endpoint, kMIDIPropertyModel),
                uniqueID: integerProperty(endpoint, kMIDIPropertyUniqueID)
            )
        )
    }

    private static func stringProperty(_ object: MIDIObjectRef, _ property: CFString) -> String? {
        var unmanaged: Unmanaged<CFString>?
        let status = MIDIObjectGetStringProperty(object, property, &unmanaged)
        guard status == noErr, let value = unmanaged?.takeRetainedValue() else {
            return nil
        }
        return value as String
    }

    private static func integerProperty(_ object: MIDIObjectRef, _ property: CFString) -> Int32? {
        var value: Int32 = 0
        let status = MIDIObjectGetIntegerProperty(object, property, &value)
        guard status == noErr else { return nil }
        return value
    }

    private static let readProc: MIDIReadProc = { packetList, refCon, _ in
        guard let refCon else { return }
        let input = Unmanaged<MagentaRT2MIDIInput>.fromOpaque(refCon).takeUnretainedValue()
        input.handle(packetList: packetList)
    }

    private func handle(packetList: UnsafePointer<MIDIPacketList>) {
        // MIDIPacket is variable-length; walking a value copy with
        // MIDIPacketNext reads past the copy's storage once numPackets > 1.
        // unsafeSequence() walks the real buffer.
        for packetPtr in packetList.unsafeSequence() {
            let length = Int(packetPtr.pointee.length)
            guard length > 0 else { continue }
            let dataOffset = MemoryLayout<MIDIPacket>.offset(of: \.data) ?? 0
            let data = UnsafeRawBufferPointer(
                start: UnsafeRawPointer(packetPtr) + dataOffset,
                count: length
            )
            let bytes = Array(data)
            rawPacketHandler?(bytes)
            parser.consume(bytes) { [self] event in
                handle(event: event)
            }
        }
    }

    private func handle(event: MagentaRT2MIDIEvent) {
        eventHandler?(event)
        switch event {
        case .noteOn(let channel, let note, _):
            guard accepts(channel: channel) else { return }
            let adjusted = Int(note) + configuration.noteOffset
            guard (0..<132).contains(adjusted) else { return }
            controls.enqueueNoteOn(Int32(adjusted))
        case .noteOff(let channel, let note):
            guard accepts(channel: channel) else { return }
            let adjusted = Int(note) + configuration.noteOffset
            guard (0..<132).contains(adjusted) else { return }
            controls.enqueueNoteOff(Int32(adjusted))
        case .controlChange(let channel, let controller, let value):
            guard accepts(channel: channel), let mapping = ccMappings[controller] else { return }
            mapping.apply(rawValue: value, to: controls)
        }
    }

    private func accepts(channel: Int) -> Bool {
        configuration.channel.map { $0 == channel } ?? true
    }
}
#else
final class MagentaRT2MIDIInput: @unchecked Sendable {
    let sourceDisplayName = ""

    init(
        configuration: MagentaRT2MIDIInputConfiguration,
        controls: MagentaRT2LiveControlQueue,
        eventHandler: (@Sendable (MagentaRT2MIDIEvent) -> Void)? = nil,
        rawPacketHandler: (@Sendable ([UInt8]) -> Void)? = nil
    ) throws {
        _ = configuration
        _ = controls
        _ = eventHandler
        _ = rawPacketHandler
        throw MagentaRT2MIDIInputError.unsupportedPlatform
    }

    static func availableSources() throws -> [MagentaRT2MIDISource] {
        throw MagentaRT2MIDIInputError.unsupportedPlatform
    }

    func stop() {}
}
#endif
