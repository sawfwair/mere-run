import Foundation
import MereRunCore

final class MagentaRT2LiveControlQueue: @unchecked Sendable {
    private let lock = NSLock()
    private var controls: MagentaRT2Controls
    private var pendingPrompt: String?
    private var pendingNoteEvents: [MagentaRT2LiveNoteEvent] = []
    private var pendingNoteOn: [Int32] = []
    private var pendingNoteOff: [Int32] = []
    private var pendingOnsetMode: Int32?
    private var pendingControls = false
    private var pendingReset = false
    private(set) var shouldStop = false

    let helpText = """
    Interactive steering enabled. Commands:
      prompt <text>
      style streaming|full
      temp <value> | topk <value> | mc <value> | notes <value> | drums <value>
      noteon <0-131> | noteoff <0-131> | onset 0|1
      drumless on|off | unmask <value> | seed <value> | reset | quit | help

    """

    init(initialControls: MagentaRT2Controls) {
        controls = initialControls
    }

    func applyCommand(_ rawLine: String) -> String {
        let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty else { return "" }
        let parts = line.split(maxSplits: 1, whereSeparator: \.isWhitespace).map(String.init)
        let command = parts[0].lowercased()
        let value = parts.count > 1 ? parts[1].trimmingCharacters(in: .whitespacesAndNewlines) : ""

        lock.lock()
        defer { lock.unlock() }

        switch command {
        case "prompt":
            guard !value.isEmpty else { return "usage: prompt <text>" }
            pendingPrompt = value
            return "queued prompt"
        case "style", "style-conditioning":
            guard let parsed = MagentaRT2StyleConditioning(rawValue: value.lowercased()) else {
                return "usage: style streaming|full"
            }
            controls.styleConditioning = parsed
            pendingControls = true
            return "style-conditioning \(parsed.rawValue)"
        case "temp", "temperature":
            guard let parsed = Float(value) else { return "usage: temp <value>" }
            controls.temperature = parsed
            pendingControls = true
            return "temperature \(parsed)"
        case "topk", "top-k":
            guard let parsed = Int32(value), parsed >= 0 else { return "usage: topk <value>" }
            controls.topK = parsed
            pendingControls = true
            return "top-k \(parsed)"
        case "mc", "musiccoca", "cfg-musiccoca":
            guard let parsed = Float(value) else { return "usage: mc <value>" }
            controls.cfgMusicCoCa = parsed
            pendingControls = true
            return "cfg-musiccoca \(parsed)"
        case "notes", "cfg-notes":
            guard let parsed = Float(value) else { return "usage: notes <value>" }
            controls.cfgNotes = parsed
            pendingControls = true
            return "cfg-notes \(parsed)"
        case "drums", "cfg-drums":
            guard let parsed = Float(value) else { return "usage: drums <value>" }
            controls.cfgDrums = parsed
            pendingControls = true
            return "cfg-drums \(parsed)"
        case "drumless":
            guard let parsed = parseBool(value) else { return "usage: drumless on|off" }
            controls.drumless = parsed
            pendingControls = true
            return "drumless \(parsed ? "on" : "off")"
        case "unmask", "unmask-width":
            guard let parsed = Int32(value), parsed >= 0 else { return "usage: unmask <value>" }
            controls.unmaskWidth = parsed
            pendingControls = true
            return "unmask-width \(parsed)"
        case "seed", "seed-rotation":
            guard let parsed = Int32(value) else { return "usage: seed <value>" }
            controls.seedRotation = parsed
            pendingControls = true
            return "seed-rotation \(parsed)"
        case "noteon", "note-on":
            guard let parsed = Self.parseMIDINote(value) else { return "usage: noteon <0-131>" }
            pendingNoteOn.append(parsed)
            return "note on \(parsed)"
        case "noteoff", "note-off":
            guard let parsed = Self.parseMIDINote(value) else { return "usage: noteoff <0-131>" }
            pendingNoteOff.append(parsed)
            return "note off \(parsed)"
        case "onset", "onset-mode":
            guard let parsed = Int32(value), parsed == 0 || parsed == 1 else { return "usage: onset 0|1" }
            pendingOnsetMode = parsed
            return "onset \(parsed)"
        case "reset":
            pendingReset = true
            return "queued reset"
        case "quit", "exit", "stop":
            shouldStop = true
            return "stopping"
        case "help", "?":
            return helpText
        default:
            return "unknown command: \(command)"
        }
    }

    func enqueueNoteOn(_ note: Int32) {
        guard (0..<132).contains(note) else { return }
        lock.lock()
        pendingNoteEvents.append(.on(note))
        pendingNoteOn.append(note)
        lock.unlock()
    }

    func enqueueNoteOff(_ note: Int32) {
        guard (0..<132).contains(note) else { return }
        lock.lock()
        pendingNoteEvents.append(.off(note))
        pendingNoteOff.append(note)
        lock.unlock()
    }

    func setOnsetMode(_ value: Int32) {
        guard value == 0 || value == 1 else { return }
        lock.lock()
        pendingOnsetMode = value
        lock.unlock()
    }

    func updateControls(_ update: (inout MagentaRT2Controls) -> Void) {
        lock.lock()
        update(&controls)
        pendingControls = true
        lock.unlock()
    }

    func snapshot() -> MagentaRT2LiveControlSnapshot? {
        lock.lock()
        defer { lock.unlock() }
        if shouldStop {
            return MagentaRT2LiveControlSnapshot(shouldStop: true)
        }
        guard pendingPrompt != nil
            || pendingControls
            || !pendingNoteEvents.isEmpty
            || pendingOnsetMode != nil
            || pendingReset else {
            return nil
        }
        let snapshot = MagentaRT2LiveControlSnapshot(
            prompt: pendingPrompt,
            controls: pendingControls ? controls : nil,
            noteEvents: pendingNoteEvents,
            noteOn: pendingNoteOn,
            noteOff: pendingNoteOff,
            onsetMode: pendingOnsetMode,
            resetState: pendingReset
        )
        pendingPrompt = nil
        pendingNoteEvents = []
        pendingNoteOn = []
        pendingNoteOff = []
        pendingOnsetMode = nil
        pendingControls = false
        pendingReset = false
        return snapshot
    }

    private func parseBool(_ value: String) -> Bool? {
        switch value.lowercased() {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        default:
            return nil
        }
    }

    private static func parseMIDINote(_ value: String) -> Int32? {
        guard let parsed = Int32(value), parsed >= 0, parsed < 132 else {
            return nil
        }
        return parsed
    }
}
