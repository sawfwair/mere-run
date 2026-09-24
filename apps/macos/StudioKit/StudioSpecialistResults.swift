import Foundation

/// The one JSON object in a command's captured output. The CLI's structured commands print
/// exactly one, but the text a Library row or a utility result keeps can carry stderr lines
/// around it, so decoding starts at the first brace and ends at the last.
package enum StudioStructuredOutput {
    package static func objectData(in text: String) -> Data? {
        guard let start = text.firstIndex(of: "{"), let end = text.lastIndex(of: "}"), start <= end else {
            return nil
        }
        return Data(text[start...end].utf8)
    }
}

// MARK: - speech diarize

extension StudioDiarizationDocument {
    /// One voice in the recording: how it is named on the page, how long it held the floor, and
    /// how many turns it took. Ordered by speaker index, so the lanes never reorder between runs.
    package struct Speaker: Equatable, Identifiable {
        package let id: Int
        package let name: String
        package let talkTime: TimeInterval
        package let turnCount: Int

        package var talkTimeDescription: String { StudioTimeFormat.string(talkTime) }
    }

    /// Reads the timeline `speech diarize --format json` wrote. RTTM output is not JSON, so a run
    /// that asked for it reads as nil and the page shows the file instead.
    package static func load(from url: URL) -> StudioDiarizationDocument? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Self.self, from: data)
    }

    package var speakers: [Speaker] {
        var talkTime: [Int: TimeInterval] = [:]
        var turns: [Int: Int] = [:]
        for segment in segments {
            talkTime[segment.speakerIndex, default: 0] += max(0, segment.endSeconds - segment.startSeconds)
            turns[segment.speakerIndex, default: 0] += 1
        }
        return talkTime.keys.sorted().map { index in
            Speaker(
                id: index,
                name: "Speaker \(index + 1)",
                talkTime: talkTime[index] ?? 0,
                turnCount: turns[index] ?? 0
            )
        }
    }

    /// "2 speakers · 14 turns · 3:12"
    package var summary: String {
        let speakers = speakerCount == 1 ? "1 speaker" : "\(speakerCount) speakers"
        let turns = segments.count == 1 ? "1 turn" : "\(segments.count) turns"
        return "\(speakers) · \(turns) · \(StudioTimeFormat.string(durationSeconds))"
    }
}

// MARK: - vision face detect

/// What `mere.run vision face detect --json-output` writes (`FaceAnalysisResult`): the picture's
/// stored size and one record per face, numbered the way `--face-index` counts them, each with
/// its score, bounding box, and five landmarks in stored pixels. The Vision Lab overlay draws
/// these and the click-to-pick face picker selects among them.
package struct StudioFaceOverlayResult: Decodable, Equatable {
    package struct Record: Decodable, Equatable {
        package struct Detection: Decodable, Equatable {
            package struct Box: Decodable, Equatable {
                package let x: Double
                package let y: Double
                package let width: Double
                package let height: Double
            }

            package struct Point: Decodable, Equatable {
                package let x: Double
                package let y: Double
            }

            package let score: Double
            package let boundingBox: Box
            package let landmarks: [Point]
        }

        package let index: Int
        package let detection: Detection
        /// The normalized ArcFace vector, when the run asked for embeddings.
        package let embedding: [Float]?
    }

    package let width: Int
    package let height: Int
    package let faces: [Record]

    /// The file `--json-output` wrote, or nil when it is not a face document.
    package static func load(from url: URL) -> StudioFaceOverlayResult? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Self.self, from: data)
    }
}

// MARK: - music analyze

/// What `mere.run music analyze` prints: one JSON object (`MusicAnalyzeOutput`) naming the file
/// it read, the model, how much of the file it listened to, and what ACE-Step understood about
/// it. Decoded by name so the page can lay it out; the keys are the CLI's own camelCase.
package struct StudioMusicAnalysisDocument: Decodable, Equatable {
    package struct Metadata: Decodable, Equatable {
        package let caption: String?
        package let lyrics: String?
        package let bpm: Int?
        package let durationSeconds: Double?
        package let keyscale: String?
        package let language: String?
        package let timesignature: String?
    }

    package let audio: String
    package let model: String
    package let inputDurationSeconds: Double
    package let analyzedDurationSeconds: Double
    package let metadata: Metadata
    /// The language model's whole reply, when the run asked to keep it.
    package let rawLMOutput: String?
    /// The serialized audio codes, when the run asked to keep them.
    package let audioCodes: String?

    package static func decode(_ text: String) -> StudioMusicAnalysisDocument? {
        guard let data = StudioStructuredOutput.objectData(in: text) else { return nil }
        return try? JSONDecoder().decode(Self.self, from: data)
    }

    /// "120 BPM", or nil when the model gave no tempo.
    package var tempoDescription: String? {
        metadata.bpm.map { "\($0) BPM" }
    }

    /// "English" for a language code the system knows, otherwise what the model said.
    package var languageDescription: String? {
        guard let language = metadata.language?.trimmingCharacters(in: .whitespacesAndNewlines),
              !language.isEmpty else { return nil }
        return Locale.current.localizedString(forLanguageCode: language) ?? language
    }

    /// "0:30 of 3:12" when the run listened to less than the whole file, else its length.
    package var analyzedDescription: String {
        let analyzed = StudioTimeFormat.string(analyzedDurationSeconds)
        let input = StudioTimeFormat.string(inputDurationSeconds)
        return analyzedDurationSeconds + 0.5 < inputDurationSeconds ? "\(analyzed) of \(input)" : input
    }

    /// The caption and lyrics as the page shows them: trimmed, and absent when blank.
    package var caption: String? { Self.prose(metadata.caption) }
    package var lyrics: String? { Self.prose(metadata.lyrics) }

    private static func prose(_ text: String?) -> String? {
        guard let trimmed = text?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }
}

// MARK: - sfx clap score

/// `mere.run sfx clap score` prints one JSON object (`score`, `prompt`, `audio`, `model`); the
/// score is read from it by name, from the last line that decodes, rather than guessed from the
/// text around it.
package enum StudioCLAPScore {
    package struct Output: Decodable, Equatable {
        package let score: Double
        package let prompt: String?
        package let audio: String?
        package let model: String?
    }

    package static func parse(_ text: String) -> Double? {
        decode(text)?.score
    }

    package static func decode(_ text: String) -> Output? {
        text.components(separatedBy: .newlines).reversed().lazy.compactMap { line in
            try? JSONDecoder().decode(Output.self, from: Data(line.utf8))
        }.first
    }
}

// MARK: - music transcribe

package struct StudioMIDINote: Identifiable, Equatable {
    package let id: Int
    package let startTick: Int
    package let durationTicks: Int
    package let pitch: Int
    package let velocity: Int
    package let channel: Int

    package init(id: Int, startTick: Int, durationTicks: Int, pitch: Int, velocity: Int, channel: Int) {
        self.id = id
        self.startTick = startTick
        self.durationTicks = durationTicks
        self.pitch = pitch
        self.velocity = velocity
        self.channel = channel
    }
}

/// What `mere.run music transcribe --format midi` writes, read far enough for a piano roll: the
/// header's format, track count, and ticks per quarter, every note-on/off pair as a note, and
/// the first tempo event.
package struct StudioMIDISummary: Equatable {
    package let format: Int
    package let trackCount: Int
    package let ticksPerQuarter: Int
    package let notes: [StudioMIDINote]
    package let tempoMicrosecondsPerQuarter: Int?

    package var pitchRange: ClosedRange<Int>? {
        guard let minimum = notes.map(\.pitch).min(),
              let maximum = notes.map(\.pitch).max() else { return nil }
        return minimum...maximum
    }

    package var totalTicks: Int {
        notes.map { $0.startTick + $0.durationTicks }.max() ?? 0
    }

    /// "128 notes · 4 tracks"
    package var summary: String {
        let noteCount = notes.count == 1 ? "1 note" : "\(notes.count) notes"
        let tracks = trackCount == 1 ? "1 track" : "\(trackCount) tracks"
        return "\(noteCount) · \(tracks)"
    }

    package static func load(from url: URL) -> StudioMIDISummary? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return decode(data)
    }

    // swiftlint:disable:next function_body_length
    package static func decode(_ data: Data) -> StudioMIDISummary? {
        guard data.count >= 14, String(data: data.prefix(4), encoding: .ascii) == "MThd" else {
            return nil
        }
        let headerLength = readUInt32(data, at: 4)
        guard headerLength >= 6, data.count >= 8 + Int(headerLength) else { return nil }
        let format = Int(readUInt16(data, at: 8))
        let trackCount = Int(readUInt16(data, at: 10))
        let division = Int(readUInt16(data, at: 12))
        let ticksPerQuarter = division & 0x8000 == 0 ? division : 480
        var offset = 8 + Int(headerLength)
        var notes: [StudioMIDINote] = []
        var tempo: Int?
        var nextID = 0

        for _ in 0..<trackCount {
            guard offset + 8 <= data.count,
                  String(data: data[offset..<(offset + 4)], encoding: .ascii) == "MTrk" else {
                break
            }
            let length = Int(readUInt32(data, at: offset + 4))
            let end = min(data.count, offset + 8 + length)
            var cursor = offset + 8
            var tick = 0
            var runningStatus: UInt8?
            var openNotes: [Int: (tick: Int, velocity: Int)] = [:]

            while cursor < end {
                guard let delta = readVariableLength(data, cursor: &cursor, end: end) else { break }
                tick += delta
                guard cursor < end else { break }
                var status = data[cursor]
                if status & 0x80 != 0 {
                    cursor += 1
                    if status < 0xF0 { runningStatus = status }
                } else if let runningStatus {
                    status = runningStatus
                } else {
                    break
                }

                if status == 0xFF {
                    guard cursor < end else { break }
                    let type = data[cursor]
                    cursor += 1
                    guard let metaLength = readVariableLength(data, cursor: &cursor, end: end),
                          cursor + metaLength <= end else { break }
                    if type == 0x51, metaLength == 3 {
                        tempo = Int(data[cursor]) << 16
                            | Int(data[cursor + 1]) << 8
                            | Int(data[cursor + 2])
                    }
                    cursor += metaLength
                    continue
                }
                if status == 0xF0 || status == 0xF7 {
                    guard let systemLength = readVariableLength(data, cursor: &cursor, end: end),
                          cursor + systemLength <= end else { break }
                    cursor += systemLength
                    continue
                }

                let command = status & 0xF0
                let channel = Int(status & 0x0F)
                let dataLength = command == 0xC0 || command == 0xD0 ? 1 : 2
                guard cursor + dataLength <= end else { break }
                let first = Int(data[cursor])
                let second = dataLength == 2 ? Int(data[cursor + 1]) : 0
                cursor += dataLength

                let key = channel * 128 + first
                if command == 0x90, second > 0 {
                    openNotes[key] = (tick, second)
                } else if command == 0x80 || (command == 0x90 && second == 0),
                          let opened = openNotes.removeValue(forKey: key) {
                    notes.append(
                        StudioMIDINote(
                            id: nextID,
                            startTick: opened.tick,
                            durationTicks: max(1, tick - opened.tick),
                            pitch: first,
                            velocity: opened.velocity,
                            channel: channel
                        )
                    )
                    nextID += 1
                }
            }
            offset = end
        }
        return StudioMIDISummary(
            format: format,
            trackCount: trackCount,
            ticksPerQuarter: ticksPerQuarter,
            notes: notes.sorted {
                $0.startTick == $1.startTick ? $0.pitch < $1.pitch : $0.startTick < $1.startTick
            },
            tempoMicrosecondsPerQuarter: tempo
        )
    }

    private static func readUInt16(_ data: Data, at offset: Int) -> UInt16 {
        guard offset + 1 < data.count else { return 0 }
        return UInt16(data[offset]) << 8 | UInt16(data[offset + 1])
    }

    private static func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
        guard offset + 3 < data.count else { return 0 }
        return UInt32(data[offset]) << 24
            | UInt32(data[offset + 1]) << 16
            | UInt32(data[offset + 2]) << 8
            | UInt32(data[offset + 3])
    }

    private static func readVariableLength(
        _ data: Data,
        cursor: inout Int,
        end: Int
    ) -> Int? {
        var value = 0
        for _ in 0..<4 {
            guard cursor < end else { return nil }
            let byte = data[cursor]
            cursor += 1
            value = (value << 7) | Int(byte & 0x7F)
            if byte & 0x80 == 0 { return value }
        }
        return value
    }
}

// MARK: - run inspect

/// What `mere.run run inspect <reference> --json` prints. The command answers in one of three
/// shapes, chosen by what the reference is: a Relay or SSH job (`WorkflowRemoteJob`), a local
/// graph run directory (`GraphRunManifest`), or the inspection envelope every other local path
/// gets (`RunInspectionEnvelope`, whose result carries an image run, a transcription run, a
/// training run directory, a structured report, or a run plan). Each is decoded by name, in the
/// snake_case the CLI writes with ISO 8601 dates, keeping only what the Runs page shows.
package enum StudioRunInspection: Equatable {
    case remoteJob(RemoteJob)
    case graphRun(GraphRun)
    case local(Envelope)

    /// A file a job or node produced, as the graph contracts describe it.
    package struct Artifact: Decodable, Equatable, Identifiable {
        package let name: String
        package let kind: String
        package let path: String
        package let sizeBytes: Int64

        package var id: String { "\(name):\(path)" }

        enum CodingKeys: String, CodingKey {
            case name
            case kind
            case path
            case sizeBytes = "size_bytes"
        }
    }

    package struct RemoteJob: Decodable, Equatable {
        package let jobID: String
        package let jobReference: String
        package let state: String
        package let executor: String
        package let runDirectory: String?
        package let createdAt: Date?
        package let updatedAt: Date?
        package let artifacts: [Artifact]
        package let error: String?

        enum CodingKeys: String, CodingKey {
            case jobID = "job_id"
            case jobReference = "job_reference"
            case state
            case executor
            case runDirectory = "run_directory"
            case createdAt = "created_at"
            case updatedAt = "updated_at"
            case artifacts
            case error
        }
    }

    package struct GraphRun: Decodable, Equatable {
        package struct Node: Decodable, Equatable, Identifiable {
            package let id: String
            package let kind: String
            package let state: String
            package let startedAt: Date?
            package let completedAt: Date?
            package let attempt: Int
            package let maxAttempts: Int
            package let artifacts: [Artifact]
            package let error: String?

            enum CodingKeys: String, CodingKey {
                case id
                case kind
                case state
                case startedAt = "started_at"
                case completedAt = "completed_at"
                case attempt
                case maxAttempts = "max_attempts"
                case artifacts
                case error
            }
        }

        package let jobID: String
        package let graphName: String
        package let state: String
        package let createdAt: Date
        package let updatedAt: Date
        package let attempt: Int
        package let nodes: [Node]
        package let outputs: [Artifact]
        package let error: String?

        enum CodingKeys: String, CodingKey {
            case jobID = "job_id"
            case graphName = "graph_name"
            case state
            case createdAt = "created_at"
            case updatedAt = "updated_at"
            case attempt
            case nodes
            case outputs
            case error
        }
    }

    package struct Envelope: Decodable, Equatable {
        package struct Diagnostic: Decodable, Equatable, Identifiable {
            package let id: String
            package let severity: String
            package let title: String
            package let message: String
        }

        /// A file an image or transcription run recorded (`RunArtifact`, camelCase).
        package struct RecordArtifact: Decodable, Equatable {
            package let url: URL
            package let byteCount: Int64
        }

        package struct Issue: Decodable, Equatable {
            package let code: String
            package let message: String
        }

        package struct ImageRun: Decodable, Equatable {
            package struct Effective: Decodable, Equatable {
                package let seed: UInt64?
            }

            package let state: String
            package let modelSelector: String
            package let createdAt: Date
            package let updatedAt: Date
            package let effective: Effective?
            package let artifacts: [RecordArtifact]
            package let issue: Issue?
        }

        package struct TranscriptionRun: Decodable, Equatable {
            package struct Plan: Decodable, Equatable {
                package struct Decision: Decodable, Equatable {
                    package let backend: String
                }

                package let modelID: String
                package let decision: Decision
            }

            package let state: String
            package let createdAt: Date
            package let updatedAt: Date
            package let effective: Plan?
            package let artifacts: [RecordArtifact]
            package let issue: Issue?
        }

        package struct RunDirectory: Decodable, Equatable {
            package struct Manifest: Decodable, Equatable {
                package let createdAt: Date
                package let format: String
                package let model: String
                package let step: Int
                package let totalSteps: Int
                package let seed: UInt64

                enum CodingKeys: String, CodingKey {
                    case createdAt = "created_at"
                    case format
                    case model
                    case step
                    case totalSteps = "total_steps"
                    case seed
                }
            }

            package struct Events: Decodable, Equatable {
                package let count: Int
                package let types: [String]
            }

            package struct DirectoryArtifact: Decodable, Equatable, Identifiable {
                package let kind: String
                package let name: String
                package let path: String
                package let exists: Bool
                package let sizeBytes: Int64?

                package var id: String { path }

                enum CodingKeys: String, CodingKey {
                    case kind
                    case name
                    case path
                    case exists
                    case sizeBytes = "size_bytes"
                }
            }

            package struct Metrics: Decodable, Equatable {
                package let lossPointCount: Int
                package let latestStep: Int?
                package let latestLoss: Float?
                package let minLoss: Float?
                package let sampleImageCount: Int
                package let checkpointCount: Int
                package let adapterCount: Int

                enum CodingKeys: String, CodingKey {
                    case lossPointCount = "loss_point_count"
                    case latestStep = "latest_step"
                    case latestLoss = "latest_loss"
                    case minLoss = "min_loss"
                    case sampleImageCount = "sample_image_count"
                    case checkpointCount = "checkpoint_count"
                    case adapterCount = "adapter_count"
                }
            }

            package let path: String
            package let status: String
            package let manifest: Manifest?
            package let events: Events
            package let artifacts: [DirectoryArtifact]
            package let metrics: Metrics
        }

        package struct Report: Decodable, Equatable {
            package let command: [String]
            package let mode: String
            package let status: String
            package let createdAt: Date?
            package let summary: String
            package let diagnosticCount: Int
            package let actionCount: Int

            enum CodingKeys: String, CodingKey {
                case command
                case mode
                case status
                case createdAt = "created_at"
                case summary
                case diagnosticCount = "diagnostic_count"
                case actionCount = "action_count"
            }
        }

        package struct Plan: Decodable, Equatable {
            package let kind: String
            package let command: [String]
            package let createdAt: Date?

            enum CodingKeys: String, CodingKey {
                case kind
                case command
                case createdAt = "created_at"
            }
        }

        package struct Result: Decodable, Equatable {
            package let kind: String
            package let path: String
            package let imageRun: ImageRun?
            package let transcriptionRun: TranscriptionRun?
            package let runDirectory: RunDirectory?
            package let report: Report?
            package let plan: Plan?

            enum CodingKeys: String, CodingKey {
                case kind
                case path
                case imageRun = "image_run"
                case transcriptionRun = "transcription_run"
                case runDirectory = "run_directory"
                case report
                case plan
            }
        }

        package let summary: String
        package let status: String
        package let createdAt: Date
        package let result: Result
        package let diagnostics: [Diagnostic]

        enum CodingKeys: String, CodingKey {
            case summary
            case status
            case createdAt = "created_at"
            case result
            case diagnostics
        }
    }

    /// Decodes whichever shape `text` holds. The three are told apart by the keys only one of them
    /// carries (`graph_name`, `job_reference`, `summary` with a `result`); anything else — an
    /// error message, an older CLI's text — reads as nil and the page shows the raw output.
    package static func decode(_ text: String) -> StudioRunInspection? {
        guard let data = StudioStructuredOutput.objectData(in: text) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if let graph = try? decoder.decode(GraphRun.self, from: data) { return .graphRun(graph) }
        if let job = try? decoder.decode(RemoteJob.self, from: data) { return .remoteJob(job) }
        if let envelope = try? decoder.decode(Envelope.self, from: data) { return .local(envelope) }
        return nil
    }

    // MARK: Presentation

    /// The rows the Runs page draws, whichever shape answered: a state, the facts worth a tile, the
    /// steps a graph ran, the files it left, and what went wrong.
    package struct Presentation: Equatable {
        package struct Fact: Equatable, Identifiable {
            package let label: String
            package let value: String

            package var id: String { label }
        }

        package struct Step: Equatable, Identifiable {
            package let id: String
            package let title: String
            package let detail: String
            package let state: String
        }

        package struct Output: Equatable, Identifiable {
            package let path: String
            package let detail: String
            /// False for a file the CLI reported missing, so the page offers no Reveal for it.
            package let exists: Bool

            package var id: String { path }
            package var name: String { URL(fileURLWithPath: path).lastPathComponent }
        }

        /// The run's state as the CLI names it ("finished", "succeeded", "failed", "ok").
        package let state: String
        package let title: String
        package let facts: [Fact]
        package let steps: [Step]
        package let outputs: [Output]
        package let problems: [String]
    }

    package var presentation: Presentation {
        switch self {
        case .remoteJob(let job):
            var facts = [Presentation.Fact(label: "Executor", value: job.executor)]
            facts.append(contentsOf: Self.timingFacts(started: job.createdAt, updated: job.updatedAt))
            if let directory = job.runDirectory { facts.append(.init(label: "Run directory", value: directory)) }
            return Presentation(
                state: job.state,
                title: job.jobID,
                facts: facts,
                steps: [],
                outputs: job.artifacts.map(Self.output),
                problems: [job.error].compactMap { $0 }
            )
        case .graphRun(let run):
            var facts = [
                Presentation.Fact(label: "Graph", value: run.graphName),
                Presentation.Fact(label: "Attempt", value: String(run.attempt)),
            ]
            facts.append(contentsOf: Self.timingFacts(started: run.createdAt, updated: run.updatedAt))
            let steps = run.nodes.map { node in
                Presentation.Step(
                    id: node.id,
                    title: node.id,
                    detail: Self.nodeDetail(node),
                    state: node.state
                )
            }
            // The run's error usually quotes the failing node's; the node is listed once.
            let problems = [run.error].compactMap { $0 } + run.nodes.compactMap { node in
                guard let error = node.error, run.error?.contains(error) != true else { return nil }
                return "\(node.id): \(error)"
            }
            return Presentation(
                state: run.state,
                title: run.jobID,
                facts: facts,
                steps: steps,
                outputs: run.outputs.map(Self.output),
                problems: problems
            )
        case .local(let envelope):
            return Self.localPresentation(envelope)
        }
    }

    private static func localPresentation(_ envelope: Envelope) -> Presentation {
        let result = envelope.result
        var problems = envelope.diagnostics
            .filter { $0.severity == "blocker" || $0.severity == "warning" }
            .map { "\($0.title): \($0.message)" }
        if let image = result.imageRun {
            var facts = [Presentation.Fact(label: "Model", value: image.modelSelector)]
            if let seed = image.effective?.seed { facts.append(.init(label: "Seed", value: String(seed))) }
            facts.append(contentsOf: timingFacts(started: image.createdAt, updated: image.updatedAt))
            if let issue = image.issue { problems.insert(issue.message, at: 0) }
            return Presentation(
                state: image.state,
                title: envelope.summary,
                facts: facts,
                steps: [],
                outputs: image.artifacts.map(output),
                problems: problems
            )
        }
        if let transcription = result.transcriptionRun {
            var facts: [Presentation.Fact] = []
            if let plan = transcription.effective {
                facts.append(.init(label: "Model", value: plan.modelID))
                facts.append(.init(label: "Backend", value: plan.decision.backend))
            }
            facts.append(contentsOf: timingFacts(started: transcription.createdAt, updated: transcription.updatedAt))
            if let issue = transcription.issue { problems.insert(issue.message, at: 0) }
            return Presentation(
                state: transcription.state,
                title: envelope.summary,
                facts: facts,
                steps: [],
                outputs: transcription.artifacts.map(output),
                problems: problems
            )
        }
        if let directory = result.runDirectory {
            var facts: [Presentation.Fact] = []
            if let manifest = directory.manifest {
                facts.append(.init(label: "Model", value: manifest.model))
                facts.append(.init(label: "Format", value: manifest.format))
                facts.append(.init(label: "Progress", value: "\(manifest.step) of \(manifest.totalSteps) steps"))
                facts.append(.init(label: "Seed", value: String(manifest.seed)))
                facts.append(.init(label: "Started", value: dateDescription(manifest.createdAt)))
            }
            facts.append(.init(label: "Events", value: String(directory.events.count)))
            let metrics = directory.metrics
            if let latestLoss = metrics.latestLoss, let latestStep = metrics.latestStep {
                facts.append(.init(label: "Latest loss", value: String(format: "%.4f at step %d", latestLoss, latestStep)))
            }
            if let minLoss = metrics.minLoss {
                facts.append(.init(label: "Lowest loss", value: String(format: "%.4f", minLoss)))
            }
            if metrics.checkpointCount > 0 {
                facts.append(.init(label: "Checkpoints", value: String(metrics.checkpointCount)))
            }
            if metrics.sampleImageCount > 0 {
                facts.append(.init(label: "Samples", value: String(metrics.sampleImageCount)))
            }
            return Presentation(
                state: directory.status,
                title: envelope.summary,
                facts: facts,
                steps: [],
                outputs: directory.artifacts.map { artifact in
                    Presentation.Output(
                        path: artifact.path,
                        detail: [artifact.kind, artifact.sizeBytes.map(byteDescription)].compactMap { $0 }
                            .joined(separator: " · "),
                        exists: artifact.exists
                    )
                },
                problems: problems
            )
        }
        if let report = result.report {
            var facts = [
                Presentation.Fact(label: "Command", value: report.command.joined(separator: " ")),
                Presentation.Fact(label: "Mode", value: report.mode.replacingOccurrences(of: "_", with: " ")),
                Presentation.Fact(label: "Diagnostics", value: String(report.diagnosticCount)),
                Presentation.Fact(label: "Actions", value: String(report.actionCount)),
            ]
            if let created = report.createdAt { facts.append(.init(label: "Written", value: dateDescription(created))) }
            return Presentation(
                state: report.status,
                title: report.summary,
                facts: facts,
                steps: [],
                outputs: [],
                problems: problems
            )
        }
        if let plan = result.plan {
            var facts = [
                Presentation.Fact(label: "Plan", value: plan.kind),
                Presentation.Fact(label: "Command", value: plan.command.joined(separator: " ")),
            ]
            if let created = plan.createdAt { facts.append(.init(label: "Written", value: dateDescription(created))) }
            return Presentation(
                state: envelope.status,
                title: envelope.summary,
                facts: facts,
                steps: [],
                outputs: [],
                problems: problems
            )
        }
        return Presentation(
            state: envelope.status,
            title: envelope.summary,
            facts: [Presentation.Fact(label: "Kind", value: result.kind.replacingOccurrences(of: "_", with: " "))],
            steps: [],
            outputs: [],
            problems: problems
        )
    }

    // MARK: Formatting

    /// "Started" and "Updated" as a date and time, and "Duration" between them.
    private static func timingFacts(started: Date?, updated: Date?) -> [Presentation.Fact] {
        var facts: [Presentation.Fact] = []
        if let started { facts.append(.init(label: "Started", value: dateDescription(started))) }
        if let updated { facts.append(.init(label: "Updated", value: dateDescription(updated))) }
        if let started, let updated, updated > started {
            facts.append(.init(label: "Duration", value: StudioTimeFormat.string(updated.timeIntervalSince(started))))
        }
        return facts
    }

    /// "kind · 2 attempts · 1:12" — what a graph node did beyond its id and state.
    private static func nodeDetail(_ node: GraphRun.Node) -> String {
        var parts = [node.kind]
        if node.attempt > 1 { parts.append("attempt \(node.attempt) of \(node.maxAttempts)") }
        if let started = node.startedAt, let completed = node.completedAt, completed > started {
            parts.append(StudioTimeFormat.string(completed.timeIntervalSince(started)))
        }
        if !node.artifacts.isEmpty {
            parts.append(node.artifacts.count == 1 ? "1 file" : "\(node.artifacts.count) files")
        }
        return parts.joined(separator: " · ")
    }

    private static func output(_ artifact: Artifact) -> Presentation.Output {
        Presentation.Output(
            path: artifact.path,
            detail: "\(artifact.kind) · \(byteDescription(artifact.sizeBytes))",
            exists: true
        )
    }

    private static func output(_ artifact: Envelope.RecordArtifact) -> Presentation.Output {
        Presentation.Output(path: artifact.url.path, detail: byteDescription(artifact.byteCount), exists: true)
    }

    private static func byteDescription(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    package static func dateDescription(_ date: Date) -> String {
        date.formatted(date: .abbreviated, time: .shortened)
    }
}
