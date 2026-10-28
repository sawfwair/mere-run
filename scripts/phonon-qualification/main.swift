import Foundation
import MLX
@testable import AudioPhononModel

// Qualification-only executable. Built against the repository's model sources in release mode.
struct Prompt: Decodable { let id: String; let text: String }
struct Step: Codable { let latent: [Float]; let eos: Float }
struct Run: Codable {
    let id: String
    let iteration: Int
    let sampleRate, samples, frames: Int
    let totalSeconds, firstAudioSeconds: Double
    let peakMLXBytes: Int
}
struct Report: Codable {
    let noiseMode: String
    let device: String
    let temperature: Float
    let loadSeconds: Double
    let voiceLoadSeconds: Double
    let runs: [Run]
}
func now() -> Double { ProcessInfo.processInfo.systemUptime }
func writeFloats(_ values: [Float], to url: URL) throws {
    let words = values.map { $0.bitPattern.littleEndian }
    let bytes = words.withUnsafeBytes { Data($0) }
    try bytes.write(to: url)
}
let args = Array(CommandLine.arguments.dropFirst())
guard (6...7).contains(args.count), let iterations = Int(args[4]), iterations > 0,
      let temperature = Float(args[5]), ["cpu", "gpu"].contains(args[3]) else {
    fatalError("Usage: PhononQualification MODEL CORPUS OUT DEVICE ITERATIONS TEMPERATURE [NOISE_JSON]")
}
let replay = args.count == 7 ? try JSONDecoder().decode([Float].self, from: Data(contentsOf: URL(fileURLWithPath: args[6]))) : nil
if let replay, replay.isEmpty || !replay.allSatisfy(\.isFinite) { throw PhononError.invalidAsset("noise must be nonempty and finite") }
let root = URL(fileURLWithPath: args[0]), output = URL(fileURLWithPath: args[2])
let prompts = try JSONDecoder().decode([Prompt].self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let report = try Device.withDefaultDevice(Device(args[3] == "cpu" ? .cpu : .gpu)) {
    let start = now()
    let model = try PhononModel(directory: root)
    let loadSeconds = now() - start
    let voiceStart = now()
    let voice = try model.loadVoice(url: root.appendingPathComponent("default-voice.safetensors"))
    let voiceSeconds = now() - voiceStart
    var runs: [Run] = []
    func generate(_ prompt: Prompt, trace: ((MLXArray, MLXArray) -> Void)? = nil, onFrame: ([Float]) throws -> Void) throws {
        if let replay {
            var cursor = 0
            try model.generate(text: prompt.text, voice: voice, noiseSampler: {
                let values = (0..<model.config.flowLm.ldim).map { replay[(cursor + $0) % replay.count] }
                cursor += model.config.flowLm.ldim
                return MLXArray(values, [1, model.config.flowLm.ldim])
            }, onStep: trace, onFrame: onFrame)
        } else { try model.generate(text: prompt.text, voice: voice, temperature: temperature, seed: 42, onFrame: onFrame) }
    }
    for prompt in prompts {
        // A real generation warms the same public path that is timed below.
        try generate(prompt) { _ in }
        for iteration in 0..<iterations {
            let start = now()
            var first: Double?, samples: [Float] = [], frames = 0
            var steps: [Step] = []
            let trace: ((MLXArray, MLXArray) -> Void)? = replay != nil && iteration == 0 ? { latent, eos in
                steps.append(Step(latent: latent.asArray(Float.self), eos: eos.item(Float.self)))
            } : nil
            try generate(prompt, trace: trace) { frame in
                if first == nil { first = now() - start }
                samples.append(contentsOf: frame)
                frames += 1
            }
            let elapsed = now() - start
            guard let first else { throw PhononError.invalidAsset("no audio generated") }
            if iteration == 0 {
                try writeFloats(samples, to: output.appendingPathComponent(prompt.id + ".f32"))
                if replay != nil { try JSONEncoder().encode(steps).write(to: output.appendingPathComponent(prompt.id + ".steps.json")) }
            }
            runs.append(Run(id: prompt.id, iteration: iteration, sampleRate: model.sampleRate,
                            samples: samples.count, frames: frames, totalSeconds: elapsed,
                            firstAudioSeconds: first, peakMLXBytes: Memory.peakMemory))
        }
    }
    return Report(noiseMode: replay == nil ? "seeded" : "replay", device: args[3], temperature: temperature, loadSeconds: loadSeconds, voiceLoadSeconds: voiceSeconds, runs: runs)
}
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(report)
try data.write(to: output.appendingPathComponent("timings.json"))
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
