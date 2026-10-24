// Compile against MereRunCoreMLModel; see docs/runtime/embeddinggemma2.md.
import Foundation
import MereRunCoreMLModel

struct ReferenceCase: Decodable {
    let kind: String?
    let text: String?
    let file: String?
    let embedding: [Float]
}
struct TextReference: Decodable { let cases: [ReferenceCase] }
struct ImageReference: Decodable { let images: [ReferenceCase] }
struct AudioReference: Decodable { let audio: [ReferenceCase] }
struct Prompts: Decodable { let prompts: [String: String] }
struct Observation: Encodable {
    let input: String
    let modality: String
    let tokens: Int
    let cosine: Double
    let norm: Double
    let embedding: [Float]
}
struct Receipt: Encodable {
    let modelRepository = "FluidInference/embeddinggemma-2-coreml"
    let modelRevision = "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4"
    let referenceRepository = "EverPaper/embeddinggemma-2-coreai"
    let referenceRevision = "305fc2a96790d9343f38038aeca211dfca56d9a5"
    let observations: [Observation]
    let skipped: [String]
}
enum QualificationError: Error { case arguments, invalidReference, nonfiniteOutput }

@main struct QualifyCoreML {
    static func main() async throws {
        guard CommandLine.arguments.count == 3 else { throw QualificationError.arguments }
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let refs = URL(fileURLWithPath: CommandLine.arguments[2])
        let decoder = JSONDecoder()
        let model = try EmbeddingGemma2(bundle: root)
        let prompts = try decoder.decode(Prompts.self, from: Data(contentsOf: root.appending(path: "config.json"))).prompts
        var observations: [Observation] = [], skipped: [String] = []
        func observe(_ vector: [Float], reference: ReferenceCase, modality: String, input: String, tokens: Int) throws {
            guard vector.count == 768, reference.embedding.count == 768, vector.allSatisfy(\.isFinite) else {
                throw QualificationError.nonfiniteOutput
            }
            let norm = sqrt(vector.reduce(0.0) { $0 + Double($1) * Double($1) })
            let referenceNorm = sqrt(reference.embedding.reduce(0.0) { $0 + Double($1) * Double($1) })
            let dot = zip(vector, reference.embedding).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
            FileHandle.standardError.write(Data("\(modality): \(input) cosine=\(dot / (norm * referenceNorm))\n".utf8))
            observations.append(Observation(input: input, modality: modality, tokens: tokens,
                                            cosine: dot / (norm * referenceNorm), norm: norm, embedding: vector))
        }
        let texts = try decoder.decode(TextReference.self, from: Data(contentsOf: refs.appending(path: "smoke-reference.json"))).cases
        for ref in texts {
            guard let kind = ref.kind, let text = ref.text, let prefix = prompts[kind] else { throw QualificationError.invalidReference }
            let ids = model.tokenizer.encodeWithSpecials(prefix + text)
            guard ids.count <= 512 else { skipped.append("text exceeds 512 tokens: " + text); continue }
            try observe(try await model.embed(sequence: MultimodalSequence(ids: ids)), reference: ref, modality: "text",
                        input: text, tokens: ids.count)
        }
        let images = try decoder.decode(ImageReference.self, from: Data(contentsOf: refs.appending(path: "image-smoke-reference.json"))).images
        for ref in images {
            guard let file = ref.file else { throw QualificationError.invalidReference }
            let sequence = try await model.imageSequence(url: refs.appending(path: file))
            try observe(try await model.embed(sequence: sequence), reference: ref, modality: "image", input: file, tokens: sequence.ids.count)
        }
        let audio = try decoder.decode(AudioReference.self, from: Data(contentsOf: refs.appending(path: "audio-smoke-reference.json"))).audio
        for ref in audio {
            guard let file = ref.file else { throw QualificationError.invalidReference }
            do {
                let sequence = try await model.audioSequence(url: refs.appending(path: file))
                try observe(try await model.embed(sequence: sequence), reference: ref, modality: "audio", input: file, tokens: sequence.ids.count)
            } catch let error as EmbedderError where error.description.contains("at most 10 seconds") {
                skipped.append(file + ": outside the 10 ms–10 s Core ML audio budget")
            }
        }
        let receipt = Receipt(observations: observations, skipped: skipped)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        FileHandle.standardOutput.write(try encoder.encode(receipt))
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
}
