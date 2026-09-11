import Foundation

extension ImageLoRATrainingOptions {
    func resolveModelRoot(model: String?) throws -> URL {
        let resolver = ModelResolver()
        guard let model else {
            do {
                return try resolver.resolve(Self.defaultManagedModelID).rootURL
            } catch {
                throw ImageLoRATrainingIssue.modelUnavailable(id: Self.defaultManagedModelID.rawValue, role: .defaultTraining)
            }
        }

        let url = URL(fileURLWithPath: model).standardizedFileURL
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        if let id = ModelResolver.ModelID(rawValue: model) {
            do {
                return try resolver.resolve(id).rootURL
            } catch {
                throw ImageLoRATrainingIssue.modelUnavailable(id: id.rawValue, role: .training)
            }
        }
        throw ImageLoRATrainingIssue("Model path not found: \(model). Pass a local model path or a known model id.")
    }

    func resolveKleinSampleModelPath() throws -> String {
        let spec = sampleModel ?? ModelResolver.ModelID.klein9B.rawValue
        let url = URL(fileURLWithPath: spec).standardizedFileURL
        if FileManager.default.fileExists(atPath: url.path) {
            return url.path
        }

        if let id = ModelResolver.ModelID(rawValue: spec) {
            do {
                return try ModelResolver().resolve(id).rootURL.path
            } catch {
                throw ImageLoRATrainingIssue.modelUnavailable(id: id.rawValue, role: .sample)
            }
        }

        throw ImageLoRATrainingIssue("Sample model path not found: \(spec). Pass a local path or known model id.")
    }

}
