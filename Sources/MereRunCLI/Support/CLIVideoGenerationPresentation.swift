import MereRunCore

struct CLIVideoGenerationPresentation: Sendable {
    let quiet: Bool
    let progressJSON: Bool
    var write: @Sendable (String) -> Void = { CLIStderr.write($0) }

    var eventHandler: VideoGenerationOperation.EventHandler? {
        guard !quiet || progressJSON else { return nil }
        let stream = progressJSON ? JSONProgressStream(write: write) : nil
        return { event in
            switch event {
            case .diagnostic(let message):
                if !quiet { write(message) }
            case .progress(let stage, let step, let totalSteps):
                if let stream {
                    if stage == "window" {
                        stream.mark(stage: stage, step: step, totalSteps: totalSteps)
                    } else {
                        stream.report(stage: stage, step: step, totalSteps: totalSteps)
                    }
                } else if !quiet {
                    if stage == "denoising" {
                        write("Denoising \(step + 1)/\(totalSteps)\n")
                    } else if stage == "window" {
                        write("H3 window \(step + 1)/\(totalSteps)\n")
                    } else {
                        write("\(stage)\n")
                    }
                }
            case .progressFinished:
                stream?.finish()
            }
        }
    }
}
