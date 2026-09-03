import Foundation

/// Finds the files a run produced from what the request asked for and what the CLI printed.
/// Filesystem existence checks go through `MereRunFileProbing` so detection is unit-testable
/// without touching the real disk.
struct ArtifactResolver {
    let fileSystem: MereRunFileProbing

    /// How many trailing stdout lines the last-resort probe considers.
    static let stdoutCandidateLineLimit = 40

    /// The explicit `--output` path a request asked for, when its template produces one.
    static func expectedOutput(template: CommandTemplate, draft: CommandDraft) -> URL? {
        guard template.outputKind != .none, !draft.outputPath.isBlank else {
            return nil
        }
        return URL(fileURLWithPath: NSString(string: draft.outputPath).expandingTildeInPath)
    }

    /// The run's main output, if it exists yet.
    func primaryOutput(expected: URL?, stdout: String) -> URL? {
        // 1. The explicit `--output` path the request asked for, once it has landed.
        if let expected, fileSystem.fileExists(atPath: expected.path) {
            return expected
        }

        // 2. Honor the CLI's stdout contract: media commands print the artifact path as a bare
        //    line and directory/OCR commands as `input -> output` pairs, most-recent first.
        //    This is the only path that detects `input -> output` outputs at all (a whole pair
        //    line never resolves as a file), and it targets the result line rather than guessing.
        for candidate in StudioResultParser.outputPaths(fromStdout: stdout) {
            let expanded = NSString(string: candidate).expandingTildeInPath
            if fileSystem.fileExists(atPath: expanded) {
                return URL(fileURLWithPath: expanded)
            }
        }

        // 3. Fallback for commands without a clean path contract: the last trailing stdout
        //    line that happens to resolve to an existing file (e.g. a relative path).
        let candidates = stdout
            .components(separatedBy: .newlines)
            .compactMap { line -> String? in
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                return trimmed.isEmpty ? nil : trimmed
            }
            .suffix(Self.stdoutCandidateLineLimit)
            .reversed()

        for candidate in candidates {
            let expanded = NSString(string: candidate).expandingTildeInPath
            if fileSystem.fileExists(atPath: expanded) {
                return URL(fileURLWithPath: expanded)
            }
        }

        return nil
    }

    /// Every existing file the CLI reported on stdout, most-recent first.
    func reportedOutputs(stdout: String) -> [URL] {
        StudioResultParser.outputPaths(fromStdout: stdout).compactMap { candidate in
            let expanded = NSString(string: candidate).expandingTildeInPath
            guard fileSystem.fileExists(atPath: expanded) else { return nil }
            return URL(fileURLWithPath: expanded)
        }
    }
}
