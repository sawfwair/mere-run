import Foundation

/// Everything one run produced, and how the app learned about it.
package struct ArtifactResolution: Equatable {
    /// Which rung of the precedence ladder produced this resolution.
    package enum Source: Equatable {
        /// The CLI's `--receipt` line named every output and its role.
        case receipt
        /// The contract declares a file or directory output and the requested `--output` landed.
        case declaredOutput
        /// Nothing structured was available: stdout heuristics and filesystem probing.
        case probe
        /// The run produced no file (a text command, a failed run, a conversation turn).
        case none
    }

    package static let empty = ArtifactResolution(source: .none, artifacts: [])

    package let source: Source
    /// Primary artifact first, then sidecars in the order they were reported or discovered.
    package let artifacts: [Artifact]

    package var primary: URL? {
        artifacts.first { $0.role == .primary }?.url
    }

    package var sidecars: [Artifact] {
        artifacts.filter { $0.role == .sidecar }
    }
}

/// Finds the files a run produced, in this order:
///
/// 1. the CLI's `--receipt` line, which names every output, its kind and its sidecar role;
/// 2. the contract's declared output kind together with the `--output` path the request asked
///    for, once that path exists;
/// 3. the stdout path contract and filesystem probing, the fallback for commands that emit no
///    receipt and for a CLI older than `--receipt`.
///
/// Filesystem existence checks go through `MereRunFileProbing` so detection is unit-testable
/// without touching the real disk.
package struct ArtifactResolver {
    package let fileSystem: MereRunFileProbing

    /// How many trailing stdout lines the last-resort probe considers.
    package static let stdoutCandidateLineLimit = 40

    /// The explicit `--output` path a request asked for, when the command declares a file or
    /// directory output.
    package static func expectedOutput(template: CommandTemplate, draft: CommandDraft) -> URL? {
        guard template.producesOutputFile, !draft.outputPath.isBlank else {
            return nil
        }
        return URL(fileURLWithPath: NSString(string: draft.outputPath).expandingTildeInPath)
    }

    /// The run's main output, if it is known yet. Cheap enough to run on every stdout chunk: it
    /// reads the receipt and probes single paths, and never enumerates a directory.
    package func primaryOutput(expected: URL?, stdout: String) -> URL? {
        // 1. The receipt is authoritative: the CLI listed what it wrote.
        if let receipt = StudioRunReceipt.parse(stdout: stdout), let first = receipt.outputs.first {
            return first.url
        }

        // 2. The explicit `--output` path the request asked for, once it has landed.
        if let expected, fileSystem.fileExists(atPath: expected.path) {
            return expected
        }

        // 3. Honor the CLI's stdout contract: media commands print the artifact path as a bare
        //    line and directory/OCR commands as `input -> output` pairs, most-recent first.
        //    This is the only path that detects `input -> output` outputs at all (a whole pair
        //    line never resolves as a file), and it targets the result line rather than guessing.
        for candidate in StudioResultParser.outputPaths(fromStdout: stdout) {
            let expanded = NSString(string: candidate).expandingTildeInPath
            if fileSystem.fileExists(atPath: expanded) {
                return URL(fileURLWithPath: expanded)
            }
        }

        // 4. Fallback for commands without a clean path contract: the last trailing stdout
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
    package func reportedOutputs(stdout: String) -> [URL] {
        StudioResultParser.outputPaths(fromStdout: stdout).compactMap { candidate in
            let expanded = NSString(string: candidate).expandingTildeInPath
            guard fileSystem.fileExists(atPath: expanded) else { return nil }
            return URL(fileURLWithPath: expanded)
        }
    }

    /// The complete artifact list for a finished run, with each sidecar's role attached.
    /// A receipt short-circuits every probe: the CLI already listed what it wrote.
    package func resolve(
        template: CommandTemplate,
        draft: CommandDraft,
        expected: URL?,
        stdout: String
    ) -> ArtifactResolution {
        if let receipt = StudioRunReceipt.parse(stdout: stdout), !receipt.outputs.isEmpty {
            var artifacts = [Artifact(url: receipt.outputs[0].url, role: .primary)]
            artifacts += receipt.outputs.dropFirst().map {
                Artifact(url: $0.url, role: .sidecar, sidecarRole: $0.role)
            }
            return ArtifactResolution(source: .receipt, artifacts: artifacts)
        }

        let declared = expected.flatMap { fileSystem.fileExists(atPath: $0.path) ? $0 : nil }
        let primary = declared ?? primaryOutput(expected: expected, stdout: stdout)
        let discovered = StudioArtifactDiscovery.urls(
            templateID: template.id,
            draft: draft,
            primaryOutput: primary,
            reportedOutputs: reportedOutputs(stdout: stdout)
        )
        guard primary != nil || !discovered.isEmpty else {
            return .empty
        }

        var artifacts = primary.map { [Artifact(url: $0, role: .primary)] } ?? []
        artifacts += discovered
            .filter { $0 != primary }
            .map {
                Artifact(
                    url: $0,
                    role: .sidecar,
                    sidecarRole: StudioArtifactRole.inferred(for: $0, templateID: template.id, draft: draft)
                )
            }
        return ArtifactResolution(source: declared != nil ? .declaredOutput : .probe, artifacts: artifacts)
    }
}
