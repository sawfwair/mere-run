import Foundation

enum CodexSkillInstallError: LocalizedError {
    case missingSkillManifests(String)

    var errorDescription: String? {
        switch self {
        case .missingSkillManifests(let path):
            return "No skill folders containing SKILL.md were found at \(path)."
        }
    }
}

enum CodexSkillInstallOutcome: Equatable {
    case installed(names: [String], destination: URL)
    case skippedNoBundledSkills
    case failed(String)
}

enum CodexSkillInstaller {
    static func installBundledSkillsIfAvailable(
        fileManager fm: FileManager = .default,
        bundle: Bundle = .main
    ) -> CodexSkillInstallOutcome {
        guard let sourceURL = bundledSkillsURL(fileManager: fm, bundle: bundle) else {
            return .skippedNoBundledSkills
        }

        let destinationURL = defaultCodexSkillsDirectory(fileManager: fm)
        do {
            let installed = try installSkills(from: sourceURL, to: destinationURL, fileManager: fm)
            return .installed(names: installed, destination: destinationURL)
        } catch {
            return .failed(error.localizedDescription)
        }
    }

    static func installSkills(
        from sourceURL: URL,
        to destinationURL: URL,
        fileManager fm: FileManager = .default
    ) throws -> [String] {
        let skillURLs = try bundledSkillURLs(in: sourceURL, fileManager: fm)
        guard !skillURLs.isEmpty else {
            throw CodexSkillInstallError.missingSkillManifests(sourceURL.path)
        }

        try fm.createDirectory(at: destinationURL, withIntermediateDirectories: true)
        var installedNames: [String] = []
        for skillURL in skillURLs {
            let targetURL = destinationURL.appendingPathComponent(skillURL.lastPathComponent, isDirectory: true)
            try copyReplacingItem(at: skillURL, to: targetURL, fileManager: fm)
            installedNames.append(skillURL.lastPathComponent)
        }
        return installedNames.sorted()
    }

    static func bundledSkillsURL(
        fileManager fm: FileManager = .default,
        bundle: Bundle = .main
    ) -> URL? {
        let directCandidates = [
            bundle.resourceURL?.appendingPathComponent("skills", isDirectory: true),
            bundle.resourceURL?.appendingPathComponent("mere.run", isDirectory: true)
                .appendingPathComponent("skills", isDirectory: true),
        ].compactMap { $0 }

        for candidate in directCandidates where isSkillSource(candidate, fileManager: fm) {
            return candidate
        }

        let anchors: [URL] = [
            URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true),
            bundle.executableURL?.deletingLastPathComponent(),
            bundle.resourceURL,
        ].compactMap { $0 }

        var seen: Set<String> = []
        for anchor in anchors {
            guard let root = CLIResolver.nearestPackageRoot(from: anchor, fileManager: fm) else {
                continue
            }
            let candidate = root.appendingPathComponent("skills", isDirectory: true)
            let path = candidate.standardizedFileURL.path
            if seen.insert(path).inserted, isSkillSource(candidate, fileManager: fm) {
                return candidate
            }
        }

        return nil
    }

    static func defaultCodexSkillsDirectory(fileManager fm: FileManager = .default) -> URL {
        fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".codex", isDirectory: true)
            .appendingPathComponent("skills", isDirectory: true)
    }

    private static func bundledSkillURLs(in sourceURL: URL, fileManager fm: FileManager) throws -> [URL] {
        let contents = try fm.contentsOfDirectory(
            at: sourceURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        return contents
            .filter { skillURL in
                var isDirectory: ObjCBool = false
                return fm.fileExists(atPath: skillURL.path, isDirectory: &isDirectory)
                    && isDirectory.boolValue
                    && fm.fileExists(
                        atPath: skillURL.appendingPathComponent("SKILL.md", isDirectory: false).path
                    )
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private static func isSkillSource(_ url: URL, fileManager fm: FileManager) -> Bool {
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return false
        }
        return ((try? bundledSkillURLs(in: url, fileManager: fm)) ?? []).isEmpty == false
    }

    private static func copyReplacingItem(at sourceURL: URL, to destinationURL: URL, fileManager fm: FileManager) throws {
        if fm.fileExists(atPath: destinationURL.path) {
            try fm.removeItem(at: destinationURL)
        }
        try fm.copyItem(at: sourceURL, to: destinationURL)
    }
}
