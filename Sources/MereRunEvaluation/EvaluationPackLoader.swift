import Crypto
import Foundation

public struct EvaluationPackFilePin: Codable, Hashable, Sendable {
    public let relativePath: String
    public let byteCount: Int64
    public let sha256: String

    enum CodingKeys: String, CodingKey {
        case relativePath = "relative_path"
        case byteCount = "byte_count"
        case sha256
    }

    public init(relativePath: String, byteCount: Int64, sha256: String) {
        self.relativePath = relativePath
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

public struct LoadedEvaluationCase: Hashable, Sendable {
    public let specification: EvaluationCase
    public let contentSHA256: String
    public let sourceFile: String
    public let sourceLine: Int

    public init(
        specification: EvaluationCase,
        contentSHA256: String,
        sourceFile: String,
        sourceLine: Int
    ) {
        self.specification = specification
        self.contentSHA256 = contentSHA256
        self.sourceFile = sourceFile
        self.sourceLine = sourceLine
    }
}

public struct LoadedEvaluationPromptSet: Hashable, Sendable {
    public let specification: EvaluationPromptSet
    public let systemPrompt: String
    public let contentSHA256: String

    public init(
        specification: EvaluationPromptSet,
        systemPrompt: String,
        contentSHA256: String
    ) {
        self.specification = specification
        self.systemPrompt = systemPrompt
        self.contentSHA256 = contentSHA256
    }
}

public struct LoadedEvaluationPack: Sendable {
    public let rootURL: URL
    public let manifestURL: URL
    public let manifest: EvaluationPackManifest
    public let manifestSHA256: String
    public let packSHA256: String
    public let files: [EvaluationPackFilePin]
    public let cases: [LoadedEvaluationCase]
    public let promptSets: [LoadedEvaluationPromptSet]
    public let imageURLs: [String: URL]
    public let scorerExecutableURL: URL?

    public init(
        rootURL: URL,
        manifestURL: URL,
        manifest: EvaluationPackManifest,
        manifestSHA256: String,
        packSHA256: String,
        files: [EvaluationPackFilePin],
        cases: [LoadedEvaluationCase],
        promptSets: [LoadedEvaluationPromptSet],
        imageURLs: [String: URL] = [:],
        scorerExecutableURL: URL?
    ) {
        self.rootURL = rootURL
        self.manifestURL = manifestURL
        self.manifest = manifest
        self.manifestSHA256 = manifestSHA256
        self.packSHA256 = packSHA256
        self.files = files
        self.cases = cases
        self.promptSets = promptSets
        self.imageURLs = imageURLs
        self.scorerExecutableURL = scorerExecutableURL
    }

    public func promptSet(id: String) -> LoadedEvaluationPromptSet? {
        promptSets.first { $0.specification.id == id }
    }

    public func imageURL(relativePath: String) -> URL? {
        imageURLs[relativePath]
    }
}

public enum EvaluationPackError: LocalizedError, Equatable {
    case invalidPath(String)
    case invalidManifest(String)
    case invalidCase(String)
    case duplicateCaseID(String)

    public var errorDescription: String? {
        switch self {
        case .invalidPath(let detail):
            "Invalid evaluation pack path: \(detail)."
        case .invalidManifest(let detail):
            "Invalid evaluation pack manifest: \(detail)."
        case .invalidCase(let detail):
            "Invalid evaluation case: \(detail)."
        case .duplicateCaseID(let id):
            "Invalid evaluation pack: duplicate case id '\(id)'."
        }
    }
}

public enum EvaluationPackLoader {
    public static let manifestFileName = "eval-pack.json"

    public static func load(
        from path: String,
        fileManager: FileManager = .default
    ) throws -> LoadedEvaluationPack {
        try load(from: URL(fileURLWithPath: path), fileManager: fileManager)
    }

    public static func load(
        from inputURL: URL,
        fileManager: FileManager = .default
    ) throws -> LoadedEvaluationPack {
        let standardizedInput = inputURL.standardizedFileURL
        let inputValues = try resourceValues(for: standardizedInput)
        let manifestURL: URL
        let rootURL: URL
        if inputValues.isDirectory == true {
            rootURL = standardizedInput.resolvingSymlinksInPath()
            manifestURL = rootURL.appendingPathComponent(manifestFileName, isDirectory: false)
        } else if inputValues.isRegularFile == true {
            manifestURL = standardizedInput
            rootURL = standardizedInput.deletingLastPathComponent().resolvingSymlinksInPath()
        } else {
            throw EvaluationPackError.invalidPath("\(standardizedInput.path) is not a file or directory")
        }

        let manifestData = try readDeclaredFile(
            manifestURL,
            relativePath: manifestURL.lastPathComponent,
            rootURL: rootURL,
            fileManager: fileManager
        )
        let manifest: EvaluationPackManifest
        do {
            try EvaluationSchemaValidation.validateManifestJSON(manifestData)
            manifest = try JSONDecoder().decode(EvaluationPackManifest.self, from: manifestData)
        } catch {
            throw EvaluationPackError.invalidManifest("cannot decode \(manifestURL.path): \(error)")
        }
        try validate(manifest)

        var pins = [try pin(
            data: manifestData,
            relativePath: manifestURL.lastPathComponent
        )]
        var imageURLs: [String: URL] = [:]
        for relativePath in manifest.imageFiles ?? [] {
            let fileURL = try declaredFileURL(
                relativePath,
                rootURL: rootURL,
                fileManager: fileManager
            )
            let data = try Data(contentsOf: fileURL)
            pins.append(try pin(data: data, relativePath: relativePath))
            imageURLs[relativePath] = fileURL
        }
        let declaredImageFiles = Set(imageURLs.keys)
        var loadedCases: [LoadedEvaluationCase] = []
        var seenCaseIDs: Set<String> = []
        for relativePath in manifest.caseFiles {
            let fileURL = try declaredFileURL(
                relativePath,
                rootURL: rootURL,
                fileManager: fileManager
            )
            let data = try Data(contentsOf: fileURL)
            pins.append(try pin(data: data, relativePath: relativePath))
            guard let text = String(data: data, encoding: .utf8) else {
                throw EvaluationPackError.invalidCase(
                    "\(relativePath) is not UTF-8"
                )
            }
            for (offset, rawLine) in text.split(
                separator: "\n",
                omittingEmptySubsequences: false
            ).enumerated() {
                let lineNumber = offset + 1
                let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !line.isEmpty else { continue }
                guard let lineData = line.data(using: .utf8) else {
                    throw EvaluationPackError.invalidCase(
                        "\(relativePath):\(lineNumber) is not UTF-8"
                    )
                }
                let specification: EvaluationCase
                do {
                    try EvaluationSchemaValidation.validateCaseJSON(lineData)
                    specification = try JSONDecoder().decode(EvaluationCase.self, from: lineData)
                } catch {
                    throw EvaluationPackError.invalidCase(
                        "cannot decode \(relativePath):\(lineNumber): \(error)"
                    )
                }
                try validate(
                    specification,
                    scorer: manifest.scorer,
                    declaredImageFiles: declaredImageFiles,
                    source: "\(relativePath):\(lineNumber)"
                )
                guard seenCaseIDs.insert(specification.id).inserted else {
                    throw EvaluationPackError.duplicateCaseID(specification.id)
                }
                loadedCases.append(LoadedEvaluationCase(
                    specification: specification,
                    contentSHA256: try canonicalSHA256(specification),
                    sourceFile: relativePath,
                    sourceLine: lineNumber
                ))
            }
        }
        guard !loadedCases.isEmpty else {
            throw EvaluationPackError.invalidManifest("declared case files contain no cases")
        }
        let referencedImageFiles = Set(loadedCases.flatMap { loadedCase in
            loadedCase.specification.messages.compactMap(\.imageFile)
        })
        let unusedImageFiles = declaredImageFiles.subtracting(referencedImageFiles)
        guard unusedImageFiles.isEmpty else {
            throw EvaluationPackError.invalidManifest(
                "image_files contains unreferenced entries: \(unusedImageFiles.sorted().joined(separator: ", "))"
            )
        }

        var loadedPromptSets: [LoadedEvaluationPromptSet] = []
        for promptSet in manifest.promptSets {
            let fileURL = try declaredFileURL(
                promptSet.systemPromptFile,
                rootURL: rootURL,
                fileManager: fileManager
            )
            let data = try Data(contentsOf: fileURL)
            let prompt = String(decoding: data, as: UTF8.self)
            guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw EvaluationPackError.invalidManifest(
                    "prompt set \(promptSet.id) is empty"
                )
            }
            pins.append(try pin(data: data, relativePath: promptSet.systemPromptFile))
            loadedPromptSets.append(LoadedEvaluationPromptSet(
                specification: promptSet,
                systemPrompt: prompt,
                contentSHA256: sha256(data)
            ))
        }

        let scorerExecutableURL: URL?
        if manifest.scorer.kind == .externalProcess,
           let executable = manifest.scorer.executable {
            scorerExecutableURL = try declaredFileURL(
                executable,
                rootURL: rootURL,
                fileManager: fileManager
            )
            guard fileManager.isExecutableFile(atPath: scorerExecutableURL!.path) else {
                throw EvaluationPackError.invalidManifest(
                    "external scorer \(executable) is not executable"
                )
            }
            let data = try Data(contentsOf: scorerExecutableURL!)
            pins.append(try pin(data: data, relativePath: executable))
        } else {
            scorerExecutableURL = nil
        }

        let uniquePins = Dictionary(grouping: pins, by: \EvaluationPackFilePin.relativePath)
            .map { _, values in values[0] }
            .sorted { $0.relativePath < $1.relativePath }
        let packSHA256 = try canonicalSHA256(uniquePins)
        return LoadedEvaluationPack(
            rootURL: rootURL,
            manifestURL: manifestURL,
            manifest: manifest,
            manifestSHA256: sha256(manifestData),
            packSHA256: packSHA256,
            files: uniquePins,
            cases: loadedCases,
            promptSets: loadedPromptSets,
            imageURLs: imageURLs,
            scorerExecutableURL: scorerExecutableURL
        )
    }

    private static func validate(_ manifest: EvaluationPackManifest) throws {
        guard manifest.schemaVersion == EvaluationPackManifest.currentSchemaVersion else {
            throw EvaluationPackError.invalidManifest(
                "unsupported schema_version \(manifest.schemaVersion)"
            )
        }
        try validateIdentifier(manifest.id, field: "id")
        guard !manifest.version.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw EvaluationPackError.invalidManifest("version is required")
        }
        guard !manifest.description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw EvaluationPackError.invalidManifest("description is required")
        }
        guard !manifest.caseFiles.isEmpty, Set(manifest.caseFiles).count == manifest.caseFiles.count else {
            throw EvaluationPackError.invalidManifest("case_files must be non-empty and unique")
        }
        let imageFiles = manifest.imageFiles ?? []
        guard Set(imageFiles).count == imageFiles.count else {
            throw EvaluationPackError.invalidManifest("image_files must be unique")
        }
        for imageFile in imageFiles {
            try validateRelativePath(imageFile, field: "image_files")
        }
        let nonImageFiles = Set(
            manifest.caseFiles
                + manifest.promptSets.map(\.systemPromptFile)
                + [manifest.scorer.executable].compactMap { $0 }
        )
        let conflictingImageFiles = Set(imageFiles).intersection(nonImageFiles)
        guard conflictingImageFiles.isEmpty else {
            throw EvaluationPackError.invalidManifest(
                "image_files overlaps another declared file role: "
                    + conflictingImageFiles.sorted().joined(separator: ", ")
            )
        }
        let promptIDs = try uniqueIdentifiers(manifest.promptSets.map(\.id), field: "prompt_sets")
        let armIDs = try uniqueIdentifiers(manifest.arms.map(\.id), field: "arms")
        let profileIDs = try uniqueIdentifiers(manifest.samplingProfiles.map(\.id), field: "sampling_profiles")
        _ = try uniqueIdentifiers(manifest.gates.map(\.id), field: "gates")
        guard !manifest.arms.isEmpty else {
            throw EvaluationPackError.invalidManifest("at least one arm is required")
        }
        guard !manifest.samplingProfiles.isEmpty else {
            throw EvaluationPackError.invalidManifest("at least one sampling profile is required")
        }

        for promptSet in manifest.promptSets {
            try validateRelativePath(promptSet.systemPromptFile, field: "system_prompt_file")
        }
        for arm in manifest.arms {
            try validateIdentifier(arm.modelSlot, field: "arm.model_slot")
            if let adapterSlot = arm.adapterSlot {
                try validateIdentifier(adapterSlot, field: "arm.adapter_slot")
            }
            guard arm.adapterScale.isFinite, arm.adapterScale >= 0 else {
                throw EvaluationPackError.invalidManifest(
                    "arm \(arm.id) adapter_scale must be finite and non-negative"
                )
            }
            if let promptSet = arm.promptSet, !promptIDs.contains(promptSet) {
                throw EvaluationPackError.invalidManifest(
                    "arm \(arm.id) references unknown prompt_set \(promptSet)"
                )
            }
            if let selectedProfiles = arm.profileIDs {
                guard !selectedProfiles.isEmpty else {
                    throw EvaluationPackError.invalidManifest(
                        "arm \(arm.id) profile_ids cannot be empty"
                    )
                }
                guard Set(selectedProfiles).count == selectedProfiles.count else {
                    throw EvaluationPackError.invalidManifest(
                        "arm \(arm.id) profile_ids must be unique"
                    )
                }
                for profileID in selectedProfiles where !profileIDs.contains(profileID) {
                    throw EvaluationPackError.invalidManifest(
                        "arm \(arm.id) references unknown profile \(profileID)"
                    )
                }
            }
        }
        for profile in manifest.samplingProfiles {
            guard profile.temperature.isFinite, profile.temperature >= 0,
                  profile.topP.isFinite, profile.topP > 0, profile.topP <= 1,
                  profile.topK >= 0,
                  profile.minP.isFinite, profile.minP >= 0, profile.minP < 1 else {
                throw EvaluationPackError.invalidManifest(
                    "sampling profile \(profile.id) has invalid sampling values"
                )
            }
            if let effort = profile.reasoningEffort, !effort.isFinite || effort < 0 || effort > 1 {
                throw EvaluationPackError.invalidManifest(
                    "sampling profile \(profile.id) reasoning_effort must be from 0 through 1"
                )
            }
        }
        switch manifest.scorer.kind {
        case .assertions:
            guard manifest.scorer.executable == nil else {
                throw EvaluationPackError.invalidManifest(
                    "assertions scorer cannot declare an executable"
                )
            }
        case .externalProcess:
            guard let executable = manifest.scorer.executable else {
                throw EvaluationPackError.invalidManifest(
                    "external-process scorer requires executable"
                )
            }
            try validateRelativePath(executable, field: "scorer.executable")
        }
        guard manifest.scorer.timeoutSeconds.isFinite,
              manifest.scorer.timeoutSeconds > 0,
              manifest.scorer.timeoutSeconds <= 300 else {
            throw EvaluationPackError.invalidManifest(
                "scorer timeout_seconds must be greater than zero and at most 300"
            )
        }
        if let requirements = manifest.adapterRequirements,
           !requirements.requireTrainingManifest,
           requirements.requireCompletedTraining || requirements.requireBaseModelMatch {
            throw EvaluationPackError.invalidManifest(
                "completed-training and base-model requirements require a training manifest"
            )
        }
        for gate in manifest.gates {
            guard gate.threshold.isFinite else {
                throw EvaluationPackError.invalidManifest(
                    "gate \(gate.id) threshold must be finite"
                )
            }
            if gate.aggregation == .metricMean {
                guard let metricID = gate.metricID else {
                    throw EvaluationPackError.invalidManifest(
                        "metric-mean gate \(gate.id) requires metric_id"
                    )
                }
                try validateIdentifier(metricID, field: "gate.metric_id")
            } else if gate.metricID != nil {
                throw EvaluationPackError.invalidManifest(
                    "gate \(gate.id) metric_id is only valid for metric-mean"
                )
            }
            if let arms = gate.filter?.arms {
                for arm in arms where !armIDs.contains(arm) {
                    throw EvaluationPackError.invalidManifest(
                        "gate \(gate.id) references unknown arm \(arm)"
                    )
                }
            }
            if let profiles = gate.filter?.profiles {
                for profile in profiles where !profileIDs.contains(profile) {
                    throw EvaluationPackError.invalidManifest(
                        "gate \(gate.id) references unknown profile \(profile)"
                    )
                }
            }
        }
        guard manifest.defaults.trials > 0,
              manifest.defaults.maxTokens > 0,
              manifest.defaults.contextSize > 0,
              (1...20).contains(manifest.defaults.topLogprobs) else {
            throw EvaluationPackError.invalidManifest(
                "defaults require positive trials, max_tokens, context_size, and top_logprobs from 1 through 20"
            )
        }
        if manifest.defaults.responseFormat == .jsonObject,
           manifest.defaults.logprobs != .none {
            throw EvaluationPackError.invalidManifest(
                "defaults.response_format json_object requires defaults.logprobs none"
            )
        }
    }

    private static func validate(
        _ specification: EvaluationCase,
        scorer: EvaluationScorer,
        declaredImageFiles: Set<String>,
        source: String
    ) throws {
        do {
            try validateIdentifier(specification.id, field: "case.id")
        } catch {
            throw EvaluationPackError.invalidCase("\(source): \(error.localizedDescription)")
        }
        guard !specification.capabilityTags.isEmpty,
              specification.capabilityTags.allSatisfy({ !$0.isEmpty }) else {
            throw EvaluationPackError.invalidCase(
                "\(source): capability_tags must be non-empty"
            )
        }
        guard !specification.messages.isEmpty,
              specification.messages.contains(where: { $0.role == .user }),
              specification.messages.allSatisfy({ !$0.content.isEmpty }) else {
            throw EvaluationPackError.invalidCase(
                "\(source): messages require non-empty content and at least one user message"
            )
        }
        for message in specification.messages {
            guard let imageFile = message.imageFile else { continue }
            guard message.role == .user else {
                throw EvaluationPackError.invalidCase(
                    "\(source): image_file is supported only on user messages"
                )
            }
            do {
                try validateRelativePath(imageFile, field: "message.image_file")
            } catch {
                throw EvaluationPackError.invalidCase("\(source): \(error.localizedDescription)")
            }
            guard declaredImageFiles.contains(imageFile) else {
                throw EvaluationPackError.invalidCase(
                    "\(source): image_file \(imageFile) is not declared in image_files"
                )
            }
        }
        if let maxTokens = specification.maxTokens, maxTokens <= 0 {
            throw EvaluationPackError.invalidCase(
                "\(source): max_tokens must be greater than zero"
            )
        }
        let assertionIDs = specification.assertions.map(\.id)
        guard Set(assertionIDs).count == assertionIDs.count else {
            throw EvaluationPackError.invalidCase(
                "\(source): assertion ids must be unique"
            )
        }
        if scorer.kind == .assertions, specification.assertions.isEmpty {
            throw EvaluationPackError.invalidCase(
                "\(source): assertions scorer requires at least one assertion"
            )
        }
        for assertion in specification.assertions {
            do {
                try validateIdentifier(assertion.id, field: "assertion.id")
            } catch {
                throw EvaluationPackError.invalidCase("\(source): \(error.localizedDescription)")
            }
            switch assertion.kind {
            case .validJSONObject:
                guard assertion.value == nil else {
                    throw EvaluationPackError.invalidCase(
                        "\(source): valid-json-object assertion cannot declare value"
                    )
                }
            case .contains, .excludes, .regex, .notRegex:
                guard let value = assertion.value, !value.isEmpty else {
                    throw EvaluationPackError.invalidCase(
                        "\(source): assertion \(assertion.id) requires value"
                    )
                }
                if assertion.kind == .regex || assertion.kind == .notRegex {
                    do {
                        _ = try NSRegularExpression(pattern: value)
                    } catch {
                        throw EvaluationPackError.invalidCase(
                            "\(source): assertion \(assertion.id) has invalid regex"
                        )
                    }
                }
            }
        }
    }

    private static func uniqueIdentifiers(_ values: [String], field: String) throws -> Set<String> {
        for value in values {
            try validateIdentifier(value, field: field)
        }
        let result = Set(values)
        guard result.count == values.count else {
            throw EvaluationPackError.invalidManifest("\(field) ids must be unique")
        }
        return result
    }

    private static func validateIdentifier(_ value: String, field: String) throws {
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        guard !value.isEmpty,
              value.count <= 128,
              value.unicodeScalars.allSatisfy(allowed.contains) else {
            throw EvaluationPackError.invalidManifest(
                "\(field) must use 1-128 letters, numbers, dots, underscores, or hyphens"
            )
        }
    }

    private static func validateRelativePath(_ path: String, field: String) throws {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty,
              !path.hasPrefix("/"),
              !path.contains("\0"),
              !components.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }) else {
            throw EvaluationPackError.invalidManifest(
                "\(field) must be a normalized relative path without traversal"
            )
        }
    }

    private static func declaredFileURL(
        _ relativePath: String,
        rootURL: URL,
        fileManager: FileManager
    ) throws -> URL {
        try validateRelativePath(relativePath, field: "declared file")
        let unresolved = rootURL.appendingPathComponent(relativePath, isDirectory: false).standardizedFileURL
        let unresolvedValues = try resourceValues(for: unresolved)
        guard unresolvedValues.isSymbolicLink != true else {
            throw EvaluationPackError.invalidPath("\(relativePath) must not be a symbolic link")
        }
        let resolved = unresolved.resolvingSymlinksInPath()
        let rootPath = rootURL.resolvingSymlinksInPath().path
        guard resolved.path.hasPrefix(rootPath + "/") else {
            throw EvaluationPackError.invalidPath("\(relativePath) escapes the pack root")
        }
        guard unresolvedValues.isRegularFile == true,
              fileManager.fileExists(atPath: resolved.path) else {
            throw EvaluationPackError.invalidPath("\(relativePath) is not a regular file")
        }
        return resolved
    }

    private static func readDeclaredFile(
        _ url: URL,
        relativePath: String,
        rootURL: URL,
        fileManager: FileManager
    ) throws -> Data {
        if url.lastPathComponent == manifestFileName {
            let values = try resourceValues(for: url)
            guard values.isRegularFile == true, values.isSymbolicLink != true else {
                throw EvaluationPackError.invalidPath("\(relativePath) is not a regular non-symlink file")
            }
            return try Data(contentsOf: url)
        }
        return try Data(contentsOf: declaredFileURL(
            relativePath,
            rootURL: rootURL,
            fileManager: fileManager
        ))
    }

    private static func resourceValues(for url: URL) throws -> URLResourceValues {
        do {
            return try url.resourceValues(forKeys: [
                .isDirectoryKey,
                .isRegularFileKey,
                .isSymbolicLinkKey,
            ])
        } catch {
            throw EvaluationPackError.invalidPath("cannot inspect \(url.path): \(error)")
        }
    }

    private static func pin(data: Data, relativePath: String) throws -> EvaluationPackFilePin {
        EvaluationPackFilePin(
            relativePath: relativePath,
            byteCount: Int64(data.count),
            sha256: sha256(data)
        )
    }

    private static func canonicalSHA256<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return sha256(try encoder.encode(value))
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
