import Foundation

public enum EvaluationCaseSplit: String, Codable, CaseIterable, Sendable {
    case training
    case validation
    case development
    case heldOut = "held-out"
    case regression
    case stability
    case sealedFrontier = "sealed-frontier"
}

public enum EvaluationMessageRole: String, Codable, CaseIterable, Sendable {
    case system
    case user
    case assistant
    case tool
}

public struct EvaluationMessage: Codable, Hashable, Sendable {
    public let role: EvaluationMessageRole
    public let content: String

    public init(role: EvaluationMessageRole, content: String) {
        self.role = role
        self.content = content
    }
}

public enum EvaluationAssertionKind: String, Codable, CaseIterable, Sendable {
    case contains
    case excludes
    case regex
    case notRegex = "not-regex"
    case validJSONObject = "valid-json-object"
}

public struct EvaluationAssertion: Codable, Hashable, Sendable {
    public let id: String
    public let kind: EvaluationAssertionKind
    public let value: String?
    public let caseInsensitive: Bool

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case value
        case caseInsensitive = "case_insensitive"
    }

    public init(
        id: String,
        kind: EvaluationAssertionKind,
        value: String? = nil,
        caseInsensitive: Bool = false
    ) {
        self.id = id
        self.kind = kind
        self.value = value
        self.caseInsensitive = caseInsensitive
    }
}

public struct EvaluationCase: Codable, Hashable, Sendable {
    public let id: String
    public let split: EvaluationCaseSplit
    public let capabilityTags: [String]
    public let messages: [EvaluationMessage]
    public let assertions: [EvaluationAssertion]
    public let maxTokens: Int?
    public let metadata: [String: String]

    enum CodingKeys: String, CodingKey {
        case id
        case split
        case capabilityTags = "capability_tags"
        case messages
        case assertions
        case maxTokens = "max_tokens"
        case metadata
    }

    public init(
        id: String,
        split: EvaluationCaseSplit,
        capabilityTags: [String],
        messages: [EvaluationMessage],
        assertions: [EvaluationAssertion] = [],
        maxTokens: Int? = nil,
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.split = split
        self.capabilityTags = capabilityTags
        self.messages = messages
        self.assertions = assertions
        self.maxTokens = maxTokens
        self.metadata = metadata
    }
}

public struct EvaluationPromptSet: Codable, Hashable, Sendable {
    public let id: String
    public let systemPromptFile: String

    enum CodingKeys: String, CodingKey {
        case id
        case systemPromptFile = "system_prompt_file"
    }

    public init(id: String, systemPromptFile: String) {
        self.id = id
        self.systemPromptFile = systemPromptFile
    }
}

public struct EvaluationArm: Codable, Hashable, Sendable {
    public let id: String
    public let modelSlot: String
    public let adapterSlot: String?
    public let adapterScale: Double
    public let promptSet: String?
    public let profileIDs: [String]?

    enum CodingKeys: String, CodingKey {
        case id
        case modelSlot = "model_slot"
        case adapterSlot = "adapter_slot"
        case adapterScale = "adapter_scale"
        case promptSet = "prompt_set"
        case profileIDs = "profile_ids"
    }

    public init(
        id: String,
        modelSlot: String,
        adapterSlot: String? = nil,
        adapterScale: Double = 1,
        promptSet: String? = nil,
        profileIDs: [String]? = nil
    ) {
        self.id = id
        self.modelSlot = modelSlot
        self.adapterSlot = adapterSlot
        self.adapterScale = adapterScale
        self.promptSet = promptSet
        self.profileIDs = profileIDs
    }
}

public struct EvaluationSamplingProfile: Codable, Hashable, Sendable {
    public let id: String
    public let temperature: Double
    public let topP: Double
    public let topK: Int
    public let minP: Double
    public let reasoningEffort: Double?
    public let showThinking: Bool

    enum CodingKeys: String, CodingKey {
        case id
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case reasoningEffort = "reasoning_effort"
        case showThinking = "show_thinking"
    }

    public init(
        id: String,
        temperature: Double,
        topP: Double,
        topK: Int = 0,
        minP: Double = 0,
        reasoningEffort: Double? = nil,
        showThinking: Bool = false
    ) {
        self.id = id
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.reasoningEffort = reasoningEffort
        self.showThinking = showThinking
    }
}

public enum EvaluationScorerKind: String, Codable, CaseIterable, Sendable {
    case assertions
    case externalProcess = "external-process"
}

public struct EvaluationScorer: Codable, Hashable, Sendable {
    public let kind: EvaluationScorerKind
    public let executable: String?
    public let arguments: [String]
    public let timeoutSeconds: Double

    enum CodingKeys: String, CodingKey {
        case kind
        case executable
        case arguments
        case timeoutSeconds = "timeout_seconds"
    }

    public init(
        kind: EvaluationScorerKind,
        executable: String? = nil,
        arguments: [String] = [],
        timeoutSeconds: Double = 30
    ) {
        self.kind = kind
        self.executable = executable
        self.arguments = arguments
        self.timeoutSeconds = timeoutSeconds
    }
}

public enum EvaluationGateAggregation: String, Codable, CaseIterable, Sendable {
    case passRate = "pass-rate"
    case meanScore = "mean-score"
    case hardFailureCount = "hard-failure-count"
    case metricMean = "metric-mean"
}

public enum EvaluationGateComparator: String, Codable, CaseIterable, Sendable {
    case greaterThanOrEqual = ">="
    case lessThanOrEqual = "<="
    case equal = "=="
}

public struct EvaluationGateFilter: Codable, Hashable, Sendable {
    public let splits: [EvaluationCaseSplit]?
    public let arms: [String]?
    public let profiles: [String]?
    public let capabilities: [String]?

    public init(
        splits: [EvaluationCaseSplit]? = nil,
        arms: [String]? = nil,
        profiles: [String]? = nil,
        capabilities: [String]? = nil
    ) {
        self.splits = splits
        self.arms = arms
        self.profiles = profiles
        self.capabilities = capabilities
    }
}

public struct EvaluationGate: Codable, Hashable, Sendable {
    public let id: String
    public let aggregation: EvaluationGateAggregation
    public let metricID: String?
    public let comparator: EvaluationGateComparator
    public let threshold: Double
    public let required: Bool
    public let filter: EvaluationGateFilter?

    enum CodingKeys: String, CodingKey {
        case id
        case aggregation
        case metricID = "metric_id"
        case comparator
        case threshold
        case required
        case filter
    }

    public init(
        id: String,
        aggregation: EvaluationGateAggregation,
        metricID: String? = nil,
        comparator: EvaluationGateComparator,
        threshold: Double,
        required: Bool = true,
        filter: EvaluationGateFilter? = nil
    ) {
        self.id = id
        self.aggregation = aggregation
        self.metricID = metricID
        self.comparator = comparator
        self.threshold = threshold
        self.required = required
        self.filter = filter
    }
}

public enum EvaluationLogprobMode: String, Codable, CaseIterable, Sendable {
    case none
    case summary
    case tokens
    case top
}

public struct EvaluationPackDefaults: Codable, Hashable, Sendable {
    public let trials: Int
    public let maxTokens: Int
    public let contextSize: Int
    public let logprobs: EvaluationLogprobMode
    public let topLogprobs: Int

    enum CodingKeys: String, CodingKey {
        case trials
        case maxTokens = "max_tokens"
        case contextSize = "context_size"
        case logprobs
        case topLogprobs = "top_logprobs"
    }

    public init(
        trials: Int = 2,
        maxTokens: Int = 512,
        contextSize: Int = 32_768,
        logprobs: EvaluationLogprobMode = .summary,
        topLogprobs: Int = 5
    ) {
        self.trials = trials
        self.maxTokens = maxTokens
        self.contextSize = contextSize
        self.logprobs = logprobs
        self.topLogprobs = topLogprobs
    }
}

public struct EvaluationAdapterRequirements: Codable, Hashable, Sendable {
    public let requireTrainingManifest: Bool
    public let requireCompletedTraining: Bool
    public let requireBaseModelMatch: Bool

    enum CodingKeys: String, CodingKey {
        case requireTrainingManifest = "require_training_manifest"
        case requireCompletedTraining = "require_completed_training"
        case requireBaseModelMatch = "require_base_model_match"
    }

    public init(
        requireTrainingManifest: Bool,
        requireCompletedTraining: Bool = true,
        requireBaseModelMatch: Bool = true
    ) {
        self.requireTrainingManifest = requireTrainingManifest
        self.requireCompletedTraining = requireCompletedTraining
        self.requireBaseModelMatch = requireBaseModelMatch
    }
}

public struct EvaluationPackManifest: Codable, Hashable, Sendable {
    public static let currentSchemaVersion = 1

    public let schemaVersion: Int
    public let id: String
    public let version: String
    public let description: String
    public let caseFiles: [String]
    public let promptSets: [EvaluationPromptSet]
    public let arms: [EvaluationArm]
    public let samplingProfiles: [EvaluationSamplingProfile]
    public let scorer: EvaluationScorer
    public let gates: [EvaluationGate]
    public let defaults: EvaluationPackDefaults
    public let adapterRequirements: EvaluationAdapterRequirements?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case id
        case version
        case description
        case caseFiles = "case_files"
        case promptSets = "prompt_sets"
        case arms
        case samplingProfiles = "sampling_profiles"
        case scorer
        case gates
        case defaults
        case adapterRequirements = "adapter_requirements"
    }

    public init(
        schemaVersion: Int = Self.currentSchemaVersion,
        id: String,
        version: String,
        description: String,
        caseFiles: [String],
        promptSets: [EvaluationPromptSet] = [],
        arms: [EvaluationArm],
        samplingProfiles: [EvaluationSamplingProfile],
        scorer: EvaluationScorer,
        gates: [EvaluationGate],
        defaults: EvaluationPackDefaults = EvaluationPackDefaults(),
        adapterRequirements: EvaluationAdapterRequirements? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.id = id
        self.version = version
        self.description = description
        self.caseFiles = caseFiles
        self.promptSets = promptSets
        self.arms = arms
        self.samplingProfiles = samplingProfiles
        self.scorer = scorer
        self.gates = gates
        self.defaults = defaults
        self.adapterRequirements = adapterRequirements
    }
}

public struct EvaluationMetric: Codable, Hashable, Sendable {
    public let id: String
    public let value: Double

    public init(id: String, value: Double) {
        self.id = id
        self.value = value
    }
}

public struct EvaluationScorerRequest: Codable, Hashable, Sendable {
    public let schemaVersion: Int
    public let packID: String
    public let packVersion: String
    public let packSHA256: String
    public let caseID: String
    public let caseSHA256: String
    public let armID: String
    public let profileID: String
    public let trial: Int
    public let modelID: String
    public let adapterSHA256: String?
    public let response: String
    public let responseSHA256: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case packID = "pack_id"
        case packVersion = "pack_version"
        case packSHA256 = "pack_sha256"
        case caseID = "case_id"
        case caseSHA256 = "case_sha256"
        case armID = "arm_id"
        case profileID = "profile_id"
        case trial
        case modelID = "model_id"
        case adapterSHA256 = "adapter_sha256"
        case response
        case responseSHA256 = "response_sha256"
    }

    public init(
        schemaVersion: Int = 1,
        packID: String,
        packVersion: String,
        packSHA256: String,
        caseID: String,
        caseSHA256: String,
        armID: String,
        profileID: String,
        trial: Int,
        modelID: String,
        adapterSHA256: String?,
        response: String,
        responseSHA256: String
    ) {
        self.schemaVersion = schemaVersion
        self.packID = packID
        self.packVersion = packVersion
        self.packSHA256 = packSHA256
        self.caseID = caseID
        self.caseSHA256 = caseSHA256
        self.armID = armID
        self.profileID = profileID
        self.trial = trial
        self.modelID = modelID
        self.adapterSHA256 = adapterSHA256
        self.response = response
        self.responseSHA256 = responseSHA256
    }
}

public struct EvaluationScorerResponse: Codable, Hashable, Sendable {
    public let schemaVersion: Int
    public let passed: Bool
    public let score: Double
    public let metrics: [EvaluationMetric]
    public let hardFailures: [String]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case passed
        case score
        case metrics
        case hardFailures = "hard_failures"
    }

    public init(
        schemaVersion: Int = 1,
        passed: Bool,
        score: Double,
        metrics: [EvaluationMetric] = [],
        hardFailures: [String] = []
    ) {
        self.schemaVersion = schemaVersion
        self.passed = passed
        self.score = score
        self.metrics = metrics
        self.hardFailures = hardFailures
    }
}
