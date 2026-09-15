import Foundation

public enum MereRunCapabilityValueKind: String, Codable, Sendable {
    case string
    case integer
    case number
    case boolean
    case file
    case directory
    case choice
}

public struct MereRunCapabilityArgument: Codable, Equatable, Sendable {
    public let name: String
    public let label: String
    public let kind: MereRunCapabilityValueKind
    public let required: Bool
    public let repeatable: Bool

    public init(name: String, label: String, kind: MereRunCapabilityValueKind, required: Bool, repeatable: Bool = false) {
        self.name = name
        self.label = label
        self.kind = kind
        self.required = required
        self.repeatable = repeatable
    }

    private enum CodingKeys: String, CodingKey {
        case name, label, kind, required, repeatable
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        label = try container.decode(String.self, forKey: .label)
        kind = try container.decode(MereRunCapabilityValueKind.self, forKey: .kind)
        required = try container.decode(Bool.self, forKey: .required)
        repeatable = try container.decodeIfPresent(Bool.self, forKey: .repeatable) ?? false
    }
}

/// How prominently a shell should surface an option. `essential` options are the
/// two to four controls a shell keeps visible next to the prompt, `standard`
/// options fill the regular inspector sections, and `expert` options collapse
/// under an advanced disclosure.
public enum MereRunCapabilityOptionTier: String, Codable, Sendable {
    case essential
    case standard
    case expert
}

/// A numeric range hint for integer and number options. Every field is optional
/// so a contract entry can declare only a lower bound or only a step.
public struct MereRunCapabilityRange: Codable, Equatable, Sendable {
    public let min: Double?
    public let max: Double?
    public let step: Double?

    public init(min: Double? = nil, max: Double? = nil, step: Double? = nil) {
        self.min = min
        self.max = max
        self.step = step
    }
}

/// Shared group names shells use to section options. Capabilities may use other
/// strings; these are the ones the built-in shells recognize.
public enum MereRunCapabilityOptionGroup {
    public static let prompt = "Prompt"
    public static let inputs = "Inputs"
    public static let output = "Output"
    public static let modelAndAdapters = "Model & adapters"
    public static let sampling = "Sampling"
    public static let run = "Run"
}

public struct MereRunCapabilityOption: Codable, Equatable, Sendable {
    public let flag: String
    public let label: String
    public let kind: MereRunCapabilityValueKind
    public let required: Bool
    public let repeatable: Bool
    public let choices: [String]
    /// The CLI's static ArgumentParser default rendered as the CLI would parse it
    /// (`"1024"`, `"0.7"`, `"peak"`). Absent when the default is machine- or
    /// model-specific.
    public let defaultValue: String?
    /// Section name for shells; see `MereRunCapabilityOptionGroup`.
    public let group: String?
    public let tier: MereRunCapabilityOptionTier?
    public let range: MereRunCapabilityRange?
    /// Flag of another option on the same capability that must be set for this
    /// option to have any effect.
    public let dependsOn: String?

    enum CodingKeys: String, CodingKey {
        case flag
        case label
        case kind
        case required
        case repeatable
        case choices
        case defaultValue = "default_value"
        case group
        case tier
        case range
        case dependsOn = "depends_on"
    }

    public init(
        flag: String,
        label: String,
        kind: MereRunCapabilityValueKind,
        required: Bool = false,
        repeatable: Bool = false,
        choices: [String] = [],
        defaultValue: String? = nil,
        group: String? = nil,
        tier: MereRunCapabilityOptionTier? = nil,
        range: MereRunCapabilityRange? = nil,
        dependsOn: String? = nil
    ) {
        self.flag = flag
        self.label = label
        self.kind = kind
        self.required = required
        self.repeatable = repeatable
        self.choices = choices
        self.defaultValue = defaultValue
        self.group = group
        self.tier = tier
        self.range = range
        self.dependsOn = dependsOn
    }
}

extension MereRunCapabilityOption {
    /// Shared presentation for commands without a bespoke form. Authored metadata always wins.
    fileprivate func withPresentation(outputFlag: String?) -> Self {
        let section: String
        if flag == outputFlag || ["--output", "--json-output", "--mask-output-dir", "--structured-prompt-output"].contains(flag) {
            section = MereRunCapabilityOptionGroup.output
        } else if ["--prompt", "--text", "--query", "--system", "--system-prompt", "--negative-prompt", "--lyrics"].contains(flag) {
            section = MereRunCapabilityOptionGroup.prompt
        } else if ["--model", "--model-root", "--lora", "--lora-scale", "--adapter"].contains(flag) {
            section = MereRunCapabilityOptionGroup.modelAndAdapters
        } else if [.file, .directory].contains(kind) {
            section = MereRunCapabilityOptionGroup.inputs
        } else if ["--seed", "--steps", "--cfg", "--temperature", "--top-p", "--top-k", "--max-tokens", "--width", "--height"].contains(flag) {
            section = MereRunCapabilityOptionGroup.sampling
        } else {
            section = MereRunCapabilityOptionGroup.run
        }
        return Self(flag: flag, label: label, kind: kind, required: required, repeatable: repeatable,
            choices: choices, defaultValue: defaultValue, group: group ?? section,
            tier: tier ?? (required ? .essential : .standard), range: range, dependsOn: dependsOn)
    }
}

/// What a successful run leaves behind when the caller passes no destination.
/// `text` prints its result to stdout, `service` runs until it is stopped, and
/// `file` and `directory` always write the artifact, at a default path when the
/// caller names none.
public enum MereRunCapabilityOutputKind: String, Codable, Sendable {
    case text
    case file
    case directory
    case service
}

/// The artifact one run produces, and how the caller asks for it.
///
/// A `text` or `service` capability that still declares a `flag` writes that
/// artifact only when the flag is passed (`optional` is then `true`); `kind`
/// describes what the run does without it. Look `flag` up in the capability's
/// `options` to learn whether it names a file or a directory.
public struct MereRunCapabilityOutput: Codable, Equatable, Sendable {
    public let kind: MereRunCapabilityOutputKind
    /// The written artifact's extension when it is a file, and the command
    /// always writes the same one. Absent for directories and for commands
    /// whose extension follows another option (`speech diarize --format`).
    public let fileExtension: String?
    /// The option whose value names the destination. Absent when the run writes
    /// nothing, or chooses the location itself (`adapter pull`, `image run-plan`).
    public let flag: String?
    /// True when the artifact is written only if `flag` is passed.
    public let optional: Bool

    enum CodingKeys: String, CodingKey {
        case kind
        case fileExtension = "file_extension"
        case flag
        case optional
    }

    public init(
        kind: MereRunCapabilityOutputKind,
        fileExtension: String? = nil,
        flag: String? = nil,
        optional: Bool = false
    ) {
        self.kind = kind
        self.fileExtension = fileExtension
        self.flag = flag
        self.optional = optional
    }

    /// `flag` and `optional` are additive: a document written before they
    /// existed decodes with no destination flag and a mandatory artifact.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        kind = try container.decode(MereRunCapabilityOutputKind.self, forKey: .kind)
        fileExtension = try container.decodeIfPresent(String.self, forKey: .fileExtension)
        flag = try container.decodeIfPresent(String.self, forKey: .flag)
        optional = try container.decodeIfPresent(Bool.self, forKey: .optional) ?? false
    }

    /// Absent fields stay absent so a decoder that predates them sees the same
    /// JSON it always did.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(kind, forKey: .kind)
        try container.encodeIfPresent(fileExtension, forKey: .fileExtension)
        try container.encodeIfPresent(flag, forKey: .flag)
        if optional { try container.encode(true, forKey: .optional) }
    }
}

public struct MereRunCommandCapability: Codable, Equatable, Sendable {
    public let id: String
    public let command: [String]
    public let title: String
    public let summary: String
    public let arguments: [MereRunCapabilityArgument]
    public let options: [MereRunCapabilityOption]
    public let output: MereRunCapabilityOutput

    public init(
        id: String,
        command: [String],
        title: String,
        summary: String,
        arguments: [MereRunCapabilityArgument] = [],
        options: [MereRunCapabilityOption],
        output: MereRunCapabilityOutput
    ) {
        self.id = id
        self.command = command
        self.title = title
        self.summary = summary
        self.arguments = arguments
        self.options = options.map { $0.withPresentation(outputFlag: output.flag) }
        self.output = output
    }
}

public struct MereRunCapabilityDocument: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let commands: [MereRunCommandCapability]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case commands
    }

    public init(schemaVersion: Int, commands: [MereRunCommandCapability]) {
        self.schemaVersion = schemaVersion
        self.commands = commands
    }
}
