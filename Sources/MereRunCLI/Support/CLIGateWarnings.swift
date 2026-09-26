import Foundation

/// The capability gate's warnings for the command line being run: options that have no effect
/// for the selected model. `MereRunCLI.main` binds them around the command's `run()`; a command
/// run in-process (tests, workflow nodes) sees none unless its caller binds them.
enum CLIGateWarnings {
    @TaskLocal static var current: [String] = []
}

/// A machine-readable result printed with the run's gate warnings as a top-level `warnings`
/// array of strings. With no warnings the key is left out, so the JSON is byte-identical to
/// `value` alone. `value` must encode as a JSON object without a `warnings` key of its own.
struct GateWarned<Value: Encodable>: Encodable {
    let value: Value
    let warnings: [String]

    init(_ value: Value, warnings: [String] = CLIGateWarnings.current) {
        self.value = value
        self.warnings = warnings
    }

    private enum CodingKeys: String, CodingKey {
        case warnings
    }

    func encode(to encoder: any Encoder) throws {
        try value.encode(to: encoder)
        guard !warnings.isEmpty else { return }
        // The encoder hands back the object `value` just wrote, so the key joins it.
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(warnings, forKey: .warnings)
    }
}
