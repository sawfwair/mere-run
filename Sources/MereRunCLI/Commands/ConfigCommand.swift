import ArgumentParser
import Foundation
import MereRunCore

struct Config: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "config",
        abstract: "Get and set persisted mere.run configuration (e.g. Hugging Face token).",
        discussion: """
        Stores settings in ~/Library/Application Support/MereRun/config.json so they
        survive across shells without environment variables. The Hugging Face token is
        used to pull gated models (e.g. image-klein-9b).

        Resolution precedence for the HF token: --hf-token/HF_TOKEN env > config file.

        Keys: hf-token, hf-endpoint

        Examples:
          mere.run config set hf-token hf_xxxxxxxx
          mere.run config get hf-token
          mere.run config list
          mere.run config unset hf-token
          mere.run config path
        """,
        subcommands: [SetCmd.self, GetCmd.self, UnsetCmd.self, ListCmd.self, PathCmd.self]
    )

    static func resolveKey(_ raw: String) throws -> MereRunConfig.Key {
        guard let key = MereRunConfig.Key(rawValue: raw) else {
            let valid = MereRunConfig.Key.allCases.map(\.rawValue).joined(separator: ", ")
            throw ValidationError("Unknown config key '\(raw)'. Valid keys: \(valid).")
        }
        return key
    }

    struct SetCmd: ParsableCommand {
        static let configuration = CommandConfiguration(commandName: "set", abstract: "Set a config value.")
        @Argument(help: "Config key (hf-token, hf-endpoint).") var key: String
        @Argument(help: "Value to store.") var value: String
        func run() throws {
            let k = try Config.resolveKey(key)
            var cfg = MereRunConfig.load()
            cfg.set(k, value)
            try cfg.save()
            let shown = k.isSecret ? MereRunConfig.masked(value) : value
            print("Set \(k.rawValue) = \(shown)")
            print("Saved to \(MereRunConfig.fileURL.path)")
        }
    }

    struct GetCmd: ParsableCommand {
        static let configuration = CommandConfiguration(commandName: "get", abstract: "Print a config value (secrets masked).")
        @Argument(help: "Config key.") var key: String
        @Flag(name: [.long], help: "Reveal secret values in full instead of masking.") var reveal = false
        func run() throws {
            let k = try Config.resolveKey(key)
            guard let v = MereRunConfig.load().value(for: k) else {
                throw CleanExit.message("\(k.rawValue) is not set.")
            }
            print((k.isSecret && !reveal) ? MereRunConfig.masked(v) : v)
        }
    }

    struct UnsetCmd: ParsableCommand {
        static let configuration = CommandConfiguration(commandName: "unset", abstract: "Remove a config value.")
        @Argument(help: "Config key.") var key: String
        func run() throws {
            let k = try Config.resolveKey(key)
            var cfg = MereRunConfig.load()
            cfg.set(k, nil)
            try cfg.save()
            print("Unset \(k.rawValue)")
        }
    }

    struct ListCmd: ParsableCommand {
        static let configuration = CommandConfiguration(commandName: "list", abstract: "Show all config values (secrets masked).")
        func run() throws {
            let cfg = MereRunConfig.load()
            print("config: \(MereRunConfig.fileURL.path)")
            for k in MereRunConfig.Key.allCases {
                let v = cfg.value(for: k)
                let shown = v.map { k.isSecret ? MereRunConfig.masked($0) : $0 } ?? "(unset)"
                print("  \(k.rawValue) = \(shown)")
            }
        }
    }

    struct PathCmd: ParsableCommand {
        static let configuration = CommandConfiguration(commandName: "path", abstract: "Print the config file path.")
        func run() throws { print(MereRunConfig.fileURL.path) }
    }
}
