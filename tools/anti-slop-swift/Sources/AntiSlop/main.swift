import AntiSlopCore
import Foundation

// Usage:
//   anti-slop [paths...] [--disable=rule-id,rule-id] [--list-rules] [--help]
//
// Lints every .swift file under the given paths (recursively; defaults to the
// current directory). Exits 1 when any violation is found.

let usage = """
    anti-slop — opinionated rules that reject low-evidence Swift patterns.

    USAGE:
      anti-slop [PATH ...] [--disable=RULE,RULE] [--config=FILE]

    OPTIONS:
      --disable=RULES  Comma-separated rule ids to skip (additive with config).
      --config=FILE    Explicit config file; defaults to auto-discovered
                       .anti-slop.json in the current directory or an ancestor.
                       Format: { "disabled": ["rule-id", ...] }
      --list-rules     Print rule ids and summaries, then exit.
      -h, --help       Show this help.

    Paths default to the current directory. Only .swift files are linted;
    .build and .git directories are always skipped.

    Vendored, not a fixed dependency: read the rules in Sources/ and change
    them to match your team's standards.
    """

func printRuleList() {
    print("anti-slop rules:\n")
    for rule in SlopRule.allRules {
        let enabled = "  "
        print("\(enabled)\(rule.id)")
        print("    \(rule.summary)")
    }
}

struct Options {
    var paths: [String] = []
    var disabled: Set<String> = []
    var configPath: String?
}

func parseArguments(_ arguments: ArraySlice<String>) -> Options? {
    var options = Options()
    for argument in arguments {
        if argument.hasPrefix("--disable=") {
            let value = argument.dropFirst("--disable=".count)
            options.disabled.formUnion(value.split(separator: ",").map(String.init))
        } else if argument.hasPrefix("--config=") {
            options.configPath = String(argument.dropFirst("--config=".count))
        } else if argument == "--list-rules" || argument == "-h" || argument == "--help" {
            return nil // handled by caller before parsing matters
        } else if argument.hasPrefix("-") && argument != "-" {
            FileHandle.standardError.write(Data("[anti-slop] unknown option: \(argument)\n".utf8))
            exit(2)
        } else {
            options.paths.append(argument)
        }
    }
    return options
}

func swiftFiles(under path: String) -> [String] {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else {
        FileHandle.standardError.write(Data("[anti-slop] no such path: \(path)\n".utf8))
        exit(2)
    }
    guard isDirectory.boolValue else {
        return path.hasSuffix(".swift") ? [path] : []
    }

    let fm = FileManager.default
    let skippedDirectoryNames: Set<String> = [".git", ".build", ".swiftpm", "DerivedData"]
    var results: [String] = []
    let enumerator = fm.enumerator(atPath: path)

    while let element = enumerator?.nextObject() as? String {
        let components = element.split(separator: "/").map(String.init)
        if let last = components.last, skippedDirectoryNames.contains(last) {
            enumerator?.skipDescendants()
            continue
        }
        if element.hasSuffix(".swift") {
            results.append((path as NSString).appendingPathComponent(element))
        }
    }
    return results.sorted()
}

let arguments = CommandLine.arguments.dropFirst()

if arguments.contains("--list-rules") {
    printRuleList()
    exit(0)
}
if arguments.contains("--help") || arguments.contains("-h") {
    print(usage)
    exit(0)
}

guard var options = parseArguments(arguments) else {
    print(usage)
    exit(0)
}

let searchPaths = options.paths.isEmpty ? ["."] : options.paths

// Persistent config (.anti-slop.json, discovered walking up from the working
// directory) merges with --disable flags; CLI flags are additive.
do {
    options.disabled.formUnion(
        try AntiSlopConfig.resolve(
            explicitConfigPath: options.configPath,
            startingAt: FileManager.default.currentDirectoryPath
        )
    )
} catch let error as ConfigError {
    FileHandle.standardError.write(Data("[anti-slop] \(error.message)\n".utf8))
    exit(2)
} catch {
    FileHandle.standardError.write(Data("[anti-slop] config error: \(error)\n".utf8))
    exit(2)
}

let files = searchPaths.flatMap(swiftFiles)

let enabledRules = SlopRule.allRules.filter { !options.disabled.contains($0.id) }

var totalViolations: [Violation] = []
for file in files {
    guard let sourceText = try? String(contentsOfFile: file, encoding: .utf8) else {
        FileHandle.standardError.write(Data("[anti-slop] skipping \(file): not UTF-8\n".utf8))
        continue
    }
    let violations = AntiSlop.lint(
        source: sourceText,
        fileName: file,
        rules: enabledRules
    )
    totalViolations.append(contentsOf: violations)
}

totalViolations.sort { lhs, rhs in
    (lhs.fileName, lhs.line, lhs.column) < (rhs.fileName, rhs.line, rhs.column)
}

for violation in totalViolations {
    print(
        "\(violation.fileName):\(violation.line):\(violation.column): error: \(violation.description)"
    )
}

if !totalViolations.isEmpty {
    let ruleCount = enabledRules.count
    print(
        "\n[anti-slop] \(totalViolations.count) violation\(totalViolations.count == 1 ? "" : "s") across \(files.count) file\(files.count == 1 ? "" : "s") (\(ruleCount) rules)."
    )
    exit(1)
}
