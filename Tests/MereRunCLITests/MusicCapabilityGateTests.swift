import Foundation
import MereRunContract
import Testing
import XCTest

@testable import MereRunCLI
@testable import MereRunCore

// Routing the generated gate cases can't reach: ACE-Step checkpoints chosen by flags other than
// `--model`, rules that narrow one family, and defaults an ignoring runtime still accepts.

/// A local ACE-Step checkpoints root holding `decoders` beside a VAE, the minimum the CLI's
/// checkpoint lookup accepts.
private func aceStepRoot(_ decoders: String...) throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    for decoder in decoders {
        let directory = root.appendingPathComponent(decoder, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        try Data().write(to: directory.appendingPathComponent("model.safetensors"))
    }
    try FileManager.default.createDirectory(at: root.appendingPathComponent("vae"), withIntermediateDirectories: true)
    return root
}

private func report(_ commandLine: String...) throws -> MereRunFamilyResolutionReport {
    try #require(CLICapabilityGate.evaluate(commandLine: commandLine)).report
}

@Test func aceStepCheckpointFlagsChooseTheFamilyTheCommandLoads() throws {
    let root = try aceStepRoot("acestep-v15-turbo", "acestep-v15-xl-base")
    defer { try? FileManager.default.removeItem(at: root) }

    // `--checkpoints-root` wins over the default model; its first decoder is Turbo.
    let turbo = try report("music", "generate", "song", "--checkpoints-root", root.path, "--task-type", "extract")
    #expect(turbo.family == "ace-step-turbo" && turbo.source == .identified)
    #expect(turbo.violations == ["--task-type extract is not supported by ACE-Step Turbo; use text2music, repaint, cover or cover-nofsq."])

    // `--decoder-subdirectory` picks Base in the same root, which runs extract and stems.
    let base = try report(
        "music", "generate", "song", "--checkpoints-root", root.path, "--decoder-subdirectory", "acestep-v15-xl-base",
        "--task-type", "extract", "--stems", "vocals"
    )
    #expect(base.family == "ace-step-base" && base.violations.isEmpty && base.warnings.isEmpty)

    // A managed Turbo id names the family without looking at disk.
    let managed = try report("music", "generate", "song", "--model", "music-acestep-xl-turbo", "--stems", "vocals")
    #expect(managed.family == "ace-step-turbo")
    #expect(managed.violations == ["--stems is not supported by ACE-Step Turbo. It applies to ACE-Step Base."])
}

@Test func theModelStillDecidesTheRuntimeWhenACheckpointsRootIsPassed() throws {
    let yue2 = try report("music", "generate", "song", "--model", "music-yue2", "--checkpoints-root", "/tmp/checkpoints")
    #expect(yue2.family == "yue2")
    #expect(yue2.violations
        == ["--checkpoints-root is not supported by YuE2. It applies to ACE-Step Turbo, ACE-Step SFT and ACE-Step Base."])
}

@Test func familyRulesNarrowGuidanceTasksAndDurations() throws {
    let turbo = try report("music", "generate", "song", "--guidance-scale", "7", "--guidance-mode", "cfg")
    #expect(turbo.family == "ace-step-turbo" && turbo.violations.isEmpty)
    #expect(turbo.warnings == [
        "--guidance-scale 7 has no effect with ACE-Step Turbo; it runs 1. Remove --guidance-scale or pass 1.",
        "--guidance-mode has no effect with ACE-Step Turbo. It applies to ACE-Step SFT and ACE-Step Base."
    ])
    #expect(try report("music", "generate", "song", "--guidance-scale", "1").warnings.isEmpty)

    let sft = try report("music", "generate", "song", "--model", "music-acestep-xl-sft", "--task-type", "lego")
    #expect(sft.violations == ["--task-type lego is not supported by ACE-Step SFT; use text2music, repaint, cover or cover-nofsq."])
    #expect(try report("music", "generate", "song", "--model", "music-acestep-xl-base", "--task-type", "lego").violations.isEmpty)

    let yue2 = try report("music", "generate", "song", "--model", "music-yue2", "--duration", "400", "--guidance-scale", "0.5")
    #expect(yue2.violations == ["--duration 400 is not supported by YuE2; use a value from 0 to 360."])
    #expect(try report("music", "generate", "song", "--model", "music-minimax-music3", "--duration", "600")
        .violations == ["--duration 600 is not supported by MiniMax Music 3; use a value from 0 to 360."])
    #expect(try report("music", "generate", "song", "--duration", "600").violations.isEmpty)
    #expect(try report("music", "generate", "song", "--model", "music-minimax-music3", "--candidates", "2")
        .violations == ["--candidates 2 is not supported by MiniMax Music 3; it runs 1. Remove --candidates or pass 1."])
}

@Test func magentaKeepsItsRejectionsAndOnlyWarnsForWhatItIgnores() throws {
    let magenta = try report(
        "music", "generate", "song", "--model", "music-magenta-rt2-small", "--seed", "4", "--quality", "draft"
    )
    #expect(magenta.violations
        == ["--seed is not supported by Magenta RealTime 2. It applies to ACE-Step Turbo, ACE-Step SFT, ACE-Step Base, MiniMax Music 3 and YuE2."])
    #expect(magenta.warnings
        == ["--quality has no effect with Magenta RealTime 2. It applies to ACE-Step Turbo, ACE-Step SFT and ACE-Step Base."])
    #expect(try report("music", "generate", "song", "--model", "music-magenta-rt2-small", "--task-type", "cover")
        .violations == ["--task-type is not supported by Magenta RealTime 2. It applies to ACE-Step Turbo, ACE-Step SFT and ACE-Step Base."])
}

@Test func separateOverlapsFollowEachModelsChunkSize() throws {
    // 32 divides the 352,800-sample ViperX chunk but not the 485,100-sample 4-stem chunk; 11 the reverse.
    #expect(try report("music", "separate", "a.wav", "--overlap", "32").violations.isEmpty)
    #expect(try report("music", "separate", "a.wav", "--overlap", "11").violations.count == 1)
    let fourStem = try report("music", "separate", "a.wav", "--model", "music-separate-bs-roformer-4stem", "--overlap", "32")
    #expect(fourStem.family == "bs-roformer-4stem" && fourStem.violations.count == 1)
    #expect(try report("music", "separate", "a.wav", "--model", "music-separate-bs-roformer-4stem", "--overlap", "11")
        .violations.isEmpty)
}

@Test func serveScopesMiniMaxAndACEStepOptions() throws {
    let minimax = try report("music", "serve", "--model", "MiniMaxAI/MiniMax-Music3", "--checkpoints-root", "/tmp/ace")
    #expect(minimax.family == "minimax-music3" && minimax.model == "music-minimax-music3")
    #expect(minimax.violations == ["--checkpoints-root is not supported by MiniMax Music 3. It applies to ACE-Step."])
    let defaults = try report("music", "serve", "--model", "music-minimax-music3", "--decoder-subdirectory", "acestep-v15-turbo")
    #expect(defaults.violations.isEmpty && defaults.warnings.count == 1)
    #expect(try report("music", "serve", "--model", "music-minimax-music3", "--decoder-subdirectory", "custom")
        .violations.count == 1)
    #expect(try report("music", "serve", "--memory-mode", "resident").violations
        == ["--memory-mode is not supported by ACE-Step. It applies to MiniMax Music 3."])
}

@Test func transcribeVariantOnlyWarnsForAManagedModel() throws {
    let managed = try report("music", "transcribe", "a.wav", "--model", "music-muscriptor-small", "--variant", "large")
    #expect(managed.violations.isEmpty && managed.warnings == ["--variant has no effect with MuScriptor."])
    let local = try report("music", "transcribe", "a.wav", "--model-path", "/tmp/muscriptor", "--variant", "large")
    #expect(local.source == .unidentified && local.warnings.isEmpty)
}

/// `MERERUN_MUSIC_ACESTEP_ROOT` is process state, so these set and restore it serially.
final class MusicACEStepRootOverrideGateTests: XCTestCase {
    private static let key = "MERERUN_MUSIC_ACESTEP_ROOT"
    private var original: String?

    override func setUp() {
        super.setUp()
        original = ProcessInfo.processInfo.environment[Self.key]
    }

    override func tearDown() {
        if let original {
            setenv(Self.key, original, 1)
        } else {
            unsetenv(Self.key)
        }
        super.tearDown()
    }

    private func report(_ arguments: [String]) throws -> MereRunFamilyResolutionReport {
        try XCTUnwrap(CLICapabilityGate.evaluate(commandLine: ["music", "generate", "song"] + arguments)).report
    }

    /// The override root wins over a managed Turbo id's own install, so a Base checkpoint there
    /// runs the Base-only tasks and stems the id alone would refuse.
    func testAnOverrideRootDecidesTheFamilyOfAManagedOrDefaultModel() throws {
        let root = try aceStepRoot("acestep-v15-xl-base")
        defer { try? FileManager.default.removeItem(at: root) }
        setenv(Self.key, root.path, 1)

        for model in [["--model", "music-acestep"], ["--model", "music-acestep-xl-turbo"], []] {
            for extra in [["--task-type", "extract"], ["--task-type", "lego"], ["--task", "complete"], ["--stems", "vocals"]] {
                let report = try report(model + extra)
                XCTAssertEqual(report.family, "ace-step-base", "\(model + extra)")
                XCTAssertEqual(report.source, model.isEmpty ? .defaultModel : .identified, "\(model + extra)")
                XCTAssertEqual(report.violations, [], "\(model + extra)")
            }
        }
        // Other runtimes never read the override.
        XCTAssertEqual(try report(["--model", "music-yue2"]).family, "yue2")
    }

    /// A Turbo override root keeps a Base id on Turbo, as the command would load it.
    func testAnOverrideRootAppliesInBothDirections() throws {
        let root = try aceStepRoot("acestep-v15-turbo")
        defer { try? FileManager.default.removeItem(at: root) }
        setenv(Self.key, root.path, 1)

        let report = try report(["--model", "music-acestep-xl-base", "--task-type", "extract"])
        XCTAssertEqual(report.family, "ace-step-turbo")
        XCTAssertEqual(
            report.violations,
            ["--task-type extract is not supported by ACE-Step Turbo; use text2music, repaint, cover or cover-nofsq."]
        )
    }

    /// An override that holds no usable checkpoint leaves the managed id's family in place.
    func testAnUnusableOverrideFallsBackToTheManagedFamily() throws {
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        setenv(Self.key, empty.path, 1)
        let report = try report(["--model", "music-acestep-xl-sft", "--task-type", "lego"])
        XCTAssertEqual(report.family, "ace-step-sft")
        XCTAssertEqual(report.violations.count, 1)
    }
}
