#if os(Linux)
import XCTest
@testable import MereRunCore

final class LlamaCLIProcessTests: XCTestCase {
    func testChatPassesRequestedReasoningModeToIsolatedCLI() async throws {
        let fixture = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-llama-cli-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: fixture) }
        try """
        #!/bin/sh
        while [ "$#" -gt 0 ]; do
          if [ "$1" = "--reasoning" ]; then
            shift
            printf '> \\n%s\\n\\nExiting...\\n' "$1"
            exit 0
          fi
          shift
        done
        exit 64
        """.write(to: fixture, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fixture.path)

        let process = LlamaCLIProcess(executableURL: fixture)
        for (showThinking, expected) in [(false, "off"), (true, "on")] {
            let request = ChatRequest(
                messages: [.init(role: .user, content: "Ready?")],
                showThinking: showThinking
            )
            let response = try await process.chat(request: request, modelPath: "/fixture/model.gguf")
            XCTAssertEqual(response.response, expected)
        }
    }

    func testExtractPerformanceParsesLlamaTimingLine() {
        let stdout = """
        READY READY

        [ Prompt: 100.4 t/s | Generation: 65.1 t/s ]

        Exiting...
        """

        let performance = LlamaCLIProcess.extractPerformance(from: stdout)

        XCTAssertEqual(performance.promptTokensPerSecond, 100.4)
        XCTAssertEqual(performance.generationTokensPerSecond, 65.1)
    }

    func testExtractPerformanceUsesLastTimingLine() {
        let stdout = """
        [ Prompt: 1.0 t/s | Generation: 2.0 t/s ]
        [ Prompt: 3.5 t/s | Generation: 4.5 t/s ]
        """

        let performance = LlamaCLIProcess.extractPerformance(from: stdout)

        XCTAssertEqual(performance.promptTokensPerSecond, 3.5)
        XCTAssertEqual(performance.generationTokensPerSecond, 4.5)
    }
}
#endif
