import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class VisionOCRCommandParsingTests: XCTestCase {
    func testVisionOCRParsesManagedDefaultModel() throws {
        let cmd = try VisionOCR.parse([
            "/tmp/image.png",
        ])

        XCTAssertEqual(cmd.images, ["/tmp/image.png"])
        XCTAssertEqual(cmd.backend, .lighton)
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.lightOnOCR.rawValue)
        XCTAssertEqual(cmd.infinityRuntime, .native)
        XCTAssertEqual(cmd.infinityParserCLI, "parser")
        XCTAssertEqual(cmd.infinityModel, Q35Resources.infinityParser2ProInt8ModelId)
        XCTAssertEqual(cmd.infinityBackend, .vllmServer)
        XCTAssertEqual(cmd.infinityTask, .doc2json)
        XCTAssertEqual(cmd.infinityOutputFormat, .md)
        XCTAssertEqual(cmd.infinityBatchSize, 1)
        XCTAssertEqual(cmd.infinityMinPixels, Q35Generator.qwen3VLMinPixels)
        XCTAssertEqual(cmd.infinityMaxPixels, Q35Generator.qwen3VLMaxPixels)
        XCTAssertEqual(cmd.maxTokens, 4096)
        XCTAssertEqual(cmd.temperature, 0.2, accuracy: 0.0001)
        XCTAssertFalse(cmd.compare)
        XCTAssertFalse(cmd.quiet)
    }

    func testVisionOCRParsesOverrides() throws {
        let cmd = try VisionOCR.parse([
            "/tmp/one.png",
            "/tmp/two.png",
            "--backend", "glm",
            "--model", "/tmp/lighton",
            "--glmocr-cli", "/opt/bin/glmocr",
            "--glm-config", "/tmp/glm.yaml",
            "--output-dir", "/tmp/out",
            "--max-tokens", "2048",
            "--temperature", "0.4",
            "--compare",
            "--quiet",
        ])

        XCTAssertEqual(cmd.images, ["/tmp/one.png", "/tmp/two.png"])
        XCTAssertEqual(cmd.backend, .glm)
        XCTAssertEqual(cmd.model, "/tmp/lighton")
        XCTAssertEqual(cmd.glmocrCLI, "/opt/bin/glmocr")
        XCTAssertEqual(cmd.glmConfig, "/tmp/glm.yaml")
        XCTAssertEqual(cmd.outputDir, "/tmp/out")
        XCTAssertEqual(cmd.maxTokens, 2048)
        XCTAssertEqual(cmd.temperature, 0.4, accuracy: 0.0001)
        XCTAssertTrue(cmd.compare)
        XCTAssertTrue(cmd.quiet)
    }

    func testVisionOCRParsesInfinityBackendOptions() throws {
        let cmd = try VisionOCR.parse([
            "/tmp/page.png",
            "--backend", "infinity",
            "--infinity-runtime", "external",
            "--infinity-parser-cli", "/opt/bin/parser",
            "--infinity-model", "infly/Infinity-Parser2-Pro",
            "--infinity-backend", "vllm-engine",
            "--infinity-api-url", "http://127.0.0.1:8000/v1/chat/completions",
            "--infinity-api-key", "test-key",
            "--infinity-task", "doc2md",
            "--infinity-output-format", "md",
            "--infinity-batch-size", "2",
            "--infinity-model-cache-dir", "/tmp/infinity-cache",
            "--infinity-min-pixels", "4096",
            "--infinity-max-pixels", "1048576",
        ])

        XCTAssertEqual(cmd.backend, .infinity)
        XCTAssertEqual(cmd.infinityRuntime, .external)
        XCTAssertEqual(cmd.infinityParserCLI, "/opt/bin/parser")
        XCTAssertEqual(cmd.infinityModel, "infly/Infinity-Parser2-Pro")
        XCTAssertEqual(cmd.infinityBackend, .vllmEngine)
        XCTAssertEqual(cmd.infinityAPIURL, "http://127.0.0.1:8000/v1/chat/completions")
        XCTAssertEqual(cmd.infinityAPIKey, "test-key")
        XCTAssertEqual(cmd.infinityTask, .doc2md)
        XCTAssertEqual(cmd.infinityOutputFormat, .md)
        XCTAssertEqual(cmd.infinityBatchSize, 2)
        XCTAssertEqual(cmd.infinityModelCacheDir, "/tmp/infinity-cache")
        XCTAssertEqual(cmd.infinityMinPixels, 4096)
        XCTAssertEqual(cmd.infinityMaxPixels, 1_048_576)
    }

    func testVisionOCRValidatesInfinityCustomPrompt() throws {
        XCTAssertThrowsError(try VisionOCR.parse([
            "/tmp/page.png",
            "--backend", "infinity",
            "--infinity-task", "custom",
        ]))

        let cmd = try VisionOCR.parse([
            "/tmp/page.png",
            "--backend", "infinity",
            "--infinity-task", "custom",
            "--infinity-prompt", "Extract the table as Markdown.",
        ])
        XCTAssertEqual(cmd.infinityPrompt, "Extract the table as Markdown.")
    }
}
