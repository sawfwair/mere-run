import MereRunCore
import XCTest
@testable import MereRunCLI

final class MachineInferenceAdmissionTests: XCTestCase {
    private let gibibyte = UInt64(1_073_741_824)

    func testManagedEmbeddingGemmaUsesSmallAdmissionWithoutRelaxingUnknownModels() async throws {
        for selector in [["--model", EmbeddingGemma2Catalog.modelID], ["-m", EmbeddingGemma2Catalog.modelID],
                         ["--model=" + EmbeddingGemma2Catalog.modelID]] {
            XCTAssertEqual(CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "text", "embed", "--image", "photo.png"] + selector),
                           MachineInferenceRequest(label: "text embed", resourceClass: .small))
        }
        for selector in [[], ["--model", "/tmp/unknown-checkpoint"], ["--model", "text-embed-qwen3-0.6b"]] {
            XCTAssertEqual(CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "text", "embed", "hello"] + selector),
                           MachineInferenceRequest(label: "text embed", resourceClass: .standard))
        }
        let request = MachineInferenceRequest(label: "text embed", resourceClass: .small)
        let coordinator = makeCoordinator(directory: try temporaryDirectory(), processID: 301,
                                          physicalMemoryBytes: 36 * gibibyte, availableMemoryBytes: 14 * gibibyte)
        let lease = try await coordinator.acquire(request)
        XCTAssertEqual(try coordinator.snapshot().activePermits, 1)
        lease.release()
        let low = makeCoordinator(directory: try temporaryDirectory(), processID: 302,
                                  physicalMemoryBytes: 36 * gibibyte, availableMemoryBytes: 5 * gibibyte)
        do {
            let lease = try await low.acquire(request)
            lease.release()
            XCTFail("Low memory must still block the managed encoder")
        } catch let error as MachineInferenceAdmissionError {
            XCTAssertEqual(error, .insufficientMemory(available: 5 * gibibyte, required: 6 * gibibyte))
        }
    }

    func testSmallLightOnOCRAdmissionUsesTheActualSingleBackendPlan() async throws {
        let small = MachineInferenceRequest(label: "vision ocr", resourceClass: .small)
        let standard = MachineInferenceRequest(label: "vision ocr", resourceClass: .standard)
        let command = ["mere.run", "vision", "ocr", "photo.png"]
        XCTAssertEqual(CLIInferenceAdmissionClassifier.request(arguments: command), small)
        for model in ["vision-ocr-lighton", "vision-ocr-lighton-3-0.8b"] {
            for selector in [["--model", model], ["-m", model], ["--model=" + model], ["-qm", model]] {
                XCTAssertEqual(CLIInferenceAdmissionClassifier.request(arguments: command + selector), small)
                for backend in [["--backend", "lighton"], ["-b", "lighton"], ["--backend=lighton"]] {
                    XCTAssertEqual(CLIInferenceAdmissionClassifier.request(arguments: command + selector + backend), small)
                }
                for backend in [["--compare"], ["--backend", "glm"], ["-b", "infinity"], ["--backend=glm"]] {
                    XCTAssertEqual(CLIInferenceAdmissionClassifier.request(arguments: command + selector + backend), standard)
                }
            }
        }
        for model in ["vision-ocr-lighton-3-4b", "/tmp/unknown-ocr"] {
            XCTAssertEqual(CLIInferenceAdmissionClassifier.request(arguments: command + ["--model", model]), standard)
        }
        XCTAssertEqual(CLIInferenceAdmissionClassifier.request(
            arguments: ["mere.run", "--models-root=/tmp/ocr", "vision", "ocr", "photo.png"]
        ), small)
        let coordinator = makeCoordinator(directory: try temporaryDirectory(), processID: 303,
                                          physicalMemoryBytes: 36 * gibibyte, availableMemoryBytes: 12 * gibibyte)
        let lease = try await coordinator.acquire(small)
        XCTAssertEqual(try coordinator.snapshot().activePermits, 1)
        lease.release()
        let low = makeCoordinator(directory: try temporaryDirectory(), processID: 304,
                                  physicalMemoryBytes: 36 * gibibyte, availableMemoryBytes: 5 * gibibyte)
        do {
            let lease = try await low.acquire(small)
            lease.release()
            XCTFail("Small OCR still requires six GiB headroom")
        } catch let error as MachineInferenceAdmissionError {
            XCTAssertEqual(error, .insufficientMemory(available: 5 * gibibyte, required: 6 * gibibyte))
        }
    }

    /// Marigold's download sits under the 48 GiB estimate threshold, but the catalog
    /// declares a 64 GB minimum because the BF16 base is loaded before quantization.
    /// The declaration, not a command list, makes every `vision depth` run large.
    func testSingleImageDepthAdmitsAsLargeFromTheCatalogMinimumMemory() throws {
        let descriptor = try XCTUnwrap(ManagedModelCapabilityCatalog.descriptor(for: MarigoldV2Repository.modelId))
        XCTAssertGreaterThanOrEqual(descriptor.minimumUnifiedMemoryGB, MachineInferenceClass.largeDeclaredMinimumMemoryGB)
        XCTAssertLessThan(MarigoldV2Repository.estimatedDownloadBytes, Int64(48 * gibibyte))

        let large = MachineInferenceRequest(label: "vision depth", resourceClass: .large)
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "vision", "depth", "photo.png"]),
            large
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "vision", "depth", "photo.png", "--model", "vision-depth-marigold-v2"]
            ),
            large
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "vision", "depth", "photo.png", "--model", "/tmp/marigold-root", "--native"]
            ),
            large
        )
        XCTAssertNil(
            CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "vision", "depth", "photo.png", "--dry-run"])
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "vision", "caption", "photo.png"]),
            MachineInferenceRequest(label: "vision caption", resourceClass: .standard)
        )
    }

    func testSmallMusicModelsAdmitWithLessThanStandardHeadroom() async throws {
        for (offset, modelID) in ["music-acestep", "music-magenta-rt2-small"].enumerated() {
            let directory = try temporaryDirectory()
            let coordinator = makeCoordinator(
                directory: directory,
                processID: Int32(107 + offset),
                physicalMemoryBytes: 24 * gibibyte,
                availableMemoryBytes: 14 * gibibyte
            )
            let request = try XCTUnwrap(
                CLIInferenceAdmissionClassifier.request(
                    arguments: ["mere.run", "music", "generate", "test", "--model", modelID]
                )
            )

            let lease = try await coordinator.acquire(request)
            XCTAssertEqual(request.resourceClass, .small, modelID)
            XCTAssertEqual(try coordinator.snapshot().activePermits, 1, modelID)
            lease.release()
        }
    }

    func testCLIClassifierProtectsMediaAndLeavesLightweightCommandsFree() {
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "image", "generate", "--prompt", "test"]
            ),
            MachineInferenceRequest(label: "image generate", resourceClass: .standard)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "speech", "synthesize", "hello"]
            ),
            MachineInferenceRequest(label: "speech synthesize", resourceClass: .small)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "music", "generate", "test"]
            ),
            MachineInferenceRequest(label: "music generate", resourceClass: .small)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: [
                    "mere.run", "music", "generate", "test", "--model", "music-magenta-rt2-small",
                ]
            ),
            MachineInferenceRequest(label: "music generate", resourceClass: .small)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: [
                    "mere.run", "music", "generate", "test", "--model", "music-acestep-xl-sft",
                ]
            ),
            MachineInferenceRequest(label: "music generate", resourceClass: .standard)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: [
                    "mere.run", "music", "generate", "test", "--model", "music-minimax-music3",
                ]
            ),
            MachineInferenceRequest(label: "music generate", resourceClass: .standard)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "music", "train-adapter", "dataset.jsonl"]
            ),
            MachineInferenceRequest(label: "music train-adapter", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "video", "generate", "test"]
            ),
            MachineInferenceRequest(label: "video generate", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "geo", "fire", "input.safetensors"]
            ),
            MachineInferenceRequest(label: "geo fire", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "geo", "olmoearth", "input.safetensors"]
            ),
            MachineInferenceRequest(label: "geo olmoearth", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "text", "chat", "--model", "text-chat-deepseek-v4-flash"]
            ),
            MachineInferenceRequest(label: "text chat", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "text", "chat", "--model=text-chat-inkling-small"]
            ),
            MachineInferenceRequest(label: "text chat", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "text", "chat", "--model", "text-chat-laguna-xs-2-1"]
            ),
            MachineInferenceRequest(label: "text chat", resourceClass: .standard)
        )
        XCTAssertNil(
            CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "status"])
        )
        XCTAssertNil(
            CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "model", "list"])
        )
        XCTAssertNil(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "agent", "start", "--model", "vision-chat-q38-27b"]
            )
        )
        XCTAssertNil(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "api", "serve", "--model", "vision-chat-q38-27b"]
            )
        )
        XCTAssertNil(
            CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "image", "generate", "--help"])
        )
        XCTAssertNil(
            CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "model", "benchmark", "fused", "--dry-run", "--json"]
            )
        )
        XCTAssertNil(
            CLIInferenceAdmissionClassifier.request(
                arguments: [
                    "mere.run", "model", "benchmark", "fused-fixture", "selected.jsonl", "--check",
                ]
            )
        )
    }

    func testImageAndChatPreflightsDoNotReserveInferenceResources() {
        let requests = [
            ["image", "generate", "--model", "image-zimage-nano", "--prompt", "test"],
            ["text", "chat", "--model", "text-chat-gemma4-12b-4bit", "--prompt", "test"],
            ["text", "chat", "--model", "text-chat-deepseek-v4-flash", "--prompt", "test"],
        ]
        for request in requests {
            let arguments = ["mere.run", "--models-root", "/tmp/preflight-models"] + request
            XCTAssertNotNil(CLIInferenceAdmissionClassifier.request(arguments: arguments))
            XCTAssertNil(CLIInferenceAdmissionClassifier.request(
                arguments: arguments + ["--preflight", "--json"]
            ), request.joined(separator: " "))
        }
    }

    func testOfflineGuidesNeverReserveInferenceMemory() {
        for model in ["text-agent-ornith-35b-mlx", "text-chat-deepseek-v4-flash"] {
            for path in [[], ["text", "chat"]] {
                XCTAssertNil(CLIInferenceAdmissionClassifier.request(
                    arguments: ["mere.run", "guide"] + path + ["--model", model, "--json"]
                ))
            }
        }
    }

    func testAPIServerKeepsInternalConcurrencyInsideWeightedReservation() {
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.apiServerRequest(engine: .textChatGemma4),
            MachineInferenceRequest(label: "api serve text-chat-gemma4", resourceClass: .standard)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.apiServerRequest(engine: .textChatDeepseekV4Flash),
            MachineInferenceRequest(label: "api serve text-chat-deepseek-v4-flash", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.apiServerRequest(
                engine: .textChatNemotronOmni,
                modelID: "omni-chat-nemotron3-nano-30b-a3b-bf16"
            ),
            MachineInferenceRequest(label: "api serve text-chat-nemotron-omni", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.apiServerRequest(
                engine: .textChatLaguna,
                modelID: "text-chat-laguna-s-2-1"
            ),
            MachineInferenceRequest(label: "api serve text-chat-laguna", resourceClass: .large)
        )
        XCTAssertEqual(
            CLIInferenceAdmissionClassifier.apiServerRequest(
                engine: .textChatLaguna,
                modelID: "text-chat-laguna-xs-2-1"
            ),
            MachineInferenceRequest(label: "api serve text-chat-laguna", resourceClass: .standard)
        )
    }

    private func makeCoordinator(
        directory: URL,
        processID: Int32,
        bootSessionID: String = "test-boot",
        physicalMemoryBytes: UInt64 = 128 * 1_073_741_824,
        availableMemoryBytes: UInt64? = 96 * 1_073_741_824,
        availableDiskBytes: UInt64? = 100 * 1_073_741_824,
        processIsAlive: @escaping @Sendable (Int32) -> Bool = { _ in true }
    ) -> MachineInferenceCoordinator {
        MachineInferenceCoordinator(
            stateDirectory: directory,
            processID: processID,
            bootSessionID: bootSessionID,
            currentDate: { Date() },
            hostSnapshot: {
                MachineInferenceHostSnapshot(
                    physicalMemoryBytes: physicalMemoryBytes,
                    availableMemoryBytes: availableMemoryBytes,
                    memoryPressure: .nominal,
                    availableDiskBytes: availableDiskBytes
                )
            },
            processIsAlive: processIsAlive
        )
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-machine-admission-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        return url
    }

    private func waitUntil(
        timeoutNanoseconds: UInt64 = 2_000_000_000,
        condition: () throws -> Bool
    ) async throws {
        let started = ContinuousClock.now
        while try !condition() {
            if started.duration(to: .now) > .nanoseconds(Int64(timeoutNanoseconds)) {
                XCTFail("Timed out waiting for admission state")
                return
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
    }
}
