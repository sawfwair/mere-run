import Foundation
import XCTest
@testable import MereRunCLI

final class WorkflowGraphTests: XCTestCase {
    func testNVIDIAMemoryProbeParsesLargestGPU() {
        XCTAssertEqual(
            WorkflowExecutorProbe.parseNVIDIAMemoryBytes("8192\n16384 MiB\n"),
            16_384 * 1_024 * 1_024
        )
        XCTAssertNil(WorkflowExecutorProbe.parseNVIDIAMemoryBytes("not available\n"))
    }

    func testLegacyWorkerProbeConservativelyDefaultsNewCapabilities() throws {
        let probe = try WorkflowBundleCodec.decoder().decode(
            WorkflowExecutorProbe.self,
            from: Data("""
            {
              "schema_version": 1,
              "worker_version": "0.22.0",
              "contract_versions": ["mere.run/job-bundle.v1"],
              "platform": "linux",
              "architecture": "arm64",
              "accelerator_backend": "cuda",
              "memory_bytes": 1024,
              "available_disk_bytes": null,
              "node_kinds": [],
              "installed_model_ids": [],
              "providers": []
            }
            """.utf8)
        )

        XCTAssertEqual(probe.systemMemoryBytes, 0)
        XCTAssertEqual(probe.logicalCPUCores, 0)
        XCTAssertFalse(probe.networkAccess)
        XCTAssertEqual(probe.availableSecretNames, [])
    }

    func testWorkerVersionCompatibilityBlocksBeforeSubmission() {
        let probe = WorkflowExecutorProbe(
            schemaVersion: 1,
            workerVersion: "0.22.0",
            contractVersions: [WorkflowJobManifest.contractVersion],
            platform: "linux",
            architecture: "x86_64",
            acceleratorBackend: "cuda",
            memoryBytes: 16_000,
            availableDiskBytes: 100_000,
            nodeKinds: ["image.generate"],
            installedModelIDs: []
        )
        let requirements = WorkflowJobRequirements(
            minimumMereRunVersion: "0.23.0",
            nodeKinds: ["image.generate"],
            modelIDs: [],
            acceleratorBackends: ["cuda"],
            minimumAcceleratorMemoryBytes: nil
        )
        let job = WorkflowJobManifest(
            contractVersion: WorkflowJobManifest.contractVersion,
            jobID: UUID().uuidString,
            createdAt: Date(),
            graphFingerprint: String(repeating: "a", count: 64),
            inputFingerprint: String(repeating: "b", count: 64),
            requirements: requirements,
            outputs: []
        )

        XCTAssertFalse(workflowVersion(probe.workerVersion, satisfiesMinimum: requirements.minimumMereRunVersion))
        XCTAssertThrowsError(try validateWorker(probe, for: job, executor: "ssh:legacy")) { error in
            XCTAssertTrue(String(describing: error).contains("requires 0.23.0 or newer"))
        }
    }

    func testJobRequirementsRoundTripNamedSecretsAndResourcesWithoutValues() throws {
        let requirements = WorkflowJobRequirements(
            minimumMereRunVersion: "0.23.0",
            nodeKinds: ["private.publish"],
            modelIDs: [],
            secretNames: ["api-token"],
            acceleratorBackends: ["cpu"],
            minimumAcceleratorMemoryBytes: 1_024,
            minimumSystemMemoryBytes: 2_048,
            minimumDiskBytes: 4_096,
            minimumCPUCores: 4,
            networkAccess: true
        )

        let data = try WorkflowBundleCodec.encoder().encode(requirements)
        let encoded = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(encoded.contains("api-token"))
        XCTAssertFalse(encoded.contains("MERERUN_SECRET_API_TOKEN"))
        XCTAssertEqual(try WorkflowBundleCodec.decoder().decode(WorkflowJobRequirements.self, from: data), requirements)
    }

    func testDatasetDiscoveryRanksCompleteImageCaptionDirectories() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let ready = root.appendingPathComponent("ready/dataset", isDirectory: true)
        let incomplete = root.appendingPathComponent("incomplete", isDirectory: true)
        try FileManager.default.createDirectory(at: ready, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: incomplete, withIntermediateDirectories: true)
        for stem in ["one", "two"] {
            try Data([1, 2, 3]).write(to: ready.appendingPathComponent("\(stem).jpg"))
            try Data("caption".utf8).write(to: ready.appendingPathComponent("\(stem).txt"))
        }
        for stem in ["one", "two", "three"] {
            try Data([4, 5]).write(to: incomplete.appendingPathComponent("\(stem).png"))
        }
        try Data("caption".utf8).write(to: incomplete.appendingPathComponent("one.txt"))

        let result = try WorkflowDatasetDiscoverer(root: root, maxDepth: 3, limit: 10).discover()
        XCTAssertEqual(result.recommended, ready.path)
        XCTAssertEqual(result.candidates.map(\.path), [ready.path, incomplete.path])
        XCTAssertEqual(result.candidates.first?.pairedCount, 2)
        XCTAssertEqual(result.candidates.first?.ready, true)
        XCTAssertEqual(result.candidates.last?.ready, false)
    }

    func testCanonicalWorkflowFixturesDecodeAndValidate() throws {
        let fixtures = try XCTUnwrap(Bundle.module.resourceURL)
            .appendingPathComponent("Fixtures/WorkflowGraphV1", isDirectory: true)
        for name in [
            "lora-sample.workflow.json",
            "image-video.workflow.json",
            "parallel-image-video.workflow.json",
        ] {
            let graph = try WorkflowGraphDocument.load(from: fixtures.appendingPathComponent(name))
            let inputs = WorkflowInputsDocument(values: graph.inputs.reduce(into: [:]) { values, input in
                switch input.value.type {
                case .string: values[input.key] = .string("fixture")
                case .assetDirectory: values[input.key] = .string("/fixture/dataset")
                default: break
                }
            })
            XCTAssertNotEqual(WorkflowGraphValidator.validate(graph: graph, inputs: inputs).status, .blocked, name)
        }

        let invalid = try WorkflowGraphDocument.load(
            from: fixtures.appendingPathComponent("cycle.invalid.workflow.json")
        )
        let result = WorkflowGraphValidator.validate(
            graph: invalid,
            inputs: WorkflowInputsDocument(values: ["prompt": .string("fixture")])
        )
        XCTAssertTrue(result.diagnostics.contains { $0.id == "workflow_cycle" })
    }

    func testValidGraphInfersStableDependencyOrder() throws {
        let graph = try decodeGraph("""
        {
          "schema_version": 1,
          "kind": "mere.run/workflow-graph",
          "name": "image-to-video",
          "inputs": {"prompt": {"type": "string"}},
          "nodes": [
            {
              "id": "make-image",
              "kind": "image.generate",
              "arguments": {"prompt": {"$ref": "inputs.prompt"}}
            },
            {
              "id": "make-video",
              "kind": "video.generate",
              "arguments": {
                "prompt": "animate the frame",
                "image": {"$ref": "nodes.make-image.outputs.image"}
              }
            }
          ],
          "outputs": {"video": {"$ref": "nodes.make-video.outputs.video"}}
        }
        """)

        let validation = WorkflowGraphValidator.validate(
            graph: graph,
            inputs: WorkflowInputsDocument(values: ["prompt": .string("portrait")])
        )

        XCTAssertEqual(validation.status, .ok)
        XCTAssertEqual(validation.order, ["make-image", "make-video"])
        XCTAssertEqual(validation.dependencies["make-video"], ["make-image"])
    }

    func testParallelSchedulerOverlapsReadyNodesAndWaitsForDependencies() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let graph = try decodeGraph("""
        {
          "schema_version": 1,
          "kind": "mere.run/workflow-graph",
          "name": "parallel-images",
          "inputs": {},
          "execution": {"max_parallel_nodes": 2},
          "nodes": [
            {"id": "image-a", "kind": "image.generate", "arguments": {"prompt": "first", "seed": 1}},
            {"id": "image-b", "kind": "image.generate", "arguments": {"prompt": "second", "seed": 2}},
            {
              "id": "video",
              "kind": "video.generate",
              "depends_on": ["image-a", "image-b"],
              "arguments": {"prompt": "finish", "image": {"$ref": "nodes.image-a.outputs.image"}, "seed": 3}
            }
          ],
          "outputs": {"video": {"$ref": "nodes.video.outputs.video"}}
        }
        """)
        let bundle = try WorkflowBundleMaterializer(
            graph: graph,
            suppliedInputs: .init(values: [:]),
            destination: root.appendingPathComponent("bundle")
        ).materialize()
        let process = OverlapWorkflowProcessRunner()

        let outcome = try WorkflowRunner(
            bundleDirectory: bundle.directory,
            runDirectory: root.appendingPathComponent("run"),
            processRunner: process
        ).execute()

        XCTAssertEqual(outcome.state, .finished)
        XCTAssertEqual(process.maximumActiveRuns, 2)
        let firstWaveEnd = try XCTUnwrap([
            process.interval(for: "image-a")?.end,
            process.interval(for: "image-b")?.end,
        ].compactMap { $0 }.max())
        let dependentStart = try XCTUnwrap(process.interval(for: "video")?.start)
        XCTAssertGreaterThanOrEqual(dependentStart, firstWaveEnd)
    }

    func testGraphParallelismAndSecretReferencesAreValidated() throws {
        let invalid = try decodeGraph("""
        {
          "schema_version": 1,
          "kind": "mere.run/workflow-graph",
          "name": "invalid-parallelism",
          "inputs": {},
          "execution": {"max_parallel_nodes": 0},
          "nodes": [{"id": "image", "kind": "image.generate", "arguments": {"prompt": "fixture"}}],
          "outputs": {"image": {"$ref": "nodes.image.outputs.image"}}
        }
        """)
        let validation = WorkflowGraphValidator.validate(graph: invalid, inputs: .init(values: [:]))
        XCTAssertTrue(validation.diagnostics.contains { $0.id == "workflow_parallelism_invalid" })

        let value = try JSONDecoder().decode(
            WorkflowValue.self,
            from: Data(#"{"$secret":"hugging-face-token"}"#.utf8)
        )
        XCTAssertEqual(value, .secretReference("hugging-face-token"))
        XCTAssertEqual(value.secretNames, ["hugging-face-token"])
        XCTAssertEqual(
            workflowSecretEnvironmentKey("hugging-face-token"),
            "MERERUN_SECRET_HUGGING_FACE_TOKEN"
        )
    }

    func testReferenceAcceptsProviderOutputPortNamesWithUnderscores() throws {
        let reference = try WorkflowReference("nodes.prepare-data.outputs.contact_sheet")

        XCTAssertEqual(
            reference.source,
            .nodeOutput(nodeID: "prepare-data", output: "contact_sheet")
        )
    }

    func testNodeExecutionPolicySurvivesPortableMaterialization() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try singleImageGraph()
        let sourceNode = try XCTUnwrap(source.nodes.first)
        let graph = WorkflowGraphDocument(
            schemaVersion: source.schemaVersion,
            kind: source.kind,
            name: source.name,
            inputs: source.inputs,
            nodes: [WorkflowNode(
                id: sourceNode.id,
                kind: sourceNode.kind,
                provider: sourceNode.provider,
                arguments: sourceNode.arguments,
                dependsOn: sourceNode.dependsOn,
                execution: .init(maxAttempts: 3, timeoutSeconds: 120, cache: .refresh)
            )],
            outputs: source.outputs,
            metadata: source.metadata
        )

        let bundle = try WorkflowBundleMaterializer(
            graph: graph,
            suppliedInputs: .init(values: ["prompt": .string("fixture")]),
            destination: root.appendingPathComponent("bundle"),
            seed: { 77 }
        ).materialize()

        XCTAssertEqual(bundle.graph.nodes.first?.execution?.resolvedMaxAttempts, 3)
        XCTAssertEqual(bundle.graph.nodes.first?.execution?.timeoutSeconds, 120)
        XCTAssertEqual(bundle.graph.nodes.first?.execution?.resolvedCache, .refresh)
        let roundTrip = try WorkflowGraphDocument.load(from: bundle.directory.appendingPathComponent("graph.json"))
        XCTAssertEqual(roundTrip, bundle.graph)
    }

    func testInvalidNodeExecutionPolicyBlocksValidation() throws {
        let source = try singleImageGraph()
        let sourceNode = try XCTUnwrap(source.nodes.first)
        let graph = WorkflowGraphDocument(
            schemaVersion: source.schemaVersion,
            kind: source.kind,
            name: source.name,
            inputs: source.inputs,
            nodes: [WorkflowNode(
                id: sourceNode.id,
                kind: sourceNode.kind,
                provider: sourceNode.provider,
                arguments: sourceNode.arguments,
                dependsOn: sourceNode.dependsOn,
                execution: .init(maxAttempts: 0, timeoutSeconds: 0, cache: .automatic)
            )],
            outputs: source.outputs,
            metadata: source.metadata
        )

        let validation = WorkflowGraphValidator.validate(
            graph: graph,
            inputs: .init(values: ["prompt": .string("fixture")])
        )

        XCTAssertEqual(validation.status, .blocked)
        XCTAssertTrue(validation.diagnostics.contains { $0.id == "workflow_node_attempts_invalid_generate" })
        XCTAssertTrue(validation.diagnostics.contains { $0.id == "workflow_node_timeout_invalid_generate" })
    }

    func testDuplicateNodeIDsProduceDiagnosticsWithoutTrapping() throws {
        let graph = try decodeGraph("""
        {
          "schema_version": 1,
          "kind": "mere.run/workflow-graph",
          "name": "duplicates",
          "inputs": {},
          "nodes": [
            {"id": "same", "kind": "image.generate", "arguments": {"prompt": "one"}},
            {"id": "same", "kind": "image.generate", "arguments": {"prompt": "two"}}
          ],
          "outputs": {"image": {"$ref": "nodes.same.outputs.image"}}
        }
        """)

        let validation = WorkflowGraphValidator.validate(graph: graph, inputs: .init(values: [:]))

        XCTAssertEqual(validation.status, .blocked)
        XCTAssertTrue(validation.diagnostics.contains { $0.id == "workflow_node_duplicate_same" })
    }

    func testCycleUnknownNodeKindAndMissingReferenceBlockValidation() throws {
        let graph = try decodeGraph("""
        {
          "schema_version": 1,
          "kind": "mere.run/workflow-graph",
          "name": "invalid",
          "inputs": {},
          "nodes": [
            {
              "id": "first",
              "kind": "shell.exec",
              "depends_on": ["second"],
              "arguments": {}
            },
            {
              "id": "second",
              "kind": "image.generate",
              "depends_on": ["first"],
              "arguments": {"prompt": {"$ref": "inputs.missing"}}
            }
          ],
          "outputs": {"image": {"$ref": "nodes.second.outputs.image"}}
        }
        """)

        let validation = WorkflowGraphValidator.validate(graph: graph, inputs: .init(values: [:]))

        XCTAssertEqual(validation.status, .blocked)
        XCTAssertTrue(validation.diagnostics.contains { $0.id == "workflow_cycle" })
        XCTAssertTrue(validation.diagnostics.contains { $0.id == "workflow_node_kind_unsupported_first" })
        XCTAssertTrue(validation.diagnostics.contains { $0.id == "workflow_reference_invalid_second" })
    }

    func testArtifactContentTypeMismatchBlocksGraph() throws {
        let graph = try decodeGraph("""
        {
          "schema_version": 1,
          "kind": "mere.run/workflow-graph",
          "name": "wrong-artifact",
          "inputs": {"data": {"type": "asset_directory"}},
          "nodes": [
            {
              "id": "train",
              "kind": "image.train-lora",
              "arguments": {"data": {"$ref": "inputs.data"}}
            },
            {
              "id": "video",
              "kind": "video.generate",
              "arguments": {
                "prompt": "move",
                "image": {"$ref": "nodes.train.outputs.adapter"}
              }
            }
          ],
          "outputs": {"video": {"$ref": "nodes.video.outputs.video"}}
        }
        """)

        let validation = WorkflowGraphValidator.validate(
            graph: graph,
            inputs: .init(values: ["data": .string("/tmp/data")])
        )

        XCTAssertEqual(validation.status, .blocked)
        XCTAssertTrue(validation.diagnostics.contains {
            $0.id == "workflow_reference_content_type_video_image"
        })
    }

    func testTrainLoRALiteArgumentUsesPublicCLIFlag() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let node = WorkflowNode(
            id: "train",
            kind: "image.train-lora",
            arguments: [
                "data": .string("/tmp/dataset"),
                "lite": .boolean(true),
                "max_text_length": .integer(128),
                "base_quantization_bits": .integer(4),
                "rank": .integer(4),
            ],
            dependsOn: nil
        )

        let invocation = try WorkflowNodeCommandBuilder.invocation(
            node: node,
            arguments: node.arguments,
            nodeDirectory: root
        )

        XCTAssertTrue(invocation.preflightArguments.contains("--lite"))
        XCTAssertTrue(invocation.runArguments.contains("--lite"))
        XCTAssertTrue(invocation.runArguments.contains("--max-text-length"))
        XCTAssertTrue(invocation.runArguments.contains("128"))
        XCTAssertTrue(invocation.runArguments.contains("--rank"))
        XCTAssertTrue(invocation.runArguments.contains("4"))
        XCTAssertTrue(invocation.runArguments.contains("--base-quantization-bits"))
    }

    func testKreaGenerationQuantizationUsesPublicCLIFlag() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let node = WorkflowNode(
            id: "sample",
            kind: "image.generate",
            arguments: [
                "prompt": .string("a neon street portrait"),
                "model": .string("image-krea2-turbo"),
                "krea_base_quantization_bits": .integer(4),
            ],
            dependsOn: nil
        )

        let invocation = try WorkflowNodeCommandBuilder.invocation(
            node: node,
            arguments: node.arguments,
            nodeDirectory: root
        )

        XCTAssertTrue(invocation.preflightArguments.contains("--krea-base-quantization-bits"))
        XCTAssertTrue(invocation.runArguments.contains("--krea-base-quantization-bits"))
        XCTAssertTrue(invocation.runArguments.contains("4"))
    }

    func testWorkflowChildStdoutCaptureUsesUniquePath() {
        let directory = URL(fileURLWithPath: "/tmp/workflow-node", isDirectory: true)
        let first = WorkflowProcessRunner.stdoutCaptureURL(in: directory)
        let second = WorkflowProcessRunner.stdoutCaptureURL(in: directory)

        XCTAssertEqual(first.deletingLastPathComponent(), directory)
        XCTAssertTrue(first.lastPathComponent.hasPrefix(".workflow-stdout-"))
        XCTAssertNotEqual(first, second)
        XCTAssertFalse(first.lastPathComponent.contains("UUID()"))
    }

    func testWorkflowChildCapturesStderrAndTerminationReason() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let nodeDirectory = root.appendingPathComponent("run/nodes/000-fixture", isDirectory: true)
        try FileManager.default.createDirectory(at: nodeDirectory, withIntermediateDirectories: true)

        let result = try WorkflowProcessRunner().run(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "printf 'fixture failure' >&2; exit 7"],
            currentDirectory: nodeDirectory,
            timeoutSeconds: nil,
            stdoutLineHandler: nil
        )

        XCTAssertEqual(result.status, 7)
        XCTAssertEqual(result.stderr, "fixture failure")
        XCTAssertEqual(result.terminationReason, .exit)
        XCTAssertEqual(result.failureSummary, "exited with status 7. stderr: fixture failure")
    }

    func testWorkflowChildEnforcesTimeout() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let nodeDirectory = root.appendingPathComponent("run/nodes/000-fixture", isDirectory: true)
        try FileManager.default.createDirectory(at: nodeDirectory, withIntermediateDirectories: true)
        let startedAt = Date()

        XCTAssertThrowsError(try WorkflowProcessRunner().run(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "sleep 30"],
            currentDirectory: nodeDirectory,
            timeoutSeconds: 1,
            stdoutLineHandler: nil
        )) { error in
            XCTAssertEqual(error.localizedDescription, "Process timed out after 1 seconds.")
        }

        XCTAssertLessThan(Date().timeIntervalSince(startedAt), 5)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: root.appendingPathComponent("run/worker-child.pid").path
            )
        )
    }

    func testMaterializationFreezesSeedAndCanonicalFingerprints() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let graph = try singleImageGraph()
        let fixedDate = Date(timeIntervalSince1970: 1_700_000_000)
        let fixedID = UUID(uuidString: "11111111-2222-3333-4444-555555555555")!

        let bundle = try WorkflowBundleMaterializer(
            graph: graph,
            suppliedInputs: .init(values: ["prompt": .string("a lighthouse")]),
            destination: root.appendingPathComponent("bundle"),
            now: { fixedDate },
            jobID: { fixedID },
            seed: { 77 }
        ).materialize()

        XCTAssertEqual(bundle.graph.nodes[0].arguments["seed"], .integer(77))
        XCTAssertEqual(bundle.job.jobID, fixedID.uuidString.lowercased())
        XCTAssertEqual(bundle.job.graphFingerprint, try WorkflowBundleCodec.hash(bundle.graph))
        let reloadedGraph = try WorkflowGraphDocument.load(
            from: bundle.directory.appendingPathComponent("graph.json")
        )
        XCTAssertEqual(
            bundle.job.graphFingerprint,
            try WorkflowBundleCodec.hash(reloadedGraph),
            "The persisted graph must verify against its immutable job manifest."
        )
        XCTAssertEqual(bundle.job.outputs, [.init(name: "image", reference: "nodes.generate.outputs.image")])
    }

    func testDirectoryAssetsAreOrderedAndContentAddressed() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let data = root.appendingPathComponent("data", isDirectory: true)
        try FileManager.default.createDirectory(at: data, withIntermediateDirectories: true)
        try Data("caption".utf8).write(to: data.appendingPathComponent("b.txt"))
        try Data([0x89, 0x50, 0x4e, 0x47]).write(to: data.appendingPathComponent("a.png"))

        let bundle = try WorkflowBundleMaterializer(
            graph: try trainingGraph(),
            suppliedInputs: .init(values: ["data": .string(data.path)]),
            destination: root.appendingPathComponent("bundle"),
            seed: { 12 }
        ).materialize()

        let entries = try XCTUnwrap(bundle.assets.groups.first).entries
        XCTAssertEqual(entries.map(\.path), ["a.png", "b.txt"])
        XCTAssertEqual(bundle.inputs.values["data"], .string("asset://data"))
        for entry in entries {
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: bundle.directory.appendingPathComponent("assets/sha256/\(entry.digest)").path
            ))
        }
    }

    func testLoRAToSampleBundleFingerprintSurvivesDiskRoundTrip() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset", isDirectory: true)
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
        try Data([0x89, 0x50, 0x4e, 0x47]).write(to: dataset.appendingPathComponent("frame.png"))
        try Data("caption".utf8).write(to: dataset.appendingPathComponent("frame.txt"))
        let fixtures = try XCTUnwrap(Bundle.module.resourceURL)
            .appendingPathComponent("Fixtures/WorkflowGraphV1", isDirectory: true)
        let graph = try WorkflowGraphDocument.load(
            from: fixtures.appendingPathComponent("lora-sample.workflow.json")
        )
        var seeds: [Int64] = [1_205_675_978_010_373_390, 6_666_523_044_613_654_146]
        let bundle = try WorkflowBundleMaterializer(
            graph: graph,
            suppliedInputs: .init(values: [
                "dataset": .string(dataset.path),
                "prompt": .string("fixture"),
            ]),
            destination: root.appendingPathComponent("bundle"),
            seed: { seeds.removeFirst() }
        ).materialize()
        let reloadedGraph = try WorkflowGraphDocument.load(
            from: bundle.directory.appendingPathComponent("graph.json")
        )

        XCTAssertEqual(bundle.graph, reloadedGraph)
        XCTAssertEqual(bundle.job.graphFingerprint, try WorkflowBundleCodec.hash(reloadedGraph))
    }

    func testGraphEventStreamWriterEmitsOneCompleteNDJSONRecord() throws {
        let pipe = Pipe()
        let event = GraphRunEvent(
            sequence: 3,
            createdAt: Date(timeIntervalSince1970: 0),
            type: "node_finished",
            state: .finished,
            nodeID: "generate",
            message: nil
        )

        emitGraphStreamEvent(event, to: pipe.fileHandleForWriting)
        let line = String(decoding: pipe.fileHandleForReading.availableData, as: UTF8.self)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        XCTAssertTrue(line.hasSuffix("\n"))
        XCTAssertEqual(line.filter { $0 == "\n" }.count, 1)
        XCTAssertEqual(try decoder.decode(GraphRunEvent.self, from: Data(line.utf8)), event)
    }

    func testNodeAssetConstantsBecomePortableBundleAssets() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("reference.png")
        try Data([0x89, 0x50, 0x4e, 0x47]).write(to: source)
        let graph = WorkflowGraphDocument(
            schemaVersion: 1,
            kind: WorkflowGraphDocument.kind,
            name: "constant-asset",
            inputs: [:],
            nodes: [WorkflowNode(
                id: "generate",
                kind: "image.generate",
                arguments: ["prompt": .string("fixture"), "input": .string(source.path)],
                dependsOn: nil
            )],
            outputs: ["image": .reference("nodes.generate.outputs.image")],
            metadata: nil
        )
        let bundleURL = root.appendingPathComponent("bundle")
        let bundle = try WorkflowBundleMaterializer(
            graph: graph,
            suppliedInputs: WorkflowInputsDocument(values: [:]),
            destination: bundleURL,
            seed: { 7 }
        ).materialize()

        XCTAssertEqual(bundle.assets.groups.count, 1)
        let portableInput = try XCTUnwrap(bundle.graph.nodes.first?.arguments["input"]?.stringValue)
        XCTAssertTrue(portableInput.hasPrefix("asset://asset-"))
        XCTAssertFalse(String(decoding: try Data(contentsOf: bundleURL.appendingPathComponent("graph.json")), as: UTF8.self).contains(root.path))

        let process = FixtureWorkflowProcessRunner()
        let runURL = root.appendingPathComponent("run")
        let outcome = try WorkflowRunner(
            bundleDirectory: bundleURL,
            runDirectory: runURL,
            processRunner: process
        ).execute()
        XCTAssertEqual(outcome.state, .finished)
        let runArguments = try XCTUnwrap(process.arguments.last)
        let inputIndex = try XCTUnwrap(runArguments.firstIndex(of: "--input"))
        XCTAssertTrue(runArguments[inputIndex + 1].hasPrefix(runURL.appendingPathComponent("inputs").path))
        XCTAssertNotEqual(runArguments[inputIndex + 1], source.path)
    }

    func testDirectoryAssetRejectsSymlinks() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let data = root.appendingPathComponent("data", isDirectory: true)
        try FileManager.default.createDirectory(at: data, withIntermediateDirectories: true)
        let outside = root.appendingPathComponent("outside.txt")
        try Data("outside".utf8).write(to: outside)
        try FileManager.default.createSymbolicLink(
            at: data.appendingPathComponent("escape.txt"),
            withDestinationURL: outside
        )

        XCTAssertThrowsError(try WorkflowBundleMaterializer(
            graph: trainingGraph(),
            suppliedInputs: .init(values: ["data": .string(data.path)]),
            destination: root.appendingPathComponent("bundle")
        ).materialize())
    }

    func testRunnerUsesCLIAdapterAndVerifiesDeclaredOutput() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let bundle = try WorkflowBundleMaterializer(
            graph: singleImageGraph(),
            suppliedInputs: .init(values: ["prompt": .string("a lighthouse")]),
            destination: root.appendingPathComponent("bundle"),
            seed: { 99 }
        ).materialize()
        let process = FixtureWorkflowProcessRunner()

        let outcome = try WorkflowRunner(
            bundleDirectory: bundle.directory,
            runDirectory: root.appendingPathComponent("run"),
            processRunner: process
        ).execute()

        XCTAssertEqual(outcome.state, .finished)
        XCTAssertEqual(process.arguments.count, 2)
        XCTAssertTrue(process.arguments[0].suffix(2) == ["--preflight", "--json"])
        XCTAssertTrue(process.arguments[1].contains("--quiet"))
        XCTAssertEqual(outcome.outputs.count, 1)
        XCTAssertEqual(outcome.outputs[0].contentType, "image/png")
    }

    func testRunnerPersistsPreflightOutputInFailureDiagnostic() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let bundle = try WorkflowBundleMaterializer(
            graph: singleImageGraph(),
            suppliedInputs: .init(values: ["prompt": .string("a lighthouse")]),
            destination: root.appendingPathComponent("bundle"),
            seed: { 99 }
        ).materialize()
        let runDirectory = root.appendingPathComponent("run")

        let outcome = try WorkflowRunner(
            bundleDirectory: bundle.directory,
            runDirectory: runDirectory,
            processRunner: FailingPreflightWorkflowProcessRunner()
        ).execute()
        let manifest = try WorkflowBundleCodec.decoder().decode(
            GraphRunManifest.self,
            from: Data(contentsOf: runDirectory.appendingPathComponent(GraphRunManifest.filename))
        )

        XCTAssertEqual(outcome.state, .failed)
        XCTAssertEqual(
            manifest.error,
            "Node 'generate' preflight exited with status 255. stdout: fixture model failure."
        )
    }

    func testRunnerRetriesFailedNodeAndRecordsAttempt() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try singleImageGraph()
        let sourceNode = try XCTUnwrap(source.nodes.first)
        let graph = WorkflowGraphDocument(
            schemaVersion: source.schemaVersion,
            kind: source.kind,
            name: source.name,
            inputs: source.inputs,
            nodes: [WorkflowNode(
                id: sourceNode.id,
                kind: sourceNode.kind,
                provider: sourceNode.provider,
                arguments: sourceNode.arguments,
                dependsOn: sourceNode.dependsOn,
                execution: .init(maxAttempts: 2, timeoutSeconds: nil, cache: nil)
            )],
            outputs: source.outputs,
            metadata: source.metadata
        )
        let bundle = try WorkflowBundleMaterializer(
            graph: graph,
            suppliedInputs: .init(values: ["prompt": .string("a lighthouse")]),
            destination: root.appendingPathComponent("bundle"),
            seed: { 99 }
        ).materialize()
        let runDirectory = root.appendingPathComponent("run")
        let process = RetryingWorkflowProcessRunner()

        let outcome = try WorkflowRunner(
            bundleDirectory: bundle.directory,
            runDirectory: runDirectory,
            processRunner: process
        ).execute()
        let manifest = try WorkflowBundleCodec.decoder().decode(
            GraphRunManifest.self,
            from: Data(contentsOf: runDirectory.appendingPathComponent(GraphRunManifest.filename))
        )
        let events = try String(
            contentsOf: runDirectory.appendingPathComponent("events.jsonl"),
            encoding: .utf8
        )
        let eventRecords = try events.split(separator: "\n").map {
            try WorkflowBundleCodec.decoder().decode(GraphRunEvent.self, from: Data($0.utf8))
        }

        XCTAssertEqual(outcome.state, .finished)
        XCTAssertEqual(process.runCount, 2)
        XCTAssertEqual(manifest.nodes.first?.attempt, 2)
        XCTAssertEqual(manifest.nodes.first?.maxAttempts, 2)
        XCTAssertEqual(manifest.nodes.first?.exitStatus, 0)
        XCTAssertNil(manifest.nodes.first?.error)
        XCTAssertTrue(eventRecords.contains { $0.type == "node_retrying" })
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: runDirectory
                    .appendingPathComponent("nodes/000-generate/artifacts/partial.txt")
                    .path
            )
        )
    }

    func testRunnerReusesVerifiedCrossRunCacheAndRepairsCorruption() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let bundle = try WorkflowBundleMaterializer(
            graph: singleImageGraph(),
            suppliedInputs: .init(values: ["prompt": .string("a lighthouse")]),
            destination: root.appendingPathComponent("bundle"),
            seed: { 99 }
        ).materialize()
        let cacheDirectory = root.appendingPathComponent("cache", isDirectory: true)
        let firstRun = root.appendingPathComponent("run-one", isDirectory: true)
        let firstProcess = FixtureWorkflowProcessRunner()

        XCTAssertEqual(try WorkflowRunner(
            bundleDirectory: bundle.directory,
            runDirectory: firstRun,
            processRunner: firstProcess,
            cacheDirectory: cacheDirectory
        ).execute().state, .finished)
        XCTAssertEqual(firstProcess.arguments.count, 2)
        let firstManifest = try WorkflowBundleCodec.decoder().decode(
            GraphRunManifest.self,
            from: Data(contentsOf: firstRun.appendingPathComponent(GraphRunManifest.filename))
        )
        let fingerprint = try XCTUnwrap(firstManifest.nodes.first?.fingerprint)

        let secondRun = root.appendingPathComponent("run-two", isDirectory: true)
        let secondProcess = FixtureWorkflowProcessRunner()
        XCTAssertEqual(try WorkflowRunner(
            bundleDirectory: bundle.directory,
            runDirectory: secondRun,
            processRunner: secondProcess,
            cacheDirectory: cacheDirectory
        ).execute().state, .finished)
        let secondManifest = try WorkflowBundleCodec.decoder().decode(
            GraphRunManifest.self,
            from: Data(contentsOf: secondRun.appendingPathComponent(GraphRunManifest.filename))
        )
        let secondEvents = try String(
            contentsOf: secondRun.appendingPathComponent("events.jsonl"),
            encoding: .utf8
        ).split(separator: "\n").map {
            try WorkflowBundleCodec.decoder().decode(GraphRunEvent.self, from: Data($0.utf8))
        }

        XCTAssertTrue(secondProcess.arguments.isEmpty)
        XCTAssertEqual(secondManifest.nodes.first?.attempt, 0)
        XCTAssertTrue(secondEvents.contains { $0.type == "node_cache_hit" })
        XCTAssertEqual(secondManifest.outputs.first?.sha256, firstManifest.outputs.first?.sha256)

        let cacheFiles = cacheDirectory
            .appendingPathComponent(fingerprint, isDirectory: true)
            .appendingPathComponent("files", isDirectory: true)
        let cachedRelativePath = try XCTUnwrap(
            FileManager.default.subpathsOfDirectory(atPath: cacheFiles.path).first { path in
                (try? cacheFiles.appendingPathComponent(path).resourceValues(
                    forKeys: [.isRegularFileKey]
                ).isRegularFile) == true
            }
        )
        try Data("corrupt".utf8).write(to: cacheFiles.appendingPathComponent(cachedRelativePath))

        let repairedProcess = FixtureWorkflowProcessRunner()
        XCTAssertEqual(try WorkflowRunner(
            bundleDirectory: bundle.directory,
            runDirectory: root.appendingPathComponent("run-three"),
            processRunner: repairedProcess,
            cacheDirectory: cacheDirectory
        ).execute().state, .finished)
        XCTAssertEqual(repairedProcess.arguments.count, 2)
    }

    func testResumeRequiresMatchingNodeFingerprintAndArtifactDigests() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let bundle = try WorkflowBundleMaterializer(
            graph: singleImageGraph(),
            suppliedInputs: .init(values: ["prompt": .string("fixture")]),
            destination: root.appendingPathComponent("bundle"),
            seed: { 44 }
        ).materialize()
        let runDirectory = root.appendingPathComponent("run")
        let initialProcess = FixtureWorkflowProcessRunner()
        XCTAssertEqual(try WorkflowRunner(
            bundleDirectory: bundle.directory,
            runDirectory: runDirectory,
            processRunner: initialProcess
        ).execute().state, .finished)

        let resumedProcess = FixtureWorkflowProcessRunner()
        let resumedOutcome = try WorkflowRunner(
            bundleDirectory: bundle.directory,
            runDirectory: runDirectory,
            resume: true,
            processRunner: resumedProcess
        ).execute()
        let resumedManifest = try WorkflowBundleCodec.decoder().decode(
            GraphRunManifest.self,
            from: Data(contentsOf: runDirectory.appendingPathComponent(GraphRunManifest.filename))
        )
        XCTAssertEqual(resumedOutcome.state, .finished, resumedManifest.error ?? "unknown resume error")
        XCTAssertTrue(resumedProcess.arguments.isEmpty)

        let manifestURL = runDirectory.appendingPathComponent(GraphRunManifest.filename)
        var manifest = try WorkflowBundleCodec.decoder().decode(
            GraphRunManifest.self,
            from: Data(contentsOf: manifestURL)
        )
        manifest.nodes[0].fingerprint = "stale"
        try WorkflowBundleCodec.write(manifest, to: manifestURL)

        let staleProcess = FixtureWorkflowProcessRunner()
        XCTAssertEqual(try WorkflowRunner(
            bundleDirectory: bundle.directory,
            runDirectory: runDirectory,
            resume: true,
            processRunner: staleProcess
        ).execute().state, .finished)
        XCTAssertEqual(staleProcess.arguments.count, 2)
    }

    func testSSHSubmissionUsesBatchModeAndQuotesAdversarialProfilePaths() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset", isDirectory: true)
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: dataset.appendingPathComponent("frame.png"))
        try Data("caption".utf8).write(to: dataset.appendingPathComponent("frame.txt"))
        let bundle = try WorkflowBundleMaterializer(
            graph: trainingGraph(),
            suppliedInputs: .init(values: ["data": .string(dataset.path)]),
            destination: root.appendingPathComponent("bundle"),
            seed: { 7 }
        ).materialize()
        let asset = try XCTUnwrap(bundle.assets.groups.first?.entries.first)
        let probe = WorkflowExecutorProbe(
            schemaVersion: 1,
            workerVersion: MereRunCLIVersion.current,
            contractVersions: [WorkflowJobManifest.contractVersion],
            platform: "linux",
            architecture: "x86_64",
            acceleratorBackend: "cuda",
            memoryBytes: 64 * 1_024 * 1_024 * 1_024,
            availableDiskBytes: 1_000_000_000,
            nodeKinds: bundle.job.requirements.nodeKinds,
            installedModelIDs: bundle.job.requirements.modelIDs
        )
        let probeJSON = String(decoding: try WorkflowBundleCodec.encoder().encode(probe), as: UTF8.self)
        var invocations: [[String]] = []
        let runner: ([String], FileHandle?) throws -> WorkflowProcessResult = { arguments, _ in
            invocations.append(arguments)
            switch arguments.first {
            case "tar":
                let archiveFlag = try XCTUnwrap(arguments.firstIndex(of: "-czf"))
                try Data("archive".utf8).write(to: URL(fileURLWithPath: arguments[archiveFlag + 1]))
                return .init(status: 0, stdout: "")
            case "scp":
                return .init(status: 0, stdout: "")
            case "ssh":
                let command = arguments.last ?? ""
                if command.contains("graph worker probe") { return .init(status: 0, stdout: probeJSON) }
                if command.contains("pwd -P") { return .init(status: 0, stdout: "/remote/cache path") }
                if command.contains("test -f") { return .init(status: 0, stdout: asset.digest) }
                return .init(status: 0, stdout: "")
            default:
                return .init(status: 127, stdout: "")
            }
        }
        let profile = WorkflowExecutorProfile(
            name: "gpu-box",
            kind: .ssh,
            destination: "worker@example.test",
            remoteRoot: "~/cache path/it's-here",
            port: 2_222,
            identityFile: "/tmp/key with spaces",
            mereRunPath: "~/bin/mere run",
            url: nil,
            tokenFile: nil
        )

        let submitted = try SSHWorkflowExecutor(profile: profile, executableRunner: runner).submit(
            bundleDirectory: bundle.directory,
            localRunDirectory: bundle.directory
        )
        XCTAssertEqual(submitted.state, .queued)
        let sshCalls = invocations.filter { $0.first == "ssh" }
        let scpCalls = invocations.filter { $0.first == "scp" }
        XCTAssertFalse(sshCalls.isEmpty)
        XCTAssertTrue(sshCalls.allSatisfy { arguments in
            arguments.contains("BatchMode=yes")
                && arguments.contains("-p")
                && arguments.contains("2222")
                && arguments.contains("/tmp/key with spaces")
        })
        let rootCommand = try XCTUnwrap(sshCalls.first(where: { $0.last?.contains("pwd -P") == true })?.last)
        XCTAssertTrue(rootCommand.contains("$HOME/"))
        XCTAssertTrue(rootCommand.contains("cache path"))
        XCTAssertTrue(rootCommand.contains("\\''"))
        XCTAssertTrue(sshCalls.contains { $0.last?.contains("'~/bin/mere run'") == true })
        let launchCommand = try XCTUnwrap(sshCalls.first(where: { $0.last?.contains("nohup") == true })?.last)
        XCTAssertFalse(launchCommand.contains("&;"))
        XCTAssertTrue(launchCommand.contains("& pid=$!;"))
        XCTAssertEqual(scpCalls.count, 2)
        XCTAssertTrue(scpCalls.allSatisfy {
            $0.contains("BatchMode=yes") && $0.contains("-O") && $0.contains("-P")
        })
        XCTAssertTrue(scpCalls.contains { $0.contains(where: { $0.hasSuffix(asset.digest) }) })
        let tarCall = try XCTUnwrap(invocations.first(where: { $0.first == "tar" }))
        XCTAssertTrue(tarCall.contains("--no-xattrs"))
        XCTAssertEqual(shellQuote("a'b;$(touch nope)"), "'a'\\''b;$(touch nope)'")
    }

    func testSSHFetchUsesExplicitRunDirectoryEntries() {
        let reports = SSHWorkflowExecutor.fetchRelativePaths(allArtifacts: false)
        let allArtifacts = SSHWorkflowExecutor.fetchRelativePaths(allArtifacts: true)

        XCTAssertTrue(reports.contains("outputs"))
        XCTAssertFalse(reports.contains("nodes"))
        XCTAssertTrue(allArtifacts.contains("nodes"))
        XCTAssertTrue(allArtifacts.contains("actions.json"))
        XCTAssertFalse(allArtifacts.contains("."))
    }

    func testRemoteReferenceParsingIsStrict() throws {
        let reference = try WorkflowRemoteReference("relay://fleet/abc-123")
        XCTAssertEqual(reference.executorReference, "relay:fleet")
        XCTAssertEqual(reference.jobID, "abc-123")
        XCTAssertThrowsError(try WorkflowRemoteReference("relay://fleet/../escape"))
        XCTAssertThrowsError(try WorkflowRemoteReference("https://fleet/abc"))
    }

    func testExecutorProfileStoreWritesPrivateFile() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("executors.json")
        let profile = WorkflowExecutorProfile(
            name: "gpu-box",
            kind: .ssh,
            destination: "worker@example.test",
            remoteRoot: "~/.cache/mere.run",
            port: nil,
            identityFile: nil,
            mereRunPath: "mere.run",
            url: nil,
            tokenFile: nil
        )
        let profiles = WorkflowExecutorProfiles(schemaVersion: 1, profiles: [profile])
        try WorkflowExecutorProfileStore.save(profiles, to: url)

        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
        XCTAssertEqual(try WorkflowExecutorProfileStore.load(from: url), profiles)
    }

    func testRemoteJobDecodesRelayPlacementDiagnostics() throws {
        let job = try WorkflowBundleCodec.decoder().decode(
            WorkflowRemoteJob.self,
            from: Data(#"""
            {
              "job_id": "job-123",
              "job_reference": "relay://fleet/job-123",
              "state": "queued",
              "executor": "relay:fleet",
              "run_directory": null,
              "created_at": "2026-07-15T12:00:00Z",
              "updated_at": "2026-07-15T12:01:00Z",
              "artifacts": [],
              "error": null,
              "placement": {
                "connected_nodes": 1,
                "graph_worker_nodes": 1,
                "eligible_nodes": 0,
                "diagnostic": "No connected graph worker satisfies this job",
                "nodes": [{
                  "agent_id": "agent-1",
                  "device_id": "tensor",
                  "device_name": "Tensor",
                  "status": "online",
                  "eligible": false,
                  "blockers": [{
                    "code": "model_missing",
                    "message": "Required model image-krea2-raw is not installed"
                  }]
                }]
              }
            }
            """#.utf8)
        )

        XCTAssertEqual(job.placement?.eligibleNodes, 0)
        XCTAssertEqual(job.placement?.nodes.first?.deviceID, "tensor")
        XCTAssertEqual(job.placement?.nodes.first?.blockers.first?.code, "model_missing")
    }

    private func singleImageGraph() throws -> WorkflowGraphDocument {
        try decodeGraph("""
        {
          "schema_version": 1,
          "kind": "mere.run/workflow-graph",
          "name": "one-image",
          "inputs": {"prompt": {"type": "string"}},
          "nodes": [{
            "id": "generate",
            "kind": "image.generate",
            "arguments": {"prompt": {"$ref": "inputs.prompt"}}
          }],
          "outputs": {"image": {"$ref": "nodes.generate.outputs.image"}}
        }
        """)
    }

    private func trainingGraph() throws -> WorkflowGraphDocument {
        try decodeGraph("""
        {
          "schema_version": 1,
          "kind": "mere.run/workflow-graph",
          "name": "train",
          "inputs": {"data": {"type": "asset_directory"}},
          "nodes": [{
            "id": "train",
            "kind": "image.train-lora",
            "arguments": {"data": {"$ref": "inputs.data"}}
          }],
          "outputs": {"adapter": {"$ref": "nodes.train.outputs.adapter"}}
        }
        """)
    }

    private func decodeGraph(_ json: String) throws -> WorkflowGraphDocument {
        try JSONDecoder().decode(WorkflowGraphDocument.self, from: Data(json.utf8))
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

private final class FixtureWorkflowProcessRunner: WorkflowProcessRunning {
    var arguments: [[String]] = []

    func run(arguments: [String], currentDirectory: URL) throws -> WorkflowProcessResult {
        self.arguments.append(arguments)
        if arguments.contains("--preflight") {
            return WorkflowProcessResult(status: 0, stdout: "{}")
        }
        guard let outputIndex = arguments.firstIndex(of: "--output"), outputIndex + 1 < arguments.count else {
            return WorkflowProcessResult(status: 2, stdout: "")
        }
        let output = URL(fileURLWithPath: arguments[outputIndex + 1])
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data([0x89, 0x50, 0x4e, 0x47]).write(to: output)
        return WorkflowProcessResult(status: 0, stdout: "ok")
    }
}

private final class OverlapWorkflowProcessRunner: WorkflowProcessRunning, @unchecked Sendable {
    struct Interval {
        let start: Date
        let end: Date
    }

    private let lock = NSLock()
    private var activeRuns = 0
    private var maximumRuns = 0
    private var intervals: [String: Interval] = [:]

    var maximumActiveRuns: Int {
        lock.withLock { maximumRuns }
    }

    func interval(for nodeID: String) -> Interval? {
        lock.withLock { intervals[nodeID] }
    }

    func run(arguments: [String], currentDirectory: URL) throws -> WorkflowProcessResult {
        if arguments.contains("--preflight") {
            return WorkflowProcessResult(status: 0, stdout: "{}")
        }
        let nodeID = currentDirectory.lastPathComponent
            .split(separator: "-", maxSplits: 1)
            .last
            .map(String.init) ?? currentDirectory.lastPathComponent
        let start = Date()
        lock.withLock {
            activeRuns += 1
            maximumRuns = max(maximumRuns, activeRuns)
        }
        Thread.sleep(forTimeInterval: 0.12)
        guard let outputIndex = arguments.firstIndex(of: "--output"),
              outputIndex + 1 < arguments.count else {
            return WorkflowProcessResult(status: 2, stdout: "")
        }
        let output = URL(fileURLWithPath: arguments[outputIndex + 1])
        try FileManager.default.createDirectory(
            at: output.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data([0x01, 0x02, 0x03, 0x04]).write(to: output)
        let end = Date()
        lock.withLock {
            activeRuns -= 1
            intervals[nodeID] = Interval(start: start, end: end)
        }
        return WorkflowProcessResult(status: 0, stdout: "ok")
    }
}

private struct FailingPreflightWorkflowProcessRunner: WorkflowProcessRunning {
    func run(arguments: [String], currentDirectory: URL) throws -> WorkflowProcessResult {
        WorkflowProcessResult(status: 255, stdout: "fixture model failure")
    }
}

private final class RetryingWorkflowProcessRunner: WorkflowProcessRunning {
    private(set) var runCount = 0

    func run(arguments: [String], currentDirectory: URL) throws -> WorkflowProcessResult {
        if arguments.contains("--preflight") {
            return WorkflowProcessResult(status: 0, stdout: "{}")
        }
        runCount += 1
        guard let outputIndex = arguments.firstIndex(of: "--output"), outputIndex + 1 < arguments.count else {
            return WorkflowProcessResult(status: 2, stdout: "")
        }
        let output = URL(fileURLWithPath: arguments[outputIndex + 1])
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        if runCount == 1 {
            let artifacts = currentDirectory.appendingPathComponent("artifacts", isDirectory: true)
            try FileManager.default.createDirectory(at: artifacts, withIntermediateDirectories: true)
            try Data("partial".utf8).write(to: artifacts.appendingPathComponent("partial.txt"))
            try Data("incomplete".utf8).write(to: output)
            return WorkflowProcessResult(status: 75, stdout: "temporary failure")
        }
        try Data([0x89, 0x50, 0x4e, 0x47]).write(to: output)
        return WorkflowProcessResult(status: 0, stdout: "ok")
    }
}
