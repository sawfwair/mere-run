@testable import StudioKit
@testable import StudioUI
import XCTest

final class StudioOperationsAndHealthTests: XCTestCase {
    func testDecodesLocalAndRelayRunListsFromCLIContracts() throws {
        let local = """
        {
          "summary": "Found one run",
          "result": {
            "root": "/tmp/runs",
            "scanned_directory_count": 2,
            "entries": [{
              "id": "run-1",
              "kind": "run_directory",
              "path": "/tmp/runs/run-1",
              "relative_path": "run-1",
              "status": "running",
              "state": "running",
              "summary": "Rendering",
              "created_at": "2026-07-28T12:00:00Z",
              "updated_at": "2026-07-28T12:01:00Z",
              "event_count": 4,
              "artifact_count": 1,
              "diagnostic_count": 0,
              "blocker_count": 0
            }]
          }
        }
        """
        let remote = """
        {
          "executor": "relay:fleet",
          "jobs": [{
            "job_id": "job-1",
            "job_reference": "relay://fleet/job-1",
            "state": "running",
            "executor": "relay:fleet",
            "run_directory": null,
            "created_at": "2026-07-28T12:00:00Z",
            "updated_at": "2026-07-28T12:01:00Z",
            "artifacts": [{
              "name": "preview",
              "kind": "image",
              "path": "preview.png",
              "content_type": "image/png",
              "size_bytes": 42
            }],
            "error": null
          }]
        }
        """

        let localResult = try JSONDecoder().decode(
            StudioLocalRunListEnvelope.self,
            from: try XCTUnwrap(local.data(using: .utf8))
        )
        let remoteResult = try JSONDecoder().decode(
            StudioRemoteRunList.self,
            from: try XCTUnwrap(remote.data(using: .utf8))
        )

        XCTAssertEqual(localResult.result.entries.first?.reference, "/tmp/runs/run-1")
        XCTAssertEqual(localResult.result.entries.first?.eventCount, 4)
        XCTAssertEqual(remoteResult.jobs.first?.jobReference, "relay://fleet/job-1")
        XCTAssertEqual(remoteResult.jobs.first?.artifacts.first?.contentType, "image/png")
    }

    func testDecodesExecutorProfilesWithoutFleetOrNodeSchema() throws {
        let json = """
        {
          "profiles": [
            {"name":"fleet","kind":"relay","destination":null,"url":"https://relay.mere.run"},
            {"name":"gpu","kind":"ssh","destination":"user@gpu","url":null}
          ]
        }
        """

        let profiles = try JSONDecoder().decode(
            StudioExecutorProfiles.self,
            from: try XCTUnwrap(json.data(using: .utf8))
        )

        XCTAssertEqual(profiles.profiles.map(\.reference), ["relay:fleet", "ssh:gpu"])
    }

    func testExtractsRelayDeviceAuthorizationURLFromStreamingCLIProgress() {
        let progress = "Open https://relay.mere.run/device?code=MERE-1234 and enter code MERE-1234."

        XCTAssertEqual(
            StudioOperationsJSON.firstHTTPURL(in: progress)?.absoluteString,
            "https://relay.mere.run/device?code=MERE-1234"
        )
    }

    func testDecodesPluginInstallationAndVerificationState() throws {
        let json = """
        {
          "contractVersion":"mere.run/plugin-catalog.v1",
          "updatedAt":"2026-07-28T12:00:00Z",
          "defaultChannel":"main",
          "source":"https://example.invalid/catalog.json",
          "plugins":[{
            "id":"mere-runpod",
            "name":"RunPod",
            "description":"Remote execution",
            "repo":"https://example.invalid/repo",
            "package":"mere-runpod",
            "subdirectory":"packages/mere-runpod",
            "entrypoint":"mere-runpod",
            "capabilities":["remote-runner"],
            "channels":{"main":{"manager":"pipx","spec":"git+https://example.invalid/repo","ref":"main"}},
            "installed":true,
            "verified":true,
            "installedVersion":"1.2.3",
            "installedPath":"/usr/local/bin/mere-runpod",
            "verificationError":null,
            "installCommand":"pipx install git+https://example.invalid/repo"
          }]
        }
        """

        let snapshot = try JSONDecoder().decode(
            StudioPluginCatalogSnapshot.self,
            from: try XCTUnwrap(json.data(using: .utf8))
        )

        XCTAssertEqual(snapshot.plugins.first?.installedVersion, "1.2.3")
        XCTAssertTrue(snapshot.plugins.first?.verified == true)
    }

    func testManifestHealthParserSeparatesRepairableModelsFromAbsentModels() throws {
        let json = """
        {
          "mode":"preview",
          "wrote_count":1,
          "already_count":2,
          "skipped_count":1,
          "entries":[
            {"model_id":"image-a","status":"would_write","path":"/tmp/a/mererun_model.json","message":null},
            {"model_id":"image-b","status":"ok","path":"/tmp/b/mererun_model.json","message":null},
            {"model_id":"image-c","status":"skipped","path":null,"message":"model directory not found"}
          ]
        }
        """

        let report = try XCTUnwrap(StudioModelHealthJSON.decodeRepairReport(json))

        XCTAssertEqual(report.actionableEntries.map(\.modelID), ["image-a"])
        XCTAssertTrue(report.errorEntries.isEmpty)
        XCTAssertEqual(report.alreadyCount, 2)
    }

    func testManifestRepairTemplateDefaultsToStructuredPreview() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .modelRepairManifests))
        let arguments = template.arguments(from: template.defaultDraft())

        XCTAssertEqual(arguments, ["model", "repair-manifests", "--dry-run", "--json"])
    }
}
