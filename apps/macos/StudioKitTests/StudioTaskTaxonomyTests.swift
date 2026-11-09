@testable import StudioKit
import XCTest

/// The sidebar's shape: every command the app builds has one task, task titles are unique within
/// a domain, and the capabilities that used to reach only the Command Console have a task.
final class StudioTaskTaxonomyTests: XCTestCase {
    func testTaskTitlesAreUniqueWithinEachDomain() {
        for domain in StudioDomain.allCases {
            let titles = domain.tasks.map(\.title)
            XCTAssertEqual(Set(titles).count, titles.count, "\(domain) repeats a task title: \(titles)")
        }
    }

    func testEveryTaskBelongsToTheDomainItsCommandsFileUnder() {
        for task in StudioTask.allCases {
            for template in task.variantTemplates {
                XCTAssertEqual(template.id.studioDomain, task.domain, "\(template.id) files under another domain than \(task)")
            }
        }
    }

    func testVideoEditAndCosmosOwnTheirCommands() {
        XCTAssertEqual(StudioDomain.video.tasks, [.videoGenerate, .videoEdit, .videoCosmos, .videoSubjects])
        XCTAssertEqual(StudioTask.videoEdit.variantTemplates.map(\.id), [.videoRetake, .videoDubIt])
        XCTAssertEqual(StudioTask.videoCosmos.variantTemplates.map(\.id), [.videoCosmos3])
        XCTAssertFalse(StudioTask.videoGenerate.runs(.videoRetake), "Generate's feed no longer claims Retake rows")
        for task in [StudioTask.videoEdit, .videoCosmos] {
            XCTAssertTrue(task.usesTaskDraft, "\(task)")
            XCTAssertEqual(task.archetype, .generate, "\(task)")
            XCTAssertEqual(task.generateArchetype?.primaryOutput, .video, "\(task)")
        }
    }

    func testDocumentRunsEveryOCRBackendOnTheTaskWorkspace() throws {
        XCTAssertEqual(StudioTask.visionDocument.domain, .vision)
        XCTAssertEqual(StudioTask.visionDocument.variantTemplates.map(\.id), [.visionOCR])
        XCTAssertTrue(StudioTask.visionDocument.usesTaskDraft)
        XCTAssertEqual(StudioTask.visionDocument.analyzeArchetype?.views, [.text])
        let backend = try XCTUnwrap(CommandTemplateID.visionOCR.capability?.options.first { $0.flag == "--backend" })
        XCTAssertTrue(backend.choices.contains("unlimited"))
    }

    func testDepthAndGeometrySitInThreeD() {
        XCTAssertEqual(StudioDomain.threeD.tasks, [.threeDFromImage, .visionDepth, .visionGeometry])
        XCTAssertFalse(StudioDomain.vision.tasks.contains(.visionDepth))
        XCTAssertEqual(CommandTemplateID.visionGeometryMultiview.studioDomain, .threeD)
    }

    func testServerWorldHostsTheWorldSessionServer() {
        XCTAssertEqual(StudioDomain.server.tasks, [.serverServing, .serverMusic, .serverVision, .serverWorld])
        XCTAssertEqual(CommandTemplateID.worldServe.studioTask, .serverWorld)
        XCTAssertEqual(StudioTask.serverWorld.archetype, .session)
        XCTAssertTrue(CommandTemplateID.worldServe.isResidentServer)
    }

    func testRetiredTasksDecodeToNothing() {
        for raw in ["audio.separate", "sound.condition", "sound.encode", "sound.decode"] {
            XCTAssertNil(StudioTask(rawValue: raw), raw)
        }
        XCTAssertNil(StudioDestination(rawValue: "audio/audio.separate"))
        XCTAssertNil(StudioDestination(rawValue: "vision/vision.depth"), "Depth moved to 3D")
        XCTAssertNotNil(StudioDestination(rawValue: "threeD/vision.depth"))
    }
}
