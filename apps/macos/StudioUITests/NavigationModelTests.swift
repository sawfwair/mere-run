@testable import StudioKit
@testable import StudioUI
import SwiftUI
import XCTest

@MainActor
final class NavigationModelTests: XCTestCase {
    // MARK: - Domains and tasks

    func testEveryDomainOwnsAtLeastOneTaskAndEveryTaskBelongsToItsDomain() {
        for domain in StudioDomain.allCases {
            XCTAssertFalse(domain.tasks.isEmpty, "\(domain) has no tasks")
            XCTAssertEqual(domain.defaultTask.domain, domain)
            for task in domain.tasks {
                XCTAssertEqual(task.domain, domain)
            }
        }
        XCTAssertEqual(
            StudioDomain.allCases.reduce(0) { $0 + $1.tasks.count },
            StudioTask.allCases.count
        )
    }

    func testSidebarGroupsCoverEveryDomainOnceInPlanOrder() {
        let grouped = StudioDomainGroup.allCases.flatMap(\.domains)
        XCTAssertEqual(grouped, StudioDomain.allCases)
        XCTAssertEqual(StudioDomainGroup.create.domains, [.image, .video, .music, .sound, .voice, .threeD])
        XCTAssertEqual(StudioDomainGroup.converse.domains, [.chat])
        XCTAssertEqual(StudioDomainGroup.understand.domains, [.vision, .audio, .text, .earth])
        XCTAssertEqual(StudioDomainGroup.system.domains, [.models, .server, .runs, .plugins])
    }

    func testDomainShortcutsUseCommandDigitsThenOptionCommandDigits() {
        XCTAssertEqual(StudioDomain.image.keyboardShortcut, KeyboardShortcut("1", modifiers: .command))
        XCTAssertEqual(StudioDomain.audio.keyboardShortcut, KeyboardShortcut("9", modifiers: .command))
        XCTAssertEqual(StudioDomain.text.keyboardShortcut, KeyboardShortcut("1", modifiers: [.command, .option]))
        XCTAssertEqual(StudioDomain.plugins.keyboardShortcut, KeyboardShortcut("6", modifiers: [.command, .option]))
        let shortcuts = StudioDomain.allCases.compactMap(\.keyboardShortcut)
        XCTAssertEqual(shortcuts.count, StudioDomain.allCases.count)
        XCTAssertEqual(Set(shortcuts.map { "\($0.key.character)\($0.modifiers.rawValue)" }).count, shortcuts.count)
    }

    func testTaskControlStyleFollowsTaskCount() {
        XCTAssertEqual(StudioTaskControlStyle.style(for: .runs), .none)
        XCTAssertEqual(StudioTaskControlStyle.style(for: .plugins), .none)
        XCTAssertEqual(StudioTaskControlStyle.style(for: .threeD), .none)
        XCTAssertEqual(StudioTaskControlStyle.style(for: .image), .segmented)
        XCTAssertEqual(StudioTaskControlStyle.style(for: .earth), .segmented)
        XCTAssertEqual(StudioTaskControlStyle.style(for: .music), .segmented)
        XCTAssertEqual(StudioTaskControlStyle.style(for: .sound), .segmented)
        XCTAssertEqual(StudioTaskControlStyle.style(for: .vision), .segmentedWithOverflow(visible: 5))
        // The overflow keeps every task reachable and the visible ones in domain order.
        let visible = Array(StudioDomain.vision.tasks.prefix(5))
        XCTAssertEqual(visible, [.visionRead, .visionFind, .visionSegment, .visionTrack, .visionDepth])
        XCTAssertLessThan(5, StudioDomain.vision.tasks.count)
    }

    // MARK: - Destinations

    func testDestinationRawValueRoundTripsForEveryTask() {
        for task in StudioTask.allCases {
            let destination = task.destination
            let decoded = StudioDestination(rawValue: destination.rawValue)
            XCTAssertEqual(decoded, destination, task.rawValue)
        }
        XCTAssertEqual(StudioDestination.default.rawValue, "image/image.generate")
    }

    func testDestinationRejectsMismatchedOrMalformedRawValues() {
        XCTAssertNil(StudioDestination(rawValue: "image/music.compose"))
        XCTAssertNil(StudioDestination(rawValue: "image"))
        XCTAssertNil(StudioDestination(rawValue: "nowhere/image.generate"))
        XCTAssertNil(StudioDestination(rawValue: ""))
    }

    func testEveryModeMapsToExactlyOneTaskAndBack() {
        for mode in StudioMode.allCases {
            let destination = StudioDestination(mode: mode)
            XCTAssertEqual(destination.task.mode, mode)
            XCTAssertEqual(destination, mode.task.destination)
        }
        let modeTasks = StudioTask.allCases.filter { $0.mode != nil }
        XCTAssertEqual(modeTasks.count, StudioMode.allCases.count)
        XCTAssertEqual(Set(modeTasks.compactMap(\.mode)).count, StudioMode.allCases.count)
    }

    func testThreeDAndEarthHaveNoModeBacking() {
        for task in StudioDomain.threeD.tasks + StudioDomain.earth.tasks {
            XCTAssertNil(task.mode, task.rawValue)
        }
        XCTAssertEqual(StudioMode.listen.destination, StudioDestination(domain: .audio, task: .audioTranscribe))
        XCTAssertEqual(StudioMode.sfx.destination, StudioDestination(domain: .sound, task: .soundGenerate))
        XCTAssertEqual(StudioMode.code.destination, StudioDestination(domain: .chat, task: .chatCode))
    }

    func testVisionLabVariantsRoundTripThroughToolbarTasks() {
        for variant in StudioVisionTask.allCases {
            let task = variant.studioTask
            XCTAssertEqual(task.domain, .vision)
            XCTAssertEqual(task.visionLabTask?.studioTask, task)
        }
        XCTAssertEqual(StudioVisionTask.faceCompare.studioTask, .visionFaces)
        XCTAssertEqual(StudioTask.visionFaces.visionLabTask, .faceDetect)
        XCTAssertNil(StudioTask.visionRead.visionLabTask)
    }

    // MARK: - Library attribution

    func testEveryCommandTemplateMapsToADomain() {
        var counts: [StudioDomain: Int] = [:]
        for templateID in CommandTemplateID.allCases {
            counts[templateID.studioDomain, default: 0] += 1
        }
        XCTAssertEqual(CommandTemplateID.imageReconstruct3DTrellis2.studioDomain, .threeD)
        XCTAssertEqual(CommandTemplateID.geoFlood.studioDomain, .earth)
        XCTAssertEqual(CommandTemplateID.textEmbed.studioDomain, .text)
        XCTAssertEqual(CommandTemplateID.qualityGate.studioDomain, .models)
        XCTAssertEqual(CommandTemplateID.modelBenchmarkFused.studioDomain, .models)
        XCTAssertEqual(CommandTemplateID.apiServe.studioDomain, .server)
        XCTAssertEqual(CommandTemplateID.musicServe.studioDomain, .server)
        XCTAssertEqual(CommandTemplateID.speechDiarize.studioDomain, .audio)
        XCTAssertEqual(CommandTemplateID.speechSynthesize.studioDomain, .voice)
        XCTAssertEqual(CommandTemplateID.runFetch.studioDomain, .runs)
        XCTAssertEqual(CommandTemplateID.pluginInstall.studioDomain, .plugins)
        for mode in StudioMode.allCases {
            XCTAssertEqual(mode.defaultTemplateID.studioDomain, mode.destination.domain, "\(mode)")
        }
        // Every domain files at least one command.
        XCTAssertEqual(Set(counts.keys), Set(StudioDomain.allCases))
    }

    func testLibraryRowsFileUnderTheirCommandsDomainNotTheAttributingMode() {
        let now = Date()
        func row(_ mode: StudioMode, _ templateID: CommandTemplateID?) -> StudioLibraryItem {
            StudioLibraryItem(
                id: UUID(), mode: mode, prompt: "p", inputURL: nil, outputURL: nil,
                createdAt: now, updatedAt: now, status: .completed, exitCode: 0,
                commandPreview: "", outputText: nil, customTitle: nil, templateID: templateID
            )
        }
        let mesh = row(.createImage, .imageReconstruct3D)
        let flood = row(.readImage, .geoFlood)
        let gate = row(.chat, .qualityGate)
        let legacyImage = row(.createImage, nil)
        let items = [mesh, flood, gate, legacyImage]

        XCTAssertEqual(mesh.domain, .threeD)
        XCTAssertEqual(flood.domain, .earth)
        XCTAssertEqual(gate.domain, .models)
        XCTAssertEqual(legacyImage.domain, .image)
        XCTAssertEqual(items.filter { $0.domain == .image }.map(\.id), [legacyImage.id])
        XCTAssertEqual(items.filter { $0.domain == .vision }, [])
        XCTAssertEqual(items.filter { $0.domain == .chat }, [])
    }

    func testOnlyDomainsWithAPromptModeShowTheLibraryColumn() {
        let withLibrary = StudioDomain.allCases.filter(\.hasPromptWorkspace)
        XCTAssertEqual(withLibrary, [.image, .video, .music, .sound, .voice, .chat, .vision, .audio])
        for domain in [StudioDomain.threeD, .text, .earth, .models, .server, .runs, .plugins] {
            XCTAssertFalse(domain.hasPromptWorkspace, "\(domain)")
        }
    }

    // MARK: - NavigationModel

    func testConsolePresenceGatesTheComposerSync() {
        let navigation = NavigationModel()
        XCTAssertTrue(navigation.shouldSyncComposerToConsole(requested: true))
        XCTAssertFalse(navigation.shouldSyncComposerToConsole(requested: false))

        navigation.isConsoleOpen = true
        XCTAssertFalse(navigation.shouldSyncComposerToConsole(requested: true), "raising an open console keeps its edits")

        navigation.isConsoleOpen = false
        XCTAssertTrue(navigation.shouldSyncComposerToConsole(requested: true))
    }

    func testRestoreReconcilesRememberedTaskVisionVariantAndPromptMode() {
        let navigation = NavigationModel()

        // A v1 upgrader: studio.mode says chat while the destination defaults to Image.
        XCTAssertEqual(navigation.restore(destination: .default, lastPromptMode: .chat), .createImage)
        XCTAssertEqual(navigation.destination, .default)

        // A persisted Vision ▸ Pose destination sets the rail variant and remembers the task.
        let pose = StudioDestination(domain: .vision, task: .visionPose)
        XCTAssertEqual(navigation.restore(destination: pose, lastPromptMode: .music), .music)
        XCTAssertEqual(navigation.destination, pose)
        XCTAssertEqual(navigation.visionLabTask, .pose)
        XCTAssertEqual(navigation.rememberedTasks[.vision], .visionPose)

        // A System destination keeps the persisted prompt mode.
        let models = StudioDestination(domain: .models, task: .modelsHealth)
        XCTAssertEqual(navigation.restore(destination: models, lastPromptMode: .code), .code)
        navigation.open(domain: .vision)
        XCTAssertEqual(navigation.destination.task, .visionPose)
    }

    func testVisionLabVariantFollowsTheToolbarAndTheRail() {
        let navigation = NavigationModel(destination: StudioDestination(domain: .vision, task: .visionRead))
        navigation.open(task: .visionFaces)
        XCTAssertEqual(navigation.visionLabTask, .faceDetect)

        navigation.selectVisionLabVariant(.faceCompare)
        XCTAssertEqual(navigation.visionLabTask, .faceCompare)
        XCTAssertEqual(navigation.destination.task, .visionFaces)

        navigation.open(task: .visionFaces)
        XCTAssertEqual(navigation.visionLabTask, .faceCompare, "re-picking the same toolbar task keeps the variant")

        navigation.selectVisionLabVariant(.geometryMultiview)
        XCTAssertEqual(navigation.destination.task, .visionGeometry)
        navigation.open(task: .visionRead)
        XCTAssertEqual(navigation.visionLabTask, .geometryMultiview, "prompt tasks leave the lab variant alone")
    }

    func testOpenDomainRemembersTheLastTaskShownThere() {
        let navigation = NavigationModel()
        navigation.open(task: .musicTranscribe)
        navigation.open(domain: .models)
        XCTAssertEqual(navigation.destination, StudioDestination(domain: .models, task: .modelsInstalled))

        navigation.open(domain: .music)
        XCTAssertEqual(navigation.destination.task, .musicTranscribe)

        navigation.open(domain: .music)
        XCTAssertEqual(navigation.destination.task, .musicTranscribe, "reopening the current domain keeps its task")
    }

    func testOpeningALibraryRowSwitchesToItsModeAndShowsTheLibrary() {
        let navigation = NavigationModel(destination: .default)
        navigation.showLibrary = false
        let id = UUID()

        navigation.open(libraryItem: id, mode: .music)

        XCTAssertEqual(navigation.destination, StudioDestination(domain: .music, task: .musicCompose))
        XCTAssertEqual(navigation.selectedLibraryID, id)
        XCTAssertTrue(navigation.showLibrary)
    }

    func testDeepLinkImportRoutesToTheImportedRow() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("navigation-import-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let artifactURL = root.appendingPathComponent("clip.mp4")
        try Data("video".utf8).write(to: artifactURL)
        let receipt = StudioLibraryImportReceipt(
            version: StudioLibraryImportReceipt.currentVersion,
            id: UUID(),
            source: .raycast,
            kind: .video,
            prompt: "a paper boat",
            artifactPath: artifactURL.path,
            createdAt: Date()
        )
        let receiptURL = root.appendingPathComponent("receipt.json")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(receipt).write(to: receiptURL)
        let library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))

        var components = URLComponents()
        components.scheme = MereRunDeepLink.scheme
        components.host = "library"
        components.path = "/import"
        components.queryItems = [URLQueryItem(name: "receipt", value: receiptURL.path)]
        let link = try XCTUnwrap(components.url)

        let navigation = NavigationModel()
        navigation.open(deepLink: link, library: library)

        XCTAssertNil(navigation.deepLinkError)
        XCTAssertEqual(navigation.destination, StudioDestination(domain: .video, task: .videoGenerate))
        XCTAssertEqual(navigation.selectedLibraryID, receipt.id)
        XCTAssertEqual(library.items.first?.id, receipt.id)
    }

    func testDeepLinkErrorsSurfaceWithoutMovingTheDestination() throws {
        let navigation = NavigationModel(destination: StudioDestination(domain: .chat, task: .chatChat))
        let library = StudioLibraryStore(
            libraryURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("navigation-error-\(UUID().uuidString).json")
        )
        let link = try XCTUnwrap(URL(string: "mererun://generate?path=/tmp/output.png"))

        navigation.open(deepLink: link, library: library)

        XCTAssertEqual(navigation.deepLinkError, MereRunDeepLinkError.unsupportedRoute.errorDescription)
        XCTAssertEqual(navigation.destination, StudioDestination(domain: .chat, task: .chatChat))
    }
}
