@testable import MereRunApp
import Foundation
import XCTest

final class StudioModelsPresenterTests: XCTestCase {
    private func row(
        _ id: String,
        category: String,
        status: String = "installed",
        size: String = "2.1 GB",
        title: String? = nil,
        supported: Bool? = nil,
        estimatedDownloadBytes: Int64? = nil
    ) -> StudioModelInventoryRow {
        StudioModelInventoryRow(
            id: id,
            category: category,
            status: status,
            size: size,
            usageTerms: nil,
            title: title,
            estimatedDownloadBytes: estimatedDownloadBytes,
            supported: supported
        )
    }

    private func item(
        mode: StudioMode = .createImage,
        commandPreview: String,
        status: StudioLibraryStatus = .completed,
        createdAt: Date,
        seconds: TimeInterval = 3.4,
        templateID: CommandTemplateID? = nil,
        draft: CommandDraft? = nil,
        model: String? = nil
    ) -> StudioLibraryItem {
        StudioLibraryItem(
            id: UUID(),
            mode: mode,
            prompt: "",
            inputURL: nil,
            outputURL: nil,
            createdAt: createdAt,
            updatedAt: createdAt.addingTimeInterval(seconds),
            status: status,
            exitCode: status == .failed ? 1 : 0,
            commandPreview: commandPreview,
            outputText: nil,
            templateID: templateID,
            commandDraft: draft,
            model: model
        )
    }

    // MARK: Families and chips

    func testFamiliesGroupCategoriesTheWayTheSidebarDoes() {
        XCTAssertEqual(StudioModelFamily.from(category: "image"), .image)
        XCTAssertEqual(StudioModelFamily.from(category: "image-3d"), .threeD)
        XCTAssertEqual(StudioModelFamily.from(category: "text-chat"), .chat)
        XCTAssertEqual(StudioModelFamily.from(category: "text-code"), .chat)
        XCTAssertEqual(StudioModelFamily.from(category: "omni-chat"), .chat)
        XCTAssertEqual(StudioModelFamily.from(category: "text-embed"), .text)
        XCTAssertEqual(StudioModelFamily.from(category: "vision-chat"), .vision)
        XCTAssertEqual(StudioModelFamily.from(category: "vision-segment"), .vision)
        XCTAssertEqual(StudioModelFamily.from(category: "speech-tts"), .voice)
        XCTAssertEqual(StudioModelFamily.from(category: "speech-asr"), .audio)
        XCTAssertEqual(StudioModelFamily.from(category: "audio"), .audio)
        XCTAssertEqual(StudioModelFamily.from(category: "sfx"), .sound)
        XCTAssertEqual(StudioModelFamily.from(category: "music"), .music)
        XCTAssertEqual(StudioModelFamily.from(category: "video"), .video)
        XCTAssertEqual(StudioModelFamily.from(category: "something-new"), .other)
    }

    func testChipRowListsOnlyPresentFamiliesInSidebarOrder() {
        let rows = [
            row("speech-tts-kokoro", category: "speech-tts"),
            row("text-chat-qwen", category: "text-chat"),
            row("image-zimage-nano", category: "image"),
            row("vision-segment-sam", category: "vision-segment"),
        ]
        XCTAssertEqual(StudioModelsPresenter.families(in: rows), [.image, .voice, .chat, .vision])
    }

    func testFilterMatchesFamilyTitleAndDisplayName() {
        let rows = [
            row("image-zimage-nano", category: "image", title: "Zimage Nano"),
            row("text-chat-qwen3.6-4b", category: "text-chat", title: "Qwen3.6 4B"),
        ]
        XCTAssertEqual(StudioModelsPresenter.filter(rows, family: .chat, query: "").map(\.id), ["text-chat-qwen3.6-4b"])
        XCTAssertEqual(StudioModelsPresenter.filter(rows, family: nil, query: "nano").map(\.id), ["image-zimage-nano"])
        XCTAssertEqual(StudioModelsPresenter.filter(rows, family: nil, query: "Chat").map(\.id), ["text-chat-qwen3.6-4b"])
        XCTAssertTrue(StudioModelsPresenter.filter(rows, family: .image, query: "qwen").isEmpty)
    }

    func testListShowsInstalledRowsAndTheRowBeingPulled() {
        let rows = [
            row("image-zimage-nano", category: "image"),
            row("vision-chat-qwen3.6-vl-4b", category: "vision-chat", status: "missing", size: "—"),
            row("text-chat-qwen3.6-4b", category: "text-chat", status: "missing", size: "—"),
        ]
        let listed = StudioModelsPresenter.listRows(rows, pullingIDs: ["vision-chat-qwen3.6-vl-4b"])
        XCTAssertEqual(listed.map(\.id), ["image-zimage-nano", "vision-chat-qwen3.6-vl-4b"])
    }

    // MARK: Row status and meta

    func testRowStatusAndMetaFollowInstallPullAndSupport() {
        let installed = row("image-zimage-nano", category: "image")
        let flagged = row("speech-asr-parakeet-tdt", category: "speech-asr", size: "2.4 GB", supported: false)
        let pulling = row("vision-chat-qwen3.6-vl-4b", category: "vision-chat", status: "missing", size: "—")
        let missing = row("text-chat-qwen3.6-4b", category: "text-chat", status: "missing", size: "—", estimatedDownloadBytes: 2_600_000_000)
        let job = StudioModelsJob(
            kind: .pull,
            modelID: pulling.id,
            subject: "Qwen3.6-VL 4B",
            progress: StudioRunProgress(label: pulling.id, fractionCompleted: 0.25, detail: "1.2 GB / 4.8 GB")
        )

        XCTAssertEqual(StudioModelsPresenter.status(of: installed, job: job), .installed)
        XCTAssertEqual(StudioModelsPresenter.status(of: flagged, job: nil), .attention)
        XCTAssertEqual(StudioModelsPresenter.status(of: pulling, job: job), .pulling(0.25))
        XCTAssertEqual(StudioModelsPresenter.status(of: missing, job: job), .missing)

        XCTAssertEqual(StudioModelsPresenter.meta(for: installed, status: .installed), "Image · 2.1 GB")
        XCTAssertEqual(StudioModelsPresenter.meta(for: flagged, status: .attention), "Audio · 2.4 GB")
        XCTAssertEqual(StudioModelsPresenter.meta(for: pulling, status: .pulling(0.25)), "Vision · pulling 25%")
        XCTAssertEqual(StudioModelsPresenter.meta(for: pulling, status: .pulling(nil)), "Vision · pulling…")
        XCTAssertEqual(StudioModelsPresenter.meta(for: missing, status: .missing), "Chat · 2.6 GB download")
        XCTAssertEqual(
            StudioModelsPresenter.meta(for: row("x", category: "image", size: "not measured"), status: .installed),
            "Image"
        )
    }

    func testSizeChipCombinesQuantizationAndSize() {
        let installed = row("image-zimage-nano", category: "image")
        let facts = StudioModelInfoFacts(root: nil, quantization: "Q4", hasManifest: true, isValid: true)
        XCTAssertEqual(StudioModelsPresenter.sizeChip(for: installed, facts: facts), "Q4 · 2.1 GB")
        XCTAssertEqual(StudioModelsPresenter.sizeChip(for: installed, facts: nil), "2.1 GB")
        XCTAssertNil(StudioModelsPresenter.sizeChip(for: row("x", category: "image", size: "—"), facts: nil))
    }

    // MARK: Toolbar subtitle

    func testInventorySummarySubtitleReportsCountAndStoreSize() {
        XCTAssertEqual(StudioModelInventorySummary(installedCount: 0, storageBytes: nil).subtitle, "No models installed")
        XCTAssertEqual(StudioModelInventorySummary(installedCount: 3, storageBytes: nil).subtitle, "3 installed")
        let sized = StudioModelInventorySummary(installedCount: 92, storageBytes: 48_000_000_000)
        XCTAssertEqual(sized.subtitle, "92 installed · 48 GB on this Mac")
    }

    // MARK: Defaults

    func testDefaultDomainTitlesNameThePromptDomainsWhoseTemplateDefaultsToTheModel() {
        XCTAssertEqual(StudioModelsPresenter.defaultDomainTitles(for: "image-zimage-nano"), ["Image"])
        XCTAssertEqual(StudioModelsPresenter.defaultDomainTitles(for: StudioChatDefaults.fallbackModelID), ["Chat"])
        XCTAssertTrue(StudioModelsPresenter.defaultDomainTitles(for: "no-such-model").isEmpty)
    }

    // MARK: Library-derived facts

    func testUsageCountsRunsThatReferenceTheModelByArgumentDraftOrThread() {
        let now = Date()
        let items = [
            item(commandPreview: "mere.run image generate --model image-zimage-nano --size 768x512", createdAt: now.addingTimeInterval(-300)),
            item(commandPreview: "mere.run image generate --model=image-zimage-nano", createdAt: now.addingTimeInterval(-200)),
            item(commandPreview: "mere.run image generate", createdAt: now.addingTimeInterval(-100), draft: {
                var draft = CommandDraft()
                draft.model = "image-zimage-nano"
                return draft
            }()),
            item(mode: .chat, commandPreview: "mere.run text chat", createdAt: now.addingTimeInterval(-50), model: "image-zimage-nano"),
            item(commandPreview: "mere.run image generate --model image-zimage-nano-xl", createdAt: now.addingTimeInterval(-10)),
        ]
        let usage = try? XCTUnwrap(StudioModelsPresenter.usage(of: "image-zimage-nano", in: items))
        XCTAssertEqual(usage?.runs, 4)
        XCTAssertEqual(usage?.lastUsed, items[3].updatedAt)
        XCTAssertNil(StudioModelsPresenter.usage(of: "text-chat-qwen3.6-4b", in: items))
    }

    func testUsageLineUsesTimeTodayAndDateOtherwise() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        let now = calendar.date(from: DateComponents(year: 2026, month: 9, day: 3, hour: 15))!
        let today = StudioModelsPresenter.usageLine(
            StudioModelUsageSummary(lastUsed: now.addingTimeInterval(-3600), runs: 214),
            now: now,
            calendar: calendar
        )
        XCTAssertTrue(today.hasSuffix(" · 214 runs"), today)
        XCTAssertFalse(today.contains("Sep"), today)

        let earlier = StudioModelsPresenter.usageLine(
            StudioModelUsageSummary(lastUsed: now.addingTimeInterval(-4 * 86_400), runs: 1),
            now: now,
            calendar: calendar
        )
        XCTAssertTrue(earlier.hasSuffix(" · 1 run"), earlier)
        XCTAssertTrue(earlier.contains("30"), earlier)
    }

    func testLastRunDurationUsesTheLatestCompletedNonConversationRun() {
        let now = Date()
        let items = [
            item(commandPreview: "mere.run image generate --model image-zimage-nano", createdAt: now.addingTimeInterval(-500), seconds: 12),
            item(commandPreview: "mere.run image generate --model image-zimage-nano", createdAt: now.addingTimeInterval(-100), seconds: 3.4),
            item(commandPreview: "mere.run image generate --model image-zimage-nano", status: .failed, createdAt: now.addingTimeInterval(-50), seconds: 1),
        ]
        XCTAssertEqual(StudioModelsPresenter.lastRunDuration(for: "image-zimage-nano", in: items), "3.4 s")
        XCTAssertNil(StudioModelsPresenter.lastRunDuration(for: "other", in: items))
        XCTAssertEqual(StudioModelsPresenter.duration(95), "1 min 35 s")
    }

    func testGateAndBenchmarkLinesComeFromTheLatestLibraryRuns() {
        let now = Date()
        let gateDate = now.addingTimeInterval(-86_400 * 4)
        var lite = CommandDraft()
        lite.benchmarkSuite = "lite"
        let items = [
            item(mode: .chat, commandPreview: "mere.run gate", status: .failed, createdAt: now.addingTimeInterval(-86_400 * 9), templateID: .qualityGate),
            item(mode: .chat, commandPreview: "mere.run gate", createdAt: gateDate, templateID: .qualityGate),
            item(mode: .chat, commandPreview: "mere.run model benchmark fused", createdAt: gateDate, templateID: .modelBenchmarkFused, draft: lite),
        ]
        let gate = StudioModelsPresenter.gateLine(in: items)
        XCTAssertEqual(gate?.ok, true)
        XCTAssertEqual(gate?.text, "Quality gate passed · \(StudioModelsPresenter.shortDate(gateDate.addingTimeInterval(3.4)))")
        XCTAssertEqual(
            StudioModelsPresenter.benchmarkLine(in: items),
            "\(StudioModelsPresenter.shortDate(gateDate.addingTimeInterval(3.4))) · Lite suite"
        )
        XCTAssertNil(StudioModelsPresenter.gateLine(in: []))
        XCTAssertNil(StudioModelsPresenter.benchmarkLine(in: []))
    }

    // MARK: Jobs

    func testLibraryPullJobDescribesTheRunningComposerPull() {
        let now = Date()
        let rows = [row("vision-chat-qwen3.6-vl-4b", category: "vision-chat", status: "missing", size: "—", title: "Qwen3.6-VL 4B")]
        let pull = item(
            mode: .readImage,
            commandPreview: "mere.run model pull vision-chat-qwen3.6-vl-4b",
            status: .running,
            createdAt: now,
            templateID: .modelPull
        )
        let progress = StudioRunProgress(label: "vision-chat-qwen3.6-vl-4b", fractionCompleted: 0.25, detail: "1.2 GB / 4.8 GB")
        let job = StudioModelsPresenter.libraryPullJob(in: [pull], rows: rows, progressByRequestID: [pull.id: progress])

        XCTAssertEqual(job?.kind, .pull)
        XCTAssertEqual(job?.modelID, "vision-chat-qwen3.6-vl-4b")
        XCTAssertEqual(job?.label, "Models · Pull Qwen3.6-VL 4B")
        XCTAssertEqual(job?.detail, "1.2 GB / 4.8 GB")
        XCTAssertEqual(job?.fraction, 0.25)
        XCTAssertEqual(job?.libraryItemID, pull.id)
        XCTAssertNil(StudioModelsPresenter.libraryPullJob(in: [], rows: rows, progressByRequestID: [:]))
    }

    func testJobLabelsAndCancellingDetail() {
        let optimize = StudioModelsJob(kind: .optimize, modelID: "video-minimax", subject: "MiniMax H3")
        XCTAssertEqual(optimize.label, "Models · Optimize MiniMax H3")
        XCTAssertNil(optimize.detail)
        let cleanup = StudioModelsJob(kind: .cleanup, modelID: nil, subject: "model storage")
        XCTAssertEqual(cleanup.label, "Models · Clean up model storage")
        let cancelling = StudioModelsJob(kind: .pull, modelID: "x", subject: "X", isCancelling: true)
        XCTAssertEqual(cancelling.detail, "Cancelling…")
    }

    // MARK: model info facts

    func testInfoFactsParseRootQuantizationManifestAndValidation() {
        let output = """
        Model Root: /Users/me/Library/Application Support/MereRun/models/image-zimage-nano
        Model ID: image-zimage-nano
        Source: primary

        Manifest (local)
          schemaVersion: 2
          precision: bf16
          quantization: bits=4 groupSize=64 scheme=affine

        Validation
          isValid: true
        """
        let facts = StudioModelsPresenter.facts(fromInfo: output)
        XCTAssertEqual(facts.root, "/Users/me/Library/Application Support/MereRun/models/image-zimage-nano")
        XCTAssertEqual(facts.quantization, "Q4")
        XCTAssertEqual(facts.hasManifest, true)
        XCTAssertEqual(facts.isValid, true)
        XCTAssertEqual(facts.verifiedLine, "manifest ok · validation passed")

        let missing = StudioModelsPresenter.facts(fromInfo: "Model Root: /m\n\nManifest: (missing)\n\nValidation\n  isValid: false")
        XCTAssertEqual(missing.hasManifest, false)
        XCTAssertEqual(missing.verifiedLine, "manifest missing")
        XCTAssertNil(StudioModelInfoFacts().verifiedLine)
        XCTAssertTrue(StudioModelInfoFacts().isEmpty)
    }

    func testAbbreviatedPathAndSourceLine() {
        XCTAssertEqual(
            StudioModelsPresenter.abbreviatedPath("/Users/me/Library/Application Support/MereRun/models", home: "/Users/me"),
            "~/Library/Application Support/MereRun/models"
        )
        XCTAssertEqual(StudioModelsPresenter.abbreviatedPath("/Volumes/Models", home: "/Users/me"), "/Volumes/Models")
        let sourced = StudioModelInventoryRow(
            id: "image-zimage-nano", category: "image", status: "installed", size: "2.1 GB",
            usageTerms: nil, sourceRepository: "mere-run/zimage-nano-q4"
        )
        XCTAssertEqual(StudioModelsPresenter.sourceLine(for: sourced), "huggingface.co/mere-run/zimage-nano-q4")
        XCTAssertNil(StudioModelsPresenter.sourceLine(for: row("x", category: "image")))
    }

    func testMemoryLineAndAdapterMeta() {
        let both = StudioModelInventoryRow(
            id: "x", category: "image", status: "installed", size: "1 GB", usageTerms: nil,
            minimumUnifiedMemoryGB: 8, recommendedUnifiedMemoryGB: 16
        )
        XCTAssertEqual(StudioModelsPresenter.memoryLine(for: both), "8 GB min · 16 GB recommended")
        XCTAssertNil(StudioModelsPresenter.memoryLine(for: row("x", category: "image")))
        XCTAssertEqual(
            StudioModelsPresenter.adapterMeta(format: "lora", byteCount: 50_331_648, version: "2", installed: true),
            "LoRA · 50.3 MB · v2"
        )
        XCTAssertEqual(
            StudioModelsPresenter.adapterMeta(format: "safetensors", byteCount: 0, version: "", installed: false),
            "SAFETENSORS · not pulled"
        )
    }
}
