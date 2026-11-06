@testable import StudioKit
@testable import StudioUI
import Foundation
import SwiftUI
import XCTest

@MainActor
final class StudioAudioRecorderTests: XCTestCase {
    private var root: URL!

    override func setUp() async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("microphone-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDown() async throws { try FileManager.default.removeItem(at: root) }

    func testRepeatedStartWhilePermissionIsPendingRequestsOnlyOnce() async {
        let permission = MicrophonePermissionGate()
        let device = TestMicrophoneRecording()
        var opened = 0
        let recorder = StudioAudioRecorder(requestPermission: { await permission.request() }, makeRecording: { _ in
            opened += 1
            return device
        })
        let first = Task { await recorder.start(domain: .voice, privateDirectory: root) }
        await permission.waitUntilRequested()
        let second = Task { await recorder.start(domain: .voice, privateDirectory: root) }
        // Join both starts after granting every request, including any duplicate.
        permission.resolveAll(true)
        await first.value
        await second.value
        XCTAssertEqual(permission.requestCount, 1)
        XCTAssertEqual(opened, 1)
        XCTAssertTrue(recorder.isRecording)
        recorder.discard()
    }

    func testDiscardRejectsLatePermissionWithoutCreatingAFile() async {
        let permission = MicrophonePermissionGate()
        var opened = 0
        let recorder = StudioAudioRecorder(requestPermission: { await permission.request() }, makeRecording: { _ in
            opened += 1
            return TestMicrophoneRecording()
        })
        let starting = Task { await recorder.start(domain: .voice, privateDirectory: root) }
        await permission.waitUntilRequested()
        recorder.discard()
        permission.resolveAll(true)
        await starting.value
        XCTAssertEqual(opened, 0)
        XCTAssertFalse(recorder.isRecording)
        XCTAssertNil(recorder.lastRecordingURL)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Recordings").path))
        recorder.discard()
    }

    func testStopRejectsLatePermission() async {
        let permission = MicrophonePermissionGate()
        var opened = 0
        let recorder = StudioAudioRecorder(requestPermission: { await permission.request() }, makeRecording: { _ in
            opened += 1
            return TestMicrophoneRecording()
        })
        let starting = Task { await recorder.start(domain: .audio, privateDirectory: root) }
        await permission.waitUntilRequested()
        XCTAssertNil(recorder.stop())
        permission.resolveAll(true)
        await starting.value
        XCTAssertEqual(opened, 0)
        XCTAssertFalse(recorder.isRecording)
        recorder.discard()
    }

    func testCallerCancellationRejectsLatePermission() async {
        let permission = MicrophonePermissionGate()
        var opened = 0
        let recorder = StudioAudioRecorder(requestPermission: { await permission.request() }, makeRecording: { _ in
            opened += 1
            return TestMicrophoneRecording()
        })
        let starting = Task { await recorder.start(domain: .audio, privateDirectory: root) }
        await permission.waitUntilRequested()
        starting.cancel()
        permission.resolveAll(true)
        await starting.value
        XCTAssertEqual(opened, 0)
        XCTAssertFalse(recorder.isRecording)
        XCTAssertNil(recorder.errorMessage)
        recorder.discard()
    }

    func testStartDuringRecordingKeepsTheOwnedDeviceAndFile() async {
        let device = TestMicrophoneRecording()
        var requests = 0
        var opened = 0
        let recorder = StudioAudioRecorder(requestPermission: { requests += 1; return true }, makeRecording: { url in
            opened += 1
            try Data("recording".utf8).write(to: url)
            return device
        })
        await recorder.start(domain: .voice, privateDirectory: root)
        let file = recorder.lastRecordingURL
        device.currentTime = 2.5
        await recorder.start(domain: .voice, privateDirectory: root)
        XCTAssertEqual(requests, 1)
        XCTAssertEqual(opened, 1)
        XCTAssertEqual(recorder.stop(), file)
        XCTAssertEqual(recorder.duration, 2.5)
        XCTAssertEqual(device.stops, 1)
        XCTAssertFalse(recorder.isRecording)
        recorder.discard()
        XCTAssertEqual(device.stops, 1)
        XCTAssertNil(recorder.lastRecordingURL)
        XCTAssertEqual(recorder.duration, 0)
        if let file { XCTAssertFalse(FileManager.default.fileExists(atPath: file.path)) }
    }

    func testCancelledAttemptCannotReplaceARecordingStartedLater() async {
        let permission = MicrophonePermissionGate()
        let device = TestMicrophoneRecording()
        var requests = 0
        var opened = 0
        let recorder = StudioAudioRecorder(requestPermission: {
            requests += 1
            return requests == 1 ? await permission.request() : true
        }, makeRecording: { _ in opened += 1; return device })
        let old = Task { await recorder.start(domain: .voice, privateDirectory: root) }
        await permission.waitUntilRequested()
        recorder.cancelPendingStart()
        await recorder.start(domain: .voice, privateDirectory: root)
        let currentFile = recorder.lastRecordingURL
        permission.resolveAll(true)
        await old.value
        XCTAssertEqual(opened, 1)
        XCTAssertEqual(recorder.lastRecordingURL, currentFile)
        XCTAssertTrue(recorder.isRecording)
        XCTAssertNil(recorder.errorMessage)
        recorder.discard()
    }

    func testPermissionDenialCanBeRetriedWithoutOpeningTheMicrophone() async {
        var granted = false
        var opened = 0
        let recorder = StudioAudioRecorder(requestPermission: { granted }, makeRecording: { _ in
            opened += 1
            return TestMicrophoneRecording()
        })
        await recorder.start(domain: .voice, privateDirectory: root)
        XCTAssertEqual(opened, 0)
        XCTAssertFalse(recorder.isRecording)
        XCTAssertNotNil(recorder.errorMessage)
        granted = true
        await recorder.start(domain: .voice, privateDirectory: root)
        XCTAssertEqual(opened, 1)
        XCTAssertTrue(recorder.isRecording)
        XCTAssertNil(recorder.errorMessage)
        recorder.discard()
    }

    func testCancelledQueuedStartDoesNotRequestPermission() async {
        var requests = 0
        var opened = 0
        let recorder = StudioAudioRecorder(requestPermission: { requests += 1; return true }, makeRecording: { _ in
            opened += 1
            return TestMicrophoneRecording()
        })
        let starting = Task { await recorder.start(domain: .audio, privateDirectory: root) }
        starting.cancel()
        await starting.value
        XCTAssertEqual(requests, 0)
        XCTAssertEqual(opened, 0)
        XCTAssertFalse(recorder.isStarting)
        XCTAssertFalse(recorder.isRecording)
    }

    func testLatePermissionDenialCannotAddAnErrorToANewerRecording() async {
        let permission = MicrophonePermissionGate()
        var requests = 0
        let recorder = StudioAudioRecorder(requestPermission: {
            requests += 1
            return requests == 1 ? await permission.request() : true
        }, makeRecording: { _ in TestMicrophoneRecording() })
        let old = Task { await recorder.start(domain: .voice, privateDirectory: root) }
        await permission.waitUntilRequested()
        recorder.cancelPendingStart()
        await recorder.start(domain: .voice, privateDirectory: root)
        permission.resolveAll(false)
        await old.value
        XCTAssertTrue(recorder.isRecording)
        XCTAssertNil(recorder.errorMessage)
        recorder.discard()
    }

    func testDeviceStartupFailureReleasesTheStartForRetry() async {
        var attempts = 0
        let recorder = StudioAudioRecorder(requestPermission: { true }, makeRecording: { _ in
            attempts += 1
            if attempts == 1 { throw CocoaError(.fileWriteUnknown) }
            return TestMicrophoneRecording()
        })
        await recorder.start(domain: .voice, privateDirectory: root)
        XCTAssertFalse(recorder.isStarting)
        XCTAssertFalse(recorder.isRecording)
        XCTAssertNil(recorder.lastRecordingURL)
        XCTAssertNotNil(recorder.errorMessage)
        await recorder.start(domain: .voice, privateDirectory: root)
        XCTAssertTrue(recorder.isRecording)
        XCTAssertNil(recorder.errorMessage)
        recorder.discard()
    }

    func testPermissionPendingPopoverSnapshots() async throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_STUDIO_SNAPSHOT_DIR"] else {
            throw XCTSkip("Set MERERUN_STUDIO_SNAPSHOT_DIR to render the microphone permission popover.")
        }
        for appearance in StudioSnapshotAppearance.allCases {
            let permission = MicrophonePermissionGate()
            let recorder = StudioAudioRecorder(requestPermission: { await permission.request() }, makeRecording: { _ in
                XCTFail("a pending permission must not open the microphone")
                return TestMicrophoneRecording()
            })
            let starting = Task { await recorder.start(domain: .voice, privateDirectory: root) }
            await permission.waitUntilRequested()
            XCTAssertTrue(recorder.isStarting)
            let view = StudioAudioRecordingPopover(domain: .voice, recorder: recorder) { _ in
                XCTFail("a pending permission cannot deliver a recording")
            }
            let rendered = Result {
                try StudioSnapshotRenderer.writePNG(view, size: CGSize(width: 352, height: 220), appearance: appearance,
                    to: URL(fileURLWithPath: path).appendingPathComponent("microphone-permission-\(appearance.rawValue).png"))
            }
            recorder.cancelPendingStart()
            permission.resolveAll(false)
            await starting.value
            _ = try rendered.get()
            XCTAssertFalse(recorder.isRecording)
        }
    }
}

@MainActor
private final class TestMicrophoneRecording: StudioMicrophoneRecording {
    var currentTime: TimeInterval = 0
    private(set) var stops = 0
    func stop() { stops += 1 }
}

@MainActor
private final class MicrophonePermissionGate {
    private var waiters: [CheckedContinuation<Bool, Never>] = []
    private var requested: CheckedContinuation<Void, Never>?
    private var resolution: Bool?
    private(set) var requestCount = 0

    func request() async -> Bool {
        requestCount += 1
        requested?.resume()
        requested = nil
        if let resolution { return resolution }
        return await withCheckedContinuation { waiters.append($0) }
    }

    func waitUntilRequested() async {
        if requestCount > 0 { return }
        await withCheckedContinuation { requested = $0 }
    }

    func resolveAll(_ granted: Bool) {
        resolution = granted
        let pending = waiters
        waiters = []
        for waiter in pending { waiter.resume(returning: granted) }
    }
}
