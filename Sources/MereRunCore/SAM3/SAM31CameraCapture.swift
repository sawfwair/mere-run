import Foundation

#if canImport(AVFoundation)
import AVFoundation

public enum SAM31CameraCapture {
    public enum CaptureError: LocalizedError, Sendable {
        case cameraUnavailable(Int)
        case failedToCreateInput
        case failedToCreateOutputDirectory(URL)
        case failedToRecord(URL, reason: String)

        public var errorDescription: String? {
            switch self {
            case .cameraUnavailable(let index):
                return "No video camera available at index \(index)."
            case .failedToCreateInput:
                return "Failed to create an AVCaptureDeviceInput for the selected camera."
            case .failedToCreateOutputDirectory(let url):
                return "Failed to create capture output directory: \(url.path)"
            case .failedToRecord(let url, let reason):
                return "Failed to record camera capture to \(url.path): \(reason)"
            }
        }
    }

    public static func record(
        cameraIndex: Int,
        durationSeconds: TimeInterval,
        outputURL: URL
    ) throws {
        let fileManager = FileManager.default
        do {
            try fileManager.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        } catch {
            throw CaptureError.failedToCreateOutputDirectory(outputURL.deletingLastPathComponent())
        }
        try? fileManager.removeItem(at: outputURL)

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .continuityCamera, .external],
            mediaType: .video,
            position: .unspecified
        )
        let devices = discovery.devices
        guard cameraIndex >= 0, cameraIndex < devices.count else {
            throw CaptureError.cameraUnavailable(cameraIndex)
        }

        let session = AVCaptureSession()
        let device = devices[cameraIndex]
        guard let input = try? AVCaptureDeviceInput(device: device) else {
            throw CaptureError.failedToCreateInput
        }
        guard session.canAddInput(input) else {
            throw CaptureError.failedToCreateInput
        }
        session.addInput(input)

        let output = AVCaptureMovieFileOutput()
        guard session.canAddOutput(output) else {
            throw CaptureError.failedToCreateInput
        }
        session.addOutput(output)

        let delegate = CameraRecordingDelegate()
        session.startRunning()
        output.startRecording(to: outputURL, recordingDelegate: delegate)
        Thread.sleep(forTimeInterval: max(durationSeconds, 0.1))
        output.stopRecording()
        delegate.wait()
        session.stopRunning()

        if let error = delegate.recordingError {
            throw CaptureError.failedToRecord(outputURL, reason: error.localizedDescription)
        }
    }
}

private final class CameraRecordingDelegate: NSObject, AVCaptureFileOutputRecordingDelegate {
    private let semaphore = DispatchSemaphore(value: 0)
    fileprivate var recordingError: Error?

    func fileOutput(
        _ output: AVCaptureFileOutput,
        didFinishRecordingTo outputFileURL: URL,
        from connections: [AVCaptureConnection],
        error: Error?
    ) {
        recordingError = error
        semaphore.signal()
    }

    fileprivate func wait() {
        semaphore.wait()
    }
}
#else
import MediaIO

public enum SAM31CameraCapture {
    public enum CaptureError: LocalizedError, Sendable {
        case cameraUnavailable(Int)
        case failedToCreateInput
        case failedToCreateOutputDirectory(URL)
        case failedToRecord(URL, reason: String)

        public var errorDescription: String? {
            switch self {
            case .cameraUnavailable(let index):
                return "No video camera available at index \(index)."
            case .failedToCreateInput:
                return "Failed to create a camera capture input for this platform."
            case .failedToCreateOutputDirectory(let url):
                return "Failed to create capture output directory: \(url.path)"
            case .failedToRecord(let url, let reason):
                return "Failed to record camera capture to \(url.path): \(reason)"
            }
        }
    }

    public static func record(
        cameraIndex: Int,
        durationSeconds: TimeInterval,
        outputURL: URL
    ) throws {
        let fileManager = FileManager.default
        do {
            try fileManager.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        } catch {
            throw CaptureError.failedToCreateOutputDirectory(outputURL.deletingLastPathComponent())
        }
        try? fileManager.removeItem(at: outputURL)

        let devicePath = "/dev/video\(cameraIndex)"
        guard fileManager.fileExists(atPath: devicePath) else {
            throw CaptureError.cameraUnavailable(cameraIndex)
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: MediaTool.ffmpegPath)
        process.arguments = [
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "v4l2",
            "-t", String(max(durationSeconds, 0.1)),
            "-i", devicePath,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            outputURL.path,
        ]

        let stderrPipe = Pipe()
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            throw CaptureError.failedToRecord(outputURL, reason: error.localizedDescription)
        }
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let details = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "ffmpeg failed"
            throw CaptureError.failedToRecord(outputURL, reason: details)
        }
    }
}
#endif
