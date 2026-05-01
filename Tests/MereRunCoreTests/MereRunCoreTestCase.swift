import MLX
import XCTest

class MereRunCoreTestCase: XCTestCase {
    override func invokeTest() {
        let env = ProcessInfo.processInfo.environment
        let debugRaw = (env["MERERUN_TEST_DEBUG_MLX"] ?? "").lowercased()
        let debugEnabled = debugRaw == "1" || debugRaw == "true" || debugRaw == "yes"
        if debugEnabled {
            print("[MereRunCoreTestCase] invokeTest \(type(of: self)) \(name)")
        }

        // IMPORTANT: MLX will attempt to initialize Metal resources while creating
        // default CPU/GPU streams. Install the metallib/bundle first.
        MLXTestSupport.ensureMetalLibraryAvailable()

        let rawDevice = (env["MERERUN_TEST_MLX_DEVICE"] ?? "").lowercased()
        let deviceType: DeviceType = (rawDevice == "gpu") ? .gpu : .cpu
        let device = Device(deviceType)

        Device.withDefaultDevice(device) {
            super.invokeTest()
        }
    }
}
