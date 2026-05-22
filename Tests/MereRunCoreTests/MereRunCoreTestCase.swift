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

        #if os(Linux)
        let raw = (env["MERERUN_TEST_LINUX_MLX"] ?? "").lowercased()
        let enabled = raw == "1" || raw == "true" || raw == "yes"
        guard enabled else {
            if debugEnabled {
                print("[MereRunCoreTestCase] skipping Linux MLX test \(type(of: self)) \(name)")
            }
            return
        }

        Device.withDefaultDevice(Device(.cpu)) {
            super.invokeTest()
        }
        #else
        // IMPORTANT: MLX will attempt to initialize Metal resources while creating
        // default CPU/GPU streams. Install the metallib/bundle first.
        MLXTestSupport.ensureMetalLibraryAvailable()

        let rawDevice = (env["MERERUN_TEST_MLX_DEVICE"] ?? "").lowercased()
        let deviceType: DeviceType = (rawDevice == "gpu") ? .gpu : .cpu
        let device = Device(deviceType)

        Device.withDefaultDevice(device) {
            super.invokeTest()
        }
        #endif
    }
}
