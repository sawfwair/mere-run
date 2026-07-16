import Foundation
import MLX

public enum NativeMLXRuntime {
    public static var backendFamilyDescription: String {
        #if os(macOS) || os(iOS)
        return "native MLX/Metal"
        #else
        return "native MLX"
        #endif
    }

    public static var defaultDeviceType: String {
        Device.defaultDevice().deviceType?.rawValue ?? "unknown"
    }

    public static var backendDescription: String {
        let device = defaultDeviceType
        #if os(macOS) || os(iOS)
        let accelerator = device == "gpu" ? "Metal" : "CPU"
        return "native MLX/\(accelerator) (default device: \(device))"
        #else
        return "native MLX (default device: \(device))"
        #endif
    }
}
