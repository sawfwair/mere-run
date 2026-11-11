import Foundation

enum MiniMaxH3ImageProfile {
    private struct Event: Encodable {
        let event = "h3_image_profile"
        let phase: String
        let step: Int
        let block: Int
        let seconds: Double
    }

    static func install(on model: MiniMaxH3ImageTransformer) {
        guard ProcessInfo.processInfo.environment["MERERUN_H3_IMAGE_PROFILE"] == "1" else { return }
        model.profile = { phase, step, block, seconds in
            let event = Event(phase: phase, step: step, block: block, seconds: seconds)
            if let data = try? JSONEncoder().encode(event) {
                FileHandle.standardError.write(data + Data([10]))
            }
        }
    }
}
