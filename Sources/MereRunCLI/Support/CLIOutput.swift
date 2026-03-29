import Foundation

enum CLIOutput {
    static func resolveOutputURL(
        _ output: String?,
        defaultPrefix: String,
        defaultExtension: String
    ) -> URL {
        if let output, !output.isEmpty {
            let url = URL(fileURLWithPath: output).standardizedFileURL
            if url.pathExtension.isEmpty {
                return url.appendingPathExtension(defaultExtension)
            }
            return url
        }

        let timestamp = Int(Date().timeIntervalSince1970)
        let filename = "\(defaultPrefix)-\(timestamp).\(defaultExtension)"
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(filename)
    }
}
