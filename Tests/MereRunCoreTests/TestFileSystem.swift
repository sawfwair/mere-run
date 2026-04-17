import Foundation

enum TestFileSystem {
    static func makeTempDir(prefix: String = "mererun-tests") throws -> URL {
        let base = FileManager.default.temporaryDirectory
        let dir = base
            .appendingPathComponent(prefix, isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func createDirectory(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    static func writeFile(_ url: URL, contents: Data = Data()) throws {
        let dir = url.deletingLastPathComponent()
        try createDirectory(dir)
        try contents.write(to: url, options: [.atomic])
    }
}

