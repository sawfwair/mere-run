import Foundation

enum CLIStderr {
    static func write(_ text: String) {
        FileHandle.standardError.write(Data(text.utf8))
    }
}
