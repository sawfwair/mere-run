import Foundation

#if canImport(CONNXRuntime) && os(macOS)
import CONNXRuntime
import COnnxRuntimeCoreML
import Darwin

final class FaceONNXSession {
    private let api: UnsafePointer<OrtApi>
    private var environment: OpaquePointer?
    private var session: OpaquePointer?
    private var memoryInfo: OpaquePointer?

    init(modelURL: URL, executionProvider: FaceExecutionProvider) throws {
        guard let api = OrtGetApiBase()?.pointee.GetApi(UInt32(ORT_API_VERSION)) else {
            throw FaceAnalysisError.runtimeFailure("could not load ONNX Runtime API")
        }
        self.api = api

        try checked(api.pointee.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "mere.run.face", &environment))
        var options: OpaquePointer?
        try checked(api.pointee.CreateSessionOptions(&options))
        defer { api.pointee.ReleaseSessionOptions(options) }
        try checked(api.pointee.SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL))
        let savedStandardOutput: Int32 = executionProvider == .coreML ? Self.redirectStandardOutputToStandardError() : -1
        defer { Self.restoreStandardOutput(savedStandardOutput) }
        if executionProvider == .coreML {
            let status = MereRunOrtAppendCoreML(options, 0x010)
            try checked(status)
        }
        let createStatus = modelURL.path.withCString { path in
            api.pointee.CreateSession(environment, path, options, &session)
        }
        try checked(createStatus)
        try checked(api.pointee.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo))
    }

    deinit {
        api.pointee.ReleaseMemoryInfo(memoryInfo)
        api.pointee.ReleaseSession(session)
        api.pointee.ReleaseEnv(environment)
    }

    func run(input: [Float], shape: [Int64], inputName: String, outputNames: [String]) throws -> [[Float]] {
        guard let memoryInfo, let session else {
            throw FaceAnalysisError.runtimeFailure("ONNX session is not initialized")
        }
        var mutableInput = input
        var inputValue: OpaquePointer?
        try mutableInput.withUnsafeMutableBytes { buffer in
            try shape.withUnsafeBufferPointer { shapeBuffer in
                try checked(api.pointee.CreateTensorWithDataAsOrtValue(
                    memoryInfo,
                    buffer.baseAddress,
                    buffer.count,
                    shapeBuffer.baseAddress,
                    shapeBuffer.count,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                    &inputValue
                ))
            }
        }
        defer { api.pointee.ReleaseValue(inputValue) }

        let inputCString = strdup(inputName)
        let outputCStrings = outputNames.map { strdup($0) }
        defer {
            free(inputCString)
            outputCStrings.forEach { free($0) }
        }
        let inputNamePointers: [UnsafePointer<CChar>?] = [
            inputCString.map { UnsafePointer<CChar>($0) },
        ]
        let outputNamePointers: [UnsafePointer<CChar>?] = outputCStrings.map {
            $0.map { UnsafePointer<CChar>($0) }
        }
        var outputs = [OpaquePointer?](repeating: nil, count: outputNames.count)

        try inputNamePointers.withUnsafeBufferPointer { inputNamesBuffer in
            try outputNamePointers.withUnsafeBufferPointer { outputNamesBuffer in
                try outputs.withUnsafeMutableBufferPointer { outputBuffer in
                    let inputValues: [OpaquePointer?] = [inputValue]
                    try inputValues.withUnsafeBufferPointer { inputValuesBuffer in
                        try checked(api.pointee.Run(
                            session,
                            nil,
                            inputNamesBuffer.baseAddress,
                            inputValuesBuffer.baseAddress,
                            1,
                            outputNamesBuffer.baseAddress,
                            outputNamesBuffer.count,
                            outputBuffer.baseAddress
                        ))
                    }
                }
            }
        }
        defer { outputs.forEach { api.pointee.ReleaseValue($0) } }
        return try outputs.map(readFloatTensor)
    }

    private func readFloatTensor(_ value: OpaquePointer?) throws -> [Float] {
        guard let value else { throw FaceAnalysisError.invalidModelOutput("missing output tensor") }
        var typeAndShape: OpaquePointer?
        try checked(api.pointee.GetTensorTypeAndShape(value, &typeAndShape))
        defer { api.pointee.ReleaseTensorTypeAndShapeInfo(typeAndShape) }
        var count = 0
        try checked(api.pointee.GetTensorShapeElementCount(typeAndShape, &count))
        var data: UnsafeMutableRawPointer?
        try checked(api.pointee.GetTensorMutableData(value, &data))
        guard let floats = data?.assumingMemoryBound(to: Float.self) else {
            throw FaceAnalysisError.invalidModelOutput("output tensor has no data")
        }
        return Array(UnsafeBufferPointer(start: floats, count: count))
    }

    private func checked(_ status: OpaquePointer?) throws {
        guard let status else { return }
        defer { api.pointee.ReleaseStatus(status) }
        let message = api.pointee.GetErrorMessage(status).map(String.init(cString:)) ?? "unknown ONNX Runtime error"
        throw FaceAnalysisError.runtimeFailure(message)
    }

    private static func redirectStandardOutputToStandardError() -> Int32 {
        fflush(stdout)
        let saved = dup(STDOUT_FILENO)
        if saved >= 0 { _ = dup2(STDERR_FILENO, STDOUT_FILENO) }
        return saved
    }

    private static func restoreStandardOutput(_ saved: Int32) {
        guard saved >= 0 else { return }
        fflush(stdout)
        _ = dup2(saved, STDOUT_FILENO)
        close(saved)
    }
}
#else
final class FaceONNXSession {
    init(modelURL: URL, executionProvider: FaceExecutionProvider) throws {
        throw FaceAnalysisError.unavailableRuntime(
            "Face analysis currently requires the macOS ONNX Runtime build; model: \(modelURL.path)"
        )
    }

    func run(input: [Float], shape: [Int64], inputName: String, outputNames: [String]) throws -> [[Float]] {
        throw FaceAnalysisError.unavailableRuntime("Face analysis ONNX Runtime is unavailable on this platform.")
    }
}
#endif
