#if os(macOS)
import Foundation
@preconcurrency import llama

public enum LlamaError: Error {
    case couldNotInitializeContext
    case decodeFailed
}

/// Thread-safe wrapper for llama.cpp context using GGUF model inference.
/// Uses a serial dispatch queue internally for thread safety.
public final class LlamaContext: @unchecked Sendable {
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vocab: OpaquePointer?
    private var sampling: UnsafeMutablePointer<llama_sampler>?
    private var tokensList: [llama_token]

    /// Temporary buffer for multi-byte UTF-8 characters.
    private var temporaryInvalidCChars: [CChar]

    /// Serial queue for thread-safe access.
    private let queue = DispatchQueue(label: "com.mererun.llamacontext")

    /// Maximum generation length.
    private var nLen: Int32 = 8192
    /// Current position in generation.
    private var nCur: Int32 = 0
    /// Number of tokens decoded.
    private var nDecode: Int32 = 0
    /// Whether generation has completed.
    public private(set) var isDone: Bool = false

    private init(model: OpaquePointer, context: OpaquePointer) {
        self.model = model
        self.context = context
        self.tokensList = []
        self.temporaryInvalidCChars = []
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)
        self.vocab = llama_model_get_vocab(model)
    }

    deinit {
        if let s = sampling { llama_sampler_free(s) }
        if let m = model { llama_model_free(m) }
        if let c = context { llama_free(c) }
        llama_backend_free()
    }

    /// Create a new LlamaContext from a GGUF model file.
    public static func createContext(
        path: String,
        contextSize: UInt32 = 32768,
        temperature: Float = 1.0,
        topP: Float = 0.95,
        topK: Int32 = 40,
        seed: UInt32 = 1234
    ) async throws -> LlamaContext {
        llama_backend_init()

        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
        modelParams.n_gpu_layers = 0
        #endif

        guard let model = llama_model_load_from_file(path, modelParams) else {
            throw LlamaError.couldNotInitializeContext
        }

        let nThreads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = contextSize
        ctxParams.n_batch = contextSize  // Allow full context in single batch
        ctxParams.n_threads = Int32(nThreads)
        ctxParams.n_threads_batch = Int32(nThreads)

        guard let context = llama_init_from_model(model, ctxParams) else {
            llama_model_free(model)
            throw LlamaError.couldNotInitializeContext
        }

        let llamaContext = LlamaContext(model: model, context: context)
        llamaContext.configureSampler(temperature: temperature, topP: topP, topK: topK, seed: seed)
        return llamaContext
    }

    /// Configure the sampling chain with new parameters.
    public func configureSampler(
        temperature: Float = 1.0,
        topP: Float = 0.95,
        topK: Int32 = 40,
        seed: UInt32 = 1234
    ) {
        queue.sync {
            if let s = sampling { llama_sampler_free(s) }
            let sparams = llama_sampler_chain_default_params()
            sampling = llama_sampler_chain_init(sparams)
            llama_sampler_chain_add(sampling, llama_sampler_init_top_k(topK))
            llama_sampler_chain_add(sampling, llama_sampler_init_top_p(topP, 1))
            llama_sampler_chain_add(sampling, llama_sampler_init_temp(temperature))
            llama_sampler_chain_add(sampling, llama_sampler_init_dist(seed))
        }
    }

    /// Set the maximum generation length.
    public func setMaxTokens(_ maxTokens: Int) {
        queue.sync { nLen = Int32(maxTokens) }
    }

    /// Get the number of tokens decoded in the last generation.
    public func getTokensDecoded() -> Int {
        queue.sync { Int(nDecode) }
    }

    /// Model description string.
    public func modelInfo() -> String {
        queue.sync {
            guard let m = model else { return "" }
            let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
            result.initialize(repeating: Int8(0), count: 256)
            defer { result.deallocate() }

            let nChars = llama_model_desc(m, result, 256)
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))

            var swiftString = ""
            for char in bufferPointer {
                swiftString.append(Character(UnicodeScalar(UInt8(char))))
            }
            return swiftString
        }
    }

    /// Initialize completion with a prompt.
    public func completionInit(text: String) throws {
        try queue.sync {
            guard let ctx = context, let v = vocab else {
                throw LlamaError.couldNotInitializeContext
            }

            tokensList = tokenize(text: text, addBos: true, vocab: v)
            temporaryInvalidCChars = []

            // Use llama_batch_get_one for simple single-sequence decoding
            let tokenCount = Int32(tokensList.count)
            let batch = tokensList.withUnsafeMutableBufferPointer { ptr in
                llama_batch_get_one(ptr.baseAddress, tokenCount)
            }

            if llama_decode(ctx, batch) != 0 {
                throw LlamaError.decodeFailed
            }

            nCur = Int32(tokensList.count)
            isDone = false
            nDecode = 0
        }
    }

    /// Generate the next token.
    public func completionLoop() throws -> String {
        try queue.sync {
            guard let ctx = context, let v = vocab, let s = sampling else {
                throw LlamaError.couldNotInitializeContext
            }

            let newTokenId = llama_sampler_sample(s, ctx, -1)

            if llama_vocab_is_eog(v, newTokenId) || nCur >= nLen {
                isDone = true
                let bytes = temporaryInvalidCChars + [0]
                let newTokenStr = String(decoding: bytes.map { UInt8(bitPattern: $0) }, as: UTF8.self)
                temporaryInvalidCChars.removeAll()
                return newTokenStr
            }

            let newTokenCChars = tokenToPiece(token: newTokenId, vocab: v)
            temporaryInvalidCChars.append(contentsOf: newTokenCChars)

            let newTokenStr: String
            let bytes = temporaryInvalidCChars + [0]
            let uint8Bytes = bytes.map { UInt8(bitPattern: $0) }
            if let nullIndex = uint8Bytes.firstIndex(of: 0) {
                let truncated = Array(uint8Bytes.prefix(nullIndex))
                if String(bytes: truncated, encoding: .utf8) != nil {
                    newTokenStr = String(decoding: truncated, as: UTF8.self)
                    temporaryInvalidCChars.removeAll()
                } else {
                    newTokenStr = ""
                }
            } else {
                newTokenStr = ""
            }

            // Decode single token using llama_batch_get_one
            var singleToken = [newTokenId]
            let batch = singleToken.withUnsafeMutableBufferPointer { ptr in
                llama_batch_get_one(ptr.baseAddress, 1)
            }

            nDecode += 1
            nCur += 1

            if llama_decode(ctx, batch) != 0 {
                throw LlamaError.decodeFailed
            }

            return newTokenStr
        }
    }

    /// Clear the context for a new generation.
    public func clear() {
        queue.sync {
            tokensList.removeAll()
            temporaryInvalidCChars.removeAll()
            isDone = false
            nCur = 0
            nDecode = 0
            if let ctx = context {
                llama_memory_clear(llama_get_memory(ctx), true)
            }
        }
    }

    // MARK: - Private

    private func tokenize(text: String, addBos: Bool, vocab v: OpaquePointer) -> [llama_token] {
        let utf8Count = text.utf8.count
        let nTokens = utf8Count + (addBos ? 1 : 0) + 1
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: nTokens)
        let tokenCount = llama_tokenize(v, text, Int32(utf8Count), tokens, Int32(nTokens), addBos, false)

        var swiftTokens: [llama_token] = []
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }

        tokens.deallocate()
        return swiftTokens
    }

    private func tokenToPiece(token: llama_token, vocab v: OpaquePointer) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer { result.deallocate() }

        let nTokens = llama_token_to_piece(v, token, result, 8, 0, false)

        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer { newResult.deallocate() }

            let nNewTokens = llama_token_to_piece(v, token, newResult, -nTokens, 0, false)
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
}
#endif
