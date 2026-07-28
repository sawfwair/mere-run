#if canImport(llama)
import Foundation
#if os(Linux)
import Glibc
#else
import Darwin
#endif
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

    #if os(Linux)
    private static func linuxGPULayerCount(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int32 {
        if let rawValue = environment["MERERUN_LLAMA_GPU_LAYERS"] {
            let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
            if let parsed = Int32(trimmed) {
                return max(0, parsed)
            }
        }

        if environment["MERERUN_LINUX_ACCEL"]?.lowercased() == "cuda" {
            return 999
        }

        return 0
    }

    private static func loadLinuxModel(path: String, params: llama_model_params) -> OpaquePointer? {
        guard params.n_gpu_layers == 0 else {
            return llama_model_load_from_file(path, params)
        }

        var cpuParams = params
        var noDevices: [ggml_backend_dev_t?] = [nil]
        return noDevices.withUnsafeMutableBufferPointer { buffer in
            cpuParams.devices = buffer.baseAddress
            return llama_model_load_from_file(path, cpuParams)
        }
    }
    #endif

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
        minP: Float = 0,
        seed: UInt32 = 1234
    ) async throws -> LlamaContext {
        llama_backend_init()

        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
        modelParams.n_gpu_layers = 0
        #elseif os(Linux)
        modelParams.n_gpu_layers = linuxGPULayerCount()
        #endif

        #if os(Linux)
        let loadedModel = loadLinuxModel(path: path, params: modelParams)
        #else
        let loadedModel = llama_model_load_from_file(path, modelParams)
        #endif

        guard let model = loadedModel else {
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
        llamaContext.configureSampler(
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP,
            seed: seed
        )
        return llamaContext
    }

    /// Configure the sampling chain with new parameters.
    public func configureSampler(
        temperature: Float = 1.0,
        topP: Float = 0.95,
        topK: Int32 = 40,
        minP: Float = 0,
        seed: UInt32 = 1234
    ) {
        queue.sync {
            if let s = sampling { llama_sampler_free(s) }
            let sparams = llama_sampler_chain_default_params()
            sampling = llama_sampler_chain_init(sparams)
            llama_sampler_chain_add(sampling, llama_sampler_init_top_k(topK))
            llama_sampler_chain_add(sampling, llama_sampler_init_top_p(topP, 1))
            if minP > 0 {
                llama_sampler_chain_add(sampling, llama_sampler_init_min_p(minP, 1))
            }
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

    /// Format chat messages with the model's embedded GGUF chat template when
    /// llama.cpp recognizes it, falling back to the legacy Qwen-style prompt.
    public func chatPrompt(for messages: [ChatMessage]) -> String {
        queue.sync {
            if let templated = applyModelChatTemplate(messages) {
                return templated
            }
            return Self.qwenStylePrompt(messages)
        }
    }

    /// Initialize completion with a prompt.
    public func completionInit(text: String) throws {
        try queue.sync {
            guard let ctx = context, let v = vocab else {
                throw LlamaError.couldNotInitializeContext
            }

            llama_memory_clear(llama_get_memory(ctx), true)
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

            let maxGeneratedTokens = nLen
            nCur = Int32(tokensList.count)
            nLen = nCur + maxGeneratedTokens
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
                let newTokenStr = String(
                    decoding: temporaryInvalidCChars.map { UInt8(bitPattern: $0) },
                    as: UTF8.self
                )
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

            if Self.isRenderedStopToken(newTokenStr) {
                isDone = true
                return ""
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

    private static func isRenderedStopToken(_ token: String) -> Bool {
        TextGenerationStopSequences.defaultRenderedChatStops.contains(
            token.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    private func applyModelChatTemplate(_ messages: [ChatMessage]) -> String? {
        guard let model,
              let template = llama_model_chat_template(model, nil) else {
            return nil
        }
        var rolePointers: [UnsafeMutablePointer<CChar>] = []
        var contentPointers: [UnsafeMutablePointer<CChar>] = []
        rolePointers.reserveCapacity(messages.count)
        contentPointers.reserveCapacity(messages.count)
        defer {
            for pointer in rolePointers { free(pointer) }
            for pointer in contentPointers { free(pointer) }
        }

        var chatMessages: [llama_chat_message] = []
        chatMessages.reserveCapacity(messages.count)
        for message in messages {
            guard let role = strdup(message.role.rawValue),
                  let content = strdup(message.content) else {
                return nil
            }
            rolePointers.append(role)
            contentPointers.append(content)
            chatMessages.append(
                llama_chat_message(
                    role: UnsafePointer(role),
                    content: UnsafePointer(content)
                )
            )
        }

        let required = chatMessages.withUnsafeBufferPointer { buffer in
            llama_chat_apply_template(template, buffer.baseAddress, buffer.count, true, nil, 0)
        }
        guard required > 0 else { return nil }

        var output = [CChar](repeating: 0, count: Int(required) + 1)
        let written = output.withUnsafeMutableBufferPointer { outputBuffer in
            chatMessages.withUnsafeBufferPointer { chatBuffer in
                llama_chat_apply_template(
                    template,
                    chatBuffer.baseAddress,
                    chatBuffer.count,
                    true,
                    outputBuffer.baseAddress,
                    Int32(outputBuffer.count)
                )
            }
        }
        guard written > 0, written < output.count else { return nil }
        output[Int(written)] = 0
        return String(
            decoding: output.prefix(Int(written)).map { UInt8(bitPattern: $0) },
            as: UTF8.self
        )
    }

    private static func qwenStylePrompt(_ messages: [ChatMessage]) -> String {
        var prompt = ""

        for message in messages {
            switch message.role {
            case .system:
                prompt += "<|im_start|>system\n\(message.content)<|im_end|>\n"
            case .user:
                prompt += "<|im_start|>user\n\(message.content)<|im_end|>\n"
            case .assistant:
                prompt += "<|im_start|>assistant\n\(message.content)<|im_end|>\n"
            case .tool:
                prompt += "<|im_start|>tool\n\(message.content)<|im_end|>\n"
            }
        }

        prompt += "<|im_start|>assistant\n"
        return prompt
    }
}
#endif
