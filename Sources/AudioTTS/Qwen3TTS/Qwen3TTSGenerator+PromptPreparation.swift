import AudioQwen3TTSModel
import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCore
import AudioCodecs

extension Qwen3TTSGenerator {
    func prepareGenerationInputs(
        text: String,
        language: String,
        speaker: String?,
        instruct: String?,
        speakerHintTokens: [Int]?,
        referencePromptTokens: [Int]?,
        tokenizer: Qwen3TTSTokenizer,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        config: Qwen3TTSModelConfig
    ) -> (inputEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {
        let talkerConfig = config.talkerConfig
        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let inputIds = MLXArray(tokenizer.encode(chatText).map { Int32($0) }).reshaped(1, -1)

        let ttsTokens = MLXArray([
            Int32(config.ttsBosTokenId),
            Int32(config.ttsEosTokenId),
            Int32(config.ttsPadTokenId)
        ]).reshaped(1, 3)
        let ttsEmbeds = talker.textProjection(talker.getTextEmbeddings()(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0..<1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1..<2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2..<3, 0...]

        var languageId: Int?
        if language.lowercased() != "auto", let map = talkerConfig.codecLanguageId {
            languageId = map[language.lowercased()]
        }

        var codecPrefill: [Int]
        if let languageId {
            codecPrefill = [
                talkerConfig.codecThinkId,
                talkerConfig.codecThinkBosId,
                languageId,
                talkerConfig.codecThinkEosId
            ]
        } else {
            codecPrefill = [
                talkerConfig.codecNoThinkId,
                talkerConfig.codecThinkBosId,
                talkerConfig.codecThinkEosId
            ]
        }

        if let speaker,
           let spkMap = talkerConfig.spkId,
           let spkIds = spkMap[speaker.lowercased()] {
            codecPrefill.append(contentsOf: spkIds)
        }
        if let speakerHintTokens, !speakerHintTokens.isEmpty {
            codecPrefill.append(contentsOf: speakerHintTokens)
        }
        if let referencePromptTokens, !referencePromptTokens.isEmpty {
            codecPrefill.append(contentsOf: referencePromptTokens)
        }

        var codecEmbed = talker.getInputEmbeddings()(MLXArray(codecPrefill.map { Int32($0) }).reshaped(1, -1))
        let codecSuffix = talker.getInputEmbeddings()(MLXArray([
            Int32(talkerConfig.codecPadId),
            Int32(talkerConfig.codecBosId)
        ]).reshaped(1, 2))
        codecEmbed = MLX.concatenated([codecEmbed, codecSuffix], axis: 1)

        var instructEmbed: MLXArray?
        if let instruct, !instruct.isEmpty {
            let instructText = "<|im_start|>user\n\(instruct)<|im_end|>\n"
            let instructIds = MLXArray(tokenizer.encode(instructText).map { Int32($0) }).reshaped(1, -1)
            instructEmbed = talker.textProjection(talker.getTextEmbeddings()(instructIds))
        }

        let roleEmbed = talker.textProjection(talker.getTextEmbeddings()(inputIds[0..., 0..<3]))
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, codecEmbed.dim(1) - 2, ttsPadEmbed.dim(2)])
        var combinedEmbed = MLX.concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedEmbed = combinedEmbed + codecEmbed[0..., 0..<(codecEmbed.dim(1) - 1), 0...]

        var inputEmbeds: MLXArray
        if let instructEmbed {
            inputEmbeds = MLX.concatenated([instructEmbed, roleEmbed, combinedEmbed], axis: 1)
        } else {
            inputEmbeds = MLX.concatenated([roleEmbed, combinedEmbed], axis: 1)
        }

        let firstTextEmbed = talker.textProjection(talker.getTextEmbeddings()(inputIds[0..., 3..<4]))
            + codecEmbed[0..., (codecEmbed.dim(1) - 1)..<codecEmbed.dim(1), 0...]
        inputEmbeds = MLX.concatenated([inputEmbeds, firstTextEmbed], axis: 1)

        let totalTokens = inputIds.dim(1)
        let trailingStart = 4
        let trailingEnd = max(trailingStart, totalTokens - 5)
        let trailingEmbed: MLXArray
        if trailingEnd > trailingStart {
            trailingEmbed = talker.textProjection(talker.getTextEmbeddings()(inputIds[0..., trailingStart..<trailingEnd]))
        } else {
            trailingEmbed = MLXArray.zeros([1, 0, talkerConfig.hiddenSize])
        }

        return (inputEmbeds, MLX.concatenated([trailingEmbed, ttsEosEmbed], axis: 1), ttsPadEmbed)
    }

    func prepareICLGenerationInputs(
        text: String,
        refText: String,
        referenceCodes: MLXArray,
        language: String,
        speakerEmbedding: MLXArray?,
        tokenizer: Qwen3TTSTokenizer,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        config: Qwen3TTSModelConfig
    ) throws -> (inputEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray, referenceCodesBQT: MLXArray) {
        let talkerConfig = config.talkerConfig

        guard referenceCodes.ndim == 3 else {
            throw Qwen3TTSError.invalidCloneReference("Reference codec tensor must be rank-3.")
        }

        let refCodesBQT: MLXArray
        if referenceCodes.dim(1) == talkerConfig.numCodeGroups {
            refCodesBQT = referenceCodes.asType(.int32)
        } else {
            refCodesBQT = referenceCodes.transposed(0, 2, 1).asType(.int32)
        }
        guard refCodesBQT.dim(2) > 0 else {
            throw Qwen3TTSError.invalidCloneReference("Reference audio produced empty codec frames.")
        }

        let refChat = "<|im_start|>assistant\n\(refText)<|im_end|>\n"
        let refIds = MLXArray(tokenizer.encode(refChat).map { Int32($0) }).reshaped(1, -1)
        let refTextEnd = max(3, refIds.dim(1) - 2)
        let refTextIds: MLXArray = refTextEnd > 3 ? refIds[0..., 3..<refTextEnd] : MLXArray.zeros([1, 0], dtype: .int32)

        let targetChat = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let targetIds = MLXArray(tokenizer.encode(targetChat).map { Int32($0) }).reshaped(1, -1)
        let textIdsEnd = max(3, targetIds.dim(1) - 5)
        let textIds: MLXArray = textIdsEnd > 3 ? targetIds[0..., 3..<textIdsEnd] : MLXArray.zeros([1, 0], dtype: .int32)

        let ttsTokens = MLXArray([
            Int32(config.ttsBosTokenId),
            Int32(config.ttsEosTokenId),
            Int32(config.ttsPadTokenId)
        ]).reshaped(1, 3)
        let ttsEmbeds = talker.textProjection(talker.getTextEmbeddings()(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0..<1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1..<2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2..<3, 0...]

        let combinedTextIds = MLX.concatenated([refTextIds, textIds], axis: 1)
        var textEmbed = talker.textProjection(talker.getTextEmbeddings()(combinedTextIds))
        textEmbed = MLX.concatenated([textEmbed, ttsEosEmbed], axis: 1)

        let firstCodebookCodes = refCodesBQT[0..., 0, 0...]
        var refCodecEmbed = talker.getInputEmbeddings()(firstCodebookCodes)
        let availableGroups = min(talkerConfig.numCodeGroups, refCodesBQT.dim(1))
        if availableGroups > 1 {
            for groupIdx in 1..<availableGroups {
                refCodecEmbed = refCodecEmbed + talker.codePredictor.codecEmbedding[groupIdx - 1](refCodesBQT[0..., groupIdx, 0...])
            }
        }

        let codecBosEmbed = talker.getInputEmbeddings()(MLXArray([Int32(talkerConfig.codecBosId)]).reshaped(1, 1))
        let codecEmbedICL = MLX.concatenated([codecBosEmbed, refCodecEmbed], axis: 1)
        let codecPadEmbed = talker.getInputEmbeddings()(MLXArray([Int32(talkerConfig.codecPadId)]).reshaped(1, 1))
        let textWithCodecPad = textEmbed + broadcast(codecPadEmbed, to: [1, textEmbed.dim(1), codecPadEmbed.dim(2)])
        let codecWithTextPad = codecEmbedICL + broadcast(ttsPadEmbed, to: [1, codecEmbedICL.dim(1), ttsPadEmbed.dim(2)])
        let iclInputEmbed = MLX.concatenated([textWithCodecPad, codecWithTextPad], axis: 1)
        let trailingTextHidden = ttsPadEmbed

        var languageId: Int?
        if language.lowercased() != "auto", let map = talkerConfig.codecLanguageId {
            languageId = map[language.lowercased()]
        }

        var codecPrefill: [Int]
        if let languageId {
            codecPrefill = [talkerConfig.codecThinkId, talkerConfig.codecThinkBosId, languageId, talkerConfig.codecThinkEosId]
        } else {
            codecPrefill = [talkerConfig.codecNoThinkId, talkerConfig.codecThinkBosId, talkerConfig.codecThinkEosId]
        }

        var codecPrefixEmbed = talker.getInputEmbeddings()(MLXArray(codecPrefill.map { Int32($0) }).reshaped(1, -1))
        let codecPrefixSuffix = talker.getInputEmbeddings()(MLXArray([
            Int32(talkerConfig.codecPadId),
            Int32(talkerConfig.codecBosId)
        ]).reshaped(1, 2))

        if let speakerEmbedding {
            let speakerEmbed: MLXArray
            if speakerEmbedding.ndim == 1 {
                speakerEmbed = speakerEmbedding.reshaped(1, 1, speakerEmbedding.dim(0))
            } else if speakerEmbedding.ndim == 2 {
                speakerEmbed = speakerEmbedding.reshaped(speakerEmbedding.dim(0), 1, speakerEmbedding.dim(1))
            } else {
                speakerEmbed = speakerEmbedding
            }

            if speakerEmbed.dim(2) == talkerConfig.hiddenSize {
                codecPrefixEmbed = MLX.concatenated([codecPrefixEmbed, speakerEmbed, codecPrefixSuffix], axis: 1)
            } else {
                codecPrefixEmbed = MLX.concatenated([codecPrefixEmbed, codecPrefixSuffix], axis: 1)
            }
        } else {
            codecPrefixEmbed = MLX.concatenated([codecPrefixEmbed, codecPrefixSuffix], axis: 1)
        }

        let roleEmbed = talker.textProjection(talker.getTextEmbeddings()(targetIds[0..., 0..<3]))
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, codecPrefixEmbed.dim(1) - 2, ttsPadEmbed.dim(2)])
        var combinedPrefix = MLX.concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedPrefix = combinedPrefix + codecPrefixEmbed[0..., 0..<(codecPrefixEmbed.dim(1) - 1), 0...]

        let inputEmbeds = MLX.concatenated([roleEmbed, combinedPrefix, iclInputEmbed], axis: 1)
        return (inputEmbeds, trailingTextHidden, ttsPadEmbed, refCodesBQT)
    }

}
