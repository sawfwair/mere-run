import MLXNN
import XCTest
@testable import MereRunCore

final class LFM2TextLoRAInjectorTests: MereRunCoreTestCase {
    func testTargetMatcherAcceptsOfficialOutputAliasOnlyUnderAttention() {
        XCTAssertTrue(LFM2TextLoRAInjector.shouldTarget(
            path: "model.layers.2.self_attn.out_proj",
            suffixes: ["o_proj"]
        ))
        XCTAssertFalse(LFM2TextLoRAInjector.shouldTarget(
            path: "model.layers.0.conv.out_proj",
            suffixes: ["out_proj"]
        ))
    }

    func testInjectsOnlyAttentionProjections() throws {
        let module = ToyLFM2Layer()
        let layers = try LFM2TextLoRAInjector.inject(
            into: module,
            rank: 2,
            targetSuffixes: ["q_proj", "o_proj"]
        )

        XCTAssertEqual(
            Set(layers.keys),
            ["self_attn.q_proj", "self_attn.out_proj"]
        )
        XCTAssertTrue(module.namedModules().contains { path, child in
            path == "self_attn.q_proj" && child is LoRALinear
        })
        XCTAssertTrue(module.namedModules().contains { path, child in
            path == "self_attn.out_proj" && child is LoRALinear
        })
        XCTAssertTrue(module.namedModules().contains { path, child in
            path == "conv.out_proj" && child is Linear
        })
    }

    func testAdapterKeyNormalizesWriterAndPEFTPrefixes() {
        XCTAssertEqual(
            LFM2TextLoRAAdapter.adapterKey(
                for: "model.layers.2.self_attn.out_proj"
            ),
            "layers.2.self_attn.out_proj"
        )
        XCTAssertEqual(
            LFM2TextLoRAAdapter.adapterKey(
                for: "model.model.layers.2.self_attn.o_proj"
            ),
            "layers.2.self_attn.out_proj"
        )
    }

    func testAssistantTargetClosesA1BThinkingPrefix() {
        XCTAssertEqual(
            LFM2TextSFTTokenizer.assistantTargetText(
                content: "Copper Finch prefers direct answers.",
                reasoningContent: nil,
                generationPromptSuffix: "<think>"
            ),
            "</think>\nCopper Finch prefers direct answers."
        )
        XCTAssertEqual(
            LFM2TextSFTTokenizer.assistantTargetText(
                content: "Copper Finch prefers direct answers.",
                reasoningContent: "Apply the persona instruction.",
                generationPromptSuffix: "<think>"
            ),
            "Apply the persona instruction.</think>\nCopper Finch prefers direct answers."
        )
    }
}

private final class ToyLFM2Layer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: ToyLFM2Attention
    @ModuleInfo(key: "conv") var convolution: ToyLFM2Convolution

    override init() {
        self._selfAttention.wrappedValue = ToyLFM2Attention()
        self._convolution.wrappedValue = ToyLFM2Convolution()
        super.init()
    }
}

private final class ToyLFM2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    override init() {
        self._qProj.wrappedValue = Linear(4, 4, bias: false)
        self._outProj.wrappedValue = Linear(4, 4, bias: false)
        super.init()
    }
}

private final class ToyLFM2Convolution: Module {
    @ModuleInfo(key: "out_proj") var outProj: Linear

    override init() {
        self._outProj.wrappedValue = Linear(4, 4, bias: false)
        super.init()
    }
}
