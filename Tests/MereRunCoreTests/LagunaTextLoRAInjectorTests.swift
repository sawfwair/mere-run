import MLXNN
import XCTest
@testable import MereRunCore

final class LagunaTextLoRAInjectorTests: MereRunCoreTestCase {
    func testTargetMatcherAcceptsLagunaAttentionProjectionSuffixes() {
        XCTAssertTrue(LagunaTextLoRAInjector.shouldTarget(
            path: "layers.0.self_attn.q_proj",
            suffixes: ["q_proj"]
        ))
        XCTAssertTrue(LagunaTextLoRAInjector.shouldTarget(
            path: "layers.0.self_attn.o_proj",
            suffixes: ["o_proj"]
        ))
        XCTAssertFalse(LagunaTextLoRAInjector.shouldTarget(
            path: "layers.0.mlp.gate_proj",
            suffixes: ["q_proj"]
        ))
    }

    func testInjectsOnlyRequestedToyAttentionLayers() throws {
        let module = ToyLagunaAttentionModule()
        let layers = try LagunaTextLoRAInjector.inject(
            into: module,
            rank: 2,
            targetSuffixes: ["q_proj", "v_proj"]
        )

        XCTAssertEqual(Set(layers.keys), ["self_attn.q_proj", "self_attn.v_proj"])
        XCTAssertTrue(module.namedModules().contains { path, child in
            path == "self_attn.q_proj" && child is LoRALinear
        })
        XCTAssertTrue(module.namedModules().contains { path, child in
            path == "self_attn.v_proj" && child is LoRALinear
        })
    }
}

private final class ToyLagunaAttentionModule: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: ToyLagunaSelfAttention

    override init() {
        self._selfAttention.wrappedValue = ToyLagunaSelfAttention()
        super.init()
    }
}

private final class ToyLagunaSelfAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    override init() {
        self._qProj.wrappedValue = Linear(4, 4, bias: false)
        self._vProj.wrappedValue = Linear(4, 4, bias: false)
        self._gateProj.wrappedValue = Linear(4, 4, bias: false)
        super.init()
    }
}
