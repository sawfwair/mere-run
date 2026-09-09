import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func denoiseLTXHDRICLoRAStage2(
    initialLatent: MLXArray,
    phases: [LTXHDRICLoRAStage2Phase],
    referenceVideos: [LTXReferenceVideoConditioningInput],
    referenceLatents: [MLXArray],
    videoContext: MLXArray,
    transformer: LTXUnifiedAVTransformerV2,
    fps: Double,
    seed: Int
) throws -> MLXArray {
    precondition(referenceVideos.count == referenceLatents.count)
    var phaseLatent = initialLatent
    for (phaseIndex, phase) in phases.enumerated() {
        let sigmas = try validatedLTXSigmaSchedule(phase.sigmas)
        let tiling = phase.tiling
        let temporal = splitLTXByCount(
            numTiles: tiling.frameTiles,
            overlap: tiling.frameOverlap,
            dimensionSize: phaseLatent.dim(2)
        )
        let vertical = splitLTXByCount(
            numTiles: tiling.heightTiles,
            overlap: tiling.heightOverlap,
            dimensionSize: phaseLatent.dim(3)
        )
        let horizontal = splitLTXByCount(
            numTiles: tiling.widthTiles,
            overlap: tiling.widthOverlap,
            dimensionSize: phaseLatent.dim(4)
        )
        let output = MLX.zeros(phaseLatent.shape, dtype: phaseLatent.dtype)
        var tileIndex = 0

        for tIndex in temporal.starts.indices {
            let tStart = temporal.starts[tIndex]
            let tEnd = temporal.ends[tIndex]
            let tMask = computeTrapezoidalMask1D(
                length: tEnd - tStart,
                rampLeft: temporal.leftRamps[tIndex],
                rampRight: temporal.rightRamps[tIndex],
                leftStartsFromZero: false
            )
            for hIndex in vertical.starts.indices {
                let hStart = vertical.starts[hIndex]
                let hEnd = vertical.ends[hIndex]
                let hMask = computeTrapezoidalMask1D(
                    length: hEnd - hStart,
                    rampLeft: vertical.leftRamps[hIndex],
                    rampRight: vertical.rightRamps[hIndex],
                    leftStartsFromZero: false
                )
                for wIndex in horizontal.starts.indices {
                    let wStart = horizontal.starts[wIndex]
                    let wEnd = horizontal.ends[wIndex]
                    let wMask = computeTrapezoidalMask1D(
                        length: wEnd - wStart,
                        rampLeft: horizontal.leftRamps[wIndex],
                        rampRight: horizontal.rightRamps[wIndex],
                        leftStartsFromZero: false
                    )
                    let tileLatent = phaseLatent[
                        0...,
                        0...,
                        tStart..<tEnd,
                        hStart..<hEnd,
                        wStart..<wEnd
                    ]
                    let positions = createPositionGrid(
                        batchSize: 1,
                        numFrames: tEnd - tStart,
                        height: hEnd - hStart,
                        width: wEnd - wStart,
                        temporalScale: 8,
                        spatialScale: 32,
                        fps: Float(fps),
                        causalFix: true
                    )
                    var state = LTX25VideoTokenState(
                        initialLatent: tileLatent,
                        positions: positions
                    )
                    if phase.usesICLoRAConditioning {
                        for (reference, latent) in zip(referenceVideos, referenceLatents) {
                            let downscale = reference.downscaleFactor
                            let refTStart = min(tStart, latent.dim(2) - 1)
                            let refTEnd = min(max(refTStart + 1, tEnd), latent.dim(2))
                            let refHStart = min(hStart / downscale, latent.dim(3) - 1)
                            let refHEnd = min(max(refHStart + 1, hEnd / downscale), latent.dim(3))
                            let refWStart = min(wStart / downscale, latent.dim(4) - 1)
                            let refWEnd = min(max(refWStart + 1, wEnd / downscale), latent.dim(4))
                            state.appendReferenceLatent(
                                latent[
                                    0...,
                                    0...,
                                    refTStart..<refTEnd,
                                    refHStart..<refHEnd,
                                    refWStart..<refWEnd
                                ],
                                downscaleFactor: downscale,
                                temporalScaleFactor: reference.temporalScaleFactor,
                                strength: reference.strength,
                                attentionStrength: reference.attentionStrength,
                                fps: fps
                            )
                        }
                    }
                    MLXRandom.seed(UInt64(bitPattern: Int64(seed &+ tileIndex)))
                    state.addNoise(scale: sigmas[0])
                    let rope = precomputeSplitRope(
                        positions: state.positions,
                        dim: 4096,
                        theta: 10_000,
                        maxPos: [20, 2048, 2048],
                        numHeads: 32
                    )
                    state = denoiseLTX25VideoTokenLoop(
                        videoState: state,
                        videoRope: rope,
                        videoContext: videoContext,
                        transformer: transformer,
                        sigmas: sigmas,
                        ancestralNoiseSeed: seed &+ tileIndex,
                        ancestralEta: 0
                    )
                    let blend = MLXArray(tMask).asType(output.dtype).reshaped(1, 1, tMask.count, 1, 1)
                        * MLXArray(hMask).asType(output.dtype).reshaped(1, 1, 1, hMask.count, 1)
                        * MLXArray(wMask).asType(output.dtype).reshaped(1, 1, 1, 1, wMask.count)
                    output[0..., 0..., tStart..<tEnd, hStart..<hEnd, wStart..<wEnd] =
                        output[0..., 0..., tStart..<tEnd, hStart..<hEnd, wStart..<wEnd]
                        + state.mainLatent() * blend
                    MLX.eval(output)
                    Memory.clearCache()
                    tileIndex += 1
                }
            }
        }
        phaseLatent = output
        MLX.eval(phaseLatent)
        _ = phaseIndex
    }
    return phaseLatent
}
