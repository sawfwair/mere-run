import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func ltx25UsesDistilledAncestralStage1(
    isLTX25: Bool,
    isFullTwoStage: Bool,
    usesDFR: Bool,
    usesHDRICLoRA: Bool,
    usesRetake: Bool,
    usesDubIt: Bool,
    hasReferenceVideos: Bool
) -> Bool {
    isLTX25
        && !isFullTwoStage
        && !usesDFR
        && !usesHDRICLoRA
        && !usesRetake
        && !usesDubIt
        && !hasReferenceVideos
}
