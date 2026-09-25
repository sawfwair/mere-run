import Foundation

extension MereRunCapabilityCatalog {
    enum TripoSRFamily: String, MereRunFamilyID {
        case tripoSR = "triposr"
    }

    enum Trellis2Family: String, MereRunFamilyID {
        case trellis2
    }

    enum InstantMeshFamily: String, MereRunFamilyID {
        case instantMesh = "instantmesh"
    }

    enum ImageValidateFamily: String, MereRunFamilyID {
        case validationSuites = "validation-suites"
    }

    static let imageReconstruct3DRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("image-3d-triposr")],
        families: [.init(TripoSRFamily.tripoSR, title: "TripoSR", models: ["image-3d-triposr"])]
    )

    static let imageReconstruct3DTrellis2Routing = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("image-3d-trellis2-4b")],
        families: [.init(Trellis2Family.trellis2, title: "TRELLIS.2", models: ["image-3d-trellis2-4b"])]
    )

    static let imageReconstruct3DMultiviewRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("image-3d-instantmesh-base")],
        families: [.init(InstantMeshFamily.instantMesh, title: "InstantMesh", models: ["image-3d-instantmesh-base"])]
    )

    /// No `--model`: `--family` chooses which suite runs, over the same option surface.
    static let imageValidateRouting = MereRunCapabilityRouting(
        modelFlags: [],
        families: [.init(ImageValidateFamily.validationSuites, title: "Image validation suites", models: [])]
    )
}
