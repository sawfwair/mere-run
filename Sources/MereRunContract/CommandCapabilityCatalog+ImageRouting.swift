import Foundation

extension MereRunCapabilityCatalog {
    /// `image generate` runtimes. The CLI reads the family and engine from the model's
    /// `mererun_model.json`; FLUX.2-dev shares the Klein engine but ignores negative prompts, and
    /// Qwen-Image-Edit Lightning is the edit engine with fixed steps and guidance.
    enum ImageGenerateFamily: String, MereRunFamilyID {
        case flux1
        case klein
        case flux2Dev = "flux2-dev"
        case zimage
        case hidream
        case sensenova
        case krea
        case qwen21 = "qwen-21"
        case qwenEdit = "qwen-edit"
        case qwenEditLightning = "qwen-edit-lightning"
        case ideogram
    }

    /// `image train-lora` trainers: Krea 2 trains the Raw base, and every FLUX.2 Klein model goes
    /// through the Klein trainer. Only the Klein base models are listed; the distilled ones are
    /// identified from their manifest and get the same surface.
    enum ImageTrainLoRAFamily: String, MereRunFamilyID {
        case krea
        case klein
    }

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

    static let imageGenerateRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("image-zimage-nano")],
        families: [
            .init(ImageGenerateFamily.flux1, title: "FLUX.1-dev", models: ["image-flux1-dev"]),
            .init(
                ImageGenerateFamily.klein,
                title: "FLUX.2 Klein",
                models: [
                    "image-klein-nano", "image-klein-max", "image-klein-9b", "image-klein-base", "image-klein-base-9b",
                    "image-bonsai-binary", "image-bonsai-ternary"
                ]
            ),
            .init(ImageGenerateFamily.flux2Dev, title: "FLUX.2-dev", models: ["image-flux2-dev"]),
            .init(
                ImageGenerateFamily.zimage,
                title: "Z-Image",
                models: ["image-zimage-nano", "image-zimage-max", "image-zimage-base"]
            ),
            .init(ImageGenerateFamily.hidream, title: "HiDream-O1", models: ["image-hidream-o1", "image-hidream-o1-dev"]),
            .init(ImageGenerateFamily.sensenova, title: "SenseNova U1.5", models: ["image-sensenova-u1-5-8b-mot"]),
            .init(ImageGenerateFamily.krea, title: "Krea 2", models: ["image-krea2-turbo", "image-krea2-raw"]),
            .init(ImageGenerateFamily.qwen21, title: "Qwen-Image 2.1", models: ["image-qwen-21"]),
            .init(ImageGenerateFamily.qwenEdit, title: "Qwen-Image-Edit", models: ["image-qwen-edit-2511"]),
            .init(
                ImageGenerateFamily.qwenEditLightning,
                title: "Qwen-Image-Edit Lightning",
                models: ["image-qwen-edit-2511-lightning"]
            ),
            .init(ImageGenerateFamily.ideogram, title: "Ideogram 4", models: ["image-ideogram4-sdnq-uint4"])
        ],
        excludedModels: .models(
            ["image-klein-shared"],
            reason: "It holds the components the Klein models share; pick a Klein model such as `image-klein-nano`."
        )
    )

    static let imageTrainLoRARouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [
            .init(whenAny: [.init(flag: "--recipe", values: ["klein-fast-style"])], models: ["image-klein-base-9b"]),
            .always("image-krea2-raw")
        ],
        families: [
            .init(ImageTrainLoRAFamily.krea, title: "Krea 2", models: ["image-krea2-raw"]),
            .init(ImageTrainLoRAFamily.klein, title: "FLUX.2 Klein", models: ["image-klein-base-9b", "image-klein-base"])
        ],
        excludedModels: .models(
            ["image-krea2-turbo"],
            reason: "Krea 2 Turbo is distilled and can't be trained; train on `image-krea2-raw` and load the LoRA on Turbo."
        ).and(
            ["image-zimage-base"],
            reason: "mere.run trains Krea 2 and FLUX.2 Klein LoRAs only; use `image-krea2-raw` or `image-klein-base-9b`."
        )
    )

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
