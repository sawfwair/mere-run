import Foundation

extension MereRunCapabilityCatalog {
    public static let openWebUIQuickstart = MereRunCommandCapability(
        id: "open-webui.quickstart",
        command: ["open-webui", "quickstart"],
        title: "Open WebUI quickstart",
        summary: "Launch and configure Open WebUI against the local API and model suite.",
        options: [
            .init(flag: "--host", label: "API host", kind: .string),
            .init(flag: "--port", label: "API port", kind: .integer),
            .init(
                flag: "--engine",
                label: "Engine",
                kind: .choice,
                choices: [
                    "text-chat-q36", "text-code", "text-chat-klein", "text-chat-gemma4",
                    "text-chat-laguna", "text-chat-lfm2", "text-chat-deepseek-v4-flash"
                ]
            ),
            .init(flag: "--webui-host", label: "WebUI host", kind: .string),
            .init(flag: "--webui-port", label: "WebUI port", kind: .integer),
            .init(flag: "--container-name", label: "Container", kind: .string),
            .init(flag: "--volume-name", label: "Volume", kind: .string),
            .init(flag: "--image", label: "Docker image", kind: .string),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--text-model", label: "Text model", kind: .string),
            .init(flag: "--vision-model", label: "Vision model", kind: .string),
            .init(flag: "--embedding-model", label: "Embedding model", kind: .string),
            .init(flag: "--image-model", label: "Image model", kind: .string),
            .init(flag: "--tts-model", label: "TTS model", kind: .string),
            .init(flag: "--stt-model", label: "STT model", kind: .string),
            .init(flag: "--tts-format", label: "TTS format", kind: .string),
            .init(flag: "--admin-email", label: "Admin email", kind: .string),
            .init(flag: "--admin-password", label: "Admin password", kind: .string),
            .init(flag: "--wait-seconds", label: "Health wait", kind: .integer),
            .init(flag: "--pull", label: "Pull models", kind: .boolean),
            .init(flag: "--accept-model-license", label: "Accept model terms", kind: .boolean),
            .init(flag: "--skip-server", label: "Skip API server", kind: .boolean),
            .init(flag: "--skip-docker", label: "Skip Docker", kind: .boolean),
            .init(flag: "--skip-configure", label: "Skip configure", kind: .boolean),
            .init(flag: "--reset", label: "Reset", kind: .boolean),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .service)
    )

    public static let apiServe = MereRunCommandCapability(
        id: "api.serve",
        command: ["api", "serve"],
        title: "API server",
        summary: "Serve installed models through OpenAI-compatible local APIs.",
        options: [
            .init(flag: "--image-run-records", label: "Image run records", kind: .directory, group: Group.output, tier: .expert),
            .init(flag: "--transcription-run-records", label: "Transcription run records", kind: .directory, group: Group.output, tier: .expert),
            .init(
                flag: "--warmup", label: "Warm model before serving", kind: .boolean, group: Group.run, tier: .expert
            ),
            .init(
                flag: "--no-warmup", label: "Skip model warmup", kind: .boolean, group: Group.run, tier: .expert
            ),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(
                flag: "--engine",
                label: "Engine",
                kind: .choice,
                choices: [
                    "text-chat-q36", "text-code", "text-chat-klein", "text-chat-gemma4",
                    "text-chat-laguna", "text-chat-lfm2", "text-chat-deepseek-v4-flash"
                ]
            ),
            .init(flag: "--lora", label: "Adapter", kind: .string),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--rate-limit-per-minute", label: "Rate limit", kind: .integer),
            .init(flag: "--max-active-requests", label: "Active requests", kind: .integer),
            .init(flag: "--memory-guard", label: "Memory guard", kind: .choice, choices: ["off", "safe", "balanced", "aggressive", "custom"]),
            .init(flag: "--memory-guard-custom-ceiling-gb", label: "Memory ceiling", kind: .number),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--kv-bits", label: "KV bits", kind: .number),
            .init(flag: "--kv-quant-scheme", label: "KV scheme", kind: .string),
            .init(flag: "--kv-group-size", label: "KV group", kind: .integer),
            .init(flag: "--quantized-kv-start", label: "KV start", kind: .integer),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .service)
    )
}
