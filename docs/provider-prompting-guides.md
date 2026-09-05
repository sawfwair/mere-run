# Provider prompting guide tracker

This tracker is for contributors who maintain model-specific examples. Use it to
find provider sources, update the offline handbook, and record recipe validation.
For instructions on reading guides, see [Cookbooks](./cookbooks.md).

Inventory snapshot: September 4, 2026, repository commit
`b4a692ec144f2e5f3c61504e657dbbb1989c6c09`. The table accounts for all 139 managed model IDs in the [canonical catalog](./model-sources.md#canonical-managed-model-ids),
grouped into 59 rows. Each exact ID appears once. This inventory excludes aliases, separately managed
adapters, and hidden companion artifacts.

## Status and coverage

Source discovery and local recipe validation have separate statuses. Table 1
defines the source statuses and records their coverage.

<table>
  <caption>Table 1. Source status definitions and coverage</caption>
  <thead>
    <tr>
      <th scope="col">Source status</th>
      <th scope="col">Meaning</th>
      <th scope="col">Managed IDs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Guide</td>
      <td>A provider-authored prompting guide, cookbook, or prompt-preparation resource was retrieved. Applicability can still be family-level.</td>
      <td>38</td>
    </tr>
    <tr>
      <td>Reference</td>
      <td>A provider model card, repository, format reference, or examples entry point was retrieved. A dedicated guide and detailed recipe extraction remain open.</td>
      <td>57</td>
    </tr>
    <tr>
      <td>Research pending</td>
      <td>Provider guidance has not been verified in this pass. This doesn&#x27;t mean no guide exists.</td>
      <td>12</td>
    </tr>
    <tr>
      <td>No free-text task</td>
      <td>The managed entry is a component or task-specific input pipeline. Track input preparation and conditioning instead of a generative prompt recipe.</td>
      <td>32</td>
    </tr>
  </tbody>
</table>

All 59 local recipes have Drafted status and are bundled for offline reading.
To read a recipe in macOS Studio, in **Help**, select **mere.run Guide**.
Each recipe includes original examples and distinguishes retrieved provider
material from local workflow advice. The handbook has no recorded inference
validation. Source research gaps remain separate from recipe validation.

Provider documentation can describe controls that the local runtime doesn't
expose. Before applying provider advice, check the exact checkpoint, chat
format, input type, and local command. Quantization reduces weight precision
and can change results. Fine-tuning, distillation, and hosted prompt rewriting
can also affect how a recipe transfers.

The Checked column records source retrieval for Guide and Reference rows.
"Not verified" means that no provider source was verified. Source links point
to provider documentation, repositories, or model cards. Local recipe links
point to the bundled Markdown files in this repository.

## Image

Table 2 lists provider sources and local recipes for image models.

<table>
  <caption>Table 2. Image model guide coverage</caption>
  <thead>
    <tr>
      <th scope="col">Family and provider</th>
      <th scope="col">Managed model IDs</th>
      <th scope="col">Source status and links</th>
      <th scope="col">Scope and next action</th>
      <th scope="col">Local recipe</th>
      <th scope="col">Checked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FLUX.2 Klein (Black Forest Labs)</td>
      <td><code>image-klein-nano</code>, <code>image-klein-max</code>, <code>image-klein-9b</code>, <code>image-klein-base</code>, <code>image-klein-base-9b</code></td>
      <td>Guide. <a href="https://docs.bfl.ai/guides/prompting_summary">FLUX prompting guide</a></td>
      <td>Family guidance; map base versus distilled settings and local reference-image controls.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-flux2-klein.md">FLUX.2 Klein (Black Forest Labs) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>FLUX.2 dev (Black Forest Labs)</td>
      <td><code>image-flux2-dev</code></td>
      <td>Guide. <a href="https://docs.bfl.ai/guides/prompting_summary">FLUX prompting guide</a></td>
      <td>Use the dev-relevant sections; hosted pro and max controls require separate compatibility review.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-flux2-dev.md">FLUX.2 dev (Black Forest Labs) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>FLUX.1 dev (Black Forest Labs)</td>
      <td><code>image-flux1-dev</code></td>
      <td>Guide. <a href="https://docs.bfl.ai/guides/prompting_summary">FLUX prompting guide</a></td>
      <td>The guide explicitly covers FLUX.1; extract generation examples for dev.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-flux1-dev.md">FLUX.1 dev (Black Forest Labs) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Klein shared components</td>
      <td><code>image-klein-shared</code></td>
      <td>No free-text task</td>
      <td>Component bundle; track prompting with the consuming Klein model.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-klein-components.md">Klein shared components guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>Bonsai image (PrismML)</td>
      <td><code>image-bonsai-binary</code>, <code>image-bonsai-ternary</code></td>
      <td>Research pending</td>
      <td>Find the publisher&#x27;s binary and ternary guidance; don&#x27;t assume another image family&#x27;s prompt format.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-bonsai-image.md">Bonsai image (PrismML) guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>Z-Image (Tongyi-MAI)</td>
      <td><code>image-zimage-nano</code>, <code>image-zimage-max</code>, <code>image-zimage-base</code></td>
      <td>Reference. <a href="https://github.com/Tongyi-MAI/Z-Image">Z-Image repository</a></td>
      <td>Generation examples; extract separate base and Turbo recommendations.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-z-image.md">Z-Image (Tongyi-MAI) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>HiDream O1 (HiDream-ai)</td>
      <td><code>image-hidream-o1</code>, <code>image-hidream-o1-dev</code></td>
      <td>Reference. <a href="https://huggingface.co/HiDream-ai/HiDream-O1-Image">HiDream O1 model card</a></td>
      <td>Review prompt preparation and the Dev card separately before sharing one recipe.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-hidream-o1.md">HiDream O1 (HiDream-ai) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>SenseNova U1.5 (SenseNova)</td>
      <td><code>image-sensenova-u1-5-8b-mot</code></td>
      <td>Guide. <a href="https://github.com/OpenSenseNova/SenseNova-U1/blob/refs/heads/feat/u1.5/docs/u1.5_best_practices.md">SenseNova U1.5 cookbook</a></td>
      <td>Exact family cookbook; map generation, editing, and optional prompt enhancement.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-sensenova-u15.md">SenseNova U1.5 (SenseNova) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Krea 2 (Krea)</td>
      <td><code>image-krea2-raw</code>, <code>image-krea2-turbo</code></td>
      <td>Guide. <a href="https://www.krea.ai/blog/krea-2-deep-dive-walkthrough">Krea 2 prompting guide</a></td>
      <td>Hosted-product guide; verify which exploration and style-reference techniques transfer to Raw and Turbo.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-krea2.md">Krea 2 (Krea) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Qwen Image Edit 2511 (Qwen)</td>
      <td><code>image-qwen-edit-2511</code>, <code>image-qwen-edit-2511-lightning</code></td>
      <td>Reference. <a href="https://huggingface.co/Qwen/Qwen-Image-Edit-2511">Qwen Image Edit 2511 model card</a></td>
      <td>Exact base release examples; Lightning needs its own adapter and settings review.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-qwen-image-edit.md">Qwen Image Edit 2511 (Qwen) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Ideogram 4 (Ideogram)</td>
      <td><code>image-ideogram4-sdnq-uint4</code></td>
      <td>Research pending</td>
      <td>Locate provider guidance for version 4 and distinguish it from the WaveCut quantization card; attempted guide URL could not be retrieved.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-ideogram4.md">Ideogram 4 (Ideogram) guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
  </tbody>
</table>

## Text and multimodal

Table 3 lists provider sources and local recipes for text and multimodal models.

<table>
  <caption>Table 3. Text and multimodal model guide coverage</caption>
  <thead>
    <tr>
      <th scope="col">Family and provider</th>
      <th scope="col">Managed model IDs</th>
      <th scope="col">Source status and links</th>
      <th scope="col">Scope and next action</th>
      <th scope="col">Local recipe</th>
      <th scope="col">Checked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MeBot and Psi</td>
      <td><code>text-chat-mebot</code>, <code>text-chat-psi-agent</code></td>
      <td>Research pending</td>
      <td>Catalog entries have no managed Hub source; identify checkpoint provenance before assigning provider guidance.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-mebot-psi.md">MeBot and Psi guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>Gemma 4 (Google)</td>
      <td><code>text-chat-gemma4</code>, <code>text-chat-gemma4-turbo</code>, <code>text-chat-gemma4-12b</code>, <code>text-chat-gemma4-12b-4bit</code>, <code>vision-chat-gemma4-12b</code>, <code>text-chat-gemma4-nano</code>, <code>text-chat-gemma4-max</code></td>
      <td>Reference. <a href="https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4">Gemma 4 prompt formatting</a></td>
      <td>Official roles, control tokens, and multimodal format; verify each checkpoint&#x27;s generation settings.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-gemma4.md">Gemma 4 (Google) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>DiffusionGemma (Google)</td>
      <td><code>text-chat-diffusiongemma-26b-optiq-4bit</code></td>
      <td>Research pending</td>
      <td>Find diffusion-specific prompting instructions; Gemma 4 formatting alone doesn&#x27;t establish compatibility.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-diffusiongemma.md">DiffusionGemma (Google) guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>Laguna 2.1 (poolside)</td>
      <td><code>text-chat-laguna-s-2-1</code>, <code>text-chat-laguna-xs-2-1</code></td>
      <td>Reference. <a href="https://huggingface.co/poolside/Laguna-S-2.1-NVFP4-mlx">Laguna S 2.1 model card</a></td>
      <td>Source located. Inspect linked base-model instructions and verify XS separately.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-laguna.md">Laguna 2.1 (poolside) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Inkling Small (Thinking Machines)</td>
      <td><code>text-chat-inkling-small</code></td>
      <td>Reference. <a href="https://huggingface.co/thinkingmachines/Inkling-Small">Inkling Small model card</a></td>
      <td>Source located. Extract exact chat, reasoning, and tool conventions.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-inkling.md">Inkling Small (Thinking Machines) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Muse Glimmer (Meta)</td>
      <td><code>vision-chat-muse-glimmer-30b</code></td>
      <td>Reference. <a href="https://huggingface.co/meta-models/Muse-Glimmer-30B">Muse Glimmer model card</a></td>
      <td>Source located. Review multimodal and conversation formatting for the managed quantization.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-muse-glimmer.md">Muse Glimmer (Meta) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Nemotron 3.5 Lightning (NVIDIA)</td>
      <td><code>text-chat-nemotron-35-lightning</code></td>
      <td>Reference. <a href="https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4">Lightning model card</a></td>
      <td>Exact upstream release; extract reasoning and sampling guidance.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-nemotron-lightning.md">Nemotron 3.5 Lightning (NVIDIA) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Nemotron 3 Nano Omni (NVIDIA)</td>
      <td><code>omni-chat-nemotron3-nano-30b-a3b-bf16</code></td>
      <td>Reference. <a href="https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16">Omni model card</a></td>
      <td>Exact upstream release; map modality-specific examples to local supported inputs.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-nemotron-omni.md">Nemotron 3 Nano Omni (NVIDIA) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Qwen 3.5, 3.6, and 3.8 (Qwen)</td>
      <td><code>text-chat-q36-nano</code>, <code>vision-chat-q38-27b</code>, <code>vision-chat-q38-27b-4bit</code>, <code>vision-chat-q38-flash-next-mixed</code>, <code>vision-chat-q38-flash-next-3bit</code>, <code>vision-chat-q38-flash-next-3bit-native-ple</code>, <code>vision-chat-q38-flash-next-4bit</code>, <code>text-agent-qwen35-9b</code>, <code>text-chat-q36-nano-gguf</code></td>
      <td>Reference. <a href="https://github.com/QwenLM/Qwen3.8">Official series repository</a>; <a href="https://github.com/QwenLM/Qwen-Cookbook">Qwen cookbook</a></td>
      <td>Series entry point; follow each exact model card. Track 3.8 Flash-Next separately from 27B when writing recipes.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-qwen-chat.md">Qwen 3.5, 3.6, and 3.8 (Qwen) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Bonsai text (PrismML)</td>
      <td><code>text-chat-bonsai-27b-1bit</code>, <code>text-chat-bonsai-27b-2bit</code></td>
      <td>Research pending</td>
      <td>Find publisher prompting and generation settings for each low-bit checkpoint.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-bonsai-text.md">Bonsai text (PrismML) guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>Ornith 1.5 (Ornith)</td>
      <td><code>text-agent-ornith-35b-mlx-4bit</code>, <code>text-agent-ornith-35b-mlx-6bit</code>, <code>text-agent-ornith-35b-mlx-8bit</code>, <code>text-agent-ornith-35b-mlx</code>, <code>vision-chat-ornith-35b</code></td>
      <td>Reference. <a href="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B">Ornith 1.5 model card</a></td>
      <td>Family source; map text, vision, tool templates, and quantized variants.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-ornith15.md">Ornith 1.5 (Ornith) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Ornith 1.0</td>
      <td><code>text-agent-ornith-9b</code>, <code>text-agent-ornith-35b</code></td>
      <td>Research pending</td>
      <td>Locate 1.0 publisher instructions for 9B and 35B; don&#x27;t inherit 1.5 guidance silently.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-ornith10.md">Ornith 1.0 guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>North Mini Code</td>
      <td><code>text-code-north-mini</code></td>
      <td>Research pending</td>
      <td>Resolve original publisher from the managed GGUF artifact and locate its prompting instructions.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-north-mini.md">North Mini Code guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>DeepSeek V4 Flash (DeepSeek)</td>
      <td><code>text-agent-deepseek-v4-flash</code></td>
      <td>Research pending</td>
      <td>Find V4-specific provider guidance and compare it with the managed GGUF chat template.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-deepseek-v4.md">DeepSeek V4 Flash (DeepSeek) guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>LFM2.5 text (Liquid AI)</td>
      <td><code>text-chat-lfm25-a1b-8bit</code>, <code>text-chat-lfm25-a1b-bf16</code>, <code>text-chat-lfm25-1.2b-bf16</code>, <code>text-chat-lfm25-1.2b-qad-4bit</code>, <code>text-chat-lfm25-2.6b-4bit</code>, <code>text-chat-lfm25-2.6b-qad-4bit</code>, <code>text-chat-lfm25-2.6b-bf16</code></td>
      <td>Guide. <a href="https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting">Prompting guide</a></td>
      <td>Review roles, input and output examples, and response prefixes. Use exact model cards for sampling settings.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-lfm25-text.md">LFM2.5 text (Liquid AI) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>LFM2.5 vision (Liquid AI)</td>
      <td><code>vision-chat-lfm25-3b-8bit</code></td>
      <td>Guide. <a href="https://docs.liquid.ai/lfm/key-concepts/vision-capabilities">Vision capabilities</a></td>
      <td>Includes VL-3B examples for image questions, grounding, layout, and tools; map to local controls.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-lfm25-vision.md">LFM2.5 vision (Liquid AI) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Qwen3 code (Qwen)</td>
      <td><code>text-code-qwen3</code></td>
      <td>Reference. <a href="https://github.com/QwenLM/Qwen3-Coder">Qwen3-Coder repository</a></td>
      <td>Family examples; confirm exact checkpoint before adopting tool formatting or sampling defaults.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-qwen3-code.md">Qwen3 code (Qwen) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Qwen3 text embeddings (Qwen)</td>
      <td><code>text-embed-qwen3-0.6b</code></td>
      <td>Reference. <a href="https://github.com/QwenLM/Qwen3-Embedding">Embedding usage</a></td>
      <td>Extract retrieval task instructions and query-versus-document formatting.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-qwen3-embedding.md">Qwen3 text embeddings (Qwen) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Qwen3 VL embeddings (Qwen)</td>
      <td><code>vision-embed-qwen3-vl-2b</code></td>
      <td>Reference. <a href="https://github.com/QwenLM/Qwen3-VL-Embedding">VL embedding usage</a></td>
      <td>Extract query instructions and multimodal input examples.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-qwen3-vl-embedding.md">Qwen3 VL embeddings (Qwen) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Privacy Filter</td>
      <td><code>text-anonymize-privacy-filter</code></td>
      <td>No free-text task</td>
      <td>Task-specific anonymization; track input and entity controls rather than a generative prompting recipe.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-privacy-filter.md">Privacy Filter guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
  </tbody>
</table>

## Speech

Table 4 lists provider sources and local recipes for speech models.

<table>
  <caption>Table 4. Speech model guide coverage</caption>
  <thead>
    <tr>
      <th scope="col">Family and provider</th>
      <th scope="col">Managed model IDs</th>
      <th scope="col">Source status and links</th>
      <th scope="col">Scope and next action</th>
      <th scope="col">Local recipe</th>
      <th scope="col">Checked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen3 TTS (Qwen)</td>
      <td><code>speech-tts-qwen3-nano</code>, <code>speech-tts-qwen3-customvoice</code></td>
      <td>Reference. <a href="https://github.com/QwenLM/Qwen3-TTS">Qwen3-TTS usage</a></td>
      <td>Separate VoiceDesign descriptions from CustomVoice delivery instructions and spoken text.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-qwen3-tts.md">Qwen3 TTS (Qwen) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Qwen3 ASR (Qwen)</td>
      <td><code>speech-asr-qwen3</code></td>
      <td>Reference. <a href="https://github.com/QwenLM/Qwen3-ASR">Qwen3-ASR usage</a></td>
      <td>Review context and language conditioning; distinguish transcription context from generative instructions.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-qwen3-asr.md">Qwen3 ASR (Qwen) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Parakeet and Sortformer (NVIDIA)</td>
      <td><code>speech-asr-parakeet</code>, <code>speech-diarization-sortformer</code></td>
      <td>No free-text task</td>
      <td>Transcription and speaker diarization; track audio requirements and task controls.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-parakeet-sortformer.md">Parakeet and Sortformer (NVIDIA) guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
  </tbody>
</table>

## Vision and reconstruction

Table 5 lists provider sources and local recipes for vision and reconstruction models.

<table>
  <caption>Table 5. Vision and reconstruction model guide coverage</caption>
  <thead>
    <tr>
      <th scope="col">Family and provider</th>
      <th scope="col">Managed model IDs</th>
      <th scope="col">Source status and links</th>
      <th scope="col">Scope and next action</th>
      <th scope="col">Local recipe</th>
      <th scope="col">Checked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Infinity Parser2 Pro (infly)</td>
      <td><code>vision-ocr-infinity-pro</code>, <code>vision-ocr-infinity-pro-int8</code></td>
      <td>Reference. <a href="https://huggingface.co/infly/Infinity-Parser2-Pro">Parser2 Pro model card</a></td>
      <td>Extract document parsing prompt and output format; verify the Int8 variant separately.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-infinity-parser.md">Infinity Parser2 Pro (infly) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>LightOnOCR 2 (LightOn)</td>
      <td><code>vision-ocr-lighton</code></td>
      <td>Reference. <a href="https://huggingface.co/lightonai/LightOnOCR-2-1B">LightOnOCR model card</a></td>
      <td>Review fixed document-recognition input and template conventions before suggesting user-written prompts.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-lighton-ocr.md">LightOnOCR 2 (LightOn) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>SAM 3.1 (Meta)</td>
      <td><code>vision-segment-sam31</code></td>
      <td>Reference. <a href="https://github.com/facebookresearch/sam3">SAM repository and notebooks</a></td>
      <td>Family examples for text and geometric prompts; verify SAM 3.1-specific differences.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-sam31.md">SAM 3.1 (Meta) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Falcon Perception (TII)</td>
      <td><code>vision-ground-falcon-perception</code></td>
      <td>Reference. <a href="https://huggingface.co/tiiuae/Falcon-Perception">Falcon Perception model card</a></td>
      <td>Extract referring-expression and multi-object query conventions.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-falcon-perception.md">Falcon Perception (TII) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>TerraMind fire and flood</td>
      <td><code>vision-flood-terramind-base</code>, <code>vision-fire-terramind-base</code></td>
      <td>No free-text task</td>
      <td>Managed task heads consume geospatial imagery; track bands and preprocessing in the runtime guide.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-terramind.md">TerraMind fire and flood guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>TESSERA and OlmoEarth embeddings</td>
      <td><code>vision-embed-tessera-v2-nano</code>, <code>vision-embed-tessera-v2-small</code>, <code>vision-embed-tessera-v2-medium</code>, <code>vision-embed-tessera-v2-large</code>, <code>vision-embed-tessera-v2-teacher</code>, <code>vision-embed-olmoearth-v12-nano</code>, <code>vision-embed-olmoearth-v12-tiny</code>, <code>vision-embed-olmoearth-v12-small</code>, <code>vision-embed-olmoearth-v12-base</code></td>
      <td>No free-text task</td>
      <td>Geospatial embedding inputs; track sensor, temporal, and preprocessing requirements.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-geo-embeddings.md">TESSERA and OlmoEarth embeddings guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>InsightFace Buffalo-L</td>
      <td><code>vision-face-buffalo-l</code></td>
      <td>No free-text task</td>
      <td>Face analysis from imagery; input preparation and thresholds replace a prompting cookbook.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-buffalo-l.md">InsightFace Buffalo-L guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>MoGe2, Video Depth Anything, and DA3</td>
      <td><code>vision-geometry-moge2-small</code>, <code>vision-depth-vda-small</code>, <code>vision-depth-vda-small-metric</code>, <code>vision-geometry-da3-small</code></td>
      <td>No free-text task</td>
      <td>Geometry and depth estimation; track image and video preparation, scale, and camera conventions.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-depth-geometry.md">MoGe2, Video Depth Anything, and DA3 guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>TripoSR, InstantMesh, and TRELLIS.2</td>
      <td><code>image-3d-triposr</code>, <code>image-3d-instantmesh-base</code>, <code>image-3d-trellis2-4b</code></td>
      <td>No free-text task</td>
      <td>Managed image-conditioned reconstruction; track source views, backgrounds, and masks.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-reconstruction3d.md">TripoSR, InstantMesh, and TRELLIS.2 guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
  </tbody>
</table>

## Music and audio

Table 6 lists provider sources and local recipes for music and audio models.

<table>
  <caption>Table 6. Music and audio model guide coverage</caption>
  <thead>
    <tr>
      <th scope="col">Family and provider</th>
      <th scope="col">Managed model IDs</th>
      <th scope="col">Source status and links</th>
      <th scope="col">Scope and next action</th>
      <th scope="col">Local recipe</th>
      <th scope="col">Checked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACE-Step 1.5 (ACE-Step)</td>
      <td><code>music-acestep</code>, <code>music-acestep-xl-base</code>, <code>music-acestep-xl-sft</code>, <code>music-acestep-xl-turbo</code>, <code>music-acestep-xl-turbo-lm4b</code>, <code>music-acestep-lm-1.7b</code>, <code>music-acestep-lm-4b</code></td>
      <td>Guide. <a href="https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md">Provider tutorial</a></td>
      <td>Separate captions, lyrics, and metadata. Check XL base, fine-tuned, and turbo variants and auxiliary language model settings.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-ace-step.md">ACE-Step 1.5 (ACE-Step) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>MiniMax Music3 (MiniMax)</td>
      <td><code>music-minimax-music3</code></td>
      <td>Guide. <a href="https://github.com/MiniMax-AI/MiniMax-Music3/tree/main/skills/music-caption-rewriter">Caption rewriter</a>; <a href="https://huggingface.co/MiniMaxAI/MiniMax-Music3">model card</a></td>
      <td>Provider prompt enhancement resource; map structured captions and separate lyrics to the local interface.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-minimax-music3.md">MiniMax Music3 (MiniMax) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Magenta RealTime 2 (Google)</td>
      <td><code>music-magenta-rt2-small</code>, <code>music-magenta-rt2-base</code></td>
      <td>Reference. <a href="https://huggingface.co/google/magenta-realtime-2">RealTime 2 model card</a></td>
      <td>Source located. Extract style conditioning and live prompt-change examples.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-magenta-rt2.md">Magenta RealTime 2 (Google) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Muscriptor</td>
      <td><code>music-muscriptor-small</code>, <code>music-muscriptor-medium</code>, <code>music-muscriptor-large</code></td>
      <td>No free-text task</td>
      <td>Music transcription; track audio preparation and output conventions.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-muscriptor.md">Muscriptor guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>RoFormer separation and cleanup</td>
      <td><code>music-separate-bs-roformer-viperx-1297</code>, <code>music-separate-bs-roformer-4stem</code>, <code>music-separate-mel-roformer-dereverb</code>, <code>music-separate-mel-roformer-denoise</code></td>
      <td>No free-text task</td>
      <td>Audio separation, denoising, and dereverberation; track stem and model selection.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-roformer.md">RoFormer separation and cleanup guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>AP-BWE and UniverSR</td>
      <td><code>audio-enhance-ap-bwe-16kto48k</code>, <code>audio-enhance-universr-audio</code></td>
      <td>No free-text task</td>
      <td>Audio enhancement; track sample rates and signal preparation.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-audio-enhance.md">AP-BWE and UniverSR guide (draft)</a></td>
      <td>Not verified</td>
    </tr>
    <tr>
      <td>Woosh generators (Sony AI)</td>
      <td><code>sfx-woosh-dflow</code>, <code>sfx-woosh-flow</code>, <code>sfx-woosh-vflow-8s</code>, <code>sfx-woosh-dvflow-8s</code></td>
      <td>Reference. <a href="https://github.com/SonyResearch/Woosh">Woosh repository</a></td>
      <td>Generation examples; separate text-only, video-conditioned, and duration-specific recipes.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-woosh.md">Woosh generators (Sony AI) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Woosh auxiliary models</td>
      <td><code>sfx-woosh-clap</code>, <code>sfx-woosh-synchformer</code></td>
      <td>Reference. <a href="https://github.com/SonyResearch/Woosh">Woosh repository</a></td>
      <td>Scoring and conditioning components. associate text guidance with the consuming generator or CLAP query.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-woosh-components.md">Woosh auxiliary models guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>MMAudio</td>
      <td><code>sfx-mmaudio-large-44k-v2</code></td>
      <td>Reference. <a href="https://github.com/hkchengrex/MMAudio">MMAudio repository</a></td>
      <td>Extract sound description and negative-prompt examples; verify text-only versus video conditioning.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-mmaudio.md">MMAudio guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
  </tbody>
</table>

## Video

Table 7 lists provider sources and local recipes for video models.

<table>
  <caption>Table 7. Video model guide coverage</caption>
  <thead>
    <tr>
      <th scope="col">Family and provider</th>
      <th scope="col">Managed model IDs</th>
      <th scope="col">Source status and links</th>
      <th scope="col">Scope and next action</th>
      <th scope="col">Local recipe</th>
      <th scope="col">Checked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LTX 2, 2.3, and 2.5 (Lightricks)</td>
      <td><code>video-ltx-av</code>, <code>video-ltx23-av-mlx</code>, <code>video-ltx23-full-mlx</code>, <code>video-ltx23-a2vid-mlx</code>, <code>video-ltx25-distilled-bf16</code>, <code>video-ltx25-full-bf16</code></td>
      <td>Guide. <a href="https://docs.ltx.io/api-documentation/implementation-guides/prompting-guide">LTX prompting guide</a></td>
      <td>Shot, action, camera, and audio guidance; record version and hosted-only features before local adaptation.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-ltx.md">LTX 2, 2.3, and 2.5 (Lightricks) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Wan 2.2 (Wan)</td>
      <td><code>video-wan22-ti2v-5b-mlx</code></td>
      <td>Reference. <a href="https://github.com/Wan-Video/Wan2.2">Wan2.2 repository</a></td>
      <td>TI2V-5B examples and prompt expansion; retain exact 5B controls.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-wan22.md">Wan 2.2 (Wan) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>MiniMax H3 FL2VA and FastH3 (MiniMax)</td>
      <td><code>video-minimax-h3-fl2va-mlx</code>, <code>video-minimax-h3-fl2va-bf16-mlx</code>, <code>video-minimax-h3-fl2va-8bit-mlx</code>, <code>video-minimax-h3-fasth3-vsa-datafree-mlx</code></td>
      <td>Guide. <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md">Base video prompt guide</a></td>
      <td>Exact H3 guide covers first-frame and last-frame alignment and audiovisual fields; review FastH3 settings separately.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-minimax-h3.md">MiniMax H3 FL2VA and FastH3 (MiniMax) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>MiniMax H3 Ref2VA (MiniMax)</td>
      <td><code>video-minimax-h3-ref2va-mlx</code></td>
      <td>Guide. <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md">Reference video prompt guide</a></td>
      <td>Use the reference-specific guide; validate reference ordering and identity descriptions locally.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-minimax-h3-ref.md">MiniMax H3 Ref2VA (MiniMax) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>Cosmos3 Edge (NVIDIA)</td>
      <td><code>video-cosmos3-edge-mlx</code></td>
      <td>Guide. <a href="https://github.com/nvidia/cosmos-framework/blob/main/docs/prompt_upsampling.md">Cosmos3 prompt upsampling</a>; <a href="https://huggingface.co/nvidia/Cosmos3-Edge">Cosmos3 Edge model card</a></td>
      <td>Provider JSON prompt preparation; separate image and video generation from reasoner and action tasks.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-cosmos3.md">Cosmos3 Edge (NVIDIA) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>SCAIL-2 (Z.ai)</td>
      <td><code>video-scail2-14b-mlx</code></td>
      <td>Reference. <a href="https://huggingface.co/zai-org/SCAIL-2">SCAIL-2 model card</a></td>
      <td>Source located. Review reference image, motion, masks, and any prompt requirements.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-scail2.md">SCAIL-2 (Z.ai) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
    <tr>
      <td>DreamX World (GD-ML)</td>
      <td><code>video-dreamx-world-5b-ar-mlx</code></td>
      <td>Reference. <a href="https://huggingface.co/GD-ML/DreamX-World-5B">DreamX World model card</a></td>
      <td>Source located. Extract causal continuation and action-conditioning conventions.</td>
      <td><a href="https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Guides/handbook-dreamx.md">DreamX World (GD-ML) guide (draft)</a></td>
      <td>2026-09-04</td>
    </tr>
  </tbody>
</table>

## Maintain the tracker

When a managed model or source changes, update the tracker in this order:

1. Reconcile the exact IDs with the `ManagedModelCatalog.allSpecs` property and
   the canonical catalog.
2. If a version needs a different prompt format, split its coverage into a
   separate row.
3. From the original publisher's model card, review the linked prompting
   sources. Quantization mirrors establish artifact provenance unless their
   authors also publish independent prompting guidance.
4. Record the source status, retrieval date, version scope, and next action.
   Identify failed retrievals and incomplete research explicitly.
5. Update the local recipe with prompt structure, supported controls, sampling
   settings, examples, and known failure cases.
6. In the Local recipe column, link the recipe with Drafted status.
7. Validate the recipe with the exact managed model ID.
8. Record the checkpoint revision, command, parameters, seed when applicable,
   and reviewed output.
9. Link the validation evidence in the Local recipe column.
10. Change the recipe status to Validated.

Start recipe validation with SenseNova U1.5, the two H3 guides, the Liquid AI
text and vision guides, ACE-Step, MiniMax Music3, and Cosmos3 prompt
preparation. These sources describe model-family conventions. Before using
Krea or LTX hosted controls, verify that the corresponding local control exists.
