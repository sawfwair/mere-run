# MereRunEvaluation

`MereRunEvaluation` defines the public, runtime-neutral contract for external
evaluation packs. It deliberately contains no bundled domain packs, product
prompts, personas, policies, private datasets, or domain scorers.

An owning repository keeps its pack content and optional scorer executable.
`mere.run` loads only the files declared by that pack, verifies their hashes,
and records a content-addressed plan and report. Packs are never copied into
this repository or installed into the public model catalog.

The version 1 boundary supports text chat and image-conditioned VLM chat:

- named model and adapter slots;
- prompt sets kept in external files;
- manifest-declared local images attached to user messages;
- matched evaluation arms;
- deterministic and sampled profiles;
- explicit data splits, including sealed frontier data;
- built-in response assertions or an opt-in external scorer process;
- typed gates and promotion eligibility.

The JSON contract is the interoperability surface. The Swift types and loader
exist so the CLI and other Swift clients enforce the same validation rules.
Image references are normalized relative paths, never URLs or machine-local
absolute paths. The loader pins the image bytes into the aggregate pack
identity before the runner loads a model.

For a JSON object response, set `defaults.response_format` to `json_object` and
set `defaults.logprobs` to `none`. The setting is part of the run plan and uses
the shared structured-output capability checks.
