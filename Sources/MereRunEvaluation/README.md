# MereRunEvaluation

`MereRunEvaluation` defines the public, runtime-neutral contract for external
evaluation packs. It deliberately contains no bundled domain packs, product
prompts, personas, policies, private datasets, or domain scorers.

An owning repository keeps its pack content and optional scorer executable.
`mere.run` loads only the files declared by that pack, verifies their hashes,
and records a content-addressed plan and report. Packs are never copied into
this repository or installed into the public model catalog.

The v1 boundary is text-chat evaluation. It supports:

- named model and adapter slots;
- prompt sets kept in external files;
- matched evaluation arms;
- deterministic and sampled profiles;
- explicit data splits, including sealed frontier data;
- built-in response assertions or an opt-in external scorer process;
- typed gates and promotion eligibility.

The JSON contract is the interoperability surface. The Swift types and loader
exist so the CLI and other Swift clients enforce the same validation rules.
