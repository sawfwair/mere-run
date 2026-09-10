# Durable execution storage

Use this library for run-directory leases, file fingerprints, atomic record
writes, and terminal states. It depends on Foundation, platform file APIs, and
Crypto. It does not load models or define CLI or HTTP behavior.

Operation families own their versioned records, preparation rules, outputs,
and retry policies. Image and file transcription runs use the same storage
primitives while preserving their separate schemas.

`RunDirectoryLease` uses a nonblocking process lock to distinguish active runs
from abandoned records. The descriptor closes across process execution and
when its owner releases it. Recovery acquires the same lease before changing
a nonterminal record to interrupted.

`RunArtifact` hashes file contents in bounded chunks. `RunRecordCodec` writes
records atomically with private file permissions. Callers validate the record
version before recovery; an unknown version must remain untouched.
