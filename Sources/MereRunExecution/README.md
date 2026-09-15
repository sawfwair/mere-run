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
records atomically with private file permissions. On Apple volumes that report
support for file protection, it applies `completeUnlessOpen` to a new private
temporary file before writing sensitive bytes. On every filesystem, it writes
and synchronizes through the open descriptor, then renames the file over the
previous record and synchronizes the directory. Volumes without file protection
retain private permissions (`0600`); they do not provide protection based on the
device's lock state. Errors when applying supported protection propagate to the
caller. Replacement does not open the previous protected record. `writeArtifact`
returns the committed bytes' hash and size without reopening the new file.

`readData` reports permission-denied reads with an unlock-or-permissions
instruction. A failed read never establishes interruption or authorizes a
record rewrite. Callers validate the record version before recovery; an unknown
version must remain untouched. Reading closed protected records and assets can
still require unlocking the device. Storage unit tests cover atomic replacement,
permissions, protection metadata, hashes, and failure cleanup; they do not
simulate device lock transitions.

To exercise storage on a mounted test volume, set `MERERUN_TEST_STORAGE_ROOT`
when running `swift test --filter RunStorageTests`. The tests create and remove
their own directories under that path. Use a volume that supports file
protection and one that does not to cover both storage capabilities.
