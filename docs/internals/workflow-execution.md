# Workflow execution ownership

`WorkflowRunner` schedules nodes, applies retry policy, and orders run events.
Three support types own the resources used by that scheduler:

- `WorkflowRunStore` initializes and resumes a run, persists its manifest, and
  appends synchronized events. A run-directory lease excludes a second worker
  before cancellation markers or child registrations can be cleared.
- `WorkflowArtifactStore` localizes inputs, verifies declared outputs, manages
  node-cache entries, and checks output digests before reuse. Resolved artifact
  paths must remain within their owning directories, including through symlinks.
- `WorkflowProcessRunner` registers each child and adapts output to workflow
  events. It uses `BoundedProcessRunner` for nonblocking pipe drainage, deadlines,
  cancellation, and process-group cleanup.

## Resume a run

A resumed worker validates the run contract, graph fingerprint, and input
fingerprints before resetting run state. It rejects an active run lease or live
registered children. Rejected resumes preserve the previous manifest, events,
and cancellation marker. The lease remains held until the execution driver
finishes its children and terminal persistence.

Each node still requires matching provider pins, normalized arguments, model
provenance, and output digests for reuse. Existing localized input files must
match the bundle's declared digests. Cache formats and relay wire contracts
remain unchanged.

## Own child processes

The shared runner drains stdout and stderr without blocking cancellation on a
silent pipe. A throwing start or output callback terminates and waits for the
process group before propagating the error. Timeouts and cancellation also
terminate descendants that retain a pipe after the group leader exits.

Workflow child stdout has a 16 MiB limit. Exceeding it fails the node and stops
the child; partial output is never returned as a successful value. Write larger
results as declared artifacts. Stderr retains its existing 16 KiB diagnostic
tail while streaming diagnostics to the worker's stderr.

The workflow parent orchestrates child commands without acquiring their model
reservation. Each native CLI child retains its existing process admission.
This extraction does not introduce an in-process inference path or another
admission policy.

## Validation boundaries

Tests use local shell children to cover timeouts, cancellation, output limits,
callback failures, and descendant cleanup. Run and artifact fixtures cover
exclusive ownership, rejected resumes, cache reuse, changed inputs, and
symlink confinement. They do not establish remote-worker availability or real
model execution quality.
