# Workflow execution ownership

`WorkflowRunner` schedules workflow graph nodes, applies retry policy, and
orders run events. Use this map when you change how `graph run` and
`graph worker` own run directories, artifacts, and child processes. All of
these types live in `Sources/MereRunCLI/Support`. For the public graph and
run-directory contract, see [Portable workflows](../workflows.md).

## Ownership

| Owner | Responsibility |
| --- | --- |
| `WorkflowRunner` | Node scheduling, retry policy, and run-event ordering |
| `WorkflowRunStore` | Initialize and resume a run, persist its manifest, append synchronized events, and hold the run-directory lease |
| `WorkflowArtifactStore` | Localize inputs, verify declared outputs, manage node-cache entries, and check output digests before reuse |
| `WorkflowProcessRunner` | Register each child process and adapt its output to workflow events |
| `BoundedProcessRunner` | Nonblocking pipe drainage, deadlines, cancellation, and process-group cleanup |
| `WorkflowRunRecovery` | Mark abandoned runs as interrupted during local and worker inspection |

The run-directory lease keeps a second worker out before cancellation markers or
child registrations can be cleared. Resolved artifact paths must stay inside
their owning directories, including through symlinks.

## Resume a run

A resumed worker validates the run contract, graph fingerprint, and input
fingerprints before it resets run state. It rejects a run with an active lease
or live registered children, and a rejected resume leaves the previous
manifest, events, and cancellation marker intact. The lease stays held until
the execution driver finishes its children and terminal persistence.

Resume resets graph output references, and rerun nodes clear their prior
artifact references before preflight. Verified completed nodes stay reusable.
Reuse still requires matching provider pins, normalized arguments, model
provenance, and output digests, and existing localized input files must match
the bundle's declared digests. Cache formats and relay wire contracts are
unchanged.

## Recover an abandoned run

`WorkflowRunRecovery` takes the same lease during inspection. It marks
abandoned execution as failed with `interrupted_at`, keeping the v1 state
vocabulary. Live children and remote executor states prevent recovery. The run
store validates event sequences and saves an incomplete final write to a
separate fragment before it repairs the append position.

## Execution and runtime lifetime

The shared runner drains stdout and stderr without letting a silent pipe block
cancellation. A throwing start or output callback stops and waits for the
process group before the error propagates. The synchronous workflow caller
forwards task cancellation to parallel workers through a shared cancellation
signal, and waits for every child outcome before it records run completion.
Timeouts and cancellation also stop descendants that keep a pipe open after the
group leader exits.

Child stdout has a 16 MiB limit. Exceeding it fails the node and stops the
child; partial output never becomes a successful value. Write larger results
as declared artifacts. Stderr keeps a 16 KiB diagnostic tail and streams
diagnostics to the worker's stderr.

The workflow parent orchestrates child commands without acquiring their model
reservation. Each native CLI child keeps its own process admission; there's no
in-process inference path or separate admission policy.

## Validate a change

Tests use local shell children to cover timeouts, cancellation, output limits,
callback failures, and descendant cleanup. Run and artifact fixtures cover
exclusive ownership, rejected resumes, cache reuse, changed inputs, and symlink
confinement. They don't establish remote-worker availability or real model
execution quality.
