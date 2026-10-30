# Bonsai 2 Small selected-runtime quality — 2026-10-16

The selected native Swift/MLX runtime was rerun once on an Apple M3 Max with the same 48 frozen quality requests. Strict scoring passes **37/48**: 2B 11/16, 4B 13/16, and 9B 13/16. This closes the missing fresh quality run after the [scheduling and transform changes](./bonsai2-small-throughput-2026-10-16.md); it does not establish broad model quality.

The executable SHA-256 is `a1e3224fae3c4c85d54bcae618de8602c5cbfdd041105292f535ffbef25d54b3`, preserved with its runtime resources on T7. Its Bonsai runtime source is committed in `3b059874`; the subsequent main merge retains those runtime source hashes. The [machine-readable summary](./bonsai2-small-selected-quality-2026-10-16-summary.json) retains input, collector, executable, and response identities without publishing private prompt data.

## Conditions

Apple M3 Max, 36 GiB unified memory, macOS 26.6.2. All checkpoints, source metadata, images, raw results, and preserved executable resources remain on T7. All 48 request-body hashes match the preceding frozen trial. Each model has eight grounded cases, two code cases with five executable checks each, four vision cases, arithmetic, and reasoning.

The resident loopback API uses serial requests, an 8,192-token context, catalog/default thinking, temperature 1.0, top-p 0.95, top-k 20, min-p 0, no penalties, and no seed control. Every quality request allows 4,096 output tokens. No output grammar is enforced. Models use the original 2-bit weights. The collector and strict scorers are unchanged; no sampled failure is retried.

CPU-only compilation overlapped some quality requests. Saved wall times are diagnostic fields and are not comparable throughput measurements. The quality gate is based on answers and token budgets.

## Results

The following tables show strict scores, outer-fence recovery, and token-cap counts per model.

| Model | Strict cases | Text cases | Grounded | Code checks | Vision strict | Arithmetic | Reasoning |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2B | 11/16 | 10/12 | 7/8 | 9/10 | 1/4 | Pass | Pass |
| 4B | 13/16 | 10/12 | 6/8 | 10/10 | 3/4 | Pass | Pass |
| 9B | 13/16 | 11/12 | 7/8 | 10/10 | 2/4 | Pass | Pass |

| Model | Vision after one outer-fence removal | Requests reaching the 4,096-token cap |
| --- | ---: | ---: |
| 2B | 3/4 | 1 |
| 4B | 3/4 | 1 |
| 9B | 4/4 | 0 |

Fence removal is a separate diagnostic on saved answers and does not change strict scores. The earlier sampled run scored 34/48. Both are single unseeded trials, so their difference is not evidence of a quality improvement caused by the performance changes.

## Retained failures

- **2B:** the grounded arithmetic case reaches the token cap; interval-merging code incorrectly combines touching endpoints; shape counts are correct but the heading is wrong; chart and photo answers require outer-fence removal to parse as JSON.
- **4B:** email JSON extraction reaches the token cap; the grounded arithmetic answer uses a named-count formula that fails the original literal-operand checks; shape counts are correct but the heading is wrong.
- **9B:** the grounded arithmetic answer uses a named-count formula that fails the original literal-operand checks; shape and chart answers require outer-fence removal to parse as JSON.

The two named-count formulas are semantically valid: they give 17 and `inbox_count + sent_count`. The frozen scorer additionally requires literal `12` and `5`. Those cases remain failures in the primary score, and the scorer limitation is recorded separately. Remaining task, formatting, and budget failures are not silently repaired by the runtime.

## Evidence and scope

Evidence root: `/Volumes/T7/mere-run-qualification/bonsai2-small-closeout-2026-10-16`. `selected-quality/api-quality` retains every request, answer, score, timing, and model-status receipt. `selected-quality-verification.json` verifies the frozen requests and artifacts; `closeout-source-verification.json` freshly verifies every checkpoint file, pinned upstream runtime blob, and all 937 native MLX source files used for the reference backend build.

Normal machine and request admission remain enabled. Admission-only startup rejections are retained separately and contain no quality samples. The actual collection uses the preserved production executable; subsequent numerical backend probes and optimized diagnostic test builds have separate identities.

The selected runtime separately passes all 18 published-upstream parity cases and 930 teacher-forced positions with unchanged gates. Sampled task scoring, numerical parity, throughput, and repository validation are distinct results. These observations do not qualify long-context behavior, other hardware, or unrestricted task quality.
