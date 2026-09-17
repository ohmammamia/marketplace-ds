# Limitations

1. **Synthetic text is more regular than real text.** Real messages are longer, mix
   several issues, and use slang and typos; expect lower F1 and a larger review queue.
2. **Name detection is a gazetteer + context rule.** Production needs a NER model plus the
   platform's own user-name table (registered names are the ones that appear). A name
   that is neither in the gazetteer nor preceded by a context word still survives.
3. **Redaction recall is measured against generated PII only.** The generator emits
   identifiers in a fixed set of formats, so the reported recall is an upper bound on
   real-world performance. The pattern set covers separated and international phone
   numbers, lower-case postcodes and possessives, and the pipeline gate re-applies
   those same patterns to the redacted text — but unusual formats will still pass.
4. **Single label per message.** Multi-issue messages are forced into one category;
   a multi-label setup is a straightforward extension once reviewed data supports it.
5. **Emergence needs volume.** Weekly counts of 3–10 per category make the Poisson
   test coarse; with real volume, daily resolution and CUSUM would be better.
   The test also treats the baseline rate as known rather than estimated, and assumes
   no overdispersion. Both make p-values mildly optimistic; BH correction across
   categories mitigates the multiplicity but not the model mis-specification. A
   negative-binomial or two-sample Poisson test would address the rest.
6. **Severity is a judgement.** The rubric is explicit but still a policy; disagreement
   is expected and is the point of making it visible.
7. **No causal claim.** "Refund complaints doubled" is a signal for an owner to
   investigate, not evidence that a specific change caused it.
8. **English only.** Non-English records (~1%) are excluded, not translated.
