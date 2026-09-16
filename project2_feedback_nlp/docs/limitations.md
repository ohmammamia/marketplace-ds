# Limitations

1. **Synthetic text is more regular than real text.** Real messages are longer, mix
   several issues, and use slang and typos; expect lower F1 and a larger review queue.
2. **Name detection is a gazetteer + context rule.** Production needs a NER model plus the
   platform's own user-name table (registered names are the ones that appear).
3. **Single label per message.** Multi-issue messages are forced into one category;
   a multi-label setup is a straightforward extension once reviewed data supports it.
4. **Emergence needs volume.** Weekly counts of 3–10 per category make the Poisson
   test coarse; with real volume, daily resolution and CUSUM would be better.
5. **Severity is a judgement.** The rubric is explicit but still a policy; disagreement
   is expected and is the point of making it visible.
6. **No causal claim.** "Refund complaints doubled" is a signal for an owner to
   investigate, not evidence that a specific change caused it.
7. **English only.** Non-English records (~1%) are excluded, not translated.
