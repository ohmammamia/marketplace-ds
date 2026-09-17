# Decision log — Project 2

### D1 Redact before anything else
- **Evidence** 19% of kept records contain an identifier.
- **Alternatives** redact only in outputs.
- **Why** models memorise; notebooks leak; the only safe place to cut is the entrance. An assertion fails the pipeline if a phone/email pattern survives.

### D2 Topics are evidence, not the taxonomy
- **Evidence** NMF topic 5 mixes praise for tidy work with genuine safety complaints; several topics split one operational issue.
- **Why** categories must map to an owner and an action; a topic model does not know who owns anything.

### D3 NMF over LDA / embeddings
- **Evidence** median 16 tokens, ~1.3k docs; coherence peaks at k=8 (3.06) where stability is 0.58.
- **Why** readable, deterministic with a seed, no GPU. **Would change if** corpus > ~50k docs or many multi-issue messages.

### D4 Weak supervision + human review, not LLM labelling
- **Evidence** weak-only macro-F1 0.70 < rules 0.77 < weak+reviewed 0.87.
- **Alternatives** zero-shot LLM labels.
- **Why** reviewed labels are cheap (250 items), auditable and improve the model measurably; LLM labels would need the same review to be trusted and add a dependency.

### D5 Abstention threshold 0.5
- **Evidence** 10% of records fall below and go to the review queue.
- **Why** ops preferred fewer, more reliable auto-labels; threshold is config, reviewed quarterly against queue size.

### D6 Volume-normalised emergence with a Poisson test
- **Alternatives** raw count growth; CUSUM.
- **Why** total volume rises with marketing; a ratio against expected share isolates *relative* growth. CUSUM would be the next step for daily data.

### D7 Multiplicative priority with a written severity rubric
- **Alternatives** learnt weights; frequency only.
- **Why** no ground truth for priority; severity is a business judgement made explicit and versioned.
