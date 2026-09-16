# Dataset

The allocation instance is built from the same cleaned tables as Project 1
(`project1_matching/src/data.py`) for one planning week:

| element | source |
|---|---|
| leads L | `leads.csv` rows with `created_at` in the week |
| providers I, capacity c_i, credits | `providers.csv` |
| feasible pairs E | Project 1 candidate generation |
| match quality q_ij | Project 1 `model_logreg.joblib` (P(purchase \| notified first)); falls back to the rule score |
| distance d_ij | zone grid |

Demo peak week 2026-01-19: 164 leads (73 priority), 220 providers, capacity 635,
5,895 feasible pairs. Synthetic only — see `common/synthetic.py`.
