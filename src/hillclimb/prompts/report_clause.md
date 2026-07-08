You may ALSO write `eval_result.json` — a diagnostic breakdown of your own
validation results. The orchestrator feeds it back to future improvement
attempts on this solution; it never affects scoring. Shape (include only the
sections you can compute honestly from your own validation split):

```json
{"split": "validation",
 "report": {
   "version": 1,
   "overall": {"score": <your val_score>, "n_scored": <validation rows>},
   "segment_label": "<what one segment is, e.g. store, class, month>",
   "zones": [{"zone": "<segment>", "score": <metric on that segment>, "n_scored": <rows>}, ...],
   "worst_origins": [{"asof": "<worst case id>", "zone": "<segment>", "score": <float>}, ...],
   "residual_bias": {"mean_error": <mean(pred - actual)>, "mean_abs_error": <float>, "mean_actual": <float>}
 }}
```

Order `zones` worst-first (largest error first) and keep every list short (<= 10 entries).
