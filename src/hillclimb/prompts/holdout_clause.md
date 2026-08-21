
## Hidden holdout

This problem has a holdout split you never see. After the validation run above, the
orchestrator re-runs the verifier against hidden data in a directory you have no access
to, and the selected solution is chosen with that score in the blend. Solutions that fit
the validation split specifically — memorized answers, constants tuned to it, anything
keyed to the exact instances — score well here and lose. Aim for a {{metric_name}} that
generalizes.
