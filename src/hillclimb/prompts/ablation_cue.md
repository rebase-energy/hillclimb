Attribute first, then change:

1. **Ablation study.** Break `solution.py` into its 3-6 major components
   (feature groups, model choice, preprocessing, validation scheme,
   post-processing). Estimate each component's contribution to the validation
   score: disable or simplify one component at a time and re-run a FAST
   evaluation (subsample rows / cut estimators so each run takes a minute or
   two — never full training). When an evaluation breakdown is shown above,
   let it focus the study: the largest error contributors point at the
   components worth measuring. Record findings in `ablation.md` — one line
   per component with its measured (or reasoned) impact — and mark the
   component you chose to target. If prior ablation findings are shown above,
   do NOT redo them; extend only what is missing.
2. **Targeted refinement.** Pick the highest-leverage component (largest
   impact or clearest weakness) and confine your ONE change to it. Name the
   component in the first line of `notes.md` (`<component>: <hypothesis>`).
