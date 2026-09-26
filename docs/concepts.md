# Concepts

The hierarchy the UI and the on-disk metadata share, coarse to fine.

```text
Run
└── Search
    └── Candidate
        └── Trial
            └── Replicate
```

- **Problem**: reusable definition under `problems/<id>/`. See
  [problems.md](problems.md).
- **Run**: one invocation of hillclimb. A single-problem run contains one
  search; a suite run contains one search per suite entry.
- **Search**: one search worker (engine process) exploring one problem with
  one climber (see [climbers.md](climbers.md)).
- **Candidate**: an immutable code artifact produced by an operator. Any change
  to the code — however small — is a new candidate with a new id.
- **Trial**: one parameter set of a candidate's code (`params`). A candidate
  that declares no tunable parameters has exactly one trial; a tuned
  candidate has several, and its score is the best trial's.
- **Replicate**: one seeded execution of a trial. A trial's score is the
  median of its replicates, so seed variance is measured, never climbed.

`hillclimb watch` opens on the Runs screen. Metadata carries
`schema_version: 3` (v2 folders keep loading); directories from the pre-v2
flat layout are ignored. How a run is laid out on disk is in
[hillclimb-dir.md](hillclimb-dir.md).
