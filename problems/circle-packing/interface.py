"""Machine-checked output format (hillclimb spaces). Format only — the
geometry (containment, overlaps) stays with the verifier."""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={
        "id": spaces.Int(values=range(26), unique=True),
        "x": spaces.Float(low=0.0, high=1.0),
        "y": spaces.Float(low=0.0, high=1.0),
        "r": spaces.Float(low=0.0),
    },
    n_rows=26,
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
