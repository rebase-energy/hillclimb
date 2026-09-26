"""Machine-checked output format (hillclimb spaces). Format only — convex
position, the hull and the smallest triangle stay with the verifier; the
score is affine-invariant, so the coordinates carry no bounds."""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={
        "id": spaces.Int(values=range(13), unique=True),
        "x": spaces.Float(),
        "y": spaces.Float(),
    },
    n_rows=13,
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
