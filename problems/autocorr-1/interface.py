"""Machine-checked output format (hillclimb spaces). Format only — the
ratio, the mass constraint and the not-all-zero rule stay with the verifier.
"""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={
        "id": spaces.Int(values=range(600), unique=True),
        "value": spaces.Float(low=0.0),
    },
    n_rows=600,
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
