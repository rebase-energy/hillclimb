"""Machine-checked output format (hillclimb spaces). Format only — the
ratio, the mass constraint and the not-all-zero rule stay with the verifier.
"""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={
        "id": spaces.Int(values=range(400), unique=True),
        "value": spaces.Float(),
    },
    n_rows=400,
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
