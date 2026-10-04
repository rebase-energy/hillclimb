"""Machine-checked output format (hillclimb spaces). Format only — the
distinct-differences rule and the mark at 0 stay with the verifier.
"""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={
        "id": spaces.Int(values=range(20), unique=True),
        "mark": spaces.Int(low=0, high=999999),
    },
    n_rows=20,
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
