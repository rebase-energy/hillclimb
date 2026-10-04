"""Machine-checked output format (hillclimb spaces). Format only — the
capacities and the value are the verifier's.
"""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={
        "id": spaces.Int(values=range(250), unique=True),
        "take": spaces.Int(values=(0, 1)),
    },
    n_rows=250,
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
