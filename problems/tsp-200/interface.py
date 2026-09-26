"""Machine-checked output format (hillclimb spaces). Format only — that the
tour visits every city once and its length stay with the verifier."""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={
        "position": spaces.Int(values=range(200), unique=True),
        "city": spaces.Int(values=range(200), unique=True),
    },
    n_rows=200,
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
