"""Machine-checked output format (hillclimb spaces). Format only — each row
is normalized to the unit sphere and the energy is computed by the verifier.
"""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={
        "id": spaces.Int(values=range(50), unique=True),
        "x": spaces.Float(),
        "y": spaces.Float(),
        "z": spaces.Float(),
    },
    n_rows=50,
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
