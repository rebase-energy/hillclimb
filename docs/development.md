# Development

Running the test suite and the one real agent call that checks the stream contract.

```bash
uv sync                  # editable install; the `tui` extra builds plotui and needs a Rust toolchain
uv run pytest            # test suite (fake backends, no agent calls)
uv run hillclimb smoke   # one real claude call: verifies auth + stream contract
```
