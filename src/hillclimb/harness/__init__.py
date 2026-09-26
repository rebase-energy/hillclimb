"""The harness: the fixed core every search runs on — one process, one journal
writer, no policy. Import from the submodules (`hillclimb.harness.core.Harness`,
`hillclimb.harness.candidate.Candidate`, ...): this package imports nothing, so
harness <-> modules stays a lazy, one-way dependency and `hillclimb.harness.x`
never loads the engine.
"""
