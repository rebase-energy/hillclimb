"""Bundled climber libraries: packages that bring a whole climber (a loop,
its operator, its scoring view), not one module."""

from hillclimb.modules import refs

refs.register("loop", "gepa", "hillclimb.climbers.gepa.loop:GepaLoop")  # optional extra
