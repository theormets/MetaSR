"""
Runs at Docker build time to precompile Julia / SymbolicRegression.jl.
Bakes the 120-180 s JIT cost into the image so every container start skips it.
"""
import numpy as np
from pysr import PySRRegressor

print("Precompiling Julia / SymbolicRegression.jl ...")
m = PySRRegressor(
    niterations=1,
    populations=1,
    maxsize=6,
    procs=0,
    multithreading=False,
    verbosity=0,
    random_state=0,
)
rng = np.random.default_rng(0)
m.fit(rng.standard_normal((12, 2)), rng.standard_normal(12))
print("Julia precompile done.")