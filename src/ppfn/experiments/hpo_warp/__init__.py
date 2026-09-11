"""Real-benchmark diffeomorphism-complexity assessment.

See `docs/experiments/hpo-warp-complexity.md` for the design this implements:
for pairs of tasks (e.g. two OpenML datasets under the same HPOBench ML
benchmark) that share a config space, fit the smallest warp `T` (plus a
monotone y-correction `h`) such that `y_A(x) ~= h(y_B(T(x)))`, report its
severity and shape-complexity separately, and diagnose whether a
diffeomorphism explains the pair at all.

Not part of the trained registration model (`src/ppfn/model/registration/`)
or its prior (`src/ppfn/prior/registration/`) -- this is a standalone
measurement study over real data.
"""
