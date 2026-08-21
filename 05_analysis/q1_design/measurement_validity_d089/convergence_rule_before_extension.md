# D-089 stability-extension rule — frozen before realizations 20–49

The unchanged J082/J012, S/N `[5,10]`, `faint_only`/`matched`, Yang/Green pilot is
extended from the already completed common `M=20` prefix to a common `M=50`. The original
base seed and namespace `p3sf:q1:d088:stability:v1` remain unchanged. Realizations 0–19 are
not replaced, and no object receives a different M.

For every object/rung/arm/criterion group, report at prefixes 20, 40, and 50:

- `p_i = n_CL/n_classifiable`, undefined if no realization is classifiable;
- `u_i = n_unclassifiable/M`;
- classifiable count;
- mean, standard deviation, median, q10 and q90 of the finite source measurement.

`M=50` is acceptable only if all groups satisfy both final-step conditions:

1. `|p_50-p_40| <= 0.10` wherever both are defined; if either is undefined, both must be
   undefined and `u` must satisfy condition 2.
2. `|u_50-u_40| <= 0.10`.

Measurement summaries are descriptive diagnostics rather than extra hard gates because no
source paper supplies a convergence tolerance for them. Any gross numerical tail or
boundary pile-up is reported and may force a practical no-go even when the two categorical
conditions pass. The M=20-to-M=40 changes are reported as an additional stability check but
do not replace the prospectively frozen final-step rule.

If any categorical final-step condition fails, final M remains unfrozen and the outcome is a
practical no-go rather than an object-specific M or a post hoc larger pilot.
