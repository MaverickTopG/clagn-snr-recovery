# D-095 Methods/reproducibility disclosure — D-094 SDSS mask-contract incident

The initial D-094 production invocation used an SDSS reader that combined positive-IVAR
validity with SDSS survey bitmask rejection. This was inconsistent with the prospectively
frozen D-089/D-093 fitting contract, in which positive inverse variance alone determines
SDSS pixel validity and survey bitmasks remain QC/provenance metadata. For the historical
bright endpoint of J233602 this extra rejection removed every usable pixel from the
rest-frame 4700--5100 Angstrom degradation window.

The defect was detected by the production-QC gate, before any D-094 scientific aggregate
was accepted or interpreted. The rejected invocation had 50 failed inputs, all the
J233602 matched-bright S/N=5 realizations. Consequently, `FULL_Q1_EXECUTION_COMPLETE` was
not recorded.

The repair population was selected from frozen instrument provenance alone, without
consulting a classifier label or recovery statistic. Every task whose
`instrument_survey` contained SDSS/SDSS-V was invalidated: exactly 6,305 of the 8,412
tasks. The exact complementary 2,107 unaffected tasks were retained: 2,106 LAMOST DR11
and one DESI EDR input. Every affected task was rerun; no scientifically convenient
subset was selected.

The matrix, task identities, seed base, seed namespace, realization indices and target
rungs were unchanged. D-095 independently recomputed all 8,350 degraded-task seeds from
the SHA-256 construction and found exact identity for every task, with no seed collision.
It also reconfirmed identical faint task IDs, seeds and spectrum-array checksums across
the faint-only and matched arms. The frozen engineering audit records
`seeds_changed=false` and `matrix_changed=false`.

After the complete provenance-selected rerun, all 8,412 fit inputs completed, all 25,050
classifier rows were terminal, every production-QC invariant passed, and the corrected
raw products were checksummed before interpretation. The incident therefore represents
a detected and fully enumerated implementation-contract defect, not an
outcome-conditioned rerun or a post-hoc scientific exclusion.

Machine-readable evidence is in `mask_contract_incident_audit_d095.json` and the frozen
D-094 `engineering_repair_audit_d094.json`.
