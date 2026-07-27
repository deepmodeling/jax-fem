# T013/T020 Exact Release Red-Green Evidence

**Date**: 2026-07-24
**Scope**: T013 failing-test gate and T020 explicit release cell/anchor set
**Claim boundary**: static P0 implementation evidence, not a completed release
validation run and not anchor-sensitivity evidence

## Frozen formal input

The formal 29,568-cell C3D8 workflow uses
`cases/kaess_2023/inputs/release-cellset.json`:

- mesh SHA-256:
  `d2283a3d3f9f133d80bb630ac8f4068f15d266a538aaceb14f8d27f9f9c08cb8`;
- release artifact SHA-256:
  `3eed36636b680e8ba865de0920a153c81abd42a21dafcc71ac9d9596d4d23437`;
- generator SHA-256:
  `073bd378be20482bb6183b224bb4fbeef606bb7d6edcf121eb46434cd150b196`;
- 1,920 W1/W2 cells removed with exactly zero mechanics contribution;
- 1,920 W3 root cells retained;
- removed and retained-root sets are disjoint and exactly partition all
  non-powder support-band cells;
- 189 W3 root-bottom nodes, canonical ID-list SHA-256
  `b2cc148b3d226c3e8e70a3fbe7cc95f90dc8b3fdae83bbe2785405e37b18ebb9`;
- 192 physical release constraints: all 189 root-bottom `uz` DOFs plus
  exactly three in-plane rigid-body DOFs.

The registered primary uses nodes `[231, 239]`: node 231 constrains `ux,uy`
and node 239 constrains `uy`. The artifact also freezes the other three
corner variants. Every variant has rigid-body constraint rank 6.

## RED evidence

Commit `1f684ea` introduced the original T013 release tests before an exact
cell-set loader or validator existed.

The additional minimal-anchor tests were then run before implementation:

```text
5 failed
```

Four failures reported a missing
`make_root_minimal_release_mechanics_bc`; the fifth showed that
`paper_minimal_root` was not a valid CLI mode. Further test-first slices
exposed:

- two failures because exact release had no fail-closed configuration
  validator;
- one failure because build-stage anchor selection could not be restricted
  to the retained W3 root;
- one failure because release VTU output could not carry auditable anchor
  point fields;
- one failure because the content-addressed artifact did not yet contain an
  anchor protocol.

These were contract/physics failures, not path or import failures.

## Important correction discovered during review

The first exact-cell implementation retained the legacy box anchor: all 189
W3 bottom nodes were fixed in all three directions, for 567 physical DOFs.
This overconstrained in-plane contraction by 375 DOFs.

Replacing it only at release time with 192 zero-valued DOFs would still have
been wrong: the new W3 in-plane anchors could reset their cooling-complete
displacements and inject an artificial constraint response. The corrected
formal path therefore:

1. selects the in-plane anchors from the hashed W3 root at build startup;
2. preserves those exact in-plane DOF pairs through build, cooling and
   release;
3. removes only bottom-normal restraints belonging to deleted material;
4. verifies that every physical release DOF is a subset of the build-stage
   physical DOFs.

No new zero-displacement anchor is introduced by the cut.

## Fail-closed implementation

The implementation rejects:

- empty, duplicate or out-of-range cell IDs and malformed packed masks;
- mesh SHA/count mismatch and non-zero packed-mask padding;
- overlap between removed and retained-root cells;
- cuts through protected part cells or outside removable support;
- retained roots outside support;
- corner-only, disconnected or non-manifold load-bearing topology;
- anchors outside the retained root;
- duplicate/invalid actual FE DOF pairs and rigid-body rank below 6;
- exact cell sets paired with legacy rigid-body or fully clamped box anchors;
- exact cell sets paired with a non-paper-minimal build boundary;
- any runtime root node, coordinate, DOF or corner variant that differs from
  the hashed anchor protocol.

For each corner variant, removing any one of the three in-plane constraints
drops the rigid-body rank and is rejected.

The formal launcher uses `paper_minimal_root` plus the exact artifact.
Non-reference meshes remain explicitly labelled diagnostic-only and use the
legacy geometric cut/box anchor.

## Visual and machine-readable evidence

`release.vtu` contains:

- cell field `release_removed`;
- point field `release_bottom_uz`;
- point fields `release_anchor_ux` and `release_anchor_uy`.

`used_config.json` records the artifact SHA, root-bottom IDs, resolved anchor
nodes and coordinates, all 192 physical DOF pairs, rigid-body rank and the
build-to-release continuity declaration. Strict-active-domain constraints
used only to eliminate inactive zero rows remain separate from these 192
physical constraints.

## GREEN verification

The focused release/BC/provenance/audit/source-manifest regression completed:

```text
69 passed, 1 skipped
```

The generator was executed again in the WSL `jax-fem-env`; byte comparison
against the frozen artifact succeeded and reproduced:

```text
3eed36636b680e8ba865de0920a153c81abd42a21dafcc71ac9d9596d4d23437
```

`bash -n cases/kaess_2023/run_kaess_phase2.sh` also passed.

## Remaining claim boundary

T020 closes the static explicit-input and rejection implementation. It does
not by itself close FR-017, PAR011 or PAR027. Those still require:

- an approved anchor-sensitivity threshold;
- four anchor variants using the same thermal history but each preserving its
  own in-plane anchors continuously through build, cooling and release (a
  mechanical checkpoint with different zero anchors must not be reused);
- far-field stress and bending comparisons;
- a real zero-load/no-residual-stress release control and pre/post solve
  evidence showing no singular pivot or rigid drift;
- constraint-jump/work diagnostics.

Until those numerical artifacts exist, the anchor assumption remains
`missing/open` and no paper-level release claim is promoted.
