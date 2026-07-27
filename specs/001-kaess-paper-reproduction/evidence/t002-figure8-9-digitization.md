# T002 Figure 8/9 Digitization Evidence

**Task**: T002 / PAR003
**Date**: 2026-07-27
**Claim boundary**: public paper-to-code numerical reference data only. This
evidence is not experimental validation and does not promote the benchmark
claim level.

## Result

The published PDF vector objects now provide independently readable,
content-addressed reference data:

- Figure 8b, 150 C standard case: 41 samples on
  `cantilever_z_mm = 0.00, -0.01, ..., -0.40`;
- Figure 9a: eight maximum-front-bending values versus build-plate
  temperature;
- Figure 9b: six fixed-speed and six constant-line-energy values versus laser
  power.

Every CSV row carries its complete process condition, physical value, source
PDF coordinate, digitized-artifact evidence id, source-PDF evidence id and
hash, digitization method, unit-bearing column, reading bound and uncertainty
semantics.

## Authoritative source

| Item | Frozen value |
|---|---|
| Paper | Kaess et al., *Materials* 16(6):2321 (2023) |
| Repository PDF | `cases/kaess_2023/references/cases/kaess_2023_paper.pdf` |
| PDF SHA-256 | `8bf5e7c2f744a55b9c4e40bbe5f17a5cb6e1649d668132edc961ecb5c3d5e941` |
| Figure 8 page/object | printed page 10, PyMuPDF page index 9, drawing index 53 |
| Figure 9 page/objects | printed page 11, PyMuPDF page index 10, drawing indices 19, 44 and 51 |

No OCR or external LLM service was used. The curves and markers are native
PDF vector paths.

## Axis calibration and sampling

### Figure 8b

- PDF x: `388.855377 -> 484.732971 pt`
- physical x: `cantilever_z = 0.0 -> -0.4 mm`
- PDF y: `114.762787 -> 211.186386 pt`
- physical y: `sigma_x = 1500 -> -500 MPa`
- method:
  `pdf_vector_axis_calibrated_piecewise_linear_v1`

The original 15 vertices form 14 line segments. They were first transformed
with the published axes and then sampled on the exact 0.01 mm grid. The first
sample requires only a 0.000228 mm endpoint extension, far below the frozen
0.02 mm reading bound.

The resulting standard-case landmarks are:

- grid surface: `223.988751 MPa`;
- grid maximum: `676.141100 MPa at -0.15 mm`;
- first zero crossing: `-0.2275573767 mm`;
- minimum: `-452.435753 MPa at -0.30 mm`;
- second zero crossing: `-0.3964694106 mm`.

### Figure 9

Figure 9a calibration:

- PDF x `217.847992 -> 332.687988 pt` maps to `0 -> 1000 C`;
- PDF y `118.213028 -> 233.833038 pt` maps to `16 -> 0 um`.

Figure 9b calibration:

- PDF x `404.388000 -> 518.807983 pt` maps to `0 -> 400 W`;
- PDF y `117.673050 -> 232.933044 pt` maps to `16 -> 0 um`.

The frozen method id is `pdf_vector_axis_calibrated_v1`.

Figure 9a contains a plotted 50 C marker even though Table 3 omits 50 C. The
figure point is retained and explicitly registered as figure evidence; it is
not attributed to Table 3.

## Reading-bound semantics

The CSV uses `uncertainty_kind=absolute_symmetric_reading_bound`:

| Quantity | Bound |
|---|---:|
| Figure 8 cantilever z | 0.02 mm |
| Figure 8 sigma_x | 50 MPa |
| Figure 9 maximum bending | 0.3 um |

These are conservative plot-reading bounds. They are not standard
uncertainties, standard deviations or author-reported error bars. A later
conversion to a statistical uncertainty model requires a separately approved
distribution assumption.

## Content identities

| Artifact | SHA-256 |
|---|---|
| `fig8_sigma_x.csv` | `dff873ec9ac9177f93bf24388a83972f9f206bddb5bf4666a79a724e63ffe2e9` |
| `fig9_bending.csv` | `11782e2230de0447c1eeed73cb7f095506a8e43724f377e6a87151a91d060362` |
| `kaess_2023.json` | `0413764a4c350b059ae7d2fa6a738b933520164874e6c5551bc0849d17f5f287` |
| `source-manifest.yaml` | `82165bda8e0030a8c46fa4c4748d7968c9178cf2df40d7a5bf87bb7f0ce037e4` |

The source manifest binds the CSV hashes to the paper PDF evidence id, drawing
indices, calibration endpoints, method ids, output row counts and reading
bounds. It also freezes the Figure 8 endpoint policy as a first-segment linear
extension of at most 0.000228 mm.

## TDD evidence

RED, before replacing the anchor-only data:

```text
python -m pytest -q \
  tests/unit/test_kaess_reference_data.py \
  tests/contract/test_kaess_source_manifest.py

5 failed, 2 passed in 0.41s
```

The failures identified the old CSV schemas, missing 41/20-point data,
anchor-only metadata and missing `figure-8-digitized-curve` evidence id.

GREEN, after freezing the vector data and hash chain:

```text
python -m pytest -q \
  tests/unit/test_kaess_reference_data.py \
  tests/contract/test_kaess_source_manifest.py

8 passed in 0.34s
```

Affected cross-contract regression:

```text
python -m pytest -q tests/contract/test_kaess_contracts.py

114 passed in 30.76s
```

The tests verify exact schemas, row counts, full parameter sets, finite
values, unique keys, process conditions, line-energy identities, PDF
coordinate recalibration, both Figure 8 zero crossings, uncertainty semantics,
metadata-to-CSV consistency and content-addressed manifest bindings.

Independent review found that the first GREEN draft used one ambiguous
`source_evidence_id/source_artifact_sha256` pair for two different identities.
The final CSV schema therefore separates:

- `digitized_evidence_id`;
- `source_pdf_evidence_id`;
- `source_pdf_sha256`.

The same review found a stale hard-coded Figure 9a dictionary in
`analyze_kaess.py`. A new failing test reproduced the mismatch, and the
analyzer now reads the frozen CSV:

```text
python -m pytest -q \
  tests/unit/test_kaess_analyzer_reference.py \
  tests/unit/test_kaess_reference_data.py \
  tests/contract/test_kaess_source_manifest.py

9 passed in 0.55s
```

Final repository regression after all review corrections:

```text
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
python -m pytest -q tests

593 passed, 2 skipped, 16 subtests passed in 115.99s
```

## Scope not closed by T002

Figure 9 is a parameter-domain curve of maximum front bending versus
temperature or power. It is not the spatial `front_bending_curve` along the
cantilever. The approved parity contract currently contains a separate
domain-mismatch in `figure9_bending_curve_nrmse`; this must be corrected through
an explicit G0 reapproval rather than silently changed in T002.

T002 closes the reference-data freeze only. It does not close material
approval, anchor sensitivity, CPU/GPU qualification, formal paper comparison
or experimental validation.
