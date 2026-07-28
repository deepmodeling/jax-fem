# Download checklist — AMB2018-01

The current network cannot reach NIST from this machine. Verified 2026-07-27:

```
WSL      https://www.nist.gov/ambench      000 (connection failed)
WSL      https://data.nist.gov             000
WSL      https://catalog.data.gov          000
Windows  https://www.nist.gov/ambench      timed out
Windows  https://data.nist.gov             connection closed while sending
Windows  https://catalog.data.gov          timed out
```

Reachable from here: TUNA mirrors (`mirrors.tuna.tsinghua.edu.cn`,
`pypi.tuna.tsinghua.edu.cn`) and `pypi.org/simple`. Not reachable:
`files.pythonhosted.org`, `conda.anaconda.org`, GitHub release assets, NIST.

So this has to be fetched from a network that can reach NIST, then moved in.

## Entry points

| Page | URL |
|---|---|
| AM-Bench home | https://www.nist.gov/ambench |
| AMB2018-01 description | https://www.nist.gov/ambench/amb2018-01-description |
| Benchmark test data | https://www.nist.gov/ambench/benchmark-test-data |
| Direct data links + citation guidance | https://www.nist.gov/ambench/direct-am-bench-data-links-and-referencing-guidance |
| Challenge descriptions | https://www.nist.gov/ambench/challenges-and-descriptions |
| Data.gov catalog entry | https://catalog.data.gov/dataset/additive-manufacturing-benchmark-test-series-am-bench-2018-test-descriptions-651ed |

## What to collect, and where it goes

### references/geometry/
- `AMB2018_01_Build` STL — build plate with the 4 bridge structures
- `AMB2018_01_Part` STL — a single bridge structure

Both are listed as resources on the Data.gov catalog entry above.

### references/material/
- IN625 precursor powder characterisation
- 15-5 stainless steel precursor characterisation
- Any temperature-dependent property data NIST publishes for either alloy

This set determines how much of conversion step 3 is real work versus
transcription. Collect everything available.

### references/measurements/
- `CHAL-AMB2018-01-PD` part deflection results (post wire-EDM separation)
- `CHAL-AMB2018-01-RS` residual elastic strain results (neutron and
  synchrotron X-ray diffraction)

### references/docs/
- AMB2018-01 test description page, saved as PDF or HTML
- Scan strategy documentation
- Challenge problem statements for PD and RS
- Optional but useful: the IMMI papers on AM-Bench 2018 and 2022 outcomes

## On arrival

Before anything is used:

1. `sha256sum` every file, record it in a source manifest.
2. Keep `references/` byte-exact. Never edit in place.
3. Note the license and the NIST citation requirement — the referencing
   guidance page states how AM-Bench data must be cited.
4. Only then derive anything into `inputs/` or `derived/`.

This mirrors the provenance discipline already used in
`cases/kaess_2023/inputs/source-manifest.yaml`.
