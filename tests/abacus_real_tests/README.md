# ABACUS Real Parser Tests

This directory contains infrastructure for testing the ABACUS parser against real calculation outputs from actual ABACUS executables.

## Directory Layout

```
abacus_real_tests/
  pp_orb/                 Pseudopotentials and orbitals (~340K total)
  configs/                Input files for 9 calculation types
  run_abacus_calculations.py   Script to run all calculations
  outputs/                Generated outputs (gitignored, per-version)
  reference_outputs/      Committed reference outputs for CI
```

## Supported Calculation Types

| Config | Type | Basis | Parser Features Tested |
|--------|------|-------|----------------------|
| `scf_pw_Si2` | SCF | PW | Energy, convergence, volume, fermi level |
| `scf_lcao_Si2` | SCF | LCAO | LCAO output format |
| `force_pw_Si2` | SCF+force | PW | TOTAL-FORCE blocks |
| `stress_pw_Si2` | SCF+stress | PW | TOTAL-STRESS, pressure |
| `relax_pw_Al` | relax | PW | Forces, ionic steps |
| `cell_relax_pw_Al` | cell-relax | PW | Forces + stress, trajectory |
| `band_pw_Al` | SCF+NSCF | PW | Eigenvalues, kpoints, bands |
| `dos_pw_Al` | SCF+NSCF | PW | DOS output |
| `md_lcao_Si8` | MD | LCAO | MD trajectory, forces |

## Quick Start

### Running the parser tests

```bash
# Uses committed reference outputs (works without ABACUS installed)
uv run python -m pytest tests/test_real_parser.py -v
```

Tests are automatically skipped if no output data is available.

### Generating outputs with an ABACUS binary

```bash
cd tests/abacus_real_tests

# Run with any ABACUS executable
python run_abacus_calculations.py /path/to/abacus

# Common examples
python run_abacus_calculations.py ~/.local/bin/abacus.lts
python run_abacus_calculations.py ~/.local/bin/abacus.dev
```

This creates `outputs/<version>/` with the parsed-relevant output files for each calculation.

### Running only specific calculations

```bash
python run_abacus_calculations.py ~/.local/bin/abacus.dev --only scf_pw_Si2 stress_pw_Si2
```

### Updating reference outputs for CI

After generating outputs, copy them to `reference_outputs/`:

```bash
cp -r outputs/<version> reference_outputs/
```

Then commit the updated `reference_outputs/` directory.

## Run Script Options

```
python run_abacus_calculations.py ABACUS_PATH [options]

positional:
  ABACUS_PATH              Path to ABACUS executable

optional:
  --output-dir DIR         Output directory (default: outputs/<version>)
  --nprocs N               Number of MPI processes (default: 1)
  --only CALC [CALC ...]   Run only specific calculations
  --work-dir DIR           Working directory (default: temp dir)
```

The script automatically:
- Sets `OMP_NUM_THREADS=1`
- Detects the ABACUS version from `abacus --version`
- Rewrites `pseudo_dir`/`orbital_dir` in INPUT files to absolute paths
- Handles multi-step calculations (band, DOS): runs SCF first, then NSCF with charge density
- Preserves only parser-relevant output files (excludes charge density, wavefunctions)
