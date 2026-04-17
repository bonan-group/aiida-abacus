#!/usr/bin/env python3
"""Run ABACUS calculations for parser testing.

Usage:
    python run_abacus_calculations.py /path/to/abacus [--output-dir OUTPUTS] [--nprocs N]

Examples:
    # Run with dev version
    python run_abacus_calculations.py ~/.local/bin/abacus.dev

    # Run with LTS version
    python run_abacus_calculations.py ~/.local/bin/abacus.lts

    # Run only specific calculations
    python run_abacus_calculations.py ~/.local/bin/abacus.dev --only scf_pw_Si2 stress_pw_Si2
"""

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# All calculation configs in execution order
CALCULATIONS = [
    "scf_pw_Si2",
    "scf_lcao_Si2",
    "force_pw_Si2",
    "stress_pw_Si2",
    "relax_pw_Al",
    "cell_relax_pw_Al",
    "band_pw_Al",
    "dos_pw_Al",
    "md_lcao_Si8",
]

# Multi-step calculations: list of (input_file, kpt_file) tuples
MULTI_STEP = {
    "band_pw_Al": [("INPUT_scf", "KPT_scf"), ("INPUT_nscf", "KLINES")],
    "dos_pw_Al": [("INPUT_scf", "KPT_scf"), ("INPUT_nscf", "KPT_nscf")],
}

# Files/dirs within OUT.AIIDA to keep (others like charge density are excluded)
OUT_KEEP_PATTERNS = [
    "running_*.log",
    "STRU_ION*_D",
    "STRU_ION*_D*",
    "INPUT",
    "kpoints",
    "warning.log",
    "PDOS",
    "DOS*",
    "eig*",
    "BANDS_*.dat",
    "*.dat",
]


def detect_version(abacus_path):
    """Detect ABACUS version by running it briefly."""
    try:
        result = subprocess.run(
            [str(abacus_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            check=False,
        )
        output = result.stdout + result.stderr
        for line in output.splitlines():
            if "ABACUS" in line and ("v" in line.lower() or "version" in line.lower()):
                parts = line.split()
                for part in parts:
                    if part.startswith("v") and any(c.isdigit() for c in part):
                        return part.lstrip("v").replace(".", "_")
        if output.strip():
            return "unknown"
    except Exception:
        pass
    return "unknown"


def detect_version_from_log(log_content):
    """Detect ABACUS version from running log content."""
    for line in log_content.splitlines():
        if "ABACUS" in line and ("v" in line.lower() or "version" in line.lower()):
            parts = line.split()
            for part in parts:
                if part.startswith("v") and any(c.isdigit() for c in part):
                    return part.lstrip("v").replace(".", "_")
    return None


def fix_input_paths(input_file, pp_orb_dir):
    """Rewrite pseudo_dir and orbital_dir in INPUT to absolute paths."""
    content = input_file.read_text()
    pp_abs = str(pp_orb_dir.resolve())
    content = re.sub(r"(pseudo_dir)\s+\S+", rf"\1 {pp_abs}", content)
    content = re.sub(r"(orbital_dir)\s+\S+", rf"\1 {pp_abs}", content)
    input_file.write_text(content)


def _run_abacus(cmd, work_dir, env, timeout=600):
    """Run ABACUS command in work directory."""
    subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, timeout=timeout, env=env, check=False)


def run_single_calc(abacus_path, work_dir, config_dir, calc_name, nprocs=1, pp_orb_dir=None):
    """Run a single ABACUS calculation.

    Returns True if successful.
    """
    if pp_orb_dir is None:
        pp_orb_dir = config_dir.parent.parent / "pp_orb"

    if calc_name in MULTI_STEP:
        return run_multi_step(abacus_path, work_dir, config_dir, calc_name, nprocs, pp_orb_dir)

    # Single-step calculation: copy config files
    for f in config_dir.iterdir():
        if f.name == "INPUT":
            dest = work_dir / "INPUT"
            shutil.copy2(f, dest)
            fix_input_paths(dest, pp_orb_dir)
        elif f.name == "KPT":
            shutil.copy2(f, work_dir / "KPT")
        elif f.name == "STRU":
            shutil.copy2(f, work_dir / "STRU")
        elif f.name not in {"INPUT_scf", "INPUT_nscf"}:
            shutil.copy2(f, work_dir / f.name)

    cmd = [str(abacus_path)]
    if nprocs > 1:
        cmd = ["mpirun", "-np", str(nprocs)] + cmd

    env = {**os.environ, "OMP_NUM_THREADS": "1"}

    print(f"  Running {calc_name}...", end=" ", flush=True)
    _run_abacus(cmd, work_dir, env)

    # Check if calculation completed
    out_dir = work_dir / "OUT.AIIDA"
    if not out_dir.exists():
        print("FAILED (no OUT.AIIDA)")
        return False

    log_files = list(out_dir.glob("running_*.log"))
    if not log_files:
        print("FAILED (no log file)")
        return False

    log_content = log_files[0].read_text()
    if "Total  Time  :" not in log_content:
        print("FAILED (incomplete)")
        return False

    print("OK")
    return True


def run_multi_step(abacus_path, work_dir, config_dir, calc_name, nprocs=1, pp_orb_dir=None):
    """Run a multi-step calculation (band, dos)."""
    if pp_orb_dir is None:
        pp_orb_dir = config_dir.parent.parent / "pp_orb"

    steps = MULTI_STEP[calc_name]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}

    # Copy STRU
    shutil.copy2(config_dir / "STRU", work_dir / "STRU")

    for i, (input_file, kpt_file) in enumerate(steps):
        step_name = "SCF" if i == 0 else "NSCF"

        # Setup input files
        dest_input = work_dir / "INPUT"
        shutil.copy2(config_dir / input_file, dest_input)
        fix_input_paths(dest_input, pp_orb_dir)
        if (config_dir / kpt_file).exists():
            # For kpoint_file=KLINES, copy as the filename ABACUS expects
            input_content = dest_input.read_text()
            kpt_match = re.search(r"kpoint_file\s+(\S+)", input_content)
            target_name = kpt_match.group(1) if kpt_match else "KPT"
            shutil.copy2(config_dir / kpt_file, work_dir / target_name)

        # For NSCF step, need charge density from SCF
        if i > 0:
            out_dirs = list(work_dir.glob("OUT.AIIDA*"))
            for od in out_dirs:
                for chg in od.glob("charge_density*"):
                    shutil.copy2(chg, work_dir / chg.name)

        cmd = [str(abacus_path)]
        if nprocs > 1:
            cmd = ["mpirun", "-np", str(nprocs)] + cmd

        print(f"  Running {calc_name} step {i + 1} ({step_name})...", end=" ", flush=True)
        _run_abacus(cmd, work_dir, env)

        out_dir = work_dir / "OUT.AIIDA"
        if out_dir.exists() and i == 0:
            log_files = list(out_dir.glob("running_*.log"))
            if log_files:
                log_content = log_files[0].read_text()
                if "Total  Time  :" not in log_content:
                    print("FAILED (SCF incomplete)")
                    return False

        # Check NSCF completion
        if i == len(steps) - 1:
            log_files = list(out_dir.glob("running_*.log"))
            if not log_files:
                print("FAILED (no log)")
                return False
            log_content = log_files[0].read_text()
            if "Total  Time  :" not in log_content:
                print("FAILED (NSCF incomplete)")
                return False

        print("OK")

    return True


def preserve_outputs(work_dir, output_dir, calc_name):
    """Copy relevant output files to the output directory."""
    dest = output_dir / calc_name
    dest.mkdir(parents=True, exist_ok=True)

    out_dir = work_dir / "OUT.AIIDA"
    if out_dir.exists():
        dest_out = dest / "OUT.AIIDA"
        dest_out.mkdir(parents=True, exist_ok=True)

        for item in out_dir.iterdir():
            name = item.name
            keep = any(fnmatch.fnmatch(name, pattern) for pattern in OUT_KEEP_PATTERNS)
            if keep:
                if item.is_file():
                    shutil.copy2(item, dest_out / name)
                elif item.is_dir():
                    shutil.copytree(item, dest_out / name, dirs_exist_ok=True)

    time_json = work_dir / "time.json"
    if time_json.exists():
        shutil.copy2(time_json, dest / "time.json")

    return dest


def main():
    parser = argparse.ArgumentParser(description="Run ABACUS calculations for parser testing")
    parser.add_argument("abacus_path", type=Path, help="Path to ABACUS executable")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: outputs/<version>)")
    parser.add_argument("--nprocs", type=int, default=1, help="Number of MPI processes")
    parser.add_argument("--only", nargs="+", default=None, help="Only run specific calculations")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for running calculations (default: temp dir)",
    )

    args = parser.parse_args()

    abacus_path = args.abacus_path.resolve()
    if not abacus_path.exists():
        print(f"Error: ABACUS executable not found: {abacus_path}")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    configs_dir = script_dir / "configs"
    pp_orb_dir = script_dir / "pp_orb"

    # Determine calculations to run
    calcs = args.only if args.only else CALCULATIONS
    for c in calcs:
        if c not in CALCULATIONS:
            print(f"Error: unknown calculation '{c}'")
            print(f"Available: {', '.join(CALCULATIONS)}")
            sys.exit(1)

    # Detect version
    print(f"ABACUS executable: {abacus_path}")
    version = detect_version(abacus_path)
    print(f"Detected version: {version}")

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = script_dir / "outputs" / version
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Setup work directory
    if args.work_dir:
        work_base = args.work_dir
        work_base.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_base = Path(tempfile.mkdtemp(prefix="abacus_test_"))
        cleanup = True

    print(f"\nRunning {len(calcs)} calculations...\n")

    results = {}
    for calc_name in calcs:
        config_dir = configs_dir / calc_name
        calc_work_dir = work_base / calc_name
        calc_work_dir.mkdir(parents=True, exist_ok=True)

        success = run_single_calc(abacus_path, calc_work_dir, config_dir, calc_name, args.nprocs, pp_orb_dir)

        if success:
            dest = preserve_outputs(calc_work_dir, output_dir, calc_name)
            results[calc_name] = "PASSED"

            if version == "unknown":
                for log_file in dest.rglob("running_*.log"):
                    v = detect_version_from_log(log_file.read_text())
                    if v:
                        version = v
                        break
        else:
            results[calc_name] = "FAILED"

    # Cleanup
    if cleanup:
        shutil.rmtree(work_base, ignore_errors=True)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Results for ABACUS {version}:")
    print(f"{'=' * 50}")
    passed = sum(1 for v in results.values() if v == "PASSED")
    failed = sum(1 for v in results.values() if v == "FAILED")
    for name, status in results.items():
        print(f"  {name}: {status}")
    print(f"\n{passed} passed, {failed} failed out of {len(results)}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
