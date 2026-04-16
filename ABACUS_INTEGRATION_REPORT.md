# ABACUS Workflow Capability Scan and `aiida-abacus` Integration Guide

## Scope

This report scans the local `abacus-test` and `ABACUS-agent-tools` repositories and maps their workflow and property-calculation capabilities onto the current `aiida-abacus` plugin. The goal is to identify what should be reused conceptually, what should be ported structurally, and what should be avoided because it conflicts with AiiDA's execution and provenance model.

Repositories scanned:

- [`abacus-test`](../abacus-test)
- [`ABACUS-agent-tools`](../ABACUS-agent-tools)
- [`aiida-abacus`](.)

## Executive Summary

`abacus-test` and `ABACUS-agent-tools` cover a much wider property workflow surface than the current `aiida-abacus` plugin, but both external packages are built around mutable working directories, shell execution, and post hoc file scanning. By contrast, `aiida-abacus` already has the correct AiiDA-native foundations: a single `CalcJob`, a parser, a restart-capable base work chain, a relax workflow, and a band workflow.

The integration path should therefore not be "embed these packages as-is". The right path is to use them as design references:

- Use `abacus-test` as the main source of workflow inventory, input preparation patterns, finite-difference/property decomposition, and result schema ideas.
- Use `ABACUS-agent-tools` as the main source of user-facing workflow recipes and ordering constraints, especially "prepare -> optional relax -> property".
- Implement the actual integration natively inside `aiida-abacus` as new `CalcJob`s, parsers, `calcfunction`s, and `WorkChain`s.

The most valuable first-wave additions to `aiida-abacus` are:

- EOS
- DOS/PDOS
- Elastic constants
- Phonons
- Bader charge
- Work function
- Vacancy formation energy
- Vibrational analysis

## Current `aiida-abacus` Baseline

The current plugin already has the minimum AiiDA-native architecture needed to extend property workflows:

- One ABACUS `CalcJob` in [calculations.py](/home/bonan/aiida_env/aiida-2.0/devzone/aiida-abacus-more-workflows/src/aiida_abacus/calculations.py:27)
- One ABACUS parser in [abacus.py](/home/bonan/aiida_env/aiida-2.0/devzone/aiida-abacus-more-workflows/src/aiida_abacus/parsers/abacus.py:48)
- A restart-capable base workflow in [base.py](/home/bonan/aiida_env/aiida-2.0/devzone/aiida-abacus-more-workflows/src/aiida_abacus/workflows/base.py:30)
- A relaxation workflow in [relax.py](/home/bonan/aiida_env/aiida-2.0/devzone/aiida-abacus-more-workflows/src/aiida_abacus/workflows/relax.py:34)
- A band workflow in [band.py](/home/bonan/aiida_env/aiida-2.0/devzone/aiida-abacus-more-workflows/src/aiida_abacus/workflows/band.py:31)

Important current capabilities:

- `AbacusCalculation` already models ABACUS inputs as `parameters`, `kpoints`, `structure`, `pseudos`, optional `settings`, optional `dynamics`, and optional `restart_folder` [calculations.py](/home/bonan/aiida_env/aiida-2.0/devzone/aiida-abacus-more-workflows/src/aiida_abacus/calculations.py:75)
- The parser already returns `misc`, optional `structure`, optional `kpoints`, optional `internal_parameters`, optional `bands`, and optional `trajectory` [abacus.py](/home/bonan/aiida_env/aiida-2.0/devzone/aiida-abacus-more-workflows/src/aiida_abacus/parsers/abacus.py:146)
- The base work chain already includes retry logic for incomplete runs, electronic non-convergence, and ionic non-convergence [base.py](/home/bonan/aiida_env/aiida-2.0/devzone/aiida-abacus-more-workflows/src/aiida_abacus/workflows/base.py:207)
- The relax work chain already supports iterative relaxation and optional final SCF [relax.py](/home/bonan/aiida_env/aiida-2.0/devzone/aiida-abacus-more-workflows/src/aiida_abacus/workflows/relax.py:64)
- The band work chain already composes optional relaxation, symmetry-path generation, SCF, and NSCF band evaluation [band.py](/home/bonan/aiida_env/aiida-2.0/devzone/aiida-abacus-more-workflows/src/aiida_abacus/workflows/band.py:71)

This means the core architectural gap is not low-level ABACUS execution. The gap is missing higher-level property workflows and the parsers/data models that support them.

## `abacus-test`: What It Provides

### Architectural Model

`abacus-test` has two distinct orchestration styles:

- A dflow/Bohrium submission stack around `submit`, `status`, and `download` in [abacustest.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/abacustest.py:14)
- A local model registry around `model` and `launching` in [model.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/model.py:4) and [launching.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/launching/launching.py:29)

The core pattern is file-tree oriented:

- Prepare a directory tree of ABACUS examples
- Expand parameter sweeps or finite-difference structures
- Run jobs externally
- Parse result files afterward
- Write JSON/CSV/plot/html outputs

That execution style is not AiiDA-native, but it is highly useful as a reference for workflow decomposition.

### ABACUS Calculation Types

The core ABACUS job types are encoded in `JOB_TYPES` in [model_013_inputs.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_013_inputs.py:14):

- `scf`
- `relax`
- `cell-relax`
- `md`
- `band`

These are the same primitives that AiiDA should expose as reusable building blocks.

### Higher-Level Property / Workflow Coverage

The package has explicit workflow models for:

- EOS in [model_004_Eos.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_004_Eos.py:11)
- Phonons in [model_005_Phonon.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_005_Phonon.py:11)
- Finite-difference forces in [model_006_FDForce.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_006_FDForce.py:27)
- Finite-difference stress in [model_007_FDStress.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_007_FDStress.py:13)
- Finite-difference magnetic force in [model_009_FDMagForce.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_009_FDMagForce.py:1)
- Magnetic exchange `J` in [model_010_MagJ.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_010_MagJ.py:23)
- Band preparation/post-processing in [model_012_band.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_012_band.py:1)
- Elastic constants in [model_015_elastic.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_015_elastic.py:16)
- Born effective charges in [model_016_bec.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_016_bec.py:13)
- Vacancy formation energy in [model_017_vacancy.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_017_vacancy.py:21)
- Supercell generation in [model_018_supercell.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_018_supercell.py:1)
- Vibrational analysis in [model_019_vibration.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_019_vibration.py:18)

This is the broadest workflow inventory among the three codebases.

### Input Preparation

`abacus-test` has a rich preparation layer in [prepare.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/prepare.py:14).

Key capabilities:

- Generate ABACUS inputs from template examples
- Convert structures from CIF/POSCAR/dpdata-supported formats
- Generate combinatorial parameter sweeps with `mix_input`, `mix_kpt`, and `mix_stru`
- Generate perturbed structures with `pert_stru`
- Attach pseudopotential and orbital libraries
- Convert ABACUS inputs to QE, VASP, or CP2K

This is not directly reusable as AiiDA execution code, but it is a strong reference for:

- builder-generation helpers
- provenance-preserving structure expansion steps
- parameter sweep work chains
- cross-code conversion utilities

### Result Schema and Collection

The strongest reusable concept in `abacus-test` is its result schema. `AbacusMetricEnum` in [comm_class_metrics.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/launching/comm_class_metrics.py:12) collects a broad spectrum of ABACUS outputs:

- runtime information
- convergence status
- energy and energy-per-atom
- band gap
- SCF steps
- forces
- stresses and virials
- magnetization data
- structural metrics
- memory/runtime summaries
- delta-to-reference metrics

The collection layer in [collectdata.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/collectdata.py:107), [resultAbacus.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_collectdata/resultAbacus.py:10), and the ABACUS extractors under `lib_collectdata` provides a useful blueprint for expanding `aiida-abacus` parser coverage.

### What `abacus-test` Contributes Best

Best reusable ideas from `abacus-test`:

- property workflow inventory
- finite-difference decomposition patterns
- result key taxonomy
- input preparation logic
- parameter sweep generation
- result aggregation and reporting concepts

Least reusable parts:

- dflow/Bohrium execution
- mutable working-directory orchestration
- CLI re-entry patterns
- post hoc scanning of arbitrary folders outside the AiiDA repository

## `ABACUS-agent-tools`: What It Provides

### Architectural Model

`ABACUS-agent-tools` is an MCP server package, not a scientific workflow engine in the AiiDA sense. Its CLI entry point is in [main.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/main.py:1), and it dynamically loads tool modules from `src/abacusagent/modules`.

Its value is not its execution backend. Its value is that it exposes an already-curated set of property operations and the ordering rules that connect them.

### Capability Surface

The module tree under [src/abacusagent/modules](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules:1) includes:

- input preparation and editing in [abacus.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/abacus.py:1)
- SCF in [scf.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/scf.py:1)
- relaxation in [relax.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/relax.py:1)
- MD in [md.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/md.py:1)
- band structure in [band.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/band.py:1)
- DOS/PDOS in [dos.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/dos.py:1)
- Bader charge in [bader.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/bader.py:1)
- ELF, charge-density difference, and spin density in [cube.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/cube.py:1)
- elastic properties in [elastic.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/elastic.py:1)
- EOS in [eos.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/eos.py:1)
- phonons in [phonon.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/phonon.py:1)
- vibrational analysis in [vibration.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/vibration.py:1)
- work function in [work_function.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/work_function.py:1)
- vacancy formation energy in [vacancy.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/vacancy.py:1)
- JDOS and PYATB wrappers in [pyatb.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/pyatb.py:1)
- structure generation/editing in [structure_generator.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/structure_generator.py:1) and [structure_editor.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/structure_editor.py:1)
- symmetry path helpers in [symmetry.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/symmetry.py:1)

### Workflow Rules Encoded in Prompts

The prompt contract in [prompt.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/prompt.py:1) is especially useful because it captures workflow ordering:

- `abacus_prepare` is mandatory before property calculations when only a structure is available
- relaxation is strongly encouraged before many properties
- phonon requires prior `cell-relax`
- vibrational analysis requires prior `relax`
- band/DOS/Bader/elastic workflows commonly depend on prior relaxed inputs

This ordering logic should be formalized in `aiida-abacus` as workflow specifications and validators.

### Execution Style

The package is still directory-centric and imperative. It assumes:

- ABACUS input directories on disk
- mutable input files
- shell execution
- local or Bohrium submission
- direct path returns to users

This makes it a poor direct runtime dependency for AiiDA, but a good reference for:

- top-level workflow topology
- property-specific preconditions
- output expectations
- optional branch logic

### What `ABACUS-agent-tools` Contributes Best

Best reusable ideas from `ABACUS-agent-tools`:

- a curated property function list
- property-specific ordering and prerequisites
- simple top-level workflow recipes
- user-facing abstractions around "prepare / relax / property"

Least reusable parts:

- MCP server runtime
- environment-variable driven state
- mutable directory API
- shell-and-path-based execution contracts

## Capability Comparison

### `abacus-test`

Strengths:

- widest workflow inventory
- best preparation layer
- best result taxonomy
- strong finite-difference and benchmarking coverage

Weaknesses:

- execution tied to dflow/Bohrium/local folder trees
- provenance not explicit
- parsers are permissive and ad hoc

### `ABACUS-agent-tools`

Strengths:

- clean user-facing capability list
- clear workflow ordering rules
- practical wrappers for many common properties

Weaknesses:

- not AiiDA-native
- directory/path mutation is central
- return model is path-oriented instead of node-oriented

### `aiida-abacus`

Strengths:

- correct AiiDA-native execution model
- structured inputs/outputs
- existing parser and restart logic
- existing relax and band workflows

Weaknesses:

- currently narrow property surface
- parser coverage is still modest compared with `abacus-test`
- no explicit high-level workflows for DOS, elastic, phonon, Bader, vacancy, work function, or vibration

## Recommended Integration Strategy

### Core Principle

Do not integrate either external package as a runtime dependency of `aiida-abacus`.

Instead:

- use their logic as a specification source
- reimplement workflow orchestration in AiiDA-native processes
- port parsing logic selectively into robust parsers
- convert directory mutation steps into node-producing `calcfunction`s or `WorkChain`s

### AiiDA Object Mapping

Recommended mapping of concepts:

- single ABACUS execution -> `AbacusCalculation`
- single external analysis executable such as Bader or PYATB -> dedicated `CalcJob`
- symmetry path generation / structure transforms / supercell expansion -> `calcfunction` or lightweight utility process
- SCF + NSCF + parser aggregation -> `WorkChain`
- relaxation + final SCF -> `WorkChain`
- multi-distortion or multi-supercell studies -> fan-out `WorkChain`
- JSON-like scalar summaries -> `Dict`
- bands -> `BandsData`
- trajectories -> `TrajectoryData`
- volumetric outputs such as ELF, spin density, charge density difference -> either `SinglefileData`, `FolderData`, or a dedicated volumetric data type if you want richer semantics

### What to Port First from `abacus-test`

Highest-value ideas to port first:

- metric coverage from `AbacusMetricEnum`
- finite-difference workflow decomposition
- elastic, EOS, and vibration workflow structure
- vacancy formation energy decomposition
- preparation helpers for perturbations and parameter sweeps

### What to Port First from `ABACUS-agent-tools`

Highest-value ideas to port first:

- "prepare -> optional relax -> property" workflow pattern
- property prerequisites and validation rules
- a consistent property catalog aligned with user expectations
- structure preprocessing ideas such as slab building and standard orientation as optional helpers

## Proposed `aiida-abacus` Roadmap

### Phase 1: Parser and Result-Schema Expansion

Extend the ABACUS parser so the base plugin captures more of the information already harvested by `abacus-test`.

Recommended outputs to add:

- richer convergence diagnostics
- SCF history and iteration counts
- stress, pressure, virial, and force summaries
- magnetization summaries
- band gap and electron-count summaries
- runtime and memory summaries where stable

Why first:

- every later property workflow benefits from stronger low-level parsing
- parser work is lower risk than multi-step workflow work
- it creates a stable schema for downstream work chains

### Phase 2: Property Work Chains That Reuse Existing Foundations

Implement the property workflows that are closest to the current `CalcJob` plus parser model:

- `AbacusDosWorkChain`
- `AbacusElasticWorkChain`
- `AbacusEosWorkChain`
- `AbacusBaderWorkChain`
- `AbacusWorkFunctionWorkChain`

These can be built by composing:

- optional relax
- base SCF
- optional NSCF or postprocessing
- one aggregation/finalization step

### Phase 3: Fan-Out / Finite-Difference Work Chains

Implement the workflows that naturally require controlled fan-out:

- `AbacusPhononWorkChain`
- `AbacusVibrationWorkChain`
- `AbacusVacancyFormationWorkChain`
- possibly `AbacusBecWorkChain`
- possibly finite-difference force/stress/magnetic workflows

These should be built as explicit work chains that:

- generate derived structures
- submit one calculation per distortion/supercell/configuration
- gather outputs deterministically
- run a final aggregation step

### Phase 4: Utilities and Builder Helpers

Add optional utilities inspired by both external packages:

- structure-to-builder helpers
- parameter sweep helpers
- supercell/slab/perturbation helpers
- high-symmetry path generation helpers
- cross-code conversion helpers if they are still desired

These should remain utility-level, not become the core runtime model.

## Current Branch Status

As of this branch, the implementation has moved beyond the initial parser-only foundation:

- parser/result-schema expansion is partially implemented
- `AbacusDosWorkChain` is implemented and registered
- `AbacusEosWorkChain` is implemented and registered
- EOS and workflow launch examples have been added

Implemented parser and retrieval improvements:

- richer convergence fields such as `converged`, `relax_converged`, `relax_steps`, and `scf_steps`
- parsed `volume`, `energy_ks`, flattened `force`/`stress`, `pressure`, and `virial`
- optional retrieval toggles for PDOS, `time.json`, eigenvalues, Mulliken output, and electrostatic potential
- `TimejsonParser` and `PdosParser`

Implemented workflow additions:

- `AbacusDosWorkChain` follows the intended optional-relax -> SCF -> DOS/PDOS NSCF pattern
- `AbacusEosWorkChain` establishes the fan-out pattern with isotropic scaling, branch submission, and EOS fitting

Current limitations:

- the EOS workflow does not yet perform per-point fixed-volume atomic-position relaxation for non-cubic or low-symmetry systems
- DOS currently exposes parsed PDOS data as `Dict`; it does not yet provide a richer dedicated DOS data container
- parser coverage is still missing several items from the original wishlist, especially magnetization summaries and band-gap/electron-count summaries

Still missing from the first recommended workflow wave:

- `AbacusElasticWorkChain`
- `AbacusBaderWorkChain`
- `AbacusWorkFunctionWorkChain`

Still missing from the later fan-out/external-tool backlog:

- `AbacusPhononWorkChain`
- `AbacusVibrationWorkChain`
- `AbacusVacancyFormationWorkChain`
- `AbacusBecWorkChain`

## Initial WorkChain Backlog

Recommended first implementation order:

1. `AbacusDosWorkChain`
2. `AbacusElasticWorkChain`
3. `AbacusEosWorkChain`
4. `AbacusBaderWorkChain`
5. `AbacusWorkFunctionWorkChain`
6. `AbacusPhononWorkChain`
7. `AbacusVacancyFormationWorkChain`
8. `AbacusVibrationWorkChain`
9. `AbacusBecWorkChain`

Rationale:

- DOS, EOS, elastic, and Bader are common, high-value, and relatively easy to explain and test
- phonon, vacancy, and vibration are more orchestration-heavy and benefit from the earlier parser/schema work

## Detailed Integration Notes by Capability

### DOS / PDOS

Source references:

- `ABACUS-agent-tools`: [dos.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/dos.py:1)
- current `aiida-abacus` band/NSCF pattern: [band.py](/home/bonan/aiida_env/aiida-2.0/devzone/aiida-abacus-more-workflows/src/aiida_abacus/workflows/band.py:261)

Recommended AiiDA design:

- optional relax
- SCF to generate charge density
- NSCF or DOS-mode calculation with proper retrieval
- parser for total DOS and projected DOS
- output `XyData` or a dedicated DOS data container if you decide to add one

### Elastic Constants

Source references:

- `abacus-test`: [model_015_elastic.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_015_elastic.py:16)
- `ABACUS-agent-tools`: [elastic.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/elastic.py:1)

Recommended AiiDA design:

- optional cell relaxation
- generate symmetry-reduced strain set
- submit one ABACUS job per strain
- parse stress tensors
- aggregate into elastic tensor and derived moduli in a final calcfunction or analysis process

### EOS

Source references:

- `abacus-test`: [model_004_Eos.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_004_Eos.py:11)
- `ABACUS-agent-tools`: [eos.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/eos.py:1)

Recommended AiiDA design:

- generate scaled cells from a reference structure
- run SCF or relax-on-fixed-volume for each scale
- collect energies and volumes
- fit Birch-Murnaghan in a final step

### Phonons

Source references:

- `abacus-test`: [model_005_Phonon.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_005_Phonon.py:11)
- `ABACUS-agent-tools`: [phonon.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/phonon.py:1)
- workflow prerequisite from prompt: [prompt.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/prompt.py:1)

Recommended AiiDA design:

- require or perform `cell-relax`
- build displaced supercells
- run force calculations for each displacement
- aggregate with phonopy in a final analysis step

### Bader Charge

Source references:

- `ABACUS-agent-tools`: [bader.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/bader.py:1)

Recommended AiiDA design:

- SCF with charge-density output enabled
- run Bader as a separate `CalcJob`
- return per-atom charges and optionally volumetric artifacts

### Work Function

Source references:

- `ABACUS-agent-tools`: [work_function.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/work_function.py:1)

Recommended AiiDA design:

- slab structure input
- SCF with electrostatic potential output
- parser or postprocess step to derive vacuum level and Fermi level
- return work function and metadata about vacuum direction / dipole correction

### Vacancy Formation Energy

Source references:

- `abacus-test`: [model_017_vacancy.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_017_vacancy.py:21)
- `ABACUS-agent-tools`: [vacancy.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/vacancy.py:1)

Recommended AiiDA design:

- generate pristine and defective supercells
- relax both
- calculate elemental reference energy as required by the definition used
- aggregate into vacancy formation energy with full provenance across all branches

### Vibrational Analysis

Source references:

- `abacus-test`: [model_019_vibration.py](/home/bonan/aiida_env/aiida-2.0/devzone/abacus-test/abacustest/lib_model/model_019_vibration.py:18)
- `ABACUS-agent-tools`: [vibration.py](/home/bonan/aiida_env/aiida-2.0/devzone/ABACUS-agent-tools/src/abacusagent/modules/vibration.py:1)

Recommended AiiDA design:

- require or perform molecular relax
- generate finite displacements
- run force calculations
- assemble Hessian and frequencies in the final step

## Design Constraints and Risks

### Provenance Mismatch

This is the main architectural risk.

Both external packages assume:

- mutable directories
- in-place file edits
- shell commands over local paths
- outputs returned as filesystem locations

AiiDA needs:

- immutable input nodes
- explicit process edges
- retrieved-folder parsing
- typed outputs
- deterministic branching captured in provenance

Any attempt to embed the external execution layers directly would weaken `aiida-abacus`.

### Parser Quality Gap

`abacus-test` tolerates missing files and parse failures in ways that are acceptable for screening but not for process accounting. `aiida-abacus` should keep strict parser exit codes like the current parser already does in [abacus.py](/home/bonan/aiida_env/aiida-2.0/devzone/aiida-abacus-more-workflows/src/aiida_abacus/parsers/abacus.py:101).

### Optional External Dependencies

Several advanced workflows will depend on external tools:

- Bader
- phonopy
- PYATB

These should be isolated as optional entry points and dedicated `CalcJob`s or plugin extras, not folded into the base ABACUS parser.

### Cross-Code Conversion

`abacus-test` supports conversions to VASP, QE, and CP2K. That is useful, but it is not the right first target for `aiida-abacus`. Keep the initial scope ABACUS-native.

## Concrete Initial Implementation Guide

### Step 1

Expand the ABACUS parser output schema.

Targets:

- adopt a stable subset of `abacus-test` metric keys
- expose them through `misc`
- keep strict exit handling

### Step 2

Add one small analysis workflow that reuses the current SCF/NSCF infrastructure.

Best candidate:

- `AbacusDosWorkChain`

Reason:

- it naturally reuses the existing band/NSCF pattern
- it has immediate user value
- it forces clear decisions on output data types

### Step 3

Add one fan-out property workflow.

Best candidate:

- `AbacusEosWorkChain` or `AbacusElasticWorkChain`

Reason:

- both are clean examples of structure generation plus aggregation
- they establish a repeatable pattern for later phonon and vacancy workflows

### Step 4

Add one external-tool workflow.

Best candidate:

- `AbacusBaderWorkChain`

Reason:

- it exercises multi-code provenance without requiring the complexity of phonopy first

### Step 5

Implement more complex workflows after the above patterns are stable:

- phonon
- vacancy
- vibration
- work function
- BEC

## Recommended Module Layout in `aiida-abacus`

Suggested additions:

- `src/aiida_abacus/workflows/dos.py`
- `src/aiida_abacus/workflows/eos.py`
- `src/aiida_abacus/workflows/elastic.py`
- `src/aiida_abacus/workflows/bader.py`
- `src/aiida_abacus/workflows/work_function.py`
- `src/aiida_abacus/workflows/phonon.py`
- `src/aiida_abacus/workflows/vacancy.py`
- `src/aiida_abacus/workflows/vibration.py`
- `src/aiida_abacus/calculations/bader.py` if external tools are separated into their own calcjobs
- parser helpers under `src/aiida_abacus/parsers/`
- structure/strain/displacement helpers under `src/aiida_abacus/common/` or `src/aiida_abacus/utils/`

## Final Recommendation

Use `abacus-test` as the specification source for workflow breadth and result semantics. Use `ABACUS-agent-tools` as the specification source for workflow composition and user-facing capability boundaries. Keep all new execution, parsing, and provenance handling native to `aiida-abacus`.

If only one concrete next action is taken, it should be:

- extend the parser schema
- then implement `AbacusDosWorkChain`

That gives the shortest path from the current plugin to a broader property-workflow platform without importing the architectural liabilities of the other two packages.
