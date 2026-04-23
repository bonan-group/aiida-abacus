"""Workflow for elastic tensor calculations."""

from __future__ import annotations

import pathlib

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import ToContext, WorkChain, calcfunction, if_

from aiida_abacus.common import ProtocolMixin, RelaxType, prepare_process_inputs, recursive_merge
from aiida_abacus.common.opthold import ElasticOptions, RelaxOptions

from .base import AbacusBaseWorkChain
from .relax import AbacusRelaxWorkChain


def validate_elastic_inputs(inputs, _):
    """Validate top-level elastic workflow inputs."""
    settings = inputs.get("elastic_settings")
    if isinstance(settings, orm.Dict):
        settings = settings.get_dict()
    else:
        settings = ElasticOptions().model_dump(mode="json")

    if settings.get("pre_relax", True) or settings.get("relax_internal_positions", True):
        if "relax" not in inputs:
            return "The `relax` namespace is required when `pre_relax` or `relax_internal_positions` is enabled."

    if not settings.get("relax_internal_positions", True) and "base" not in inputs:
        return "The `base` namespace is required when `relax_internal_positions` is disabled."

    return None


class AbacusElasticWorkChain(ProtocolMixin, WorkChain):
    """Workflow for elastic tensor calculations based on finite strains."""

    _protocol_tag = "elastic"

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(
            AbacusBaseWorkChain,
            namespace="base",
            exclude=("clean_workdir", "abacus.structure", "abacus.parent_folder"),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for fixed-ion elastic branches.",
            },
        )
        spec.expose_inputs(
            AbacusRelaxWorkChain,
            namespace="relax",
            exclude=("structure",),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for optional pre-relax and position-only strained branches.",
            },
        )
        spec.input("structure", valid_type=orm.StructureData, help="The input structure.")
        spec.input(
            "elastic_settings",
            help=ElasticOptions.aiida_description(),
            valid_type=orm.Dict,
            validator=ElasticOptions.aiida_validate,
            serializer=ElasticOptions.aiida_serialize,
        )
        spec.inputs.validator = validate_elastic_inputs
        spec.outline(
            cls.setup,
            if_(cls.should_run_initial_relax)(
                cls.run_initial_relax,
                cls.inspect_initial_relax,
            ),
            cls.generate_strained_structures,
            cls.run_equilibrium_branch,
            cls.inspect_equilibrium_branch,
            cls.run_strain_branches,
            cls.inspect_strain_branches,
            cls.finalize_results,
        )
        spec.output("structure", valid_type=orm.StructureData, help="Reference structure used to generate strains.")
        spec.output("elastic_parameters", valid_type=orm.Dict, help="Elastic tensor and derived moduli.")
        spec.output("branch_results", valid_type=orm.Dict, help="Raw branch metadata, stresses, and strain payload.")
        spec.exit_code(631, "ERROR_RELAX_PROCESS_FAILED", message="The initial relaxation calculation failed.")
        spec.exit_code(632, "ERROR_EQUILIBRIUM_PROCESS_FAILED", message="The equilibrium elastic branch failed.")
        spec.exit_code(633, "ERROR_STRAIN_PROCESS_FAILED", message="One or more strained elastic branches failed.")
        spec.exit_code(634, "ERROR_MISSING_BRANCH_STRESS", message="A required branch finished without stress output.")
        spec.exit_code(635, "ERROR_ELASTIC_FIT_FAILED", message="Elastic tensor fitting failed.")

    @classmethod
    def get_protocol_filepath(cls, file_alias: str | None = None) -> pathlib.Path:
        """Return the path to the YAML protocol definition."""
        return super().get_protocol_filepath(file_alias)

    @classmethod
    def get_builder_from_protocol(
        cls, code, structure, protocol=None, overrides=None, relax_type=RelaxType.POSITIONS_CELL, options=None, **kwargs
    ):
        """Return a builder prepopulated according to the chosen protocol."""
        type_check(relax_type, RelaxType)
        inputs = cls.get_protocol_inputs(protocol, overrides)
        base_overrides = inputs.get("base", {})
        relax_overrides = inputs.get("relax", {})

        pseudo_family = base_overrides.get("pseudo_family")
        if pseudo_family is not None:
            relax_overrides = recursive_merge(
                {
                    "base": {"pseudo_family": pseudo_family},
                    "base_final_scf": {"pseudo_family": pseudo_family},
                },
                relax_overrides,
            )

        builder = cls.get_builder()
        builder.structure = structure
        builder.elastic_settings = orm.Dict(dict=inputs.get("elastic_settings", {}))

        if "base" in inputs:
            base = AbacusBaseWorkChain.get_builder_from_protocol(
                code=code,
                structure=structure,
                protocol=protocol,
                overrides=base_overrides,
                options=options,
                **kwargs,
            )
            builder.base = base

        if "relax" in inputs or builder.elastic_settings.get_dict().get("pre_relax", True):
            relax = AbacusRelaxWorkChain.get_builder_from_protocol(
                code=code,
                structure=structure,
                protocol=protocol,
                overrides=relax_overrides,
                options=options,
                relax_type=relax_type,
                **kwargs,
            )
            builder.relax = relax

        return builder

    def setup(self):
        """Initialize workchain context."""
        self.ctx.elastic_settings = self.inputs.elastic_settings.get_dict()
        self.ctx.structure = self.inputs.structure
        self.ctx.base_inputs = (
            AttributeDict(self.exposed_inputs(AbacusBaseWorkChain, "base")) if "base" in self.inputs else None
        )
        self.ctx.relax_inputs = (
            AttributeDict(self.exposed_inputs(AbacusRelaxWorkChain, "relax")) if "relax" in self.inputs else None
        )
        self.ctx.strained_structures = []
        self.ctx.strain_payload = []
        self.ctx.branch_labels = []
        self.out("structure", self.ctx.structure)

    def should_run_initial_relax(self):
        """Return whether the initial relaxation should run."""
        return self.ctx.elastic_settings.get("pre_relax", True)

    def run_initial_relax(self):
        """Run an initial relaxation for the reference structure."""
        inputs = _build_relax_branch_inputs(self.ctx.relax_inputs, self.ctx.structure, "initial_relax")
        prepared = prepare_process_inputs(AbacusRelaxWorkChain, inputs)
        running = self.submit(AbacusRelaxWorkChain, **prepared)
        self.report(f"launching AbacusRelaxWorkChain<{running.pk}> for elastic pre-relax")
        return ToContext(initial_relax_workchain=running)

    def inspect_initial_relax(self):
        """Inspect the initial relaxation."""
        if not self.ctx.initial_relax_workchain.is_finished_ok:
            return self.exit_codes.ERROR_RELAX_PROCESS_FAILED
        self.ctx.structure = self.ctx.initial_relax_workchain.outputs.structure
        self.out("structure", self.ctx.structure)

    def generate_strained_structures(self):
        """Generate the strained structures and serialize the strain payload."""
        structures, strains = _generate_deformed_structures(self.ctx.structure, self.ctx.elastic_settings)
        self.ctx.strained_structures = structures
        self.ctx.strain_payload = strains
        self.ctx.branch_labels = [f"elastic_strain_{index:02d}" for index in range(len(structures))]

    def run_equilibrium_branch(self):
        """Run the equilibrium branch used to define the reference stress."""
        if self.ctx.elastic_settings.get("relax_internal_positions", True):
            inputs = self._build_position_relax_inputs(self.ctx.structure, "elastic_equilibrium")
            process = AbacusRelaxWorkChain
        else:
            inputs = self._build_base_inputs(self.ctx.structure, "elastic_equilibrium")
            process = AbacusBaseWorkChain

        prepared = prepare_process_inputs(process, inputs)
        running = self.submit(process, **prepared)
        self.report(f"launching {process.__name__}<{running.pk}> for elastic equilibrium")
        return ToContext(equilibrium_workchain=running)

    def inspect_equilibrium_branch(self):
        """Inspect the equilibrium branch result."""
        if not self.ctx.equilibrium_workchain.is_finished_ok:
            return self.exit_codes.ERROR_EQUILIBRIUM_PROCESS_FAILED

        stress = _extract_stress(self.ctx.equilibrium_workchain)
        if stress is None:
            return self.exit_codes.ERROR_MISSING_BRANCH_STRESS

        self.ctx.equilibrium_stress = stress
        self.ctx.equilibrium_pk = self.ctx.equilibrium_workchain.pk

    def run_strain_branches(self):
        """Launch all strained branches."""
        running = {}
        for label, structure in zip(self.ctx.branch_labels, self.ctx.strained_structures):
            if self.ctx.elastic_settings.get("relax_internal_positions", True):
                inputs = self._build_position_relax_inputs(structure, label)
                process = AbacusRelaxWorkChain
            else:
                inputs = self._build_base_inputs(structure, label)
                process = AbacusBaseWorkChain

            prepared = prepare_process_inputs(process, inputs)
            running[label] = self.submit(process, **prepared)
            self.report(f"launching {process.__name__}<{running[label].pk}> for elastic strain branch `{label}`")

        return ToContext(**running)

    def inspect_strain_branches(self):
        """Collect stresses from all strained branches."""
        branch_results = []

        for label, strain_payload in zip(self.ctx.branch_labels, self.ctx.strain_payload):
            workchain = self.ctx[label]
            if not workchain.is_finished_ok:
                self.report(f"Elastic branch `{label}` failed with exit status {workchain.exit_status}")
                return self.exit_codes.ERROR_STRAIN_PROCESS_FAILED

            stress = _extract_stress(workchain)
            if stress is None:
                self.report(f"Elastic branch `{label}` finished without a `stress` entry in `misc`.")
                return self.exit_codes.ERROR_MISSING_BRANCH_STRESS

            branch_results.append({"label": label, "pk": workchain.pk, "strain": strain_payload, "stress_kbar": stress})

        self.ctx.branch_results = branch_results

    def finalize_results(self):
        """Fit the elastic tensor and expose serialized branch metadata."""
        payload = {
            "equilibrium": {
                "pk": self.ctx.equilibrium_pk,
                "stress_kbar": self.ctx.equilibrium_stress,
            },
            "branches": self.ctx.branch_results,
            "fit_with_relaxed_internal_positions": self.ctx.elastic_settings.get("relax_internal_positions", True),
        }

        branch_results = orm.Dict(dict=payload)
        fit_result = fit_elastic_tensor(branch_results, self.inputs.elastic_settings)
        if not fit_result.get_dict().get("fit_success", False):
            return self.exit_codes.ERROR_ELASTIC_FIT_FAILED

        self.out("branch_results", branch_results)
        self.out("elastic_parameters", fit_result)

    def _build_base_inputs(self, structure, label):
        """Build fixed-ion branch inputs."""
        inputs = _build_base_branch_inputs(self.ctx.base_inputs, structure, label)
        parameters = inputs.abacus.parameters
        parameters.setdefault("input", {})
        parameters["input"]["cal_force"] = 1
        parameters["input"]["cal_stress"] = 1
        parameters["input"]["calculation"] = "scf"
        inputs.abacus.parameters = parameters
        return inputs

    def _build_position_relax_inputs(self, structure, label):
        """Build position-only relax inputs for an elastic branch."""
        inputs = _build_relax_branch_inputs(self.ctx.relax_inputs, structure, label)

        relax_settings = dict(inputs.relax_settings) if "relax_settings" in inputs else {}
        relax_settings = recursive_merge(
            relax_settings,
            RelaxOptions(relax_type=RelaxType.POSITIONS).model_dump(mode="json"),
        )
        inputs.relax_settings = relax_settings

        inputs.base.abacus.parameters.setdefault("input", {})
        inputs.base.abacus.parameters["input"]["cal_force"] = 1
        inputs.base.abacus.parameters["input"]["cal_stress"] = 1

        if "base_final_scf" in inputs:
            inputs.base_final_scf.abacus.parameters.setdefault("input", {})
            inputs.base_final_scf.abacus.parameters["input"]["cal_force"] = 1
            inputs.base_final_scf.abacus.parameters["input"]["cal_stress"] = 1

        return inputs


def _extract_stress(workchain):
    """Return the flattened stress payload from a finished child workchain."""
    try:
        stress = workchain.outputs.misc.get_dict().get("stress")
    except (AttributeError, KeyError):
        return None

    if stress is None or len(stress) != 9:
        return None
    return stress


def _build_base_branch_inputs(base_inputs, structure, label):
    """Build explicit branch inputs for a base-workchain branch."""
    inputs = AttributeDict()
    inputs.metadata = AttributeDict(dict(base_inputs.metadata))
    inputs.metadata.call_link_label = label
    inputs.max_iterations = base_inputs.max_iterations

    inputs.abacus = AttributeDict()
    inputs.abacus.code = base_inputs.abacus.code
    inputs.abacus.metadata = AttributeDict(dict(base_inputs.abacus.metadata))
    inputs.abacus.parameters = base_inputs.abacus.parameters.get_dict()
    inputs.abacus.pseudos = base_inputs.abacus.pseudos
    inputs.abacus.structure = structure

    if "settings" in base_inputs.abacus:
        inputs.abacus.settings = (
            base_inputs.abacus.settings.get_dict()
            if hasattr(base_inputs.abacus.settings, "get_dict")
            else dict(base_inputs.abacus.settings)
        )

    if "kpoints" in base_inputs:
        inputs.kpoints = base_inputs.kpoints
    if "kpoints_distance" in base_inputs:
        inputs.kpoints_distance = base_inputs.kpoints_distance
    if "kpoints_force_parity" in base_inputs:
        inputs.kpoints_force_parity = base_inputs.kpoints_force_parity
    if "pseudo_family" in base_inputs:
        inputs.pseudo_family = base_inputs.pseudo_family
    return inputs


def _build_relax_branch_inputs(relax_inputs, structure, label):
    """Build explicit branch inputs for a relax-workchain branch."""
    inputs = AttributeDict()
    inputs.metadata = AttributeDict(dict(relax_inputs.metadata))
    inputs.metadata.call_link_label = label
    inputs.structure = structure

    if "relax_settings" in relax_inputs:
        inputs.relax_settings = relax_inputs.relax_settings.get_dict()
    if "meta_convergence" in relax_inputs:
        inputs.meta_convergence = relax_inputs.meta_convergence
    if "max_meta_convergence_iterations" in relax_inputs:
        inputs.max_meta_convergence_iterations = relax_inputs.max_meta_convergence_iterations
    if "volume_convergence" in relax_inputs:
        inputs.volume_convergence = relax_inputs.volume_convergence
    if "clean_workdir" in relax_inputs:
        inputs.clean_workdir = relax_inputs.clean_workdir

    inputs.base = _build_base_branch_inputs(relax_inputs.base, structure, f"{label}_base")

    if "base_final_scf" in relax_inputs:
        inputs.base_final_scf = _build_base_branch_inputs(relax_inputs.base_final_scf, structure, f"{label}_final_scf")

    return inputs


def _generate_deformed_structures(structure, elastic_settings):
    """Generate strained structures using pymatgen's elastic helper stack."""
    from pymatgen.analysis.elasticity.elastic import Strain
    from pymatgen.analysis.elasticity.strain import DeformedStructureSet

    norm = float(elastic_settings["norm_strain"])
    shear = float(elastic_settings["shear_strain"])
    pymatgen_structure = structure.get_pymatgen()

    deformed_set = DeformedStructureSet(
        pymatgen_structure,
        symmetry=False,
        norm_strains=[-norm, -0.5 * norm, 0.5 * norm, norm],
        shear_strains=[-shear, -0.5 * shear, 0.5 * shear, shear],
    )

    strained_structures = [orm.StructureData(pymatgen=deformed) for deformed in deformed_set]
    strains = [Strain.from_deformation(deformation).as_dict() for deformation in deformed_set.deformations]
    return strained_structures, strains


@calcfunction
def fit_elastic_tensor(branch_results, elastic_settings):
    """Fit the elastic tensor from branch stresses and serialized strains."""
    import numpy as np
    from pymatgen.analysis.elasticity.elastic import ElasticTensor, Strain
    from pymatgen.analysis.elasticity.stress import Stress

    payload = branch_results.get_dict()
    settings = elastic_settings.get_dict()

    try:
        strains = [Strain.from_dict(entry["strain"]) for entry in payload["branches"]]
        deformed_stresses = [
            Stress(-0.1 * np.array(entry["stress_kbar"]).reshape(3, 3)) for entry in payload["branches"]
        ]
        eq_stress = Stress(-0.1 * np.array(payload["equilibrium"]["stress_kbar"]).reshape(3, 3))
        tensor = ElasticTensor.from_independent_strains(
            strains,
            deformed_stresses,
            eq_stress=eq_stress if settings.get("fit_eq_stress", True) else None,
            vasp=False,
        )
    except Exception as exception:  # pylint: disable=broad-exception-caught
        return orm.Dict(dict={"fit_success": False, "fit_error": str(exception)})

    result = {
        "fit_success": True,
        "fit_method": "stress_from_independent_strains",
        "fit_with_relaxed_internal_positions": payload["fit_with_relaxed_internal_positions"],
        "elastic_tensor": tensor.voigt.tolist(),
        "bulk_modulus_voigt": float(tensor.k_voigt),
        "shear_modulus_voigt": float(tensor.g_voigt),
        "youngs_modulus": float(tensor.y_mod),
        "poisson_ratio": float(tensor.homogeneous_poisson),
        "universal_anisotropy": float(tensor.universal_anisotropy),
        "eq_stress_gpa": eq_stress.voigt.tolist(),
        "strain_labels": [entry["label"] for entry in payload["branches"]],
    }
    return orm.Dict(dict=result)
