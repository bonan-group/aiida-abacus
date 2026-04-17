"""Workflow for equation-of-state calculations."""

import pathlib

import numpy as np
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import ToContext, WorkChain, calcfunction, if_
from ase.eos import EquationOfState

from aiida_abacus.common import ProtocolMixin, RelaxType, prepare_process_inputs
from aiida_abacus.common.opthold import EosOptions

from .base import AbacusBaseWorkChain
from .relax import AbacusRelaxWorkChain


class AbacusEosWorkChain(ProtocolMixin, WorkChain):
    """Workflow for equation-of-state calculations."""

    _protocol_tag = "eos"

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(
            AbacusBaseWorkChain,
            namespace="base",
            exclude=("clean_workdir", "abacus.structure", "abacus.parent_folder"),
            namespace_options={"help": "Inputs for the `AbacusBaseWorkChain` EOS branch calculations."},
        )
        spec.expose_inputs(
            AbacusRelaxWorkChain,
            namespace="relax",
            exclude=("structure",),
            namespace_options={
                "help": "Inputs for the optional `AbacusRelaxWorkChain` pre-relaxation.",
                "required": False,
                "populate_defaults": False,
            },
        )
        spec.input("structure", valid_type=orm.StructureData, help="The input structure.")
        spec.input(
            "eos_settings",
            help=EosOptions.aiida_description(),
            valid_type=orm.Dict,
            validator=EosOptions.aiida_validate,
            serializer=EosOptions.aiida_serialize,
        )
        spec.outline(
            cls.setup,
            if_(cls.should_do_relax)(
                cls.run_relax,
                cls.verify_relax,
            ),
            cls.generate_structures,
            cls.run_eos,
            cls.inspect_eos,
        )
        spec.output("structure", valid_type=orm.StructureData, help="Final structure used as the EOS reference.")
        spec.output("eos_parameters", valid_type=orm.Dict, help="Raw EOS samples and fitted equation-of-state data.")
        spec.exit_code(621, "ERROR_EOS_PROCESS_FAILED", message="One or more EOS branch calculations failed.")
        spec.exit_code(622, "ERROR_RELAX_PROCESS_FAILED", message="The relaxation calculation failed.")

    @classmethod
    def get_protocol_filepath(cls, file_alias: str | None = None) -> pathlib.Path:
        """Return the path to the YAML protocol definition."""
        return super().get_protocol_filepath(file_alias)

    @classmethod
    def get_builder_from_protocol(
        cls, code, structure, protocol=None, overrides=None, relax_type=RelaxType.POSITIONS_CELL, options=None, **kwargs
    ):
        """Return a builder prepopulated according to the chosen protocol."""
        inputs = cls.get_protocol_inputs(protocol, overrides)
        base = AbacusBaseWorkChain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol=protocol,
            overrides=inputs.get("base", None),
            options=options,
            **kwargs,
        )

        builder = cls.get_builder()
        builder.base = base
        builder.structure = structure
        builder.eos_settings = orm.Dict(dict=inputs.get("eos_settings", {}))

        type_check(relax_type, RelaxType)
        if relax_type != RelaxType.NONE:
            relax = AbacusRelaxWorkChain.get_builder_from_protocol(
                code=code,
                structure=structure,
                protocol=protocol,
                overrides=inputs.get("relax", None),
                options=options,
                relax_type=relax_type,
                **kwargs,
            )
            builder.relax = relax

        return builder

    def setup(self):
        """Initialize workchain context."""
        self.ctx.base_inputs = AttributeDict(self.exposed_inputs(AbacusBaseWorkChain, "base"))
        self.ctx.relax_inputs = (
            AttributeDict(self.exposed_inputs(AbacusRelaxWorkChain, "relax")) if "relax" in self.inputs else None
        )
        self.ctx.structure = self.inputs.structure
        self.ctx.eos_settings = self.inputs.eos_settings.get_dict()
        self.out("structure", self.ctx.structure)

    def should_do_relax(self):
        """Return whether the relax namespace was provided."""
        return "relax" in self.inputs

    def run_relax(self):
        """Run the optional relax workflow."""
        self.ctx.relax_inputs.structure = self.ctx.structure
        self.ctx.relax_inputs.metadata.call_link_label = "relax"
        inputs = prepare_process_inputs(AbacusRelaxWorkChain, self.ctx.relax_inputs)
        running = self.submit(AbacusRelaxWorkChain, **inputs)
        self.report(f"launching AbacusRelaxWorkChain<{running.pk}> for EOS pre-relax")
        return ToContext(relax_workchain=running)

    def verify_relax(self):
        """Verify the relax workflow."""
        if not self.ctx.relax_workchain.is_finished_ok:
            return self.exit_codes.ERROR_RELAX_PROCESS_FAILED
        self.ctx.structure = self.ctx.relax_workchain.outputs.structure
        self.out("structure", self.ctx.structure)

    def generate_structures(self):
        """Generate scaled structures for the EOS sweep."""
        scales = _build_scale_factors(self.ctx.eos_settings)
        self.ctx.scale_factors = scales
        self.ctx.scale_labels = []
        self.ctx.scaled_structures = []

        for index, scale in enumerate(scales):
            label = f"eos_workchain_{index:02d}"
            structure = scale_structure(self.ctx.structure, orm.Float(scale))
            self.ctx.scale_labels.append(label)
            self.ctx.scaled_structures.append(structure)

    def run_eos(self):
        """Launch one fixed-structure calculation per scaling factor."""
        running = {}
        for label, scale, structure in zip(self.ctx.scale_labels, self.ctx.scale_factors, self.ctx.scaled_structures):
            inputs = _build_base_branch_inputs(self.ctx.base_inputs, structure, label)
            prepared = prepare_process_inputs(AbacusBaseWorkChain, inputs)
            running[label] = self.submit(AbacusBaseWorkChain, **prepared)
            self.report(f"launching AbacusBaseWorkChain<{running[label].pk}> for EOS scale {scale:.4f}")
        return ToContext(**running)

    def inspect_eos(self):
        """Collect branch results and fit the EOS."""
        samples = []
        for label, scale, structure in zip(self.ctx.scale_labels, self.ctx.scale_factors, self.ctx.scaled_structures):
            workchain = self.ctx[label]
            if not workchain.is_finished_ok:
                self.report(f"EOS branch `{label}` failed with exit status {workchain.exit_status}")
                return self.exit_codes.ERROR_EOS_PROCESS_FAILED

            misc = workchain.outputs.misc.get_dict()
            energy = misc.get("total_energy")
            if energy is None and misc.get("energy_ks") is not None:
                energy = misc["energy_ks"]
            if energy is None:
                self.report(f"EOS branch `{label}` finished without `total_energy` or `energy_ks` in `misc`.")
                return self.exit_codes.ERROR_EOS_PROCESS_FAILED

            samples.append(
                {
                    "label": label,
                    "scale": scale,
                    "volume": misc.get("volume", structure.get_cell_volume()),
                    "energy": energy,
                    "pk": workchain.pk,
                }
            )

        eos_data = {
            "reference_volume": self.ctx.structure.get_cell_volume(),
            "fit_equation": self.ctx.eos_settings.get("fit_equation", "birchmurnaghan"),
            "samples": samples,
            "scales": [entry["scale"] for entry in samples],
            "volumes": [entry["volume"] for entry in samples],
            "energies": [entry["energy"] for entry in samples],
        }
        self.out("eos_parameters", fit_eos(orm.Dict(dict=eos_data), self.inputs.eos_settings))


def _build_scale_factors(settings):
    """Build the ordered list of scale factors for the EOS sweep."""
    start = float(settings["scale_start"])
    end = float(settings["scale_end"])
    step = float(settings["scale_step"])

    scales = []
    current = start
    while current <= end + 1.0e-10:
        scales.append(round(current, 10))
        current += step

    if not any(abs(scale - 1.0) < 1.0e-10 for scale in scales):
        scales.append(1.0)

    return sorted(set(scales))


def _build_base_branch_inputs(base_inputs, structure, label):
    """Build explicit branch inputs for a base-workchain fan-out branch."""
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


@calcfunction
def scale_structure(structure, scale_factor):
    """Scale a structure isotropically while preserving fractional coordinates."""
    atoms = structure.get_ase()
    atoms.set_cell(atoms.cell * scale_factor.value, scale_atoms=True)
    return orm.StructureData(ase=atoms)


@calcfunction
def fit_eos(eos_data, eos_settings):
    """Fit an equation of state to volume-energy data using ASE."""
    data = eos_data.get_dict()
    settings = eos_settings.get_dict()
    volumes = np.array(data["volumes"], dtype=float)
    energies = np.array(data["energies"], dtype=float)
    scales = np.array(data["scales"], dtype=float)

    minimum_index = int(np.argmin(energies))
    payload = dict(data)
    payload["minimum_index"] = minimum_index
    payload["minimum_scale"] = float(scales[minimum_index])
    payload["minimum_volume"] = float(volumes[minimum_index])
    payload["minimum_energy"] = float(energies[minimum_index])
    payload["fit_success"] = False

    if len(volumes) < 3:
        payload["fit_error"] = "At least three EOS points are required for fitting."
        return orm.Dict(dict=payload)

    try:
        eos = EquationOfState(volumes, energies, eos=settings.get("fit_equation", "birchmurnaghan"))
        equilibrium_volume, equilibrium_energy, bulk_modulus = eos.fit()
    except Exception as exception:  # pylint: disable=broad-exception-caught
        payload["fit_error"] = str(exception)
        return orm.Dict(dict=payload)

    payload["fit_success"] = True
    payload["equilibrium_volume"] = float(equilibrium_volume)
    payload["equilibrium_energy"] = float(equilibrium_energy)
    payload["bulk_modulus_ev_ang3"] = float(bulk_modulus)
    payload["bulk_modulus_gpa"] = float(bulk_modulus * 160.21766208)
    payload["equilibrium_scale"] = float((equilibrium_volume / data["reference_volume"]) ** (1.0 / 3.0))
    return orm.Dict(dict=payload)
