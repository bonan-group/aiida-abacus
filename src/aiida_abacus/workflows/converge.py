"""Workflow for convergence testing of plane-wave cutoff and k-point spacing."""

from copy import deepcopy

import numpy as np
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, append_, calcfunction

from aiida_abacus.common import ProtocolMixin, prepare_process_inputs
from aiida_abacus.common.opthold import ConvOptions

from .base import AbacusBaseWorkChain


class AbacusConvergenceWorkChain(ProtocolMixin, WorkChain):
    """
    A workchain to perform convergence tests for plane-wave cutoff energy and k-point spacing.

    Submits parallel ``AbacusBaseWorkChain`` calculations:
    - Cutoff convergence: varies ``ecutwfc`` at fixed k-point spacing
    - K-point convergence: varies k-point spacing at fixed ``ecutwfc``

    The ``conv_settings`` input controls the ranges and steps.
    """

    _protocol_tag = "conv"
    _sub_workchain = AbacusBaseWorkChain

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(
            AbacusBaseWorkChain,
            namespace="base",
            exclude=("clean_workdir", "abacus.structure", "abacus.parent_folder"),
            namespace_options={"help": "Inputs for the `AbacusBaseWorkChain` convergence calculations."},
        )
        spec.input("structure", valid_type=orm.StructureData, help="The input structure.")
        spec.input(
            "conv_settings",
            help=ConvOptions.aiida_description(),
            valid_type=orm.Dict,
            validator=ConvOptions.aiida_validate,
            serializer=ConvOptions.aiida_serialize,
        )
        spec.outline(cls.setup, cls.launch_conv_calcs, cls.analyse)
        spec.exit_code(
            401,
            "ERROR_SUBWORKFLOW_ERRORED",
            message="At least one of the launched sub-workchains has failed.",
        )
        spec.output(
            "cutoff_conv_data",
            valid_type=orm.Dict,
            required=False,
            help="Summary of cutoff energy convergence results.",
        )
        spec.output(
            "kpoints_conv_data",
            valid_type=orm.Dict,
            required=False,
            help="Summary of k-point spacing convergence results.",
        )

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        structure,
        protocol=None,
        overrides=None,
        options=None,
        **kwargs,
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
        builder.conv_settings = orm.Dict(dict=inputs.get("conv_settings", {}))
        return builder

    def setup(self):
        """Generate the cutoff energy and k-point spacing lists from convergence settings."""
        settings = self.inputs.conv_settings.get_dict()
        self.ctx.settings = settings

        # Plane-wave cutoff energies in Ry (start < stop, increment)
        start = settings["cutoff_start"]
        stop = settings["cutoff_stop"]
        if start < stop:
            cutoff_list = [start]
            cut = start
            while True:
                cut += settings["cutoff_step"]
                if cut < stop:
                    cutoff_list.append(cut)
                else:
                    cutoff_list.append(stop)
                    break
        else:
            cutoff_list = []

        # K-point spacing in 1/Angstrom (start > stop, decrement from coarse to fine)
        start = settings["kspacing_start"]
        stop = settings["kspacing_stop"]
        if start > stop:
            spacing = start
            kspacing_list = [spacing]
            while True:
                spacing -= settings["kspacing_step"]
                if spacing > stop:
                    kspacing_list.append(spacing)
                else:
                    kspacing_list.append(stop)
                    break
        else:
            kspacing_list = []

        self.ctx.cutoff_list = cutoff_list
        self.ctx.kspacing_list = kspacing_list

    def launch_conv_calcs(self):
        """Submit parallel convergence calculations."""
        # Resolve default fixed values
        if not self.ctx.cutoff_list:
            cut_k = 60.0
        else:
            cut_k = min(self.ctx.cutoff_list)
        if not self.ctx.kspacing_list:
            k_cut = 0.15
        else:
            k_cut = min(self.ctx.kspacing_list)

        cutoff_for_kconv = self.ctx.settings.get("cutoff_kconv", cut_k)
        kspacing_for_cutoffconv = self.ctx.settings.get("kspacing_cutconv", k_cut)

        base_inputs = AttributeDict(self.exposed_inputs(AbacusBaseWorkChain, "base"))
        original_label = base_inputs.metadata.get("label", "")

        # Cutoff energy convergence tests: fix k-point spacing, vary ecutwfc
        for idx, cut in enumerate(self.ctx.cutoff_list):
            label = f"cutconv_{idx:02d}" if not original_label else f"{original_label}_cutconv_{idx:02d}"
            inputs = _build_conv_branch_inputs(
                base_inputs,
                structure=self.inputs.structure,
                ecutwfc=cut,
                kpoints_distance=orm.Float(kspacing_for_cutoffconv),
                label=label,
            )
            prepared = prepare_process_inputs(AbacusBaseWorkChain, inputs)
            running = self.submit(AbacusBaseWorkChain, **prepared)
            self.report(f"Submitted {running} with cutoff energy {cut:.1f} Ry.")
            self.to_context(cutoff_conv_workchains=append_(running))

        # K-point spacing convergence tests: fix ecutwfc, vary kpoints_distance
        for idx, kspacing in enumerate(self.ctx.kspacing_list):
            label = f"kconv_{idx:02d}" if not original_label else f"{original_label}_kconv_{idx:02d}"
            inputs = _build_conv_branch_inputs(
                base_inputs,
                structure=self.inputs.structure,
                ecutwfc=cutoff_for_kconv,
                kpoints_distance=orm.Float(kspacing),
                label=label,
            )
            prepared = prepare_process_inputs(AbacusBaseWorkChain, inputs)
            running = self.submit(AbacusBaseWorkChain, **prepared)
            self.report(f"Submitted {running} with k-point spacing {kspacing:.3f} 1/Angstrom.")
            self.to_context(kpoints_conv_workchains=append_(running))

    def analyse(self):
        """Collect and summarize the convergence results from all sub-workchains."""

        def get_maximum(values):
            """Get the maximum norm of a list of vectors."""
            if values is None:
                return None
            return float(np.amax(np.linalg.norm(values, axis=1)))

        def collect_data(workchain):
            """Collect convergence data from a workchain output."""
            output = workchain.outputs.misc.get_dict()
            data = {}
            data["energy"] = output.get("total_energy")
            if output.get("forces") is not None:
                data["maximum_force"] = get_maximum(output["forces"])
            if output.get("stress") is not None:
                data["maximum_stress"] = get_maximum(output["stress"])
            data["volume"] = output.get("volume")
            return data

        def unpack(name, input_data):
            """Unpack a dict with numeric keys into parallel lists."""
            output_dict = {name: []}
            for key, data in sorted(input_data.items()):
                output_dict[name].append(key)
                for key_, value in data.items():
                    if key_ not in output_dict:
                        output_dict[key_] = []
                    output_dict[key_].append(value)
            return output_dict

        exit_code = None

        # Collect cutoff convergence data
        cutoff_data = {}
        cutoff_miscs = {}
        if "cutoff_conv_workchains" in self.ctx:
            for iwork, workchain in enumerate(self.ctx.cutoff_conv_workchains):
                if workchain.exit_status != 0:
                    exit_code = self.exit_codes.ERROR_SUBWORKFLOW_ERRORED
                    self.report(f"Skipping workchain {workchain} with exit status {workchain.exit_status}")
                    continue
                params = workchain.inputs.abacus.parameters
                if isinstance(params, orm.Dict):
                    params = params.get_dict()
                cutoff = params["input"]["ecutwfc"]
                cutoff_data[cutoff] = collect_data(workchain)
                cutoff_data[cutoff]["mesh"] = workchain.called[0].inputs.kpoints.get_kpoints_mesh()[0]
                cutoff_miscs[f"workchain_{iwork}"] = workchain.outputs.misc

        # Collect k-point spacing convergence data
        kspacing_data = {}
        kspacing_miscs = {}
        if "kpoints_conv_workchains" in self.ctx:
            for iwork, workchain in enumerate(self.ctx.kpoints_conv_workchains):
                if workchain.exit_status != 0:
                    exit_code = self.exit_codes.ERROR_SUBWORKFLOW_ERRORED
                    self.report(f"Skipping workchain {workchain} with exit status {workchain.exit_status}")
                    continue
                spacing = float(workchain.inputs.kpoints_distance)
                kspacing_data[spacing] = collect_data(workchain)
                kspacing_data[spacing]["mesh"] = workchain.called[0].inputs.kpoints.get_kpoints_mesh()[0]
                params = workchain.inputs.abacus.parameters
                if isinstance(params, orm.Dict):
                    params = params.get_dict()
                kspacing_data[spacing]["cutoff_energy"] = params["input"]["ecutwfc"]
                kspacing_miscs[f"workchain_{iwork}"] = workchain.outputs.misc

        @calcfunction
        def create_links_cutconv(**miscs):
            """Create output Dict with provenance links to misc nodes."""
            return orm.Dict(dict=unpack("cutoff_energy", cutoff_data))

        @calcfunction
        def create_links_kconv(**miscs):
            """Create output Dict with provenance links to misc nodes."""
            return orm.Dict(dict=unpack("kpoints_spacing", kspacing_data))

        if cutoff_data:
            self.out("cutoff_conv_data", create_links_cutconv(**cutoff_miscs))
        if kspacing_data:
            self.out("kpoints_conv_data", create_links_kconv(**kspacing_miscs))

        return exit_code


def _build_conv_branch_inputs(base_inputs, structure, ecutwfc, kpoints_distance, label):
    """Build inputs for a single convergence branch of ``AbacusBaseWorkChain``.

    :param base_inputs: AttributeDict of exposed base inputs (template)
    :param structure: StructureData for the calculation
    :param ecutwfc: cutoff energy in Ry to set in parameters
    :param kpoints_distance: Float node with k-point spacing in 1/Angstrom
    :param label: call_link_label string
    :return: AttributeDict of inputs ready for ``prepare_process_inputs``
    """
    inputs = AttributeDict()
    inputs.metadata = AttributeDict(deepcopy(dict(base_inputs.metadata)))
    inputs.metadata.call_link_label = label
    inputs.max_iterations = base_inputs.max_iterations

    inputs.abacus = AttributeDict()
    inputs.abacus.code = base_inputs.abacus.code
    inputs.abacus.metadata = AttributeDict(deepcopy(dict(base_inputs.abacus.metadata)))
    # Deep-copy the parameters dict and override ecutwfc
    params = deepcopy(base_inputs.abacus.parameters.get_dict())
    params.setdefault("input", {})["ecutwfc"] = ecutwfc
    inputs.abacus.parameters = params
    inputs.abacus.pseudos = base_inputs.abacus.pseudos
    inputs.abacus.structure = structure

    if "settings" in base_inputs.abacus:
        settings = base_inputs.abacus.settings
        inputs.abacus.settings = settings.get_dict() if hasattr(settings, "get_dict") else dict(settings)

    if "kpoints" in base_inputs:
        inputs.kpoints = base_inputs.kpoints
    # Always set kpoints_distance for convergence tests
    inputs.kpoints_distance = kpoints_distance

    if "kpoints_force_parity" in base_inputs:
        inputs.kpoints_force_parity = base_inputs.kpoints_force_parity
    if "pseudo_family" in base_inputs:
        inputs.pseudo_family = base_inputs.pseudo_family
    return inputs


def get_conv_data(conv_work):
    """Extract convergence data as pandas DataFrames.

    :param conv_work: WorkChainNode of a completed ``AbacusConvergenceWorkChain``
    :returns: tuple of (cutoff_dataframe, kpoints_dataframe), either may be None
    """
    import pandas as pd

    if "cutoff_conv_data" in conv_work.outputs:
        cutdf = pd.DataFrame(conv_work.outputs.cutoff_conv_data.get_dict())
        cutdf["energy_per_atom"] = cutdf["energy"] / len(conv_work.inputs.structure.sites)
        cutdf["dE_per_atom"] = cutdf["energy_per_atom"] - cutdf["energy_per_atom"].iloc[-1]
    else:
        cutdf = None

    if "kpoints_conv_data" in conv_work.outputs:
        kdf = pd.DataFrame(conv_work.outputs.kpoints_conv_data.get_dict())
        kdf["energy_per_atom"] = kdf["energy"] / len(conv_work.inputs.structure.sites)
        kdf["dE_per_atom"] = kdf["energy_per_atom"] - kdf["energy_per_atom"].iloc[-1]
    else:
        kdf = None

    return cutdf, kdf


def plot_conv_data(cdf, kdf, **kwargs):
    """Make combined plots for the convergence test results."""
    import matplotlib.pyplot as plt

    figs = []
    if cdf is not None:
        fig, axs = plt.subplots(3, 1, sharex=True, **kwargs)
        figs.append(fig)
        axs[0].plot(cdf.cutoff_energy, cdf.dE_per_atom, "-x")
        axs[0].set_ylabel("dE (Ry / atom)")
        i = 0
        if "maximum_force" in cdf.columns:
            i += 1
            axs[i].plot(cdf.cutoff_energy, cdf.maximum_force, "-x")
            axs[i].set_ylabel(r"$F_{max}$ (Ry/Bohr)")
        if "maximum_stress" in cdf.columns:
            i += 1
            axs[i].plot(cdf.cutoff_energy, cdf.maximum_stress, "-x")
            axs[i].set_ylabel(r"$S_{max}$ (kBar)")
        axs[i].set_xlabel("Cut-off energy (Ry)")
        fig.tight_layout()

    if kdf is not None:
        fig, axs = plt.subplots(3, 1, sharex=True, **kwargs)
        figs.append(fig)
        axs[0].plot(kdf.kpoints_spacing, kdf.dE_per_atom, "-x")
        axs[0].set_ylabel("dE (Ry / atom)")
        i = 0
        if "maximum_force" in kdf.columns:
            i += 1
            axs[i].plot(kdf.kpoints_spacing, kdf.maximum_force, "-x")
            axs[i].set_ylabel(r"$F_{max}$ (Ry/Bohr)")
        if "maximum_stress" in kdf.columns:
            i += 1
            axs[i].plot(kdf.kpoints_spacing, kdf.maximum_stress, "-x")
            axs[i].set_ylabel(r"$S_{max}$ (kBar)")
        axs[i].set_xticks(kdf.kpoints_spacing)
        axs[i].set_xticklabels(
            [f"{row.kpoints_spacing:.3f}\n{row.mesh}" for _, row in kdf.iterrows()],
            rotation=45,
        )
        axs[i].set_xlabel("K-point spacing (mesh)")
        fig.tight_layout()

    return figs
