"""Workflow for performing density of states calculations."""

import pathlib

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import ToContext, WorkChain, if_

from aiida_abacus.common import ProtocolMixin, RelaxType, prepare_process_inputs
from aiida_abacus.common.opthold import DosOptions

from .base import AbacusBaseWorkChain
from .relax import AbacusRelaxWorkChain


class AbacusDosWorkChain(ProtocolMixin, WorkChain):
    """Workflow for performing DOS and PDOS calculations."""

    _protocol_tag = "dos"

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(
            AbacusBaseWorkChain,
            namespace="base",
            exclude=("clean_workdir", "abacus.structure", "abacus.parent_folder"),
            namespace_options={"help": "Inputs for the SCF `AbacusBaseWorkChain`."},
        )
        spec.expose_inputs(
            AbacusRelaxWorkChain,
            namespace="relax",
            exclude=("structure",),
            namespace_options={
                "help": "Inputs for the optional `AbacusRelaxWorkChain`.",
                "required": False,
                "populate_defaults": False,
            },
        )
        spec.input("structure", valid_type=orm.StructureData, help="The input structure.")
        spec.input(
            "dos_settings",
            help=DosOptions.aiida_description(),
            valid_type=orm.Dict,
            validator=DosOptions.aiida_validate,
            serializer=DosOptions.aiida_serialize,
        )
        spec.outline(
            cls.setup,
            if_(cls.should_do_relax)(
                cls.run_relax,
                cls.verify_relax,
            ),
            cls.run_scf,
            cls.verify_scf,
            cls.run_dos,
            cls.verify_dos,
        )
        spec.output("structure", valid_type=orm.StructureData, help="Final structure used for the DOS calculation.")
        spec.output("scf_parameters", valid_type=orm.Dict, help="Parsed SCF misc output.")
        spec.output("dos_parameters", valid_type=orm.Dict, help="Parsed DOS misc output.")
        spec.output("pdos", valid_type=orm.Dict, required=False, help="Projected density of states data.")
        spec.exit_code(611, "ERROR_SCF_PROCESS_FAILED", message="The SCF calculation failed.")
        spec.exit_code(612, "ERROR_DOS_PROCESS_FAILED", message="The DOS calculation failed.")
        spec.exit_code(613, "ERROR_RELAX_PROCESS_FAILED", message="The relaxation calculation failed.")

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
        builder.dos_settings = orm.Dict(dict=inputs.get("dos_settings", {}))

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
        self.ctx.scf_inputs = AttributeDict(self.exposed_inputs(AbacusBaseWorkChain, "base"))
        self.ctx.relax_inputs = (
            AttributeDict(self.exposed_inputs(AbacusRelaxWorkChain, "relax")) if "relax" in self.inputs else None
        )
        self.ctx.structure = self.inputs.structure
        self.ctx.dos_settings = self.inputs.dos_settings.get_dict()
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
        self.report(f"launching AbacusRelaxWorkChain<{running.pk}> for DOS pre-relax")
        return ToContext(relax_workchain=running)

    def verify_relax(self):
        """Verify the relax workflow."""
        if not self.ctx.relax_workchain.is_finished_ok:
            return self.exit_codes.ERROR_RELAX_PROCESS_FAILED
        self.ctx.structure = self.ctx.relax_workchain.outputs.structure
        self.out("structure", self.ctx.structure)

    def run_scf(self):
        """Run the SCF calculation that seeds the DOS NSCF step."""
        inputs = self.ctx.scf_inputs
        inputs.abacus.structure = self.ctx.structure
        parameters = inputs.abacus.parameters.get_dict()
        parameters["input"]["out_chg"] = 1
        inputs.abacus.parameters = orm.Dict(parameters)
        prepared = prepare_process_inputs(AbacusBaseWorkChain, inputs)
        running = self.submit(AbacusBaseWorkChain, **prepared)
        self.report(f"launching AbacusBaseWorkChain<{running.pk}> for DOS SCF")
        return ToContext(scf_workchain=running)

    def verify_scf(self):
        """Verify the SCF calculation and expose its parameters."""
        if not self.ctx.scf_workchain.is_finished_ok:
            return self.exit_codes.ERROR_SCF_PROCESS_FAILED
        self.ctx.restart_folder = self.ctx.scf_workchain.outputs.remote_folder
        self.out("scf_parameters", self.ctx.scf_workchain.outputs.misc)

    def run_dos(self):
        """Run the DOS NSCF step."""
        inputs = self.ctx.scf_inputs
        inputs.abacus.structure = self.ctx.structure
        parameters = inputs.abacus.parameters.get_dict()
        parameters["input"]["calculation"] = "nscf"
        parameters["input"]["init_chg"] = "file"
        parameters["input"]["out_chg"] = 0
        parameters["input"]["out_dos"] = 1
        if self.ctx.dos_settings.get("include_pdos", True):
            parameters["input"]["out_pdos"] = 1
        inputs.abacus.parameters = parameters
        inputs.abacus.restart_folder = self.ctx.restart_folder
        inputs.abacus.settings = inputs.abacus.settings.get_dict() if "settings" in inputs.abacus else {}
        if self.ctx.dos_settings.get("include_pdos", True):
            inputs.abacus.settings["include_pdos"] = True
        if "kpoints" in inputs:
            del inputs["kpoints"]
        inputs.kpoints_distance = orm.Float(self.ctx.dos_settings["dos_kpoints_distance"])
        prepared = prepare_process_inputs(AbacusBaseWorkChain, inputs)
        running = self.submit(AbacusBaseWorkChain, **prepared)
        self.report(f"launching AbacusBaseWorkChain<{running.pk}> for DOS NSCF")
        return ToContext(dos_workchain=running)

    def verify_dos(self):
        """Verify the DOS calculation and expose outputs."""
        if not self.ctx.dos_workchain.is_finished_ok:
            return self.exit_codes.ERROR_DOS_PROCESS_FAILED
        self.out("dos_parameters", self.ctx.dos_workchain.outputs.misc)
        if "pdos" in self.ctx.dos_workchain.outputs:
            self.out("pdos", self.ctx.dos_workchain.outputs.pdos)
