from types import SimpleNamespace

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_abacus.common import RelaxType
from aiida_abacus.workflows.dos import AbacusDosWorkChain


def _instantiate_dos_workchain(abacus_inputs, abacus_kpoints):
    manager = get_manager()
    runner = manager.get_runner()
    base_inputs = abacus_inputs()
    base_inputs.pop("kpoints", None)

    inputs = AttributeDict()
    inputs.base = AttributeDict()
    inputs.base.abacus = base_inputs
    inputs.base.kpoints = abacus_kpoints
    inputs.structure = base_inputs.structure
    inputs.dos_settings = orm.Dict({"dos_kpoints_distance": 0.15, "include_pdos": True})

    workchain = instantiate_process(runner, AbacusDosWorkChain, **inputs)
    workchain.setup()
    return workchain


def test_dos_builder_from_protocol(si_structure, abacus_code, pseudo_family_v2):
    builder = AbacusDosWorkChain.get_builder_from_protocol(
        code=abacus_code,
        structure=si_structure,
        protocol="fast",
        overrides={"base": {"pseudo_family": "apns-efficiency-test"}},
        relax_type=RelaxType.NONE,
    )

    assert builder.base.abacus.code == abacus_code
    assert builder.structure == si_structure
    assert builder.dos_settings.get_dict()["include_pdos"] is True


def test_run_dos_configures_nscf_inputs(aiida_profile_clean, abacus_inputs, abacus_kpoints):
    workchain = _instantiate_dos_workchain(abacus_inputs, abacus_kpoints)
    workchain.ctx.restart_folder = SimpleNamespace()
    captured = {}

    def fake_submit(process, **inputs):
        captured["process"] = process
        captured["inputs"] = inputs
        return SimpleNamespace(pk=321)

    workchain.submit = fake_submit
    workchain.run_dos()

    dos_inputs = captured["inputs"]
    params = dos_inputs["abacus"]["parameters"].get_dict()["input"]

    assert captured["process"] is not None
    assert params["calculation"] == "nscf"
    assert params["init_chg"] == "file"
    assert params["out_chg"] == 0
    assert params["out_dos"] == 1
    assert params["out_pdos"] == 1
    assert dos_inputs["abacus"]["settings"].get_dict()["include_pdos"] is True
    assert "kpoints" not in dos_inputs
    assert dos_inputs["kpoints_distance"].value == 0.15


def test_verify_dos_exposes_outputs(aiida_profile_clean, abacus_inputs, abacus_kpoints):
    workchain = _instantiate_dos_workchain(abacus_inputs, abacus_kpoints)
    pdos = orm.Dict({"energy": [0.0], "nspin": 1, "orbitals": []})
    misc = orm.Dict({"fermi_level": 1.2})
    workchain.ctx.dos_workchain = SimpleNamespace(
        is_finished_ok=True,
        outputs=AttributeDict({"misc": misc, "pdos": pdos}),
    )

    result = workchain.verify_dos()

    assert result is None
    assert workchain.outputs["dos_parameters"].get_dict()["fermi_level"] == 1.2
    assert workchain.outputs["pdos"].get_dict()["nspin"] == 1
