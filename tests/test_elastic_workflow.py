from types import SimpleNamespace

import numpy as np
import pytest
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

pytest.importorskip("pymatgen")

from pymatgen.analysis.elasticity.elastic import ElasticTensor, Strain

from aiida_abacus.common import RelaxType
from aiida_abacus.workflows.base import AbacusBaseWorkChain
from aiida_abacus.workflows.elastic import AbacusElasticWorkChain, fit_elastic_tensor
from aiida_abacus.workflows.relax import AbacusRelaxWorkChain


def _make_base_namespace(abacus_inputs, abacus_kpoints):
    base_inputs = abacus_inputs()
    base_inputs.pop("kpoints", None)
    parameters = base_inputs.parameters.get_dict()
    parameters.setdefault("input", {})
    parameters["input"].setdefault("calculation", "scf")
    base_inputs.parameters = orm.Dict(dict=parameters)

    namespace = AttributeDict()
    namespace.abacus = base_inputs
    namespace.kpoints = abacus_kpoints
    namespace.max_iterations = orm.Int(5)
    return namespace


def _make_relax_namespace(abacus_inputs, abacus_kpoints):
    relax = AttributeDict()
    relax.base = _make_base_namespace(abacus_inputs, abacus_kpoints)
    relax.base_final_scf = _make_base_namespace(abacus_inputs, abacus_kpoints)
    relax.relax_settings = orm.Dict(dict={"perform": True, "relax_type": RelaxType.POSITIONS_CELL.value})
    return relax


def _instantiate_elastic_workchain(abacus_inputs, abacus_kpoints, si_structure):
    manager = get_manager()
    runner = manager.get_runner()

    inputs = AttributeDict()
    inputs.base = _make_base_namespace(abacus_inputs, abacus_kpoints)
    inputs.relax = _make_relax_namespace(abacus_inputs, abacus_kpoints)
    inputs.structure = si_structure
    inputs.elastic_settings = orm.Dict(
        dict={
            "norm_strain": 0.01,
            "shear_strain": 0.01,
            "pre_relax": False,
            "relax_internal_positions": True,
            "fit_eq_stress": True,
        }
    )

    workchain = instantiate_process(runner, AbacusElasticWorkChain, **inputs)
    workchain.setup()
    return workchain


def test_elastic_builder_from_protocol(si_structure, abacus_code, pseudo_family_v2):
    builder = AbacusElasticWorkChain.get_builder_from_protocol(
        code=abacus_code,
        structure=si_structure,
        protocol="fast",
        overrides={"base": {"pseudo_family": "apns-efficiency-test"}},
        relax_type=RelaxType.POSITIONS_CELL,
    )

    assert builder.structure == si_structure
    assert builder.elastic_settings.get_dict()["pre_relax"] is False
    assert "relax" in builder
    assert builder.base.pseudo_family.value == "apns-efficiency-test"
    assert builder.relax.base.pseudo_family.value == "apns-efficiency-test"
    assert builder.relax.base_final_scf.pseudo_family.value == "apns-efficiency-test"


def test_generate_strained_structures_creates_reference_set(
    aiida_profile_clean, abacus_inputs, abacus_kpoints, si_structure
):
    workchain = _instantiate_elastic_workchain(abacus_inputs, abacus_kpoints, si_structure)

    workchain.generate_strained_structures()

    assert len(workchain.ctx.strained_structures) == 24
    assert len(workchain.ctx.strain_payload) == 24
    assert workchain.ctx.branch_labels[0] == "elastic_strain_00"


def test_run_strain_branches_use_relax_workchain(aiida_profile_clean, abacus_inputs, abacus_kpoints, si_structure):
    workchain = _instantiate_elastic_workchain(abacus_inputs, abacus_kpoints, si_structure)
    workchain.generate_strained_structures()
    captured = []

    def fake_submit(process, **inputs):
        captured.append((process, inputs))
        return SimpleNamespace(pk=500 + len(captured))

    workchain.submit = fake_submit
    workchain.run_strain_branches()

    assert len(captured) == 24
    assert all(process is AbacusRelaxWorkChain for process, _ in captured)
    assert captured[0][1]["relax_settings"].get_dict()["relax_type"] == RelaxType.POSITIONS.value
    assert captured[0][1]["base"]["abacus"]["parameters"].get_dict()["input"]["cal_force"] == 1
    assert captured[0][1]["base"]["abacus"]["parameters"].get_dict()["input"]["cal_stress"] == 1


def test_run_strain_branches_use_base_workchain_when_fixed_ion(
    aiida_profile_clean, abacus_inputs, abacus_kpoints, si_structure
):
    workchain = _instantiate_elastic_workchain(abacus_inputs, abacus_kpoints, si_structure)
    workchain.ctx.elastic_settings["relax_internal_positions"] = False
    workchain.generate_strained_structures()
    captured = []

    def fake_submit(process, **inputs):
        captured.append((process, inputs))
        return SimpleNamespace(pk=700 + len(captured))

    workchain.submit = fake_submit
    workchain.run_strain_branches()

    assert len(captured) == 24
    assert all(process is AbacusBaseWorkChain for process, _ in captured)
    assert captured[0][1]["abacus"]["parameters"].get_dict()["input"]["calculation"] == "scf"
    assert captured[0][1]["abacus"]["parameters"].get_dict()["input"]["cal_force"] == 1
    assert captured[0][1]["abacus"]["parameters"].get_dict()["input"]["cal_stress"] == 1


def test_fit_elastic_tensor_from_mocked_stresses(aiida_profile_clean):
    tensor = ElasticTensor.from_voigt(np.diag([200.0, 210.0, 220.0, 80.0, 85.0, 90.0]))
    strains = [
        Strain.from_voigt([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        Strain.from_voigt([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        Strain.from_voigt([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]),
        Strain.from_voigt([0.0, 0.0, 0.0, 0.01, 0.0, 0.0]),
        Strain.from_voigt([0.0, 0.0, 0.0, 0.0, 0.01, 0.0]),
        Strain.from_voigt([0.0, 0.0, 0.0, 0.0, 0.0, 0.01]),
    ]
    branch_payload = {
        "equilibrium": {"pk": 1, "stress_kbar": [0.0] * 9},
        "branches": [],
        "fit_with_relaxed_internal_positions": True,
    }

    for index, strain in enumerate(strains):
        stress_gpa = tensor.calculate_stress(strain)
        branch_payload["branches"].append(
            {
                "label": f"elastic_strain_{index:02d}",
                "pk": 100 + index,
                "strain": strain.as_dict(),
                "stress_kbar": (-10.0 * stress_gpa).reshape(-1).tolist(),
            }
        )

    result = fit_elastic_tensor(
        orm.Dict(dict=branch_payload),
        orm.Dict(dict={"fit_eq_stress": True}),
    )

    assert result.get_dict()["fit_success"] is True
    assert len(result.get_dict()["elastic_tensor"]) == 6
    assert result.get_dict()["fit_with_relaxed_internal_positions"] is True
