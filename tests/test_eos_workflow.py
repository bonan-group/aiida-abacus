from types import SimpleNamespace

import pytest
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_abacus.common import RelaxType
from aiida_abacus.workflows.eos import AbacusEosWorkChain, fit_eos


def _instantiate_eos_workchain(abacus_inputs, abacus_kpoints):
    manager = get_manager()
    runner = manager.get_runner()
    base_inputs = abacus_inputs()
    base_inputs.pop("kpoints", None)

    inputs = AttributeDict()
    inputs.base = AttributeDict()
    inputs.base.abacus = base_inputs
    inputs.base.kpoints = abacus_kpoints
    inputs.structure = base_inputs.structure
    inputs.eos_settings = orm.Dict(
        dict={
            "scale_start": 0.96,
            "scale_end": 1.04,
            "scale_step": 0.02,
            "fit_equation": "birchmurnaghan",
        }
    )

    workchain = instantiate_process(runner, AbacusEosWorkChain, **inputs)
    workchain.setup()
    return workchain


def test_eos_builder_from_protocol(si_structure, abacus_code, pseudo_family_v2):
    builder = AbacusEosWorkChain.get_builder_from_protocol(
        code=abacus_code,
        structure=si_structure,
        protocol="fast",
        overrides={"base": {"pseudo_family": "apns-efficiency-test"}},
        relax_type=RelaxType.NONE,
    )

    assert builder.base.abacus.code == abacus_code
    assert builder.structure == si_structure
    assert builder.eos_settings.get_dict()["scale_start"] == 0.96


def test_generate_structures_scales_reference_structure(aiida_profile_clean, abacus_inputs, abacus_kpoints):
    workchain = _instantiate_eos_workchain(abacus_inputs, abacus_kpoints)

    workchain.generate_structures()

    assert workchain.ctx.scale_factors == [0.96, 0.98, 1.0, 1.02, 1.04]
    reference_volume = workchain.ctx.structure.get_cell_volume()
    assert len(workchain.ctx.scaled_structures) == 5
    assert workchain.ctx.scaled_structures[0].get_cell_volume() == pytest.approx(reference_volume * 0.96**3)
    assert workchain.ctx.scaled_structures[2].get_cell_volume() == pytest.approx(reference_volume)


def test_run_eos_submits_all_scaled_structures(aiida_profile_clean, abacus_inputs, abacus_kpoints):
    workchain = _instantiate_eos_workchain(abacus_inputs, abacus_kpoints)
    workchain.generate_structures()
    captured = []

    def fake_submit(process, **inputs):
        captured.append((process, inputs))
        return SimpleNamespace(pk=200 + len(captured))

    workchain.submit = fake_submit
    workchain.run_eos()

    assert len(captured) == 5
    assert all(process is not None for process, _ in captured)
    assert captured[0][1]["metadata"]["call_link_label"] == "eos_workchain_00"
    vol_last = captured[-1][1]["abacus"]["structure"].get_cell_volume()
    vol_first = captured[0][1]["abacus"]["structure"].get_cell_volume()
    assert vol_last > vol_first


def test_inspect_eos_collects_branch_results(aiida_profile_clean, abacus_inputs, abacus_kpoints):
    workchain = _instantiate_eos_workchain(abacus_inputs, abacus_kpoints)
    workchain.generate_structures()

    for index, (label, scale, structure) in enumerate(
        zip(workchain.ctx.scale_labels, workchain.ctx.scale_factors, workchain.ctx.scaled_structures)
    ):
        energy = (scale - 1.01) ** 2 - 10.0
        setattr(
            workchain.ctx,
            label,
            SimpleNamespace(
                pk=500 + index,
                is_finished_ok=True,
                outputs=AttributeDict(
                    {"misc": orm.Dict(dict={"volume": structure.get_cell_volume(), "total_energy": energy})}
                ),
            ),
        )

    result = workchain.inspect_eos()

    assert result is None
    eos_parameters = workchain.outputs["eos_parameters"].get_dict()
    assert eos_parameters["fit_success"] is True
    assert len(eos_parameters["samples"]) == 5
    assert eos_parameters["minimum_scale"] == pytest.approx(1.0, abs=0.03)


def test_inspect_eos_falls_back_to_structure_volume(aiida_profile_clean, abacus_inputs, abacus_kpoints):
    workchain = _instantiate_eos_workchain(abacus_inputs, abacus_kpoints)
    workchain.generate_structures()

    for index, (label, scale, structure) in enumerate(
        zip(workchain.ctx.scale_labels, workchain.ctx.scale_factors, workchain.ctx.scaled_structures)
    ):
        energy = (scale - 1.0) ** 2 - 5.0
        setattr(
            workchain.ctx,
            label,
            SimpleNamespace(
                pk=700 + index,
                is_finished_ok=True,
                outputs=AttributeDict({"misc": orm.Dict(dict={"energy_ks": energy})}),
            ),
        )

    result = workchain.inspect_eos()

    assert result is None
    eos_parameters = workchain.outputs["eos_parameters"].get_dict()
    assert eos_parameters["fit_success"] is True
    assert eos_parameters["samples"][0]["volume"] == pytest.approx(workchain.ctx.scaled_structures[0].get_cell_volume())


def test_inspect_eos_uses_energy_ks_in_ev(aiida_profile_clean, abacus_inputs, abacus_kpoints):
    workchain = _instantiate_eos_workchain(abacus_inputs, abacus_kpoints)
    workchain.generate_structures()

    total_energy_ev = -8.0

    for index, label in enumerate(workchain.ctx.scale_labels):
        setattr(
            workchain.ctx,
            label,
            SimpleNamespace(
                pk=800 + index,
                is_finished_ok=True,
                outputs=AttributeDict({"misc": orm.Dict(dict={"energy_ks": total_energy_ev})}),
            ),
        )

    result = workchain.inspect_eos()

    assert result is None
    eos_parameters = workchain.outputs["eos_parameters"].get_dict()
    assert eos_parameters["samples"][0]["energy"] == pytest.approx(total_energy_ev)


def test_fit_eos_reports_insufficient_points(aiida_profile_clean):
    result = fit_eos(
        orm.Dict(
            dict={
                "reference_volume": 10.0,
                "fit_equation": "birchmurnaghan",
                "samples": [],
                "scales": [0.98, 1.0],
                "volumes": [9.4, 10.0],
                "energies": [-10.0, -10.1],
            }
        ),
        orm.Dict(dict={"fit_equation": "birchmurnaghan"}),
    )

    assert result.get_dict()["fit_success"] is False
    assert "fit_error" in result.get_dict()
