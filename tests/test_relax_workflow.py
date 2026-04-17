from types import SimpleNamespace

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_abacus.common import RelaxType
from aiida_abacus.workflows.relax import AbacusRelaxWorkChain


def _instantiate_relax_workchain(abacus_inputs, abacus_kpoints, si_structure):
    manager = get_manager()
    runner = manager.get_runner()

    base_inputs = abacus_inputs(parameters={"input": {"calculation": "cell-relax"}})
    base_inputs.pop("kpoints", None)

    inputs = AttributeDict()
    inputs.base = AttributeDict()
    inputs.base.abacus = base_inputs
    inputs.base.kpoints = abacus_kpoints
    inputs.base.max_iterations = orm.Int(5)
    inputs.base_final_scf = AttributeDict()
    inputs.base_final_scf.abacus = abacus_inputs(parameters={"input": {"calculation": "scf"}})
    inputs.base_final_scf.abacus.pop("kpoints", None)
    inputs.base_final_scf.kpoints = abacus_kpoints
    inputs.base_final_scf.max_iterations = orm.Int(5)
    inputs.structure = si_structure

    workchain = instantiate_process(runner, AbacusRelaxWorkChain, **inputs)
    workchain.setup()
    workchain.ctx.iteration = 1
    workchain.ctx.workchains = []
    return workchain


def _instantiate_relax_workchain_with_calculation(abacus_inputs, abacus_kpoints, si_structure, calculation):
    manager = get_manager()
    runner = manager.get_runner()

    base_inputs = abacus_inputs(parameters={"input": {"calculation": calculation}})
    base_inputs.pop("kpoints", None)

    inputs = AttributeDict()
    inputs.base = AttributeDict()
    inputs.base.abacus = base_inputs
    inputs.base.kpoints = abacus_kpoints
    inputs.base.max_iterations = orm.Int(5)
    inputs.base_final_scf = AttributeDict()
    inputs.base_final_scf.abacus = abacus_inputs(parameters={"input": {"calculation": "scf"}})
    inputs.base_final_scf.abacus.pop("kpoints", None)
    inputs.base_final_scf.kpoints = abacus_kpoints
    inputs.base_final_scf.max_iterations = orm.Int(5)
    inputs.structure = si_structure
    if calculation == "cell-relax":
        inputs.relax_settings = orm.Dict(dict={"relax_type": RelaxType.POSITIONS_CELL.value})
    elif calculation == "relax":
        inputs.relax_settings = orm.Dict(dict={"relax_type": RelaxType.POSITIONS.value})
    else:
        inputs.relax_settings = orm.Dict(dict={"perform": False, "relax_type": RelaxType.NONE.value})

    workchain = instantiate_process(runner, AbacusRelaxWorkChain, **inputs)
    workchain.setup()
    return workchain


def test_inspect_relax_handles_missing_number_of_bands(
    aiida_profile_clean, abacus_inputs, abacus_kpoints, si_structure
):
    workchain = _instantiate_relax_workchain(abacus_inputs, abacus_kpoints, si_structure)
    relaxed_structure = orm.StructureData(ase=si_structure.get_ase())
    child = SimpleNamespace(
        is_excepted=False,
        is_killed=False,
        is_failed=False,
        outputs=AttributeDict(
            {
                "structure": relaxed_structure,
                "misc": orm.Dict(dict={"total_energy": -10.0}),
            }
        ),
    )
    workchain.ctx.workchains.append(child)

    result = workchain.inspect_relax()

    assert result is None
    assert workchain.ctx.current_structure == relaxed_structure
    assert workchain.ctx.current_number_of_bands is None


def test_setup_keeps_final_scf_for_variable_cell_relaxation(
    aiida_profile_clean, abacus_inputs, abacus_kpoints, si_structure
):
    workchain = _instantiate_relax_workchain_with_calculation(abacus_inputs, abacus_kpoints, si_structure, "cell-relax")

    assert "final_scf_inputs" in workchain.ctx
    assert workchain.ctx.meta_convergence is True


def test_setup_skips_final_scf_for_positions_only_relaxation(
    aiida_profile_clean, abacus_inputs, abacus_kpoints, si_structure
):
    workchain = _instantiate_relax_workchain_with_calculation(abacus_inputs, abacus_kpoints, si_structure, "relax")

    assert "final_scf_inputs" not in workchain.ctx
    assert workchain.ctx.meta_convergence is False


def test_setup_skips_final_scf_for_scf_only(
    aiida_profile_clean, abacus_inputs, abacus_kpoints, si_structure
):
    workchain = _instantiate_relax_workchain_with_calculation(abacus_inputs, abacus_kpoints, si_structure, "scf")

    assert "final_scf_inputs" not in workchain.ctx
    assert workchain.ctx.meta_convergence is False


def test_scf_only_mode_runs_single_base_workchain_and_finishes_without_structure_output(
    aiida_profile_clean, abacus_inputs, abacus_kpoints, si_structure
):
    workchain = _instantiate_relax_workchain_with_calculation(abacus_inputs, abacus_kpoints, si_structure, "scf")

    assert workchain.should_run_relax() is True

    child = SimpleNamespace(
        is_excepted=False,
        is_killed=False,
        is_failed=False,
        outputs=AttributeDict(
            {
                "misc": orm.Dict(dict={"total_energy": -10.0}),
            }
        ),
    )
    workchain.ctx.workchains.append(child)
    workchain.ctx.iteration = 1

    result = workchain.inspect_relax()

    assert result is None
    assert workchain.ctx.is_converged is True
    assert workchain.should_run_final_scf() is False

    workchain.out = lambda *args, **kwargs: None
    workchain.out_many = lambda *args, **kwargs: None
    workchain.exposed_outputs = lambda node, process_class: {"misc": node.outputs.misc}
    workchain.results()
