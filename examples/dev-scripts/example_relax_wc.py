from aiida import orm
from aiida.engine import run_get_node
from ase.build import bulk

from aiida_abacus.workflows import AbacusRelaxWorkChain

input_parameters = {
    "symmetry": 1,
    "basis_type": "pw",
    "ecutwfc": 60,
    "scf_thr": 1e-7,
    "scf_nmax": 100,
    "device": "cpu",
    "ks_solver": "dav_subspace",
    "precision": "double",
    "calculation": "cell-relax",
    "out_stru": "1",
}


def main():
    """Launch the example relax work chain."""
    si2 = bulk("Si", "diamond", 5.4)
    orm.load_computer("localhost")
    code = orm.load_code("abacus@localhost")

    builder = AbacusRelaxWorkChain.get_builder()
    builder.base.abacus.code = code
    builder.base.abacus.metadata.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 180,
    }

    structure = orm.StructureData(ase=si2)
    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh([4, 4, 4], offset=[0, 0, 0])
    builder.base.kpoints = kpoints

    pseudo_family = orm.load_group("PseudoDojo/0.4/PBE/SR/standard/upf")
    builder.base.abacus.pseudos = pseudo_family.get_pseudos(structure=structure)
    builder.base.abacus.parameters = orm.Dict({"input": input_parameters})
    builder.base.abacus.kpoints = kpoints
    builder.structure = structure

    builder.base.abacus.metadata.description = "Test job submission with the aiida_abacus plugin"
    builder.base.abacus.metadata.label = "Abacus Si2"

    final_scf_input = dict(input_parameters)
    final_scf_input["calculation"] = "scf"
    builder.base_final_scf.abacus.parameters = orm.Dict({"input": final_scf_input})
    builder.base_final_scf.abacus.code = code
    builder.base_final_scf.kpoints = kpoints
    builder.base_final_scf.abacus.metadata.options = builder.base.abacus.metadata.options
    builder.base_final_scf.abacus.pseudos = builder.base.abacus.pseudos

    results, node = run_get_node(builder)

    misc = results["misc"].get_dict()
    print(f"Miscellaneous: {misc}")
    retrieved = results["retrieved"]
    print(f"Retrieved files: {retrieved.list_object_names()}")
    remote_folder = results["remote_folder"].entry_point
    print(f"Remote folder entry_point: {remote_folder}")
    print("Calc launch over.")


if __name__ == "__main__":
    main()
