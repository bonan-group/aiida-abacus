from aiida import orm
from aiida.engine import run_get_node
from ase.build import bulk

input_parameters = {
    "symmetry": 1,
    "basis_type": "pw",
    "ecutwfc": 60,
    "scf_thr": 1e-7,
    "scf_nmax": 100,
    "device": "cpu",
    "ks_solver": "dav_subspace",
    "precision": "double",
}


def main():
    """Launch the example SCF calculation."""
    si2 = bulk("Si", "diamond", 5.4)
    orm.load_computer("localhost")
    code = orm.load_code("abacus@localhost")

    builder = code.get_builder()
    builder.metadata.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 180,
    }

    structure = orm.StructureData(ase=si2)
    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh([4, 4, 4], offset=[0, 0, 0])

    pseudo_family = orm.load_group("PseudoDojo/0.4/PBE/SR/standard/upf")
    builder.pseudos = pseudo_family.get_pseudos(structure=structure)

    builder.parameters = orm.Dict({"input": input_parameters})
    builder.kpoints = kpoints
    builder.structure = structure
    builder.metadata.description = "Test job submission with the aiida_abacus plugin"
    builder.metadata.label = "Abacus Si2"

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
