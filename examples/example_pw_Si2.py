"""Launch a calculation using the 'aiida-abacus' plugin."""

import numpy as np
from aiida import engine, orm
from aiida.common.exceptions import NotExistent
from aiida.orm import Dict, KpointsData, StructureData, load_group

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
    """Launch the example calculation."""
    computer = orm.load_computer("localhost")

    try:
        code = orm.load_code("abacus@localhost")
    except NotExistent:
        code = orm.InstalledCode(
            label="abacus", computer=computer, filepath_executable="abacus", default_calc_job_plugin="abacus.abacus"
        )

    builder = code.get_builder()
    builder.metadata.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 180,
    }

    lattice_vectors_fractional = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    atomic_positions_fractional = [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]]
    structure = StructureData(cell=lattice_vectors_fractional)
    for pos in atomic_positions_fractional:
        structure.append_atom(position=pos, symbols="Si")

    stru_settings = {
        "LATTICE_CONSTANT": 10.2,
        "m": [[False, False, False], [True, True, True]],
    }

    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([4, 4, 4], offset=[0, 0, 0])

    pseudo_family = load_group("PseudoDojo/0.4/PBE/SR/standard/upf")
    builder.pseudos = pseudo_family.get_pseudos(structure=structure)

    all_parameters = {
        "input": input_parameters,
        "stru": stru_settings,
    }

    builder.structure = structure
    builder.kpoints = kpoints
    builder.settings = stru_settings
    builder.metadata.description = "Simple pw_Si2 job submission with the aiida_abacus plugin"

    parameters = Dict(dict=all_parameters)
    results, node = engine.run.get_node(builder, parameters=parameters)
    misc = results["misc"].get_dict()
    print(f"Miscellaneous: {misc}")
    print(f"Total energy is: {misc['total_energy']} eV")
    retrieved = results["retrieved"]
    print(f"Retrieved files: {retrieved.list_object_names()}")
    remote_folder_path = results["remote_folder"].get_remote_path()
    print(f"Remote folder path: {remote_folder_path}")
    print("Calculation over.")
    print(
        """
You can use
    `verdi process list -a`
to see the status of the calculation."""
    )


if __name__ == "__main__":
    main()
