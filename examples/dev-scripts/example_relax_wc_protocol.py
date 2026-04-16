from aiida import orm
from aiida.engine import run_get_node
from ase.build import bulk

from aiida_abacus.workflows import AbacusRelaxWorkChain


def main():
    """Launch the protocol-based relax work chain example."""
    si2 = bulk("Si", "diamond", 5.4)
    structure = orm.StructureData(ase=si2)
    orm.load_computer("localhost")
    code = orm.load_code("abacus@localhost")

    builder = AbacusRelaxWorkChain.get_builder_from_protocol(code, structure, protocol="fast")
    builder.base.abacus.metadata.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 180,
    }
    builder.base_final_scf.abacus.metadata.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 180,
    }

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
