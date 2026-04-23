"""Launch a convergence work chain using a preconfigured code and pseudo family.

Assumptions:
1. A pseudo family (e.g. `apns-efficiency-v1`) is available in the current AiiDA profile.
2. The `abacus@localhost` code is available in the current AiiDA profile.

Run with:
    AIIDA_PATH=$(pwd) uv run python examples/dev-scripts/example_converge_wc.py
"""

from aiida import orm
from aiida.engine import run_get_node
from ase.build import bulk

from aiida_abacus.workflows.converge import AbacusConvergenceWorkChain, get_conv_data


def main():
    """Launch the convergence work chain example."""
    structure = orm.StructureData(ase=bulk("Si", "diamond", 5.43))
    code = orm.load_code("abacus@localhost")

    builder = AbacusConvergenceWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        protocol="fast",
        overrides={"base": {"pseudo_family": "apns-efficiency-v1"}},
    )
    builder.base.abacus.metadata.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 600,
    }
    builder.conv_settings = orm.Dict(
        dict={
            "cutoff_start": 30.0,
            "cutoff_stop": 70.0,
            "cutoff_step": 10.0,
            "kspacing_start": 0.40,
            "kspacing_stop": 0.15,
            "kspacing_step": 0.10,
            "cutoff_kconv": 50.0,
            "kspacing_cutconv": 0.20,
        }
    )

    _results, node = run_get_node(builder)

    print(f"Convergence work chain pk: {node.pk}")
    print(f"Finished OK: {node.is_finished_ok}")

    cdf, kdf = get_conv_data(node)

    if cdf is not None:
        print("\nCutoff energy convergence:")
        print(cdf.to_string(index=False))

    if kdf is not None:
        print("\nK-point spacing convergence:")
        print(kdf.to_string(index=False))


if __name__ == "__main__":
    main()
