"""Launch a minimal elastic work chain example.

Assumptions:
1. The `apns-efficiency-v1` pseudo family is available in the current AiiDA profile.
2. The `abacus@localhost` code is available in the current AiiDA profile.

Run with:
    AIIDA_PATH=$(pwd) uv run python examples/dev-scripts/example_elastic_wc.py
"""

from aiida import orm
from aiida.engine import run_get_node
from ase.build import bulk

from aiida_abacus.common import RelaxType
from aiida_abacus.workflows.elastic import AbacusElasticWorkChain


PSEUDO_FAMILY = "apns-efficiency-v1"


def _set_branch_options(builder, max_wallclock_seconds=600):
    """Apply compact localhost options to all nested subworkflows."""
    options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": max_wallclock_seconds,
    }

    if "base" in builder:
        builder.base.abacus.metadata.options = options

    if "relax" in builder:
        builder.relax.base.abacus.metadata.options = options
        builder.relax.base_final_scf.abacus.metadata.options = options


def main():
    """Launch the elastic workflow on diamond silicon."""
    structure = orm.StructureData(ase=bulk("Si", "diamond", 5.43))
    code = orm.load_code("abacus@localhost")

    builder = AbacusElasticWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        protocol="fast",
        relax_type=RelaxType.POSITIONS_CELL,
        overrides={
            "base": {"pseudo_family": PSEUDO_FAMILY},
            "relax": {
                "base": {"pseudo_family": PSEUDO_FAMILY},
                "base_final_scf": {"pseudo_family": PSEUDO_FAMILY},
            },
        },
    )
    _set_branch_options(builder)
    builder.elastic_settings = orm.Dict(
        dict={
            "norm_strain": 0.012,
            "shear_strain": 0.012,
            "pre_relax": True,
            "relax_internal_positions": True,
            "fit_eq_stress": True,
        }
    )

    results, node = run_get_node(builder)

    print(f"Elastic work chain pk: {node.pk}")
    print(f"Finished OK: {node.is_finished_ok}")
    if not node.is_finished_ok:
        print(f"Exit status: {node.exit_status}")
        print(f"Exit message: {node.exit_message}")
        return

    elastic = results["elastic_parameters"].get_dict()
    branch_results = results["branch_results"].get_dict()

    print(f"Fit success: {elastic['fit_success']}")
    print(f"Relaxed-ion branches: {elastic['fit_with_relaxed_internal_positions']}")
    print(f"Strain branch count: {len(branch_results['branches'])}")
    print(f"Bulk modulus (Voigt): {elastic['bulk_modulus_voigt']:.6f} GPa")
    print(f"Shear modulus (Voigt): {elastic['shear_modulus_voigt']:.6f} GPa")
    print(f"Young's modulus: {elastic['youngs_modulus']:.6f} GPa")
    print(f"Poisson ratio: {elastic['poisson_ratio']:.6f}")
    print("Elastic tensor (Voigt, GPa):")
    for row in elastic["elastic_tensor"]:
        print("  " + " ".join(f"{value:10.4f}" for value in row))


if __name__ == "__main__":
    main()
