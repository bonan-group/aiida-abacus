"""Launch an EOS work chain using a preconfigured code and pseudo family.

Assumptions:
1. The `apns-efficiency-v1` pseudo family is available in the current AiiDA profile.
2. The `abacus@localhost` code is available in the current AiiDA profile.

Run with:
    AIIDA_PATH=$(pwd) uv run python examples/dev-scripts/example_eos_wc.py
"""

from aiida import orm
from aiida.engine import run_get_node
from ase.build import bulk

from aiida_abacus.common import RelaxType
from aiida_abacus.workflows.eos import AbacusEosWorkChain


def main():
    """Launch the EOS work chain example."""
    structure = orm.StructureData(ase=bulk("Si", "diamond", 5.43))
    code = orm.load_code("abacus@localhost")

    builder = AbacusEosWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        protocol="fast",
        overrides={"base": {"pseudo_family": "apns-efficiency-v1", 'parameters': {'inputs': {'ecutwfc': 60}}}},
        relax_type=RelaxType.NONE,
    )
    builder.base.abacus.metadata.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 600,
    }
    builder.eos_settings = orm.Dict(
        dict={
            "scale_start": 0.96,
            "scale_end": 1.04,
            "scale_step": 0.02,
            "fit_equation": "birchmurnaghan",
        }
    )

    results, node = run_get_node(builder)

    eos_parameters = results["eos_parameters"].get_dict()
    print(f"EOS work chain pk: {node.pk}")
    print(f"Finished OK: {node.is_finished_ok}")
    print(f"Fit success: {eos_parameters['fit_success']}")
    print(f"Reference volume: {eos_parameters['reference_volume']:.6f} A^3")
    print(f"Minimum sampled scale: {eos_parameters['minimum_scale']:.6f}")
    print(f"Minimum sampled volume: {eos_parameters['minimum_volume']:.6f} A^3")
    print(f"Minimum sampled energy: {eos_parameters['minimum_energy']:.12f} eV")

    if eos_parameters["fit_success"]:
        print(f"Equilibrium scale: {eos_parameters['equilibrium_scale']:.6f}")
        print(f"Equilibrium volume: {eos_parameters['equilibrium_volume']:.6f} A^3")
        print(f"Equilibrium energy: {eos_parameters['equilibrium_energy']:.12f} eV")
        print(f"Bulk modulus: {eos_parameters['bulk_modulus_gpa']:.6f} GPa")
    else:
        print(f"Fit error: {eos_parameters.get('fit_error', 'unknown error')}")

    print("Sampled points:")
    for sample in eos_parameters["samples"]:
        print(
            "  "
            f"{sample['label']}: scale={sample['scale']:.4f}, "
            f"volume={sample['volume']:.6f} A^3, "
            f"energy={sample['energy']:.12f} eV, "
            f"pk={sample['pk']}"
        )


if __name__ == "__main__":
    main()
