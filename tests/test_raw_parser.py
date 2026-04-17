"""
Tests for the parsers
"""

from io import StringIO

import numpy as np
import pytest

from aiida_abacus.parsers.raw_parsers import (
    AbacusRawParser,
    BandsParser,
    InternalParametersParser,
    KpointsParser,
    PdosParser,
    StruParser,
    TimejsonParser,
    WarningLogParser,
)


def test_eigenvalues(data_folder):
    parser = AbacusRawParser(data_folder / "band_Al_pw/running_scf.log")
    eigen, _occ, kpt_cart = parser.parse_eigenvalues()
    assert eigen.shape == (2, 18, 15)

    parser = AbacusRawParser(data_folder / "band_Al_pw/running_nscf.log")
    eigen, _occ, kpt_cart = parser.parse_eigenvalues()
    assert eigen.shape == (2, 61, 15)

    kpt_frac, kpt_cart = parser.parse_kpoints()
    assert kpt_frac.shape == (122, 4)
    assert kpt_cart.shape == (122, 4)
    weights = kpt_frac[:, 3]
    np.testing.assert_allclose(kpt_frac[0], [0.0, 0.0, 0.0, 0.0082])
    np.testing.assert_allclose(kpt_frac[1], [0.025, -0.025, 0.025, 0.0082])
    assert weights.shape == (122,)
    assert sum(weights) == pytest.approx(1.0, abs=1e-3)  # Allow tolerance for floating point precision


def test_kpoints_parser(data_folder):
    parser = KpointsParser(data_folder / "pw_Si2/OUT.aiida/kpoints")
    points, weights = parser.parse()
    assert len(points) == 8
    assert len(weights) == 8
    assert abs(sum(weights) - 1.0) <= 1e-4
    assert weights[0] == 0.0156
    assert points[0] == [0, 0, 0]


def test_internal_parameters_parser(data_folder):
    parser = InternalParametersParser(data_folder / "pw_Si2/OUT.aiida/INPUT")
    params = parser.parse()
    assert params["nspin"] == "1"
    assert params["lj_rcut"] == "None"
    assert params["kspacing"] == "0 0 0"


def test_bands_parser(data_folder):
    parser = BandsParser(data_folder / "band_Al_pw/BANDS_1.dat")
    kdist, eigenvalues = parser.parse()
    assert len(kdist) == 122
    assert eigenvalues.shape == (122, 15)


def test_stru_parser(data_folder):
    parser = StruParser(data_folder / "pw_Si2/STRU")
    cell, positions, species = parser.parse()
    assert species == ["Si", "Si"]
    a = 10.2 * 0.5 / 1.8897261255
    np.testing.assert_allclose(cell, np.array([[a, a, 0], [a, 0, a], [0, a, a]]))
    np.testing.assert_allclose(positions, np.array([[0, 0, 0], [0.5 * a, 0.5 * a, 0.5 * a]]))
    parser = StruParser(data_folder / "STRU_ION_D")
    cell, positions, species = parser.parse()
    assert species == ["Cd", "Cd", "Cd", "Cd", "Sn", "Sn", "Sn", "Sn"]
    a = 10.2 * 0.5 / 1.8897261255
    np.testing.assert_allclose(
        cell,
        np.array(
            [
                [6.6539429744, 0.0000000000, 0.0000000000],
                [0.0000000000, 6.6539429744, 0.0000000000],
                [0.0000000000, 0.0000000000, 13.1571816610],
            ]
        ),
    )
    np.testing.assert_allclose(positions[0], [0.0, 0.0, 13.1571816610])


def test_parse_notifications():
    parser = AbacusRawParser(
        StringIO(
            "\n".join(
                [
                    " #SCF IS CONVERGED#",
                    " !!SCF IS NOT CONVERGED!!",
                    " Relaxation is not converged",
                    " Relaxation is converged!",
                    " Relaxation is converged, but the SCF is unconverged",
                ]
            )
        )
    )

    notifications = parser.parse_notifications()

    assert [entry["name"] for entry in notifications] == [
        "scf_converged",
        "scf_not_converged",
        "ionic_not_converged",
        "ionic_converged",
        "relax_scf_not_converged",
    ]


def test_parse_notifications_preserves_repeated_order():
    parser = AbacusRawParser(
        StringIO(
            "\n".join(
                [
                    " !!SCF IS NOT CONVERGED!!",
                    " #SCF IS CONVERGED#",
                    " !!SCF IS NOT CONVERGED!!",
                ]
            )
        )
    )

    notifications = parser.parse_notifications()

    assert [entry["name"] for entry in notifications] == [
        "scf_not_converged",
        "scf_converged",
        "scf_not_converged",
    ]


def test_parse_notifications_supports_lts_scf_markers():
    parser = AbacusRawParser(
        StringIO(
            "\n".join(
                [
                    " charge density convergence is achieved",
                    " !! convergence has not been achieved @_@",
                ]
            )
        )
    )

    notifications = parser.parse_notifications()

    assert [entry["name"] for entry in notifications] == [
        "scf_converged",
        "scf_not_converged",
    ]


def test_warning_log_parser():
    parser = WarningLogParser(
        StringIO(
            "\n".join(
                [
                    " scf  warning : Threshold on eigenvalues was too large.",
                    " ignored line",
                    " driver warning : Calculation will restart",
                ]
            )
        )
    )

    notifications = parser.parse()

    assert notifications == [
        {"source": "scf", "message": "Threshold on eigenvalues was too large."},
        {"source": "driver", "message": "Calculation will restart"},
    ]


def test_parse_runtime_warnings():
    parser = AbacusRawParser(
        StringIO(
            "\n".join(
                [
                    " random line",
                    " Notice: Threshold on eigenvalues was too large.",
                    " Warning: Falling back to a slower path",
                    " Notice: Threshold on eigenvalues was too large.",
                ]
            )
        )
    )

    notifications = parser.parse_runtime_warnings()

    assert notifications == [
        {"source": "running_log", "message": "Threshold on eigenvalues was too large."},
        {"source": "running_log", "message": "Falling back to a slower path"},
    ]


def test_parse_additional_metrics_from_existing_scf_fixture(data_folder):
    parser = AbacusRawParser(data_folder / "pw_Si2/OUT.aiida/running_scf.log")
    results = parser.parse()

    assert results["volume"] == pytest.approx(39.3137)
    assert results["energy_ks"] == pytest.approx(-215.5056984087)
    assert results["converged"] is True
    assert results["scf_steps"] == 5
    assert results["pressure"] is None
    assert results["forces"] is None
    assert results["stresses"] is None


def test_parse_additional_metrics_from_existing_relax_fixture(data_folder):
    parser = AbacusRawParser(data_folder / "pw_Si2-relax/OUT.aiida/running_cell-relax.log")
    results = parser.parse()

    assert results["relax_converged"] is True
    assert results["relax_steps"] == 7
    assert len(results["largest_gradient"]) == 7
    assert len(results["largest_gradient_stress"]) == 7
    assert results["forces"] is not None
    assert results["stresses"] is not None
    assert len(results["force"]) == 6
    assert len(results["stress"]) == 9
    assert len(results["pressures"]) == len(results["stresses"])
    assert results["virial"] is not None
    assert len(results["virial"]) == 9


def test_parse_non_lts_stress_and_pressure_fixture(data_folder):
    parser = AbacusRawParser(data_folder / "pw_Si2-non-lts/OUT.aiida/running_scf.log")
    results = parser.parse()

    assert results["converged"] is True
    assert results["total_energy"] == pytest.approx(-230.2627734838431479)
    assert results["number_of_bands"] == 14
    assert results["fermi_level"] == pytest.approx(6.2945208731)
    assert results["force"] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert results["stress"] == pytest.approx(
        [-0.0296363105, 0.0, 0.0, 0.0, -0.0296363105, 0.0, 0.0, 0.0, -0.0296363105]
    )
    assert results["pressure"] == pytest.approx(-0.0296363105)
    assert results["total_pressure"] == pytest.approx(-0.029636)
    assert results["total_pressure_unit"] == "kbar"
    assert results["volume"] == pytest.approx(40.9113)
    assert results["virial"] is not None
    assert len(results["virial"]) == 9


def test_timejson_parser(data_folder):
    parser = TimejsonParser(data_folder / "pw_Si2/time.json")
    results = parser.parse()

    assert results["total_time"] == pytest.approx(1.74071)
    assert results["stress_time"] is None
    assert results["force_time"] is None


def test_pdos_parser():
    parser = PdosParser(
        StringIO(
            "\n".join(
                [
                    "<pdos>",
                    "  <nspin>2</nspin>",
                    "  <energy_values>-1.0 0.0 1.0</energy_values>",
                    '  <orbital index="1" atom_index="1" species="Si" l="1" m="0" z="0">',
                    "    <data>",
                    "      -1.0 0.1 0.2",
                    "      0.0 0.3 0.4",
                    "      1.0 0.5 0.6",
                    "    </data>",
                    "  </orbital>",
                    "</pdos>",
                ]
            )
        )
    )

    results = parser.parse()

    assert results["nspin"] == 2
    assert results["energy"] == [-1.0, 0.0, 1.0]
    assert results["orbitals"][0]["species"] == "Si"
    assert results["orbitals"][0]["data"][0] == [0.1, 0.3, 0.5]
    assert results["orbitals"][0]["data"][1] == [-0.2, -0.4, -0.6]
