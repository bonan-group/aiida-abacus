import pathlib

import pytest
from aiida import orm

from aiida_abacus.parsers.abacus import AbacusParser


@pytest.fixture
def parser_with_retrieved(calc_with_retrieved, request):
    """Fixture to create an AbacusParser instance with a given pre-computed data folder"""

    def wrapped(name, parameters=None, settings=None, parse=True):
        _relative_file_path = f"test_data/{name}"
        file_path = str(pathlib.Path(request.fspath).parent / _relative_file_path)
        node = calc_with_retrieved(file_path, parameters=parameters, settings=settings)
        parser = AbacusParser(node)
        exit_code = None
        if parse:
            exit_code = parser.parse()
        return parser, exit_code

    return wrapped


def _write_retrieved_tree(
    base_path: pathlib.Path,
    calculation: str,
    log_content: str,
    warning_content: str = "",
    extra_files: dict[str, str] | None = None,
):
    out_folder = base_path / "OUT.aiida"
    out_folder.mkdir(parents=True, exist_ok=True)
    (out_folder / f"running_{calculation}.log").write_text(log_content)
    (out_folder / "warning.log").write_text(warning_content)
    (base_path / "abacus_output").write_text("")
    for relative_path, content in (extra_files or {}).items():
        destination = base_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)


MINIMAL_STRU = """ATOMIC_SPECIES
Si 28.0855 Si.upf

NUMERICAL_ORBITAL

LATTICE_CONSTANT
1.889726125

LATTICE_VECTORS
5.1 0.0 0.0
0.0 5.1 0.0
0.0 0.0 5.1

ATOMIC_POSITIONS
Direct
Si
0.0
2
0.0 0.0 0.0 1 1 1
0.25 0.25 0.25 1 1 1
"""


def test_parser_pw_si2(calc_with_retrieved, request):
    """Test parsing pw_Si2 calculation (SCF)"""
    _relative_file_path = "test_data/pw_Si2"
    file_path = str(pathlib.Path(request.fspath).parent / _relative_file_path)
    node = calc_with_retrieved(file_path, {})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    # Check that parsing was successful
    assert exit_code is None

    # Check that basic outputs are present
    assert "misc" in parser.outputs

    misc = parser.outputs["misc"].get_dict()

    # Check for basic calculation information
    assert "fermi_level" in misc
    assert "all_forces" in misc
    assert "all_stress" in misc

    # Check for run_status
    assert "run_status" in misc
    run_status = misc["run_status"]
    assert isinstance(run_status, dict)
    assert "completed" in run_status
    assert "completion_marker_found" in run_status
    assert "termination_marker" in run_status

    # For successful parsing, calculation should be completed
    assert run_status["completed"] is True
    assert run_status["completion_marker_found"] is True
    assert run_status["termination_marker"] == "Total  Time"

    # Check for Fermi level value
    assert isinstance(misc["fermi_level"], float)
    assert misc["fermi_level"] > 0

    # Check forces array - may be empty for SCF calculation or contain final forces
    forces = misc.get("final_forces", misc.get("all_forces", []))
    if forces:  # Only check if forces are present
        assert isinstance(forces, list)
        # For Si2, should have 2 atoms if forces are calculated
        if len(forces) > 0:
            for force in forces:
                assert isinstance(force, list)
                assert len(force) == 3
                for component in force:
                    assert isinstance(component, (int, float))

    # Check stress tensor - may be empty for SCF calculation
    stress = misc.get("all_stress", [])
    if stress:  # Only check if stress is present
        assert isinstance(stress, list)
        if len(stress) > 0:
            for row in stress:
                assert isinstance(row, list)
                assert len(row) == 3
                for component in row:
                    assert isinstance(component, (int, float))

    # Check for other common ABACUS output fields
    expected_fields = ["total_energy", "number_of_bands"]
    for field in expected_fields:
        assert isinstance(misc[field], (int, float, bool, list))

    assert misc["volume"] == pytest.approx(39.3137)
    assert misc["energy_ks"] == pytest.approx(-215.5056984087)
    assert misc["converged"] is True
    assert misc["scf_steps"] == 5
    assert misc["force"] is None
    assert misc["stress"] is None
    assert misc["pressures"] is None
    assert misc["virial"] is None
    assert misc["warnings"] == [{"source": "scf", "message": "Threshold on eigenvalues was too large."}]


def test_parser_pw_si2_relax(calc_with_retrieved, request):
    """Test parsing pw_Si2-relax calculation (cell relaxation)"""
    _relative_file_path = "test_data/pw_Si2-relax"
    file_path = str(pathlib.Path(request.fspath).parent / _relative_file_path)

    # Use relax calculation type
    node = calc_with_retrieved(file_path, parameters={"input": {"calculation": "cell-relax"}})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    # Check that parsing was successful
    assert exit_code is None

    # Check that basic outputs are present
    assert "misc" in parser.outputs

    misc = parser.outputs["misc"].get_dict()

    # Check for basic calculation information
    assert "fermi_level" in misc
    assert misc.get("all_forces")
    assert misc.get("all_stress")

    # Check for run_status
    assert "run_status" in misc
    run_status = misc["run_status"]
    assert isinstance(run_status, dict)
    assert "completed" in run_status
    assert "completion_marker_found" in run_status
    assert "termination_marker" in run_status

    # For successful relaxation calculations, calculation should be completed
    assert run_status["completed"] is True
    assert run_status["completion_marker_found"] is True
    assert run_status["termination_marker"] == "Total  Time"

    # Check for Fermi level value
    assert isinstance(misc["fermi_level"], (int, float))
    assert misc["fermi_level"] > 0

    # For relaxation calculations, we expect multiple force steps
    all_forces = misc.get("all_forces", [])
    assert isinstance(all_forces, list)
    # Should have forces from multiple ionic steps for relaxation
    if len(all_forces) > 0:
        for step_forces in all_forces:
            assert isinstance(step_forces, list)
            if step_forces:  # Check if this step has forces
                for force in step_forces:
                    assert isinstance(force, list)
                    assert len(force) == 3
                    for component in force:
                        assert isinstance(component, (int, float))

    # For relaxation calculations, we expect multiple stress steps
    all_stress = misc.get("all_stress", [])
    assert isinstance(all_stress, list)
    if len(all_stress) > 0:
        for step_stress in all_stress:
            assert isinstance(step_stress, list)
            if step_stress:  # Check if this step has stress
                for row in step_stress:
                    assert isinstance(row, list)
                    assert len(row) == 3
                    for component in row:
                        assert isinstance(component, (int, float))

    # Check final forces if available
    final_forces = misc.get("final_forces")
    if final_forces is not None:
        assert isinstance(final_forces, list)
        # For Si2, should have 2 atoms
        if len(final_forces) > 0:
            for force in final_forces:
                assert isinstance(force, list)
                assert len(force) == 3

    assert misc["relax_converged"] is True
    assert misc["relax_steps"] == 7
    assert len(misc["largest_gradient"]) == 7
    assert len(misc["largest_gradient_stress"]) == 7
    assert len(misc["force"]) == 6
    assert len(misc["stress"]) == 9
    assert len(misc["forces"]) == len(misc["all_forces"])
    assert len(misc["stresses"]) == len(misc["all_stress"])
    assert len(misc["virial"]) == 9

    # Check for structure output in relaxation calculations
    if "structure" in parser.outputs:
        structure = parser.outputs["structure"]
        assert hasattr(structure, "get_pymatgen") or hasattr(structure, "get_ase")


def test_parser_energy_components(parser_with_retrieved):
    """Test that parser correctly extracts energy components"""
    parser, _ = parser_with_retrieved("pw_Si2")
    misc = parser.outputs["misc"].get_dict()

    # Check for run_status
    assert "run_status" in misc
    run_status = misc["run_status"]
    assert isinstance(run_status, dict)
    assert run_status["completed"] is True
    assert run_status["termination_marker"] == "Total  Time"

    # Check for different energy components if available
    energy_components = [
        "energy",
        "total_energy",
        "efermi",  # alternative name for fermi level
    ]

    for component in energy_components:
        if component in misc:
            # Energy may be stored as string, so convert and test
            energy_value = misc[component]
            if isinstance(energy_value, str):
                try:
                    float(energy_value)
                except ValueError:
                    pytest.fail(f"Energy component {component} is not a valid number: {energy_value}")
            else:
                assert isinstance(energy_value, (int, float))

    # Fermi level is always present in ABACUS output
    assert "fermi_level" in misc
    assert isinstance(misc["fermi_level"], float)


def test_parser_fermi_level(parser_with_retrieved):
    """Test that parser correctly extracts Fermi level"""
    parser, _ = parser_with_retrieved("pw_Si2")
    misc = parser.outputs["misc"].get_dict()

    # Check for run_status
    assert "run_status" in misc
    run_status = misc["run_status"]
    assert isinstance(run_status, dict)
    assert run_status["completed"] is True
    assert run_status["termination_marker"] == "Total  Time"

    # Check for Fermi level
    assert "fermi_level" in misc
    assert isinstance(misc["fermi_level"], float)
    # Fermi level for Si should be around a few eV
    assert -10 < misc["fermi_level"] < 10


def test_parser_pw_si2_non_lts_stress_pressure(calc_with_retrieved, request):
    """Test parsing newer non-LTS SCF output that uses #TOTAL-STRESS and #TOTAL-PRESSURE markers."""
    file_path = str(pathlib.Path(request.fspath).parent / "test_data/pw_Si2-non-lts")
    node = calc_with_retrieved(file_path, {})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is None
    misc = parser.outputs["misc"].get_dict()

    assert misc["converged"] is True
    assert misc["number_of_bands"] == 14
    assert misc["fermi_level"] == pytest.approx(6.2945208731)
    assert misc["force"] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert misc["stress"] == pytest.approx([-0.0296363105, 0.0, 0.0, 0.0, -0.0296363105, 0.0, 0.0, 0.0, -0.0296363105])
    assert misc["pressure"] == pytest.approx(-0.0296363105)
    assert misc["pressures"] == pytest.approx([-0.0296363105])
    assert misc["total_pressure"] == pytest.approx(-0.029636)
    assert misc["total_pressure_unit"] == "kbar"


def test_parser_relax_trajectory(parser_with_retrieved):
    """Test that parser correctly handles relaxation trajectory data"""
    parser, _ = parser_with_retrieved("pw_Si2-relax", parameters={"input": {"calculation": "cell-relax"}})
    misc = parser.outputs["misc"].get_dict()

    # Check for run_status
    assert "run_status" in misc
    run_status = misc["run_status"]
    assert isinstance(run_status, dict)
    assert run_status["completed"] is True
    assert run_status["termination_marker"] == "Total  Time"

    # Check for trajectory-related information in relaxation
    all_forces = misc.get("all_forces")
    all_stress = misc.get("all_stress")

    # For relaxation, we expect multiple steps
    assert len(all_forces) > 2
    assert len(all_stress) > 2

    # If we have multiple steps, this indicates trajectory data
    if len(all_forces) > 1:
        # Verify each step has proper structure
        for step_forces in all_forces:
            if step_forces:  # non-empty step
                assert isinstance(step_forces, list)

    if len(all_stress) > 1:
        # Verify each step has proper structure
        for step_stress in all_stress:
            if step_stress:  # non-empty step
                assert isinstance(step_stress, list)
    # Check that trajectory output node is present
    assert isinstance(parser.outputs.get("trajectory"), orm.TrajectoryData)
    assert len(parser.outputs["trajectory"].get_array("forces")) == len(misc.get("all_forces"))
    assert len(parser.outputs["trajectory"].get_array("stresses")) == len(misc.get("all_stress"))
    assert parser.outputs.get("trajectory").get_step_structure(1)


def test_parser_pw_si2_incomplete(calc_with_retrieved, request):
    """Test parsing incomplete pw_Si2 calculation (truncated before completion)"""
    _relative_file_path = "test_data/pw_Si2-incomplete"
    file_path = str(pathlib.Path(request.fspath).parent / _relative_file_path)

    # Use relax calculation type to match the original data
    node = calc_with_retrieved(file_path, parameters={"input": {"calculation": "cell-relax"}})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    # For incomplete calculations, parser should return ERROR_CALCULATION_INCOMPLETE
    assert exit_code is not None
    assert exit_code.status == 301  # ERROR_CALCULATION_INCOMPLETE

    # When parser exits with error, no outputs are created
    assert "misc" not in parser.outputs


def test_parser_returns_electronic_not_converged(calc_with_retrieved, tmp_path):
    file_path = tmp_path / "pw_scf_not_converged"
    _write_retrieved_tree(
        file_path,
        "scf",
        "\n".join(
            [
                " EFERMI = 1.23 eV",
                " NBANDS = 8",
                " !!SCF IS NOT CONVERGED!!",
                " Total  Time  :  1.0 s",
            ]
        ),
    )

    node = calc_with_retrieved(str(file_path))
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is not None
    assert exit_code.status == 302
    assert "misc" not in parser.outputs


def test_parser_returns_ionic_not_converged(calc_with_retrieved, tmp_path):
    file_path = tmp_path / "pw_relax_not_converged"
    _write_retrieved_tree(
        file_path,
        "cell-relax",
        "\n".join(
            [
                " EFERMI = 1.23 eV",
                " NBANDS = 8",
                " Relaxation is not converged",
                " Total  Time  :  1.0 s",
            ]
        ),
    )

    node = calc_with_retrieved(str(file_path), parameters={"input": {"calculation": "cell-relax"}})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is not None
    assert exit_code.status == 303
    assert "misc" not in parser.outputs


def test_parser_relax_trajectory_falls_back_to_final_structure(calc_with_retrieved, tmp_path):
    file_path = tmp_path / "pw_relax_only_final_structure"
    _write_retrieved_tree(
        file_path,
        "cell-relax",
        "\n".join(
            [
                " Volume (A^3) = 39.3137",
                " NBANDS = 8",
                " EFERMI = 1.23 eV",
                " STEP OF RELAXATION : 1",
                "!FINAL_ETOT_IS -10.0 eV",
                " Relaxation is converged",
                " Self-consistent calculation is converged",
                " Total  Time  :  1.0 s",
            ]
        ),
        extra_files={
            "OUT.aiida/STRU_ION_D": MINIMAL_STRU,
        },
    )

    node = calc_with_retrieved(str(file_path), parameters={"input": {"calculation": "cell-relax"}})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is None
    assert "trajectory" in parser.outputs
    trajectory = parser.outputs["trajectory"]
    assert isinstance(trajectory, orm.TrajectoryData)
    assert trajectory.get_stepids() == [0]
    assert trajectory.get_step_structure(0)


def test_parser_relax_uses_numbered_snapshot_when_plain_final_structure_is_missing(calc_with_retrieved, tmp_path):
    file_path = tmp_path / "pw_relax_numbered_final_structure_only"
    _write_retrieved_tree(
        file_path,
        "cell-relax",
        "\n".join(
            [
                " Volume (A^3) = 39.3137",
                " NBANDS = 8",
                " EFERMI = 1.23 eV",
                " STEP OF RELAXATION : 1",
                "!FINAL_ETOT_IS -10.0 eV",
                " Relaxation is converged",
                " Self-consistent calculation is converged",
                " Total  Time  :  1.0 s",
            ]
        ),
        extra_files={
            "OUT.aiida/STRU_ION1_D": MINIMAL_STRU,
        },
    )

    node = calc_with_retrieved(str(file_path), parameters={"input": {"calculation": "cell-relax"}})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is None
    assert "structure" in parser.outputs
    assert "trajectory" in parser.outputs
    trajectory = parser.outputs["trajectory"]
    assert isinstance(trajectory, orm.TrajectoryData)
    assert trajectory.get_stepids() == [0]
    assert trajectory.get_step_structure(0)


def test_parser_relax_without_any_trajectory_snapshot_does_not_crash(calc_with_retrieved, tmp_path):
    file_path = tmp_path / "pw_relax_without_trajectory_snapshot"
    _write_retrieved_tree(
        file_path,
        "cell-relax",
        "\n".join(
            [
                " Volume (A^3) = 39.3137",
                " NBANDS = 8",
                " EFERMI = 1.23 eV",
                " STEP OF RELAXATION : 1",
                "!FINAL_ETOT_IS -10.0 eV",
                " Relaxation is converged",
                " Self-consistent calculation is converged",
                " Total  Time  :  1.0 s",
            ]
        ),
    )

    node = calc_with_retrieved(str(file_path), parameters={"input": {"calculation": "cell-relax"}})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is not None
    assert exit_code.status == 300


def test_parser_returns_missing_output_files(calc_with_retrieved, tmp_path):
    file_path = tmp_path / "pw_relax_missing_final_structure"
    _write_retrieved_tree(
        file_path,
        "cell-relax",
        "\n".join(
            [
                " EFERMI = 1.23 eV",
                " NBANDS = 8",
                " Total  Time  :  1.0 s",
            ]
        ),
    )

    node = calc_with_retrieved(str(file_path), parameters={"input": {"calculation": "cell-relax"}})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is not None
    assert exit_code.status == 300
    assert "misc" not in parser.outputs


def test_parser_merges_running_log_warnings(calc_with_retrieved, tmp_path):
    file_path = tmp_path / "pw_warning_merge"
    _write_retrieved_tree(
        file_path,
        "scf",
        "\n".join(
            [
                " EFERMI = 1.23 eV",
                " NBANDS = 8",
                " Notice: Threshold on eigenvalues was too large.",
                " Total  Time  :  1.0 s",
            ]
        ),
        warning_content="driver warning : Calculation will restart\n",
    )

    node = calc_with_retrieved(str(file_path))
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is None
    assert parser.outputs["misc"].get_dict()["warnings"] == [
        {"source": "driver", "message": "Calculation will restart"},
        {"source": "running_log", "message": "Threshold on eigenvalues was too large."},
    ]


def test_parser_includes_time_json_metrics(parser_with_retrieved):
    parser, exit_code = parser_with_retrieved("pw_Si2", settings={"include_time_json": True})

    assert exit_code is None
    misc = parser.outputs["misc"].get_dict()

    assert misc["total_time"] == pytest.approx(1.74071)
    assert misc["stress_time"] is None
    assert misc["force_time"] is None


def test_parser_emits_pdos_node(calc_with_retrieved, tmp_path):
    file_path = tmp_path / "pw_pdos"
    _write_retrieved_tree(
        file_path,
        "scf",
        "\n".join(
            [
                " Volume (A^3) = 10.0",
                " E_KohnSham     -1.23       -16.0",
                " EFERMI = 1.23 eV",
                " NBANDS = 8",
                " #SCF IS CONVERGED#",
                " !FINAL_ETOT_IS  -2.0 eV",
                " Total  Time  :  1.0 s",
            ]
        ),
        extra_files={
            "OUT.aiida/PDOS": "\n".join(
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
        },
    )

    node = calc_with_retrieved(str(file_path), settings={"include_pdos": True})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is None
    assert "pdos" in parser.outputs
    pdos = parser.outputs["pdos"].get_dict()
    assert pdos["nspin"] == 2
    assert pdos["orbitals"][0]["species"] == "Si"
    assert pdos["orbitals"][0]["data"][1] == [-0.2, -0.4, -0.6]


def test_parser_returns_geometry_not_converged(calc_with_retrieved, tmp_path):
    file_path = tmp_path / "pw_geometry_not_converged"
    _write_retrieved_tree(
        file_path,
        "cell-relax",
        "\n".join(
            [
                " EFERMI = 1.23 eV",
                " NBANDS = 8",
                " Geometry relaxation is not converged",
                " Total  Time  :  1.0 s",
            ]
        ),
    )

    node = calc_with_retrieved(str(file_path), parameters={"input": {"calculation": "cell-relax"}})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is not None
    assert exit_code.status == 303


def test_parser_returns_relax_scf_not_converged(calc_with_retrieved, tmp_path):
    file_path = tmp_path / "pw_relax_scf_not_converged"
    _write_retrieved_tree(
        file_path,
        "cell-relax",
        "\n".join(
            [
                " EFERMI = 1.23 eV",
                " NBANDS = 8",
                " Relaxation is converged!",
                " Relaxation is converged, but the SCF is unconverged",
                " Total  Time  :  1.0 s",
            ]
        ),
    )

    node = calc_with_retrieved(str(file_path), parameters={"input": {"calculation": "cell-relax"}})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is not None
    assert exit_code.status == 302


def test_parser_allows_relax_with_intermediate_scf_failure_if_final_scf_converges(calc_with_retrieved, tmp_path):
    file_path = tmp_path / "pw_relax_final_scf_converged"
    _write_retrieved_tree(
        file_path,
        "cell-relax",
        "\n".join(
            [
                " EFERMI = 1.23 eV",
                " NBANDS = 8",
                " !!SCF IS NOT CONVERGED!!",
                " Relaxation is not converged yet!",
                " #SCF IS CONVERGED#",
                " Relaxation is converged!",
                " Total  Time  :  1.0 s",
            ]
        ),
    )
    (file_path / "OUT.aiida" / "STRU_ION_D").write_text(
        "\n".join(
            [
                "ATOMIC_SPECIES",
                "Si 28.0855 Si.upf",
                "",
                "NUMERICAL_ORBITAL",
                "Si.orb",
                "",
                "LATTICE_CONSTANT",
                "1.889726125457828",
                "",
                "LATTICE_VECTORS",
                "5.1 0.0 0.0",
                "0.0 5.1 0.0",
                "0.0 0.0 5.1",
                "",
                "ATOMIC_POSITIONS",
                "Cartesian_angstrom",
                "Si",
                "0.0",
                "1",
                "0.0 0.0 0.0 1 1 1",
            ]
        )
    )
    (file_path / "OUT.aiida" / "STRU_ION1_D").write_text((file_path / "OUT.aiida" / "STRU_ION_D").read_text())

    node = calc_with_retrieved(str(file_path), parameters={"input": {"calculation": "cell-relax"}})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is None
    assert "misc" in parser.outputs
    assert "structure" in parser.outputs


def test_parser_does_not_apply_scf_convergence_failure_to_nscf(calc_with_retrieved, tmp_path):
    file_path = tmp_path / "pw_nscf_not_converged_marker"
    _write_retrieved_tree(
        file_path,
        "nscf",
        "\n".join(
            [
                " EFERMI = 1.23 eV",
                " NBANDS = 8",
                " !! convergence has not been achieved @_@",
                " Total  Time  :  1.0 s",
            ]
        ),
    )

    node = calc_with_retrieved(str(file_path), parameters={"input": {"calculation": "nscf"}})
    parser = AbacusParser(node)
    exit_code = parser.parse()

    assert exit_code is None
    assert "misc" in parser.outputs
