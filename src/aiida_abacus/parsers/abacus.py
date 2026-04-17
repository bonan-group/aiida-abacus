"""
Parsers provided by aiida_abacus.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""

import re
from pathlib import PurePosixPath

import numpy as np
from aiida import orm
from aiida.common import exceptions
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

from ..common import make_retrieve_list
from .raw_parsers import (
    AbacusRawParser,
    InternalParametersParser,
    KpointsParser,
    PdosParser,
    StruParser,
    TimejsonParser,
    WarningLogParser,
)


class ParserError(RuntimeError):
    """Base exception for parser errors."""


class QuantityMissingError(ParserError):
    """A required quantity is missing from the parsed data."""


class RequiredQuantityMissingError(ParserError):
    """A required quantity that must be present is missing."""


class MissingFileError(ParserError):
    """An expected output file is missing."""


AbacusCalculation = CalculationFactory("abacus.abacus")

DEFAULT_OUTPUT_SETTINGS = {
    "bands": False,
    "internal_parameters": False,
    "kpoints": False,
    "pdos": False,
    "time_json": False,
    "eigenvalues": False,
    "mulliken": False,
}

RELAX_RUN_TYPES = {"relax", "cell-relax", "md"}
SCF_CONVERGENCE_CHECK_RUN_TYPES = {"scf", "relax", "cell-relax"}


class AbacusParser(Parser):
    """
    Parser class for parsing output of calculation.
    """

    def __init__(self, node):
        """
        Initialize Parser instance

        Checks that the ProcessNode being passed was produced by a AbacusCalculation.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.nodes.process.process.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, AbacusCalculation):
            raise exceptions.ParsingError("Can only parse AbacusCalculation")

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        output_folder = self.retrieved
        settings = {} if "settings" not in self.node.inputs else self.node.inputs.settings
        output_suffix = self.node.process_class._OUTPUT_SUFFIX
        expected_files = make_retrieve_list(self.node.inputs.parameters, settings, output_suffix)
        run_type = self.node.inputs.parameters["input"].get("calculation", "scf")
        mandatory_files = self._get_mandatory_files(run_type, output_suffix)

        # Check if the files are retrieved
        missing = []
        for name in expected_files:
            try:
                output_folder.get_object(name)
            except FileNotFoundError:
                missing.append(name)

        if missing:
            self.logger.warning(f"The following expected files are missing: {missing}")

        # Parse the calculation task output file
        main_log = next(filter(lambda x: "running_" in x, expected_files))
        misc_results = {}
        with output_folder.open(main_log, "r") as fhandle:
            raw_parser = AbacusRawParser(fhandle)
        misc_results.update(raw_parser.parse())

        # Check if calculation completed successfully using run_status from raw parser
        run_status = misc_results.get("run_status", {})
        notifications = run_status.get("notifications", [])

        if not run_status.get("completed", False):
            marker = run_status.get("termination_marker", "unknown")
            self.logger.warning(f"Calculation did not complete successfully. Termination marker: {marker}")
            return self.exit_codes.ERROR_CALCULATION_INCOMPLETE

        if run_type in SCF_CONVERGENCE_CHECK_RUN_TYPES:
            final_scf_state = self._last_notification_name(notifications, {"scf_converged", "scf_not_converged"})
            final_ionic_state = self._last_notification_name(
                notifications, {"ionic_converged", "ionic_not_converged", "geometry_not_converged"}
            )

            if any(n["name"] == "relax_scf_not_converged" for n in notifications):
                self.logger.warning("Ionic relaxation converged, but the final SCF did not converge.")
                return self.exit_codes.ERROR_ELECTRONIC_NOT_CONVERGED

            # Check for electronic convergence failure
            if final_scf_state == "scf_not_converged":
                self.logger.warning("SCF did not converge in the final relevant step.")
                return self.exit_codes.ERROR_ELECTRONIC_NOT_CONVERGED

            # Check for ionic relaxation convergence failure
            if final_ionic_state in {"ionic_not_converged", "geometry_not_converged"}:
                self.logger.warning("Ionic relaxation did not converge.")
                return self.exit_codes.ERROR_IONIC_NOT_CONVERGED

        missing_mandatory = [name for name in mandatory_files if name in missing]
        if missing_mandatory:
            self.logger.error(f"The following mandatory output files are missing: {missing_mandatory}")
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # Parse warning.log if available
        warning_notifications = self._parse_warning_log(output_folder, output_suffix)
        misc_results["warnings"] = self._merge_warnings(warning_notifications, raw_parser.parse_runtime_warnings())
        misc_results.update(self._parse_time_json(output_folder, output_suffix))

        misc_node = orm.Dict(dict=misc_results)

        # Parse the bands output if requested
        if self.check_include_node("bands"):
            eigenvalues, occupations, _ = raw_parser.parse_eigenvalues()
            kpoints_direct, _ = raw_parser.parse_kpoints()
            kcoord = kpoints_direct[:, :3]
            kweights = kpoints_direct[:, 3]
            node = orm.BandsData()
            node.set_kpoints(kcoord, weights=kweights)
            assert kcoord.shape[0] == eigenvalues.shape[1], "Inconsistent number of kpoints reported (do not use kpar)"
            node.set_bands(eigenvalues, occupations=occupations)
            node.labels = self.node.inputs.kpoints.labels
            # Record the fermi level - the unit is eV
            node.base.attributes.set("fermi_level", misc_node.get("fermi_level"))
            self.out("bands", node)

        # TODO: there could be other types that should have a output structure
        if run_type in ["relax", "cell-relax", "md"]:
            # Parse the final structure
            fname = _find_relax_structure_file(output_folder, output_suffix)
            if fname is None:
                self.logger.error("No relaxation structure snapshot was found in the retrieved files.")
                return self.exit_codes.ERROR_MISSING_OUTPUT_FILES
            with output_folder.open(fname, "r") as fhandle:
                parser = StruParser(fhandle)
                cell, positions, species = parser.parse_structure()
            node = orm.StructureData(cell=cell)
            for pos, symbol in zip(positions, species):
                node.append_atom(position=pos, symbols=symbol)
            self.out("structure", node)
            # Compose trajectory node
            trajectory = compose_trajectory(output_folder, misc_results, self.node.process_class._OUTPUT_SUFFIX)
            if trajectory is not None:
                self.out("trajectory", trajectory)

        # Parse the calculation raw parameters
        if self.check_include_node("internal_parameters"):
            fname = next(filter(lambda x: x.endswith("INPUT"), expected_files))
            with output_folder.open(fname, "r") as fhandle:
                parser = InternalParametersParser(fhandle)
            self.out("internal_parameters", orm.Dict(parser.parse()))

        # Parse the KPOINTS actually used
        if self.check_include_node("kpoints"):
            fname = next(filter(lambda x: x.endswith("kpoints"), expected_files))
            with output_folder.open(fname, "r") as fhandle:
                parser = KpointsParser(fhandle)
                coords, weights = parser.parse()
            node = orm.KpointsData()
            node.set_kpoints(coords, weights=weights)
            # Set the cell based on the  INPUT structure
            node.set_cell_from_structure(self.node.inputs.structure)
            self.out("kpoints", node)

        if self.check_include_node("pdos"):
            pdos_data = self._parse_pdos(output_folder, output_suffix)
            if pdos_data is not None:
                self.out("pdos", orm.Dict(dict=pdos_data))

        # Define the output nodes
        self.out("misc", misc_node)

    def _get_mandatory_files(self, run_type: str, output_suffix: str) -> list[str]:
        """Return files that are required for this parser invocation."""
        folder_name = f"OUT.{output_suffix}"
        mandatory = [f"{folder_name}/running_{run_type}.log"]

        if self.check_include_node("internal_parameters"):
            mandatory.append(f"{folder_name}/INPUT")
        if self.check_include_node("kpoints"):
            mandatory.append(f"{folder_name}/kpoints")

        return mandatory

    def _parse_warning_log(self, output_folder: orm.FolderData, output_suffix: str) -> list[dict]:
        """Parse warning.log if present."""
        folder_name = "OUT." + output_suffix
        warning_log_path = folder_name + "/warning.log"
        try:
            with output_folder.open(warning_log_path, "r") as fhandle:
                warning_parser = WarningLogParser(fhandle)
                return warning_parser.parse()
        except FileNotFoundError:
            return []
        except Exception as exc:
            self.logger.warning(f"Failed to parse warning.log: {exc}")
            return []

    def _parse_time_json(self, output_folder: orm.FolderData, output_suffix: str) -> dict:
        """Parse time.json if requested and available."""
        if not self.check_include_node("time_json"):
            return {}

        candidate_paths = ("time.json", f"OUT.{output_suffix}/time.json")
        for path in candidate_paths:
            try:
                with output_folder.open(path, "r") as fhandle:
                    return TimejsonParser(fhandle).parse()
            except FileNotFoundError:
                continue
            except Exception as exc:
                self.logger.warning(f"Failed to parse {path}: {exc}")
                return {}
        self.logger.warning("time.json was requested but not found in retrieved files.")
        return {}

    def _parse_pdos(self, output_folder: orm.FolderData, output_suffix: str) -> dict | None:
        """Parse PDOS file if requested and available."""
        path = f"OUT.{output_suffix}/PDOS"
        try:
            with output_folder.open(path, "r") as fhandle:
                return PdosParser(fhandle).parse()
        except FileNotFoundError:
            self.logger.warning("PDOS was requested but not found in retrieved files.")
            return None
        except Exception as exc:
            self.logger.warning(f"Failed to parse PDOS: {exc}")
            return None

    @staticmethod
    def _merge_warnings(*warning_sets: list[dict]) -> list[dict]:
        """Merge warning records while preserving input order and removing duplicates."""
        merged = []
        seen = set()
        for warning_set in warning_sets:
            for warning in warning_set:
                source = warning.get("source", "")
                message = warning.get("message", "")
                key = message
                if key in seen:
                    continue
                seen.add(key)
                merged.append({"source": source, "message": message})
        return merged

    @staticmethod
    def _last_notification_name(notifications: list[dict], names: set[str]) -> str | None:
        """Return the last matching notification name from an ordered notification list."""
        for notification in reversed(notifications):
            name = notification.get("name")
            if name in names:
                return name
        return None

    def check_include_node(self, name: str):
        """
        Check whether to include certain output node
        """

        if "settings" not in self.node.inputs:
            return DEFAULT_OUTPUT_SETTINGS[name]
        return self.node.inputs.settings.get("include_" + name, DEFAULT_OUTPUT_SETTINGS[name])


def compose_trajectory(output_folder: orm.FolderData, data_dict: dict, output_suffix=AbacusCalculation._OUTPUT_SUFFIX):
    """
    Compose a TrajectoryData node based on the retrieved data

    :param output_folder: A FolderData containing the retrieved files
    :param data_dict: The `results` dictionary retrieved
    :param output_suffix: The *suffix* used by AbacusCalculation

    :return: A orm.TrajectoryData Node.
    """
    folder_name = "OUT." + output_suffix
    traj_files = _list_relax_structure_files(output_folder, output_suffix)
    traj_files.sort(key=lambda name: int(re.search(r"STRU_ION(\d+)_D$", PurePosixPath(name).name).group(1)))
    if not traj_files:
        fallback = _find_relax_structure_file(output_folder, output_suffix)
        if fallback is None:
            return None
        traj_files = [PurePosixPath(fallback).name]
    cell_list = []
    positions_list = []
    symbols_list = []
    for traj_file in traj_files:
        with output_folder.open(folder_name + "/" + traj_file) as fhandle:
            parser = StruParser(fhandle)
            cell, positions, species = parser.parse_structure()
        cell_list.append(cell)
        positions_list.append(positions)
        symbols_list.append(species)
    if not symbols_list:
        return None
    traj = orm.TrajectoryData()
    traj.set_trajectory(symbols=symbols_list[0], cells=np.array(cell_list), positions=np.array(positions_list))
    # Set additional data
    if data_dict.get("all_forces"):
        traj.set_array("forces", np.array(data_dict["all_forces"]))
        traj.base.attributes.set("force_unit", data_dict["force_unit"])
    if data_dict.get("energies"):
        traj.set_array("energies", np.array(data_dict["energies"]))
    all_stress = data_dict.get("all_stress", data_dict.get("all_stresses"))
    if all_stress:
        traj.set_array("stresses", np.array(all_stress))
        traj.base.attributes.set("stress_unit", data_dict["stress_unit"])
    return traj


def _list_relax_structure_files(output_folder: orm.FolderData, output_suffix: str) -> list[str]:
    """Return numbered ionic snapshots available in the retrieved OUT folder."""
    folder_name = "OUT." + output_suffix
    try:
        return [
            file_name
            for file_name in output_folder.list_object_names(folder_name)
            if re.match(r"STRU_ION\d+_D$", file_name)
        ]
    except FileNotFoundError:
        return []


def _find_relax_structure_file(output_folder: orm.FolderData, output_suffix: str) -> str | None:
    """Return the best available final relaxation structure path from the retrieved OUT folder."""
    folder_name = "OUT." + output_suffix
    try:
        output_folder.get_object(f"{folder_name}/STRU_ION_D")
        return f"{folder_name}/STRU_ION_D"
    except FileNotFoundError:
        pass

    traj_files = _list_relax_structure_files(output_folder, output_suffix)
    if not traj_files:
        return None

    traj_files.sort(key=lambda name: int(re.search(r"STRU_ION(\d+)_D$", PurePosixPath(name).name).group(1)))
    return f"{folder_name}/{traj_files[-1]}"
