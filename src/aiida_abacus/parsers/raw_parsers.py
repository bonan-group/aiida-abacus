import json
import re
import xml.etree.ElementTree as ET
from logging import getLogger
from pathlib import Path
from typing import List

import numpy as np

logger = getLogger(__name__)


class BaseRawParser:
    def __init__(self, fhandle):
        """A parser for the ABACUS output file."""
        if not hasattr(fhandle, "read"):
            self.content = Path(fhandle).read_text()
        else:
            self.content = fhandle.read()
        self.lines = self.content.split("\n")


class AbacusRawParser(BaseRawParser):
    """
    A parser to process abacus output files (running_xxx.log)
    """

    def __init__(self, fhandle):
        """A parser for the ABACUS output file."""
        super().__init__(fhandle)
        self.results = {}
        self.is_parsed = False

    def parse_blocks(self) -> None:
        """
        Parse blocks from output file.
        """
        all_blocks = []
        index = 0
        while index < len(self.lines):
            header = self.lines[index].strip()
            match = re.match(r"#?TOTAL-(FORCE|STRESS)\s*\(([^)]+)\)#?$", header, flags=re.IGNORECASE)
            if match is None:
                index += 1
                continue

            block_type = match.group(1).upper()
            block_unit = match.group(2).strip()
            index += 1

            while index < len(self.lines) and (
                not self.lines[index].strip()
                or set(self.lines[index].strip()) == {"-"}
                or ("Atoms" in self.lines[index] and "Force_" in self.lines[index])
                or ("Stress_x" in self.lines[index] and "Stress_y" in self.lines[index] and "Stress_z" in self.lines[index])
            ):
                index += 1

            block_lines = []
            while index < len(self.lines):
                stripped = self.lines[index].strip()
                if (
                    not stripped
                    or set(stripped) == {"-"}
                    or stripped.startswith("#TOTAL-PRESSURE")
                    or stripped.startswith("TOTAL-PRESSURE")
                ):
                    break
                block_lines.append(stripped)
                index += 1

            all_blocks.append((block_type, block_unit, block_lines))
            continue

        all_forces = []
        all_stress = []
        # Process the blocks one by one, additional block type can be supported by adding more elifs.
        for block_type, block_unit, lines in all_blocks:
            if block_type == "FORCE":
                forces = []
                for line in lines:
                    tokens = line.split()
                    forces.append([float(token) for token in tokens[1:]])
                all_forces.append(forces)
                self.results["force_unit"] = block_unit

            if block_type == "STRESS":
                stress = []
                for line in lines:
                    stress.append([float(token) for token in line.split()])
                all_stress.append(stress)
                self.results["stress_unit"] = block_unit
        self.results["all_forces"] = all_forces
        self.results["all_stress"] = all_stress
        self.results["final_forces"] = all_forces[-1] if all_forces else None
        self.results["final_stress"] = all_stress[-1] if all_stress else None

    def parse(self) -> dict:
        """
        Parse ABACUS output file.

        :returns: parsed results as a dictionary
        """
        self.parse_blocks()
        # Parse the lines one-by-one for general information of the calculation
        self.results["energies"] = []  # Container for the per-ionic-step energies in eV
        energy_ks = []
        scf_iterations = set()
        for line in self.lines:
            stripped = line.strip()
            pressure_match = re.search(r"TOTAL-PRESSURE#?.*:\s*([-+0-9.eE]+)\s+([A-Za-z/]+)\s*$", line.strip(), re.IGNORECASE)
            if pressure_match:
                self.results["total_pressure"] = float(pressure_match.group(1))
                self.results["total_pressure_unit"] = pressure_match.group(2)
            elif "!FINAL_ETOT_IS" in line:
                self.results["total_energy"] = float(line.strip().split()[-2])
            elif "final etot is" in line:
                self.results["energies"].append(float(line.strip().split()[-2]))
            elif "NBANDS" in line:
                nbands_match = re.search(r"NBANDS\)?\s*=\s*(\d+)", stripped)
                if nbands_match:
                    self.results["number_of_bands"] = int(nbands_match.group(1))
            elif "EFERMI" in line:
                self.results["fermi_level"] = float(line.strip().split()[-2])
            elif "E_Fermi" in line:
                fermi_match = re.search(r"E_Fermi\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", stripped)
                if fermi_match:
                    self.results["fermi_level"] = float(fermi_match.group(2))
            elif "E_KohnSham" in line:
                energy_ks.append(float(line.strip().split()[-1]))
            else:
                volume_match = re.search(r"(?:cell\s+)?volume \(A\^3\)\s*=\s*([-+0-9.eE]+)", stripped, re.IGNORECASE)
                if volume_match:
                    self.results["volume"] = float(volume_match.group(1))
                elif "Largest gradient in force is" in line:
                    self.results.setdefault("largest_gradient", []).append(float(line.strip().split()[-2]))
                elif "Largest gradient is" in line:
                    self.results.setdefault("largest_gradient", []).append(float(line.strip().split()[-1]))
                elif "Largest gradient in stress is" in line:
                    self.results.setdefault("largest_gradient_stress", []).append(float(line.strip().split()[-2]))
                elif "STEP OF RELAXATION :" in line or " STEP OF ION RELAXATION : " in line:
                    self.results["relax_steps"] = int(line.strip().split()[-1])
                elif "ALGORITHM --------------- ION=" in line:
                    match = re.search(r"ION=\s*(\d+)\s+ELEC=\s*(\d+)", line)
                    if match:
                        self.results["relax_steps"] = int(match.group(1))
                        scf_iterations.add((int(match.group(1)), int(match.group(2))))

        notifications = self.parse_notifications()
        final_scf_state = self._last_notification_name(notifications, {"scf_converged", "scf_not_converged"})
        final_relax_state = self._last_notification_name(
            notifications,
            {"ionic_converged", "ionic_not_converged", "geometry_not_converged"},
        )
        self.results["converged"] = None if final_scf_state is None else final_scf_state == "scf_converged"
        self.results["relax_converged"] = None if final_relax_state is None else final_relax_state == "ionic_converged"
        self.results["energy_ks"] = energy_ks[-1] if energy_ks else None
        self.results["scf_steps"] = max((electron for _, electron in scf_iterations), default=None)
        self._normalize_force_stress()
        # Check calculation completion status
        self.results["run_status"] = self.compose_run_status()

        self.is_parsed = True
        return self.results

    def _normalize_force_stress(self) -> None:
        """Add flattened force, stress, pressure, and virial fields."""
        all_forces = self.results.get("all_forces", [])
        all_stress = self.results.get("all_stress", [])
        volume = self.results.get("volume")

        self.results["forces"] = [np.array(step).reshape(-1).tolist() for step in all_forces] or None
        self.results["force"] = self.results["forces"][-1] if self.results["forces"] else None
        self.results["stresses"] = [np.array(step).reshape(-1).tolist() for step in all_stress] or None
        self.results["stress"] = self.results["stresses"][-1] if self.results["stresses"] else None

        if self.results["stresses"]:
            pressures = []
            for stress in self.results["stresses"]:
                pressures.append((stress[0] + stress[4] + stress[8]) / 3.0)
            self.results["pressures"] = pressures
            self.results["pressure"] = pressures[-1]
        else:
            self.results["pressures"] = None
            self.results["pressure"] = self.results.get("total_pressure")

        if volume is not None and self.results["stresses"]:
            virials = [(np.array(stress) * volume * KBAR_TO_EV_PER_ANGSTROM3).tolist() for stress in self.results["stresses"]]
            self.results["virials"] = virials
            self.results["virial"] = virials[-1]
        else:
            self.results["virials"] = None
            self.results["virial"] = None

    @staticmethod
    def _last_notification_name(notifications: list[dict], names: set[str]) -> str | None:
        """Return the last matching notification name from an ordered notification list."""
        for notification in reversed(notifications):
            name = notification.get("name")
            if name in names:
                return name
        return None

    def parse_kpoints(self):
        """
        Parse the kpoints involved in the calculation

        :return: A tuple of kpoints in direct and cartesian coordinates
        """

        kdirect = BlockParser(
            self.lines, re.compile(r"^K-POINTS (DIRECT) COORDINATES"), offset=2, types=[int, float, float, float, float]
        ).parse()
        kcart = BlockParser(
            self.lines,
            re.compile(r"^K-POINTS (CARTESIAN) COORDINATES"),
            offset=2,
            types=[int, float, float, float, float],
        ).parse()
        if len(kdirect) == 0:
            raise ValueError("No kpoints data found")
        if len(kdirect) > 2:
            raise ValueError("Multiple sets of kpoints data found")
        # Take the last set of kpoint reported
        # Return an array made of kpoint coordinates and weight, remove the kpoint index
        return np.array(kdirect[-1][1])[:, 1:], np.array(kcart[-1][1])[:, 1:]

    def parse_eigenvalues(self):
        """
        Parse the eigenvalues
        :return: A tuple of eigenvalues and occupations and k-points (in cartesian coordinates)
        """

        nspins = int(re.search(r"NSPIN == (\d)", self.content).group(1))
        nkthis_procs = int(re.search(r"k-point number in this process = (\d+)", self.content).group(1))
        parser = BlockParser(
            self.lines,
            re.compile(r"^ (\d+)/(\d+) kpoint \(Cartesian\) *= *([-0-9.]+) ([-0-9.]+) ([-0-9.]+)"),
            offset=1,
            types=[int, float, float],
        )
        blocks = parser.parse()
        eigenvalues = {}
        occupations = {}
        ntot = len(blocks)
        nkpts = ntot // nspins
        # NOTE: Abacus only report the kpoint on the head MPI process!
        # TODO: Raise a PR to the developers to include all kpoints in the log file.
        if nkpts != nkthis_procs:
            logger.warning("The number of kpoint is (), but only () on this proc")
        assert ntot % nspins == 0
        kpt_cart = np.zeros((nkpts, 3))
        # Process all blocks
        for i, (key, block) in enumerate(blocks):
            ikpt = int(key[0])
            # Sanity check
            if i == 0:
                nkpt_tot = int(key[1])
                assert nkpt_tot == nkpts, "Mismatch in kpont number possible unsupported spin type"
            kpt_cart[ikpt - 1, 0] = float(key[2])
            kpt_cart[ikpt - 1, 1] = float(key[3])
            kpt_cart[ikpt - 1, 2] = float(key[4])
            # Check which spin we are with
            ispin = i // nkpts
            if ispin not in eigenvalues:
                eigenvalues[ispin] = {}
                occupations[ispin] = {}
            occ = [entry[2] for entry in block]
            energy = [entry[1] for entry in block]
            eigenvalues[ispin][ikpt] = np.array(energy)
            occupations[ispin][ikpt] = np.array(occ)
        # Construct overall block
        nkpts = len(eigenvalues[0])
        nspins = len(eigenvalues)
        assert max(eigenvalues[0].keys()) == nkpts
        eigen_arrays = []
        occ_arrays = []
        for spin in range(nspins):
            eigen_arrays.append(np.stack([eigenvalues[spin][i] for i in range(1, nkpts + 1)], axis=0))
            occ_arrays.append(np.stack([occupations[spin][i] for i in range(1, nkpts + 1)], axis=0))
        return np.stack(eigen_arrays, axis=0), np.stack(occ_arrays, axis=0), kpt_cart

    def compose_run_status(self) -> dict:
        """
        Check if the calculation completed successfully by looking for 'Total  Time'
        at the end of the running log file and compose the run status dictionary.

        Also detects convergence status from the running log.

        :returns: Dictionary with completion status information
        """
        run_status = {"completed": False, "completion_marker_found": False, "termination_marker": None}

        try:
            # Check if we have any lines to analyze
            if not self.lines:
                logger.warning("Empty file content provided for completion check")
                return run_status

            # Get the last few lines to check for completion markers
            last_lines = self.lines[-10:] if len(self.lines) >= 10 else self.lines

            # Check for ABACUS completion marker
            completion_marker = "Total  Time"  # ABACUS standard completion marker

            # Check for completion marker in the last lines
            for line in reversed(last_lines):
                line_stripped = line.strip()

                # Check for successful completion marker
                if completion_marker in line_stripped:
                    run_status["completed"] = True
                    run_status["completion_marker_found"] = True
                    run_status["termination_marker"] = completion_marker
                    logger.info(f"Found completion marker '{completion_marker}' in line: {line_stripped}")
                    break

            # If no completion marker found, calculation is incomplete
            if not run_status["completed"]:
                logger.warning(f"Completion marker '{completion_marker}' not found in log file")

            # Detect convergence status from the full log
            notifications = self.parse_notifications()
            run_status["notifications"] = notifications

        except Exception as e:
            logger.error(f"Error checking calculation completion: {e!s}")
            run_status["completed"] = False
            run_status["termination_marker"] = f"error: {e!s}"

        return run_status

    def parse_notifications(self) -> list:
        """
        Scan the running log for convergence and error notifications.

        Detects ABACUS-specific patterns:
        - SCF convergence:
          current branches: "!!SCF IS NOT CONVERGED!!" / "#SCF IS CONVERGED#"
          LTS branches: "!! convergence has not been achieved @_@" / "charge density convergence is achieved"
        - Ionic convergence: "Relaxation is (not) converged"
        - Geometry convergence: "Geometry relaxation is not converged"
        - Mixed state: "Relaxation is converged, but the SCF is unconverged"

        Preserve the full encounter order so callers can reason about the final state
        of a relaxation instead of just the presence of any earlier warning.

        :returns: List of notification dicts with 'name' and 'message' keys
        """
        notifications = []

        # Patterns to search for in the running log
        patterns = {
            "scf_not_converged": re.compile(r"!!SCF IS NOT CONVERGED!!|!!\s*convergence has not been achieved\s*@_@"),
            "scf_converged": re.compile(r"#SCF IS CONVERGED#|charge density convergence is achieved"),
            "ionic_not_converged": re.compile(r"Relaxation is not converged"),
            "ionic_converged": re.compile(r"Relaxation is converged!"),
            "geometry_not_converged": re.compile(r"Geometry relaxation is not converged"),
            "relax_scf_not_converged": re.compile(r"Relaxation is converged, but the SCF is unconverged"),
        }

        for line in self.lines:
            for name, pattern in patterns.items():
                if pattern.search(line):
                    notifications.append({"name": name, "message": line.strip()})

        return notifications

    def parse_runtime_warnings(self) -> list:
        """
        Scan the running log for warning-like messages not mirrored into ``warning.log``.

        ABACUS commonly emits numerical quality warnings as ``Notice: ...`` lines in
        ``running_*.log``. Keep these in a separate list so they can be merged into the
        parsed misc output without conflating them with convergence notifications.
        """
        patterns = (
            re.compile(r"^\s*Notice:\s*(.+)$"),
            re.compile(r"^\s*Warning:\s*(.+)$", re.IGNORECASE),
        )
        warnings = []
        seen = set()

        for line in self.lines:
            stripped = line.strip()
            for pattern in patterns:
                match = pattern.match(stripped)
                if match is None:
                    continue
                message = match.group(1).strip()
                key = ("running_log", message)
                if key not in seen:
                    seen.add(key)
                    warnings.append({"source": "running_log", "message": message})
                break

        return warnings


class BlockParser:
    """Parser to extract blocks of data"""

    DEFAULT_END_CHAR = ["------", "++++++"]

    def __init__(self, lines: List[str], key_re, offset=1, types=None, end_characters=None):
        """
        A parser to parse blocks of data by searching a title line
        Example:
            HEADER  <- header line used for matching
            XXXX       ^
            ---------  |
            A 1 B 2    | Data starts here so offset is 3
            A 1 B 2
            C 1 D 2
            ---------  <- data ends here so the end character is "-----" (default)

        :param lines: A list contains string of each line
        :param key_re: The regular expression to match the presence of the block
        :param offset: Offset from the header to the real data
        :param types: The types of the data for each line
        """
        self.lines = lines
        self.key_re = key_re
        self.types = types
        self.offset = offset
        self.blocks = []
        self.end_characters = [] if not end_characters else end_characters
        self.end_characters += self.DEFAULT_END_CHAR

    def parse(self):
        """Parse the data"""
        for i, line in enumerate(self.lines):
            m = self.key_re.match(line)
            # not matching a header line
            if m is None:
                continue
            # We have found a matched group
            block_name = m.groups()
            block_tokens = []
            j = i + self.offset
            while j < len(self.lines):
                this_line = self.lines[j].strip()
                # Break with empty line
                if not this_line:
                    break
                # Break with predefined sequence such as ---- or +++++
                if any(key in this_line for key in self.end_characters):
                    break
                tokens = self.lines[j].strip().split()
                block_tokens.append(tokens)
                j += 1
            self.blocks.append((block_name, block_tokens))

        if self.types is not None:
            self.blocks = self.convert_type()
        return self.blocks

    def convert_type(self):
        """Convert the match data to the correct type"""
        assert self.blocks
        converted = []
        for block_name, block_tokens in self.blocks:
            new_block = []
            for tokens in block_tokens:
                # Use the type constructors to convert the string data to the right type
                new_block.append([constructor(token) for constructor, token in zip(self.types, tokens)])
            converted.append([block_name, new_block])
        return converted


class TimejsonParser(BaseRawParser):
    """Parse ABACUS time.json and expose high-level timing metrics."""

    def __init__(self, fhandle):
        super().__init__(fhandle)
        self.data = json.loads(self.content)

    def parse(self) -> dict:
        total_time = self.data.get("total")
        stress_time = self._get_time("Stress_PW", "cal_stress") or self._get_time("Force_Stress_LCAO", "getForceStress")
        force_time = self._get_time("Forces", "cal_force_nl")
        return {
            "total_time": total_time,
            "stress_time": stress_time,
            "force_time": force_time,
        }

    def _get_time(self, class_name: str, func_name: str) -> float | None:
        for entry in self.data.get("sub", []):
            if entry.get("class_name") != class_name:
                continue
            for sub_entry in entry.get("sub", []):
                if sub_entry.get("name") == func_name:
                    return sub_entry.get("cpu_second")
        return None


class PdosParser(BaseRawParser):
    """Parse ABACUS PDOS XML output."""

    def parse(self) -> dict:
        root = ET.fromstring(self.content)
        nspin = int(root.findtext("nspin"))
        energy = [float(value) for value in root.findtext("energy_values", "").split()]
        orbitals = []
        for orbital in root.findall("orbital"):
            data = [[] for _ in range(nspin)]
            for line in orbital.findtext("data", "").splitlines():
                tokens = line.split()
                if not tokens:
                    continue
                for spin in range(min(nspin, len(tokens) - 1)):
                    value = float(tokens[spin + 1])
                    data[spin].append(-value if spin == 1 else value)
            orbitals.append(
                {
                    "index": int(orbital.get("index")),
                    "atom_index": int(orbital.get("atom_index")),
                    "species": orbital.get("species"),
                    "l": int(orbital.get("l")),
                    "m": int(orbital.get("m")),
                    "z": int(orbital.get("z")),
                    "data": data,
                }
            )

        return {"nspin": nspin, "energy": energy, "orbitals": orbitals}


KBAR_TO_EV_PER_ANGSTROM3 = 6.241509074e-4


class BandsParser(BaseRawParser):
    """Parser to process the BNADS_XX.dat files"""

    def parse(self):
        """Parse the bands.dat file"""
        arrays = []
        for line in self.lines:
            if not line:
                continue
            arrays.append(np.fromstring(line, sep=" ", dtype=float))
        data = np.stack(arrays, axis=0)[:, 1:]
        kdist = data[:, 0]
        eigenvalues = data[:, 1:]
        return kdist, eigenvalues


class KpointsParser(BaseRawParser):
    """
    Parse the kpoints file in the suffix.out folder
    """

    def parse(self):
        """Read the output kpoints file"""

        line = self.lines[0]
        nkpts = int(line.strip().split()[-1])
        assert self.lines[1].startswith("K-POINTS DIRECT COORDINATES")
        points = []
        weights = []
        for i in range(nkpts):
            tokens = self.lines[i + 3].strip().split()
            points.append([float(tokens[i]) for i in range(1, 4)])
            weights.append(float(tokens[4]))
        return points, weights


class InternalParametersParser(BaseRawParser):
    """
    Parse the INPUT file in the suffix.out folder
    NOTE: This does not work for a general INPUT file
    """

    def parse(self):
        """Read the output internal parameters file"""
        out_dict = {}
        # Skip the first line
        for _line in self.lines[1:]:
            if _line.startswith("#"):
                continue
            line = _line.strip()
            if not line:
                continue
            # Remove the trialing # comments
            match = re.match(r"^(.+) *#.*$", line)
            if match:
                tokens = match.group(1).split(maxsplit=1)
            else:
                tokens = line.split(maxsplit=1)
            # Add potential null value
            if len(tokens) != 2:
                tokens.append("None")
            out_dict[tokens[0].strip()] = tokens[1].strip()
        return out_dict


class StruParser(BaseRawParser):
    """
    Parse a STRU file
    """

    def parse(self):
        """
        Parse a STRU file
        :returns: A tuple of lattice vectors, positions, species.
        """
        blocks = self.parse_blocks()
        lattice_constant = float(blocks["LATTICE_CONSTANT"][0])  # In bohr
        lattice_vectors = np.array([[float(value) for value in line.split()] for line in blocks["LATTICE_VECTORS"]])
        positions = []
        species = []
        # magnetic_moments = []
        pos_block = blocks["ATOMIC_POSITIONS"]
        coord_type = pos_block[0]
        current_specie = None
        p = 1
        while p < len(pos_block):
            current_specie = pos_block[p]
            # current_magmom = float(pos_block[p+1])
            current_natoms = float(pos_block[p + 2])
            for i in range(int(current_natoms)):
                tokens = pos_block[p + 3 + i].split()
                positions.append([float(value) for value in tokens[:3]])
                species.append(current_specie)
            p += 3 + int(current_natoms)
        positions = np.array(positions)
        # Lattice vectors in angstrom
        lattice_vectors *= lattice_constant / 1.8897261255

        if coord_type == "Direct":
            positions = positions @ lattice_vectors
        elif coord_type == "Cartesian":
            positions *= lattice_constant / 1.8897261255
        elif coord_type == "Cartesian_au":
            positions *= 1.0 / 1.8897261255
        elif coord_type == "Cartesian_angstrom":
            pass
        else:
            raise ValueError(f"Unknown coordinate type {coord_type}")
        self.structure = {"lattice_vectors": lattice_vectors, "species": species, "positions": positions}
        return lattice_vectors, positions, species

    def parse_structure(self):
        """Parse for the structure"""
        return self.parse()

    def parse_blocks(self):
        """Split the file content by their blocks"""
        keywords = ["ATOMIC_SPECIES", "LATTICE_CONSTANT", "LATTICE_VECTORS", "ATOMIC_POSITIONS"]
        blocks = {}
        current_block = None
        for _line in self.lines:
            line = _line.strip()
            # Skip comment lines
            if not line or line.startswith("#"):
                continue
            # Remove any trailing comments
            line = line.split("#", maxsplit=1)[0].strip()
            # Check if we are in a block title line
            is_title = False
            for block_name in keywords:
                if block_name in line:
                    current_block = block_name
                    blocks[current_block] = []
                    is_title = True
                    continue
            if is_title:
                continue
            # We are in a block - record the content
            if current_block is not None:
                blocks[current_block].append(line)
        self.blocks = blocks
        return blocks


class WarningLogParser(BaseRawParser):
    """
    Parse ABACUS warning.log file.

    ABACUS writes warnings via WARNING() and WARNING_QUIT() functions.
    Format: <file>  warning : <description>
    """

    WARNING_PATTERN = re.compile(r"^\s*(\S+)\s+warning\s*:\s*(.+)$", re.IGNORECASE)

    def parse(self) -> list:
        """
        Parse warning.log and return a list of notification dicts.

        :returns: List of dicts with 'source', 'message' keys
        """
        notifications = []
        for line in self.lines:
            match = self.WARNING_PATTERN.match(line.strip())
            if match:
                notifications.append({"source": match.group(1), "message": match.group(2).strip()})
        return notifications
