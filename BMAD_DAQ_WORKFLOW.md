# Bmad simulations with FACET-II DAQ settings

This guide explains how to simulate the FACET-II linac in Bmad from **L0AFEND to END** with the magnet and RF settings saved in a FACET-II DAQ scan. There are two workflows:

- **A. Original workflow + DAQ settings.** The standard `initializeTao()` / `trackBeam()` flow. The only change is that the lattice is overwritten with values from a DAQ scan.
- **B. `get_tao_from_experiment()`.** One call that builds the lattice, loads the DAQ settings, prepares the beam, sets the collective effects and runs the simulation.

Examples of every option are in [`examples/Kladov_BMAD_tutorial.ipynb`](examples/Kladov_BMAD_tutorial.ipynb). How the bends are treated is shown in [`examples/Kladov_BMAD_dipole_showcase.ipynb`](examples/Kladov_BMAD_dipole_showcase.ipynb).

> This guide replaces `OUTDATED_2026-06_FACET2_S2E_Kladov_BMAD_upgrade.pdf`. That PDF describes a separate `FACET2_S2E_Kladov` package, which has since been merged into `FACET2_S2E`.

---

## 1. Installation (S3DF, from scratch)

1. **Pick a node that can see the DAQ data.** DAQ scans are stored under
   `/sdf/data/ad/fs/transition/nfs/slac/g/facet/matlab/data_prod/nas-li20-pm00/`.
   This path is **not** mounted on the `sdflogin` nodes. Work on a node where `ls` of that path succeeds, for example a compute node through Slurm or S3DF OnDemand. Without DAQ data you can work anywhere.

2. **Install conda** somewhere with enough space. Home quotas are small, so use a group directory:
   ```bash
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3-Linux-x86_64.sh -p /sdf/group/<group>/<user>/miniforge3
   ```

3. **Clone the repository** without downloading the large LFS files yet:
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/slaclab/FACET2-S2E.git
   cd FACET2-S2E
   ```
   > Until [PR #10](https://github.com/slaclab/FACET2-S2E/pull/10) is merged, the features described here exist only in that PR. To get them, run `git fetch origin pull/10/head:pr-10 && git checkout pr-10` after the commands above.

4. **Create the conda environment.** Give the location explicitly, because the `prefix:` line in the yml points to another user's directory:
   ```bash
   conda env create -f bmadQPADCondaEnv.yml -p /sdf/group/<group>/<user>/envs/bmad-qpad
   conda activate /sdf/group/<group>/<user>/envs/bmad-qpad
   ```

5. **Download the beam files** with Git LFS (git-lfs comes with the environment). A full `git lfs pull` is larger than 2 GB. This guide only needs one beam:
   ```bash
   git lfs install
   git lfs pull --include="beams/2024-12-11_Impact_OneBunch/*"
   ```

6. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```
   Editable mode matters: the package finds the lattice (`bmad/`), the configs (`setLattice_configs/`) and the scratch directory (`temp_beam/`) inside this repository.

7. **(Optional) Register the Jupyter kernel:**
   ```bash
   python -m ipykernel install --user --name bmad-qpad --display-name "bmad-qpad"
   ```

---

## 2. Choosing a DAQ scan

A scan is identified by three strings:

| Argument | Example | Meaning |
|---|---|---|
| `experiment` | `"BEAMPHYS"` | Experiment name |
| `scan_number` | `"14438"` | DAQ scan number |
| `date` | `"/2026/20260121"` | `/YYYY/YYYYMMDD` of the scan |

The file that gets read is
`<DAQ root>/<experiment><date>/<experiment>_<scan_number>/<experiment>_<scan_number>.mat`.

---

## 3. Workflow A: original workflow plus DAQ settings

Everything behaves as in the original FACET2-S2E: the lattice from `defaults.yml`, CSR only in the bunch compressors, and the beam loaded through `activeBeamFile.h5`. The DAQ values are applied on top of that lattice.

```python
import FACET2_S2E as qs
from Experimental_functions import DATASET

REPO = "/path/to/FACET2-S2E"
DAQ_ROOT = "/sdf/data/ad/fs/transition/nfs/slac/g/facet/matlab/data_prod/nas-li20-pm00"

tao = qs.initializeTao(
    filePath=REPO,
    inputBeamFilePathSuffix="/beams/2024-12-11_Impact_OneBunch/2024-12-11_oneBunch.h5",
    numMacroParticles=50000,
    csrTF=True,
)

ds = DATASET("", "BEAMPHYS", "14438", pathfull=f"{DAQ_ROOT}/BEAMPHYS/2026/20260121")

# Everything (RF, quads, sextupoles, correctors):
qs.edit_tao_based_on_experiment_database(tao, ds)            # correctors_coef=0 to skip the correctors
# ...or only the RF (energy profile), leaving the magnets untouched:
# qs.edit_energy_tao_based_on_experiment_database(tao, ds)

qs.trackBeam(tao, filepath=REPO, trackStart="L0AFEND", trackEnd="END",
             centerBC14=True, assertBC14Energy=True,
             centerBC20=True, assertBC20Energy=True)

P = qs.getBeamAtElement(tao, "END")    # openPMD ParticleGroup
```

In this workflow the beam keeps the energy stored in its file. The DAQ RF settings can shift the design energy at L0AFEND, so the beam and the lattice may start at slightly different energies. Workflow B matches them automatically.

---

## 4. Workflow B: `get_tao_from_experiment()`

```python
import FACET2_S2E as qs

REPO = "/path/to/FACET2-S2E"

tao = qs.get_tao_from_experiment(
    experiment="BEAMPHYS", scan_number="14438", date="/2026/20260121",
    filepath=REPO,                       # optional with "pip install -e ."
    start="L0AFEND", finish="END",
    file_ext=REPO + "/beams/2024-12-11_Impact_OneBunch/2024-12-11_oneBunch",   # no ".h5"
    N_to_use_from_file=5e4,
    lattice="setLattice_configs/2024-10-22_oneBunch-Copy1.yml",
    csrTF=True, lscTF=True,
    sr_wakes_on=True, lr_wakes_on=True,
)

P = qs.getBeamAtElement(tao, "END", tToZ=False)
qs.print_result_from_tao(tao, location="END", couples=[['x', 'xp'], ['y', 'yp'], ['z', 'pz']], z_from_t=True)
```

To run **without DAQ data**, leave `experiment` and `scan_number` empty. The lattice is then `defaults.yml` plus the file given in `lattice`.

The function does the following, in order:

1. Initializes Tao, applies `defaults.yml` plus `lattice`, and sets the collective effects.
2. If a scan is given, loads its settings (see Section 5).
3. Prepares the beam and writes it to `temp_beam/temp.h5` and `temp_beam/temp_e.h5`:
   - sets the beam energy to the design energy at `start` (see `energy`);
   - optionally resamples, rescales or re-centers it (`N_to_use_from_file`, `moments`, `means`);
   - sets all z to 0 and centers t.
4. Optionally runs the beam-based energy feedback (`energy_edit_on_beam`) and the dipole treatment (`tune_dipoles`).
5. Tracks from `start` to `finish` (if `run=True`). The beam at `start` and `finish` is saved to `temp_beam/<ELEMENT>temp.h5`.

### Main options

| Option | Default | Meaning |
|---|---|---|
| `filepath` | `None` | Path to this repository. If `None`, it is found from the installed package (requires `pip install -e .`). |
| `lattice` | `2024-10-22_oneBunch-Copy1.yml` | Settings applied on top of `defaults.yml`. |
| `file_ext` | `""` | Input beam, without `.h5`. If `""`, a Gaussian beam is generated from `moments`. |
| `N_to_use_from_file` | `None` | Randomly sample this many particles from the file. |
| `gaussFromExternal`, `N_in_simple_bunch` | `False`, `5e4` | Replace the file beam with a Gaussian beam of the same RMS sizes. |
| `energy` | `None` | `None`: design energy at `start`. `-1`: keep the file energy. A positive number: that energy in MeV. |
| `moments`, `means`, `charge` | — | Rescale or re-center the beam (see the docstring). |
| `csrTF`, `lscTF` | `False` | CSR and space charge. When on, they are applied to **all** elements. |
| `csr_method`, `lsc_method`, `n_bin`, `grid_size` | `1_dim`, `slice`, 32, `[32,32,32]` | Collective-effect calculation settings. |
| `sr_wakes_on`, `lr_wakes_on` | **`False`** | Short- and long-range wakes. **Off by default.** |
| `edit_only_energy_from_exp` | `False` | Load only the RF settings from the DAQ, not the magnets. |
| `correctors_coef` | **`0`** | Scale factor for the DAQ corrector values. `0` turns them off; `-1/10` uses the experimental values. |
| `correctors_from_beg` | `False` | Load the sector-10 correctors from the start of the injector instead of from the dogleg. |
| `desired_P0Cs_MeV` | `[None]*4` | Design energies other than 125 / 335 / 4500 / 10000 MeV, keeping the bend geometry of the nominal energies. |
| `edited_bunch_energy_at_checkpoints_MeV` | `[None]*4` | Force the beam energy at BX0FBEG, BC11CBEG, ENDL2F and ENDL3F_2. |
| `energy_edit_on_beam`, `desired_beam_energies_for_the_feedback` | `False` | Beam-based energy feedback: scale the cavities until the tracked beam reaches the target energies. |
| `tune_dipoles`, `tune_dipoles_to_125_335_4500_10000_MeV` | `False` | Fix the bend fields to those of the DAQ energies (or of the nominal energies). |
| `run`, `locationsToSave` | `True`, `[start, finish]` | Track now, and where to save the beam. |

The corrector defaults differ on purpose: `edit_tao_based_on_experiment_database()` loads everything, correctors included, while `get_tao_from_experiment()` leaves the correctors off because simulations usually don't want them.

### Running an already initialized lattice

With `run=False`, you can modify `tao` first and then track with:

- `qs.run_initialized_sim(tao, start, finish)` (add `treat_dipoles_TF=True` to fix the bend fields to the nominal-energy values first)
- `qs.run_initialized_sim_edit_lattice_energy_for_dipoles(tao, start, finish, desired_P0Cs_MeV=...)`
- `qs.run_initialized_sim_edit_bunch_energy(tao, start, finish, edited_bunch_energy_at_checkpoints_MeV=...)`
- `qs.set_beam(tao, file)` to load another beam.

---

## 5. What is loaded from the DAQ

Each value is the mean over all shots in the scan.

| Group | Lattice elements | DAQ source |
|---|---|---|
| Injector RF | `L0AF` phase and voltage, `L0BF` phase and voltage | `KLYS_LI10_31_SFB_PDES` (−20° offset), `KLYS_LI10_31_ADES`, `KLYS_LI10_41_SFB_PDES`, `KLYS_LI10_41_ADES`. Both voltages are then scaled together so the design energy at BX10661 is 125 MeV. |
| L1 / L2 / L3 phases | All cavities of each linac get one phase | `KLYS_LI11_11_SSSB_PDES`, `LI14_SBST_1_PHAS`, `LI19_SBST_1_PHAS` |
| L1 / L2 / L3 amplitudes | Not loaded | The gradients are set so the energies are 335 / 4500 / 10000 MeV. |
| Quadrupoles | Sector 10 (11 quads), sector 11 (11 quads), Q19851 and Q19871, and sector 20 (W-chicane Q1–Q6 including the boost supplies, final focus, spectrometer) | `*_BDES`, `*_BCON`, `*_BACT` |
| Sextupoles | S1EL, S2EL, S3EL, S3ER, S2ER, S1ER strengths | `LI20_LGPS_*_BACT` |
| Sextupole movers | X/Y offsets of S1EL, S2EL, S2ER, S1ER (S3 stays at 0) | `SIOC_SYS1_ML00_AO*` (mm) |
| Correctors | Sector 10 from XC10721 (or from the start with `correctors_from_beg`) and sector 11, scaled by `correctors_coef` | `*COR_*_BDES`, `*COR_*_BCON` |
| Dipole energies | DL10, BC11, BC20 bends (BC14 stays at 4500 MeV). Used only with `tune_dipoles=True`. | `BEND_*`, `LI20_LGPS_*` |

**Not loaded:**
- L1/L2/L3 amplitudes;
- phases of individual klystrons (one phase per linac is used);
- quadrupoles in sectors 12–18 and most of sector 19 (these are not saved in the DAQ);
- bend fields (setting `B_FIELD` in Bmad moves every downstream element);
- laser heater, XTCAV and kickers;
- the beam distribution.

These keep their values from `defaults.yml` and the `lattice` file.

---

## 6. Other functions

All are available as `qs.<name>` after `import FACET2_S2E as qs`.

| Area | Functions |
|---|---|
| Beam creation and editing | `make_simple_bunch*`, `edit_bunch_parameters`, `edit_bunch_parameters_from_PG`, `modifyInputBeamSimple`, `cut_length` |
| Energy tuning | `tune_to_P0Cs`, `edit_energy_based_on_beam_inj` / `_L1` / `_L2` / `_L3` / `_all` |
| Collective effects | `applyBMADCollectiveEffectSettings` |
| Lattice information | `get_element_array`, `get_rij`, `get_tijk` |
| Sextupoles | `setAllWChicaneSextupoles`, `setAllWChicaneSextupolesXOffsets`, `setAllWChicaneSextupolesYOffsets` |
| Scans | `make_1d_scan`, `make_comparison_dz_2nd_order` |
| Plotting | `print_result`, `print_result_from_tao`, `print_result_from_file`, `plotModKladov`, `make_a_plot`, `enable_plt_styling` |
| Microbunching | `make_modulated_bunch`, `hist`, `get_spectrum`, `print_spec`, `analyze_spec`, `get_microbunching_gain*` |

`initializeTao(autoLoadActiveFile=False, loadCustomLatticeTF=True, latticeFile=...)` sets up only the lattice and the collective effects. The beam is then loaded separately with `set_beam()`. With the default `autoLoadActiveFile=True`, `initializeTao()` and `trackBeam()` behave as before.

---

## 7. Things to know

- **Scratch files are overwritten.** To save disk space, workflow B writes intermediate beams to `<repo>/temp_beam/` under fixed names (`temp.h5`, `temp_e.h5`, `<ELEMENT>temp.h5`), overwriting them on every run. Copy any result you want to keep, and don't run several simulations from the same repository at the same time.
- **Repository path.** With an editable install (`pip install -e .`), `filepath` / `filePath` can be omitted; in a notebook, `str(Path(qs.__file__).resolve().parents[2])` gives the same path. With a regular `pip install .`, always pass it.
- **DAQ values are averaged over the whole scan.** If a magnet or phase was the scanned variable, the lattice gets its average value over all steps.
- **In workflow B, CSR and space charge apply to every element**, not only the bunch compressors. 3D methods (`steady_state_3d`, `fft_3d`) are slow. Test with few particles first and run long simulations through Slurm, not on a login node.
