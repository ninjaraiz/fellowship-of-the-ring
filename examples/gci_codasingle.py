"""
GCI study with CODA_SINGLE rings: 3 flight conditions x 5 meshes.

Layout (one GANDALF root per flight condition, as in the hand-made
``GCI/aoa_1.000_m_0.8`` case)::

    GCI/
    ├── aoa_1.000_m_0.8/
    │   ├── outputs/f0.8, f1, f2, f5, f10/   (run_sst_v4.py + run.sh + mesh)
    │   └── metadata/cases_metadata.json     (FRODO-readable, mesh_map incl.)
    ├── aoa_2.500_m_0.9/
    └── aoa_4.000_m_1.05/

The ``mesh`` factor is a design variable, so every row of ``df_cases``
is unique.  Folders only first; node assignment + Slurm submission run
at the end (``submit_cases`` actually calls ``sbatch`` — review before
running this script).
"""

import os

import pandas as pd

from FotR import GANDALF

# --------------------------------------------------------------------------
# Adjustable inputs
# --------------------------------------------------------------------------
DATASET_DIR = '/home/m.jaraiz/Documentos/DATASETS/data_TIFON'
GCI_DIR = os.path.join(DATASET_DIR, 'GCI')
SOURCES_DIR = os.path.join(DATASET_DIR, 'sources')

# Flight conditions (aoa, mach). First one matches the hand-made case.
FLCC = [(1.0, 0.8), (2.5, 0.9), (4.0, 1.05)]

# Mesh refinement factor -> mesh file (order matters: folder f* per factor).
MESHES = {
    0.8: 'mesh_TIFON_v8_f0.8.msh',
    1: 'mesh_TIFON_v8_f1.msh',
    2: 'mesh_TIFON_v8_f2.msh',
    5: 'mesh_TIFON_v8_f5.msh',
    10: 'mesh_TIFON_v8_f10.msh',
}

# Placeholder literals (cf. example_gandalf_TIFON.ipynb: c from the
# airfoil file, cm = -c/4, ALT fixed at 11000 m).
CHORD = '0.3425'
CM = '-0.085625'
ALT = '11000'

# Slurm submission (cf. example_gandalf_TIFON.ipynb).
NODES = [f'n00{n}' for n in [4, 5, 6, 7]]
CPUS_PER_JOB = 48
FILE_SH = 'run.sh'


def main():
    mesh_factors = list(MESHES)
    rings = []

    for aoa, mach in FLCC:
        root = os.path.join(GCI_DIR, f'aoa_{aoa:.3f}_m_{mach}')
        gdf = GANDALF(
            root,
            eq_type='rans',
            num_stages=2,
            version='flowsimulator2024',
            ring='coda_single',
        )
        gdf.define_cases(
            method='external',
            external_dataframe=pd.DataFrame({
                'aoa': [aoa] * len(mesh_factors),
                'mach': [mach] * len(mesh_factors),
                'mesh': mesh_factors,
            }),
        )
        gdf.generate_folders(
            base_files=['run_sst_v4.py', 'run.sh'],
            mesh_paths=[
                os.path.join(GCI_DIR, 'meshes', MESHES[m])
                for m in mesh_factors
            ],
            script_dir=SOURCES_DIR,
            folder_fmt='f{mesh:g}',
            overwrite=False,
            update_base_files=True,
            data_to_update={
                'AOA_PLACEHOLDER': 'aoa',
                'MACH_PLACEHOLDER': 'mach',
                'ALT_PLACEHOLDER': ALT,
                'PYTHON_FILE_PLACEHOLDER': 'run_sst_v4.py',
                'CHORD_PLACEHOLDER': CHORD,
                'CM_PLACEHOLDER': CM,
            },
        )
        rings.append(gdf)
        print(f'Prepared {root}: {gdf.folders_name}')

    for gdf in rings:
        gdf.assign_jobs(
            file_sh=FILE_SH,
            nodes=NODES,
            cpus_per_job=CPUS_PER_JOB,
            submit=True,
        )

    for gdf in rings:
        gdf.submit_cases()


if __name__ == '__main__':
    main()
