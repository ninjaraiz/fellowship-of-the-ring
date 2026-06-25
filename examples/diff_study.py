import sys
sys.path.append('/home/m.jaraiz/repos/pyLowOrder/')
from FotR import FRODO

def read_db(datafolder, case_idx):
    db = FRODO(root_dir = datafolder, format = 'CODA', initial_parse = True)
    
    db.extract_inputs(
        id_groups = (3,),
        cases_idx = case_idx,
        vtu_type='surface',
        verbose=False
        )

    # db.extract_inputs(
    #     id_groups = (4,),
    #     cases_idx = case_idx,
    #     vtu_type='volume',
    #     verbose=False
    #     )
    
    for stage in [0, 1]:
        db.extract_outputs(
            id_groups=(3,),
            stage=stage, cases_idx = case_idx,
            var_name_excluded = [
                'BoundaryValues_CoefSkinFrictionX',
                'BoundaryValues_CoefSkinFrictionY',
                'BoundaryValues_CoefSkinFrictionZ'
                ],
            vtu_type='surface',
            )
        
        # db.extract_outputs(
        #     id_groups=(4,),
        #     stage=stage, cases_idx = case_idx,
        #     var_name_excluded = [],
        #     vtu_type='volume',
        #     )
    
    return db

case_idx = list(range(5))
db_0 = read_db(
    datafolder = '/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_basic/',
    case_idx = case_idx,
    )

from FotR import SAM
import matplotlib.pyplot as plt
import imageio
from numpy import diff, max, linspace
list_fig = []
xyz_sort, order_sort = SAM.Weapons.sort_by_centroid(db_0.data_dict['CADGroup_3']['Coord'])
cp = db_0.data_dict['CADGroup_3']['Vars']['0']['BoundaryValues_CoefPressure'][order_sort, :]

scale = 7

case = 2

# for stencil in range(10, 300):
for radius_coef in linspace(0.1, 5, 100):
    grad_cp = SAM.DifferentialOperators.gradient(
        X = xyz_sort,
        f = cp,
        radius=radius_coef * max(diff(cp, axis=0)),
        poly_order = 2
    )

    mask_intrados = xyz_sort[:, 2] > 0
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(
        xyz_sort[mask_intrados, 0],
        xyz_sort[mask_intrados, 2],
        c='black',
        s=1
    )
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_ylim(bottom = xyz_sort[:, 2].min()*scale, top = xyz_sort[:, 2].max()*scale)

    ax_cp = ax.twinx()
    ax_cp.scatter(
        xyz_sort[mask_intrados, 0], cp[mask_intrados, case], c='green', s=2
    )
    ax_cp.spines['left'].set_position(('outward', 60))
    ax_cp.spines['left'].set_color('green')
    ax_cp.tick_params(axis='y', colors='green')
    ax_cp.invert_yaxis()
    ax_cp.yaxis.set_label_position('left')
    ax_cp.yaxis.tick_left()
    ax_cp.spines['right'].set_visible(False)

    ax_twin = ax.twinx()
    ax_twin.scatter(
        xyz_sort[mask_intrados, 0], grad_cp[0, mask_intrados, case], s=3, c='red'
    )
    ax_twin.set_yscale('symlog')
    # ax_twin.spines['right'].set_position(('outward', 30 +_ * 30))
    ax_twin.spines['right'].set_color('red')
    ax_twin.tick_params(axis='y', colors='red')
    fig.suptitle(f'Case {case} - radius_coef {radius_coef}')
    list_fig.append(fig)
    plt.close(fig)
    # fig.savefig(f'/home/m.jaraiz/repos/fellowship-of-the-ring/examples/pictures_diff/s_{stencil}.png')

import os
folder_path = '/home/m.jaraiz/repos/fellowship-of-the-ring/examples/pictures_diff'
gif_path = os.path.join(folder_path, f'case_{case}_radius.gif')
with imageio.get_writer(gif_path, mode='I', duration=1.5) as writer:
    for fig in list_fig:
        # Save the figure to a temporary file
        temp_path = os.path.join(folder_path, 'temp.png')
        fig.savefig(temp_path)
        plt.close(fig)
        # Read the image and append it to the gif
        image = imageio.imread(temp_path)
        writer.append_data(image)
        # Remove the temporary file
        os.remove(temp_path)