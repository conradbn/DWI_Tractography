import re
import os
from glob import glob
from datetime import datetime

def prep_inputs(subj, year):
    # Specify the output directory (likely on server)
    path_out = f'/Volumes/NBL_Projects/NSF_SNP/DWI_Project/Processed_Data/{subj}/{year}'

    # Specify the temporary processing directory (ideally on local hard drive)
    # Add unique date string to the folder name (down to seconds)
    if os.path.isdir('/Users/nbl_imac2'):
        path_tmp = f'/Users/nbl_imac2/Desktop/tmp_proc/tmp_{subj}_{year}'  #datetime.now().strftime('%Y-%m-%d-%H-%M-%S'
    elif os.path.isdir('/Users/nbl_imac'):
        path_tmp = f'/Users/nbl_imac/Desktop/tmp_proc/tmp_{subj}_{year}'
    else:
        print('YOU ARE NOT RUNNING ON AN NBL IMAC!! CANNOT PROCEED')

    # Specify path to raw data directory
    path_data = f'/Volumes/NBL_Projects/NSF_SNP/Subject_Data/{subj}/{year}'

    # Specify inputs (requires full paths)
    dwi = glob(f'{path_data}/**/*DTI*dir*.DCM', recursive=True)
    revPE = glob(f'{path_data}/**/*DTI_APA*.DCM', recursive=True)

    # Find the FreeSurfer data we need, based on scan ID from imaging files
    # Exclude files with "OLD" in path, as these are from FreeSurfer editing runs
    scan_id = re.findall(r'_(\d{6})', dwi[0])[0]

    struct = [fn for fn in
              glob(f'/Volumes/NBL_Projects/NSF_SNP/FreeSurfer/PRICE_NSF/{scan_id}*/**/brain.finalsurfs.mgz',
                   recursive=True)
              if 'OLD' not in fn]

    aseg = [fn for fn in
            glob(f'/Volumes/NBL_Projects/NSF_SNP/FreeSurfer/PRICE_NSF/{scan_id}*/**/aseg.mgz', recursive=True)
            if 'OLD' not in fn]

    # Set the number of streamlines to generate during tractography
    n_streamlines = '10M'

    # Set the number of threads for MRtrix3 commands to use
    # Restricting to allow for parallel processing of multiple subject
    n_threads = '4'

    return path_out, path_tmp, path_data, dwi, revPE, scan_id, struct, aseg, n_streamlines, n_threads
