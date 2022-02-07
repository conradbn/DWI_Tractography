import re
import os
from glob import glob
import numpy as np
import DWI_Tractography_Pipeline_MRtrix3 as pipeline
from setup_pipeline_for_SNP_Project import prep_inputs

# Get list of all existing subject directories where (the first) DTI scan exists
subs = glob(f'/Volumes/NBL_Projects/NSF_SNP/Subject_Data/SNP*/Y*/mri/*/DICOM/*DTI_56dir_b2000_22iso_mb2.01.DCM', recursive=True)

# Split this list into 4 groups for parallel processing
subs_split = np.array_split(subs, 4)
run_split = 0
subs_tmp = subs_split[run_split]

# Loop through data list
for s in subs_tmp:
    # Clear console for each subject
    os.system('clear')

    # Set subject and visit code
    subj = re.findall(r'SNP\d{3}', s)[0]
    year = re.findall(r'Y\d{1}', s)[0]

    # Prepare inputs (project-specific)
    path_out, path_tmp, path_data, dwi, revPE, scan_id, struct, aseg, n_streamlines, n_threads = prep_inputs(subj, year)

    # Check all the inputs are non-empty (e.g. FreeSurfer may not have been run yet)
    # Skip subject if this is the case
    inputs = [path_out, path_tmp, path_data, dwi, revPE, scan_id, struct, aseg, n_streamlines, n_threads]
    if not all(v for v in inputs):
        continue

    # Report the input files (consider verifying before proceeding)
    dwi, revPE, struct, aseg = pipeline.check_inputs(path_out, path_tmp, dwi, revPE, struct, aseg, n_streamlines)

    # Run each step of pipeline
    pipeline.copy_data_to_tmp_directory(path_tmp, dwi, revPE, struct, aseg)
    pipeline.prep_data(path_tmp, dwi, revPE, n_threads)
    pipeline.run_denoising(path_tmp, n_threads)
    pipeline.run_topup_and_eddy(path_tmp, n_threads)
    pipeline.prep_seg_for_ACT(path_tmp, aseg, struct, n_threads)
    pipeline.estimate_resp_function(path_tmp, n_threads)
    pipeline.run_csd(path_tmp, n_threads)
    pipeline.generate_streamlines_and_run_sift2(path_tmp, n_streamlines, n_threads)
    pipeline.copy_data_to_output_directory(path_out, path_tmp, n_streamlines)
    pipeline.create_post_process_QA_figure(path_out)
    pipeline.clear_tmp(path_out, path_tmp, n_streamlines)

