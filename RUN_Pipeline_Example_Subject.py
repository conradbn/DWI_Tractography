import DWI_Tractography_Pipeline_MRtrix3 as pipeline
from setup_pipeline_for_SNP_Project import prep_inputs

# Set subject and visit code
subj = 'SNP005'
year = 'Y2'

# Prepare inputs (project-specific)
path_out, path_tmp, path_data, dwi, revPE, scan_id, struct, aseg, n_streamlines = prep_inputs(subj,year)

# Report the input files (consider verifying before proceeding)
dwi, revPE, struct, aseg = pipeline.check_inputs(path_out, path_tmp, dwi, revPE, struct, aseg, n_streamlines)

# Run each step of pipeline
pipeline.copy_data_to_tmp_directory(path_tmp, dwi, revPE, struct, aseg)
pipeline.prep_data(path_tmp, dwi, revPE)
pipeline.run_denoising(path_tmp)
pipeline.run_topup_and_eddy(path_tmp)
pipeline.prep_seg_for_ACT(path_tmp, aseg, struct)
pipeline.estimate_resp_function(path_tmp)
pipeline.run_csd(path_tmp)
pipeline.generate_streamlines_and_run_sift2(path_tmp, n_streamlines)
pipeline.copy_data_to_output_directory(path_out, path_tmp, n_streamlines)
pipeline.create_post_process_QA_figure(path_out)