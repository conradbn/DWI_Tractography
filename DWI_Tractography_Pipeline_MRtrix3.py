import os
import sys
import subprocess
from subprocess import run
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

'''
DWI Tractography Pipeline using MRTrix3 (and some FSL)

Inputs = 
    raw DWI data in DICOM or NII format
    FreeSurfer processed T1w image and segmentation file

Performs full preprocessing and then series of steps to produce a whole-brain structural connectome through the 
generation of probabilistic streamlines 


NOTES FROM DEVELOPMENT:

v0.3: first working version, tested on adult, kindergarten, and infant data (output text saved 
within jupyter notebook). Issue with denoising step on the reverse phase encode volumes, large black stripes in data 
being introduced (I believe has to do with only a few volumes collected). Topup is then performed but missing data 
throws off registration 

v0.4: 
remove denoising of reverse PE data added code to run multi-threaded eddy (though currently doesn't seem to 
actually use more that 2-3 threads max, while MRtrix3 commands use all available. Some discussion on forums about 
having to actually compile eddy on Mac for OpenMP support, but not trivial). commented out the writing log to text 
file code. saving in notebook output for now. flirt command updated to use wider search and sinc interpolation added 
-force flag to make sure any existing processed data is overwritten 

v0.5:
Fix writing to text file + to jupyter cell

v0.6:
Made input format more general, requiring full paths to inputs
Also require specification of output directory + local processing directory

v0.7:
Added QA figure generation


'''

def execute(command):
    ## Shell script execution
    # Print output to notebook and save to text file
    output = run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(output.stdout)
    log = open('out_log.txt', 'a')
    print(output.stdout, file=log)


def check_inputs(path_out, path_tmp, dwi, revPE, struct, aseg, n_streamlines):
    if isinstance(revPE, list):
        print('** WARNING ** Reverse phase encoded datasets provided as a list, using the first one!')
        revPE = revPE[0]
    if isinstance(struct, list):
        print('** WARNING ** Structural datasets provided as a list, using the first one!')
        struct = struct[0]
    if isinstance(aseg, list):
        print('** WARNING ** Aseg datasets provided as a list, using the first one!')
        aseg = aseg[0]

    print(f'Path to output folder = \n   {path_out}\n')
    print(f'Path to temporary folder = \n   {path_tmp}\n')
    print(f'DWI input data = \n   ' + '\n   '.join(dwi) + '\n')
    print(f'DWI reverse phase-encoded data = \n   {revPE}\n')
    print(f'Structural image file from FreeSurfer = \n   {struct}\n')
    print(f'Segmentation file from FreeSurfer = \n   {aseg}\n')
    print(f'Generating the following # of streamlines = \n   {n_streamlines}\n')
    dwi = dwi
    return dwi, revPE, struct, aseg


def copy_data_to_tmp_directory(path_tmp, dwi, revPE, struct, aseg):
    print('Step 01: Copying data to temporary directory...')
    # Make temporary directory
    os.mkdir(str(path_tmp))
    os.chdir(str(path_tmp))
    # Check if DWI is list of files or single file
    if isinstance(dwi, list):
        dwi_all = ' '.join(dwi)
    else:
        dwi_all = dwi
    # Copy to tmp    
    execute(f'cp -va {dwi_all} {revPE} {struct} {aseg} {path_tmp}/')

    # Verify that everything is there
    # First change paths of input files to point to tmp directory
    for ii in range(len(dwi)):
        dwi[ii] = path_tmp + '/' + os.path.basename(dwi[ii])
    revPE = path_tmp + '/' + os.path.basename(revPE)
    struct = path_tmp + '/' + os.path.basename(struct)
    aseg = path_tmp + '/' + os.path.basename(aseg)
    # Check if any files are missing in the tmp directory
    if (any(not os.path.isfile(d) for d in dwi) or
            not os.path.isfile(revPE) or
            not os.path.isfile(struct) or
            not os.path.isfile(aseg)):
        # Print error and exit
        print('**ERROR**: Did not successfully find/copy one of the input files, check manually!')
        sys.exit()


def prep_data(path, dwi, revPE):
    print('Step 02: Preparing data for processing...')
    os.chdir(str(path))

    # Change paths of input files to those in the tmp directory
    for ii in range(len(dwi)):
        dwi[ii] = path + '/' + os.path.basename(dwi[ii])
    revPE = path + '/' + os.path.basename(revPE)

    if dwi[0].endswith('.DCM') or dwi[0].endswith('.dcm'):
        ################################################################
        # Prepare data for preprocessing (DICOM Data)
        # Convert from DICOM to MRtrix3 mif format
        # NOTE - Have to change DCM to dcm in file extension for MRTrix3 to recognize correctly
        for ii in range(len(dwi)):
            execute(f'cp {dwi[ii]} {dwi[ii]}.tmp.dcm')
            execute(f'mrconvert {dwi[ii]}.tmp.dcm tmp_dwi_raw{str(ii)}.mif')

            # Convert reverse PE image to mif
        execute(f'cp {revPE} {revPE}.tmp.dcm')
        execute(f'mrconvert {revPE}.tmp.dcm dwi_raw_revPE.mif')

        # Combine the primary dwi images (if multiple)
        if len(dwi) > 1:
            execute('mrcat -force -axis 3 tmp_dwi_raw*.mif dwi_raw.mif')
        else:
            execute('cp tmp_dwi_raw0.mif dwi_raw.mif')

        # Remove temporary versions of images
        execute('rm -f *tmp* */*tmp*')

    elif dwi[0].endswith('.nii') or dwi[0].endswith('.nii.gz'):
        ################################################################
        # Prepare data for preprocessing (NIFTI Data)
        ## Prep filenames
        # Modify input strings for the pipeline
        # Get the bvec and bval filenames (assumes same prefix as nii/nii.gz)
        bvec = []
        bval = []
        for ii in range(len(dwi)):
            f = dwi[ii]
            if '.gz' in f:
                bvec.append(f.replace('.nii.gz', '.bvec'))
                bval.append(f.replace('.nii.gz', '.bval'))
            else:
                bvec.append(f.replace('.nii', '.bvec'))
                bval.append(f.replace('.nii', '.bval'))

        f = revPE
        if '.gz' in f:
            bvec_revPE = revPE.replace('.nii.gz', '.bvec')
            bval_revPE = revPE.replace('.nii.gz', '.bval')
        else:
            bvec_revPE = revPE.replace('.nii', '.bvec')
            bval_revPE = revPE.replace('.nii', '.bval')

        # Convert to MRtrix3 mif format
        for ii in range(len(dwi)):
            execute(f'mrconvert -fslgrad {bvec[ii]} {bval[ii]} {dwi[ii]} tmp_dwi_raw{str(ii)}.mif')

            # Convert reverse PE image to mif
        execute(f'mrconvert -fslgrad {bvec_revPE} {bval_revPE} {revPE} dwi_raw_revPE.mif')

        # Combine the primary dwi images (if multiple)
        if len(dwi) > 1:
            execute('mrcat -force -axis 3 tmp_dwi_raw*.mif dwi_raw.mif')
        else:
            execute('cp tmp_dwi_raw0.mif dwi_raw.mif')

        # Remove temporary versions of images
        execute('rm -f *tmp* */*tmp*')

    else:
        print('**ERROR**: File extension of input data is not DCM, dcm, nii, or nii.gz!' +
              ' - currently only accept those in this pipeline')
        sys.exit()


def run_denoising(path):
    print('Step 03: Denoising the raw data...')
    os.chdir(str(path))
    #################################################################
    ## Perform initial denoising of the DWI data (per MRTrix3 recommendation)
    execute('dwidenoise -force -noise dwi_raw_noise.mif dwi_raw.mif dwi_raw_dn.mif')
    # execute('dwidenoise -noise dwi_raw_revPE_noise.mif dwi_raw_revPE.mif dwi_raw_revPE_dn.mif')

    # Get the first image (b0) of the DWI data and the reverse phase-encoded scan
    execute('dwiextract -force dwi_raw_dn.mif -bzero dwi_raw_dn_b0.mif')
    execute('dwiextract -force dwi_raw_revPE.mif -bzero dwi_raw_revPE_b0.mif')

    # Combine the initial b0 images into an image pair for topup
    execute('mrconvert -force -coord 3 0 -axes 0,1,2 dwi_raw_dn_b0.mif tmp1.mif')
    execute('mrconvert -force -coord 3 0 -axes 0,1,2 dwi_raw_revPE_b0.mif tmp2.mif')
    execute('mrcat -axis 3 tmp1.mif tmp2.mif dwi_raw_dn_b0_pair.mif')
    execute('rm -f tmp*.mif')


def run_topup_and_eddy(path):
    print('Step 04: Performing topup and eddy motion correction (this can take a few hours)...')
    os.chdir(str(path))
    #################################################################
    ## FSL Preprocessing
    # Uses a wrapper from MRtrix3 that calls eddy and topup
    # NOTE - Not currently using the new slice-to-vol motion correction,
    # as it only is implemented with CUDA version of eddy (computationally expensive). 
    # This feature could be useful in particular for the infant data. Will ultimately require the
    # --mporder and --slspec flags described here https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide
    # Also see for more info https://mrtrix.readthedocs.io/en/latest/dwi_preprocessing/dwifslpreproc.html
    # We will also likely be good to turn on --estimate_move_by_susceptibility for high motion
    execute('export OMP_NUM_THREADS=15; \
        dwifslpreproc dwi_raw_dn.mif dwi_proc.mif \
        -rpe_pair -se_epi dwi_raw_dn_b0_pair.mif -pe_dir PA -align_seepi -nocleanup \
        -export_grad_fsl dwi_raw_fsl.bvec dwi_raw_fsl.bval \
        -eddy_options " --repol --cnr_maps --data_is_shelled --slm=linear" \
        -eddyqc_all QA -nthreads 12')


def prep_seg_for_ACT(path, aseg, struct):
    print('Step 05: Preparing segmentation for anatomically constrained tractography...')
    os.chdir(str(path))
    # Change paths of input files to those in the tmp directory
    aseg = path + '/' + os.path.basename(aseg)
    struct = path + '/' + os.path.basename(struct)

    #################################################################
    ## Prepare FreeSurfer segmentation for anatomically-constrained tractography (ACT)
    # Use nii here (instead of mif) since working with FSL commands
    # Convert MGZ files to nii with FreeSurfer conversion tool
    execute(f'mri_convert {aseg} aseg.nii')
    execute(f'mri_convert {struct} struct.nii')

    # Extract b0 (assumes first image)
    execute('mrconvert -force -coord 3 0 -axes 0,1,2 dwi_proc.mif dwi_proc_b0.nii')

    # Register DWI to structural image
    execute('flirt -dof 6 -interp sinc -searchrx -180 180 -searchry -180 180 -searchrz -180 180 \
        -in dwi_proc_b0.nii -ref struct.nii \
        -out dwi_proc_b0_2struct.nii \
        -omat dwi_proc_b0_2struct.xform.fsl.txt')

    # Convert transformation matrix to MRtrix format
    execute('transformconvert dwi_proc_b0_2struct.xform.fsl.txt dwi_proc_b0.nii struct.nii \
    flirt_import dwi_proc_b0_2struct.xform.mrtrix.txt')

    # Sample structural to DWI space (using inverse transform)
    execute('mrtransform -force struct.nii \
        -linear dwi_proc_b0_2struct.xform.mrtrix.txt \
        struct.2dwi.nii -inverse')

    # Sample segmentation to DWI space (using inverse transform)    
    execute('mrtransform -force aseg.nii \
        -linear dwi_proc_b0_2struct.xform.mrtrix.txt \
        aseg.2dwi.nii -inverse')

    # Generate the 5 tissue file for ACT 
    execute('5ttgen freesurfer -lut /Applications/freesurfer/FreeSurferColorLUT.txt -force \
        aseg.2dwi.nii aseg_5ttgen.mif')

    # Generate GM/WM interface for seeding
    execute('5tt2gmwmi -force aseg_5ttgen.mif aseg_5tt_gmwmi.mif')


def estimate_resp_function(path):
    print('Step 06: Estimating tissue response functions...')
    #################################################################
    ## DWI Response Function Estimation

    # Get DWI mask
    execute('dwi2mask -force dwi_proc.mif dwi_proc_mask.mif')

    # Create FA and MD maps for future reference and QA
    execute('dwi2tensor -force dwi_proc.mif tmp.mif; \
         tensor2metric -mask dwi_proc_mask.mif -fa dwi_proc_FA.mif tmp.mif; \
         tensor2metric -mask dwi_proc_mask.mif -adc dwi_proc_MD.mif tmp.mif; \
         rm -f tmp.mif')

    # NOTE - estimated here for individual subject but,
    # in case of group analysis, best to take group averaged functions.
    execute('dwi2response -force dhollander dwi_proc.mif \
        dhollander_wm_response_sub.txt \
        dhollander_gm_response_sub.txt \
        dhollander_csf_response_sub.txt \
        -voxels dhollander_voxels_sub.mif')


def run_csd(path):
    print('Step 07: Running constrained spherical deconvolution...')
    os.chdir(str(path))
    #################################################################
    ## Run Constrained Spherical Deconvolution
    # Generates fiber orientation distribution at each voxel
    ## Multi-shell multi-tissue constrained spherical deconvolution  
    execute('dwi2fod -force msmt_csd -mask dwi_proc_mask.mif  dwi_proc.mif \
        *wm_response*.txt fod_wm.mif  \
        *gm_response*.txt fod_gm.mif  \
        *csf_response*.txt fod_csf.mif')

    ## Run the normalizaton/bias correction step
    execute('mtnormalise -force fod_wm.mif fod_wm_norm.mif \
         fod_gm.mif fod_gm_norm.mif \
         fod_csf.mif fod_csf_norm.mif \
         -mask dwi_proc_mask.mif')


def generate_streamlines_and_run_sift2(path, n_streamlines):
    print('Step 08: Generating streamlines and computing their weights...')
    os.chdir(str(path))
    #################################################################
    ## Generate Streamlines
    # NOTE - the algorithm and cutoff settings are actually the defaults,
    # but I set them here for clarity
    execute(f'tckgen -force fod_wm_norm.mif tracks_{n_streamlines}.tck \
        -backtrack -crop_at_gmwmi -info \
        -act aseg_5ttgen.mif  -seed_gmwmi aseg_5tt_gmwmi.mif \
        -algorithm iFOD2 -select {n_streamlines} -cutoff 0.05')

    ## Run SIFT2 Algorithm
    # Weights streamlines based on diffusion signal, such
    # that they are more closely related to true fiber densities. This allows
    # for more valid quantitative tractography measures.
    # NOTE - to make use of SIFT2 output, the weights txt file must be included
    # in further track quantitification tools using "-tck_weights_in" flag
    # ALSO NOTE - the I removed the -fd_scale_gm flag due to the use of
    # multi-tissue FOD estimation
    execute(f'tcksift2 -force tracks_{n_streamlines}.tck fod_wm_norm.mif tracks_{n_streamlines}_sift2_weights.txt \
        -act aseg_5ttgen.mif -info \
        -out_mu tracks_{n_streamlines}_sift2_weights_prop_coeff.txt')

    #  Generate a smaller number of streamlines for QA and viewing
    execute(f'tckedit -force -number 200k tracks_{n_streamlines}.tck tracks_{n_streamlines}_to200k.tck \
        -tck_weights_in tracks_{n_streamlines}_sift2_weights.txt \
        -tck_weights_out tracks_{n_streamlines}_to200k_sift2_weights.txt')


def copy_data_to_output_directory(path_out, path_tmp, n_streamlines):
    print('Step 09: Copying data to output directory (please manually delete temporary folder as needed)...')
    execute(f'mkdir -p {path_out}')
    execute(f'cp -va {path_tmp}/ {path_out}/')

    if os.path.exists(f'{path_out}/tracks_{n_streamlines}_to200k.tck'):
        print('** Process COMPLETE! **')
    else:
        print(f'** Process FAILED!! ** Please check the log file in {path_tmp} to determine the error.')


def create_post_process_QA_figure(path_out):
    os.chdir(path_out)

    # Load B-value info to determine the first volume of each shell
    execute('mrinfo -shell_bvalues dwi_raw.mif > bvalue_info.txt')
    execute('mrinfo -shell_sizes dwi_raw.mif >> bvalue_info.txt')
    bval = np.loadtxt('bvalue_info.txt').round()
    execute('mrinfo -shell_indices dwi_raw.mif > bvalue_inds.txt')
    bval_all = np.loadtxt('dwi_raw_fsl.bval').round()

    # Begin building image/label dictionary, later used for reference and concatenation
    image_label_dict = {}

    # Raw data example from each b-shell
    for ii in range(bval.shape[1]):
        nvols = int(bval[1, ii])
        val = int(bval[0, ii])
        firstvol = np.argwhere(bval_all == bval[0, ii])[0]
        firstvol = ' '.join(map(str, firstvol))
        print(f'Acquired {nvols} volumes at b = {val}, first {firstvol}')
        # Build dictionary for raw data inputs QA figures 
        # (note these can vary based on inputs, while the later images will not)
        image_label_dict.update({f'QA_raw_b{val}_0000.png': f'Raw data, b={val} (1/{nvols})'})
        # Raw data 
        execute(f'mrview dwi_raw.mif -mode 2 -volume {firstvol} -interpolation 0 -noannot -size 1000,1000 -autoscale \
            -capture.folder QA -capture.prefix QA_raw_b{val}_ -capture.grab -exit')

        # Noise image
    execute('mrview dwi_raw_noise.mif -mode 2 -interpolation 0 -noannot -size 1000,1000  \
        -capture.folder QA -capture.prefix QA_noise_ -capture.grab -exit')

    # B0 image pair (revPE)
    execute('mrview dwi_raw_dn_b0_pair.mif -mode 1 -plane 2 -volume 0 -interpolation 0 -noannot -size 1000,1000  \
        -capture.folder QA -capture.prefix QA_b0_ -capture.grab -exit')
    execute('mrview dwi_raw_dn_b0_pair.mif -mode 1 -plane 2 -volume 1 -interpolation 0 -autoscale -noannot -size 1000,1000 \
        -capture.folder QA -capture.prefix QA_b0_revPE_ -capture.grab -exit')

    # Topup corrected B0
    execute('mrview dwi_proc_b0.nii -mode 1 -plane 2 -interpolation 0 -autoscale -noannot -size 1000,1000  \
        -capture.folder QA -capture.prefix QA_b0_topup_ -capture.grab -exit')

    # CNR Map
    execute('mrview QA/eddy_cnr_maps.nii.gz -mode 2 -interpolation 0 -noannot -size 1000,1000 \
        -colourmap 4 -colourbar 1 -intensity_range 1,2.5 \
        -capture.folder QA -capture.prefix QA_cnr_ -capture.grab -exit')

    # Masks
    execute('mrview QA/eddy_mask.nii -mode 2 -interpolation 0 -noannot -size 1000,1000 -intensity_range 0,1 \
        -overlay.load dwi_proc_mask.mif -overlay.colourmap 3 -overlay.interpolation 0 -overlay.intensity 0,1 \
        -capture.folder QA -capture.prefix QA_masks_ -capture.grab -exit')

    # FA map (white matter should be bright), scale 0-1
    execute('mrview dwi_proc_FA.mif -mode 2 -interpolation 0 -noannot -size 1000,1000  -colourbar 1 -intensity_range 0,1 \
        -capture.folder QA -capture.prefix QA_fa_ -capture.grab -exit')

    # MD map (csf should be bright), scale 0-.003
    execute('mrview dwi_proc_MD.mif -mode 2 -interpolation 0 -noannot -size 1000,1000 -colourbar 1 -intensity_range 0,.003 \
        -capture.folder QA -capture.prefix QA_md_ -capture.grab -exit')

    # 5TT struct over b0
    execute('mrview dwi_proc_b0.nii -mode 2 -interpolation 0 -noannot -size 1000,1000 \
        -overlay.load struct.2dwi.nii -overlay.colourmap 1 -overlay.interpolation 0 -overlay.opacity 0.3 \
        -capture.folder QA -capture.prefix QA_struct2dwi_ -capture.grab -exit')

    # 5TT GMWMI
    execute('mrview dwi_proc_b0.nii -mode 2 -interpolation 0 -noannot -size 1000,1000 \
        -overlay.load aseg_5tt_gmwmi.mif -overlay.colourmap 1 -overlay.intensity 0,1 -overlay.opacity 0.8 \
        -capture.folder QA -capture.prefix QA_gmwmi_ -capture.grab -exit')

    # Dhollander voxel selection (wm=blue, csf=red, gm=green)
    execute('mrview dwi_proc_b0.nii -mode 2 -interpolation 0 -noannot -size 1000,1000 \
        -overlay.load dhollander_voxels_sub.mif -overlay.interpolation 0 \
        -capture.folder QA -capture.prefix QA_dhollander_voxels_ -capture.grab -exit')

    # ODF-related images
    execute(
        'echo "MRViewOdfScale: 4" >> ~/.mrtrix.conf')  # NOTE - to be safe this (>>) appends to the user config file. Though it does not exist by default.
    execute('mrconvert -coord 3 0 fod_wm.mif - | mrcat -force fod_csf.mif fod_gm.mif - fod_all.mif')
    execute('mrview fod_all.mif -mode 1 -plane 2 -noannot -size 1000,1000 -interpolation 0 \
        -capture.folder QA -capture.prefix QA_fod_tissuemap_ -capture.grab -exit')

    execute('mrview dwi_proc_FA.mif -mode 1 -plane 2 -noannot -size 1000,1000  \
        -odf.load_sh fod_wm_norm.mif \
        -capture.folder QA -capture.prefix QA_fod_wm_ -capture.grab -exit')

    execute('fod2dec -force fod_wm_norm.mif fod_wm_norm_dec.mif -mask dwi_proc_mask.mif')
    execute('mrview fod_wm_norm_dec.mif -mode 1 -plane 2 -noannot -size 1000,1000 -interpolation 0 \
        -capture.folder QA -capture.prefix QA_fod_dec_ -capture.grab -exit')

    # Tractography
    execute('mrview dwi_proc_b0.nii -mode 2 -interpolation 0 -noannot -size 1000,1000 \
        -tractography.load tracks_10M_to*.tck -tractography.geometry lines \
        -tractography.opacity 0.1 \
        -capture.folder QA -capture.prefix QA_tck_2D_ -capture.grab -exit')

    execute('mrview dwi_proc_b0.nii -mode 3 -interpolation 0 -noannot -size 1000,1000 -imagevisible 0 \
        -tractography.load tracks_10M_to*.tck -tractography.geometry lines \
        -tractography.opacity 0.05 \
        -capture.folder QA -capture.prefix QA_tck_3D_ -capture.grab -exit')

    execute('mrview dwi_proc_b0.nii -mode 2 -interpolation 0 -noannot -size 1000,1000 \
        -tractography.load tracks_10M_to*.tck -tractography.geometry lines \
        -tractography.tsf_load tracks_10M_to*_sift2_weights.txt -tractography.tsf_range 0,2.5 \
        -capture.folder QA -capture.prefix QA_tck_2D_weighted_ -capture.grab -exit')

    # Motion parameters
    # Load eddy motion parameter data
    eddy_params = np.loadtxt('QA/eddy_parameters')
    eddy_rms = np.loadtxt('QA/eddy_movement_rms')

    # Calculate movement metrics
    rotations = eddy_params[:, 3:6] / np.pi * 180
    avg_rotations = np.nanmean(rotations, axis=0)

    translations = eddy_params[:, 0:3]
    avg_translations = np.nanmean(translations, axis=0)

    abs_displacement = eddy_rms[:, 0]
    avg_abs_displacement = np.array([np.nanmean(abs_displacement)])
    rel_displacement = eddy_rms[:, 1]
    avg_rel_displacement = np.array([np.nanmean(rel_displacement)])
    # Configure figure
    PAGESIZE = (6, 8)
    TITLE_FONTSIZE = 16
    LABEL_FONTSIZE = 12
    PDF_DPI = 600
    VIS_PERCENTILE_MAX = 99.9
    fig = plt.figure(0, figsize=PAGESIZE)

    # Visualize rotations
    ax = plt.subplot(5, 1, 1)
    plt.plot(range(0, rotations.shape[0]), rotations[:, 0], color='b', label='x ({:.3f})'.format(avg_rotations[0]))
    plt.plot(range(0, rotations.shape[0]), rotations[:, 1], color='c', label='y ({:.3f})'.format(avg_rotations[1]))
    plt.plot(range(0, rotations.shape[0]), rotations[:, 2], color='m', label='z ({:.3f})'.format(avg_rotations[2]))
    plt.xlim((-1, rotations.shape[0] + 1))
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(labelbottom=False)
    plt.ylabel('Rotation' '\n' r'($^\circ$)', fontsize=3 * LABEL_FONTSIZE / 4)
    plt.grid()
    plt.legend(fontsize=LABEL_FONTSIZE / 2, loc='upper left')
    plt.title('Motion and Intensity', fontsize=TITLE_FONTSIZE)

    # Visualize translations
    ax = plt.subplot(5, 1, 2)
    plt.plot(range(0, translations.shape[0]), translations[:, 0], color='b',
             label='x ({:.3f})'.format(avg_translations[0]))
    plt.plot(range(0, translations.shape[0]), translations[:, 1], color='c',
             label='y ({:.3f})'.format(avg_translations[1]))
    plt.plot(range(0, translations.shape[0]), translations[:, 2], color='m',
             label='z ({:.3f})'.format(avg_translations[2]))
    plt.xlim((-1, translations.shape[0] + 1))
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(labelbottom=False)
    plt.ylabel('Translation\n(mm)', fontsize=3 * LABEL_FONTSIZE / 4)
    plt.grid()
    plt.legend(fontsize=LABEL_FONTSIZE / 2, loc='upper left')

    # Visualize Displacement (Abs, Rel)
    ax = plt.subplot(5, 1, 3)
    plt.plot(range(0, abs_displacement.shape[0]), abs_displacement, color='b',
             label='Abs. ({:.3f})'.format(avg_abs_displacement[0]))
    plt.plot(range(0, rel_displacement.shape[0]), rel_displacement, color='m',
             label='Rel. ({:.3f})'.format(avg_rel_displacement[0]))
    plt.xlim((-1, abs_displacement.shape[0] + 1))
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(labelbottom=False)
    plt.ylabel('Displacement\n(mm)', fontsize=3 * LABEL_FONTSIZE / 4)
    plt.grid()
    plt.legend(fontsize=LABEL_FONTSIZE / 2, loc='upper left')

    # Visualize median intensity
    execute('mrconvert dwi_proc.mif tmp_proc.nii')
    dwi_img = nib.load('tmp_proc.nii').get_fdata()
    execute('rm -f tmp_proc.nii')
    mask_img = nib.load('QA/eddy_mask.nii').get_fdata().astype(bool)

    median_intensities = np.zeros(dwi_img.shape[3])
    for i in range(dwi_img.shape[3]):
        dwi_vol = dwi_img[:, :, :, i]
        median_intensities[i] = np.nanmedian(dwi_vol[mask_img])

    ax = plt.subplot(5, 1, 4)
    ax.plot(range(0, dwi_img.shape[3]), median_intensities, color='c')
    ax.plot(range(0, dwi_img.shape[3]), median_intensities, '.', color='b')
    plt.xlim((-1, dwi_img.shape[3] + 1))
    ax.set_ylabel('Median\nIntensity', fontsize=3 * LABEL_FONTSIZE / 4)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(labelbottom=False)
    ax.grid()

    # Visualize B values
    bvals = np.loadtxt('dwi_raw_fsl.bval')
    ax = plt.subplot(5, 1, 5)
    ax.plot(range(0, len(bvals)), bvals, color='c', label='B Value')
    ax.plot(range(0, len(bvals)), bvals, '.', color='b', label='B Value')
    ax.set_ylabel('B Value', fontsize=3 * LABEL_FONTSIZE / 4)
    plt.xlim((-1, len(bvals) + 1))
    ax.grid()

    # Finish up motion
    ax.set_xlabel('Diffusion Volume', fontsize=LABEL_FONTSIZE)

    # Print
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig('QA/QA_motion_and_intensity', dpi=PDF_DPI, facecolor='w')
    plt.close()

    # Visualize outlier map
    def _str2list(string):
        row = []
        for i in range(len(string)):
            if not np.mod(i, 2):
                row.append(int(string[i]))
        return row

    rows = []
    with open('QA/eddy_outlier_map', 'r') as f:
        txt = f.readlines()
        for i in np.arange(1, len(txt)):
            rows.append(_str2list(txt[i].strip('\n')))
    outlier_array = np.array(rows)
    outlier_map = np.transpose(outlier_array)

    ax = plt.subplot(1, 1, 1)
    ax.matshow(outlier_map, aspect='auto', origin='lower')
    ax.set_title('Eddy Outlier Slices', fontsize=TITLE_FONTSIZE)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Diffusion Volume', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Slice', fontsize=LABEL_FONTSIZE)
    plt.xlim(0, outlier_map.shape[1])
    plt.ylim(0, outlier_map.shape[0])
    ax.grid()
    # Print
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig('QA/QA_outlier_map', dpi=PDF_DPI, facecolor='w')
    plt.close()

    # Define dictionary of QA images and the information to include at the top
    image_label_dict.update({
        'QA_noise_0000.png': 'Estimated noise map',
        'QA_b0_0000.png': 'B0 image (primary phase encode dir.)',
        'QA_b0_revPE_0000.png': 'B0 image (reverse phase encode dir.)',
        'QA_b0_topup_0000.png': 'Distortion-corrected B0 image',
        'QA_motion_and_intensity.png': '',
        'QA_outlier_map.png': '',
        'QA_cnr_0000.png': 'Contrast to noise (CNR) map',
        'QA_masks_0000.png': 'Eddy mask (white), MRtrix3 mask (red)',
        'QA_fa_0000.png': 'Fractional anisotropy (FA) map (WM should be bright)',
        'QA_md_0000.png': 'Mean diffusitivity (MD) map (CSF should be bright)',
        'QA_struct2dwi_0000.png': 'Structural registration to DWI',
        'QA_gmwmi_0000.png': 'Gray/White matter interface derived from FreeSurfer output',
        'QA_dhollander_voxels_0000.png': 'Dhollander algo. voxels (wm=blue,csf=red,gm=green)',
        'QA_fod_tissuemap_0000.png': 'Dhollander algo. tissue map (wm=blue,csf=red,gm=green)',
        'QA_fod_wm_0000.png': 'White matter fiber orientiation distributions (FODs)',
        'QA_fod_dec_0000.png': 'Directionally-encoded color map, from FODs',
        'QA_tck_2D_0000.png': 'Wholebrain tractography (colored by direction)',
        'QA_tck_2D_weighted_0000.png': 'Wholebrain tractography (colored by weight)',
        'QA_tck_3D_0000.png': 'Wholebrain tractography in 3D (colored by direction)',
    })

    # Common flags to pass to convert command
    flags = '-bordercolor white -border 1 -gravity north -fill white -pointsize 35 -annotate 0'
    # Add label to each image
    for image, label in image_label_dict.items():
        cmd = f'convert {flags} "{label}" QA/{image} QA/{image}'
        execute(f'{cmd}')
    # Stack all images vertically for final document
    lbls = ' QA/'.join(image_label_dict.keys())
    execute(f'convert -resize 1000 -append QA/{lbls} QA/QA_all.png')