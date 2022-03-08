## Requirements
`pip3 install numpy scipy numba babel joblib scipy`

## Usage example
`python3 streamlinepos_from_surfaces.py 
../../hemi_surfaces/freesurfer/surf/lh.white 
../../hemi_surfaces/freesurfer/surf/lh.area 
../../hemi_surfaces/freesurfer/surf/lh.pial 
../../hemi_surfaces/freesurfer/surf/lh.area.pial 
../../track_convert/sub-04_acq-01_rh_v1v2v3_tracking_tracto_mp2rage.tck 
streamlines_positions.mat 
equivol 
32`


Command line parameters:
1. white surface
2. area file white surface
3. pial surface
4. area file pial surface
5. streamlines file (.tck)
6. filename of output (MATLAB .mat format)
7. method: equivol or equisdist
8. number of cpus
9. optional: number of streamlines to process (for debugging)
