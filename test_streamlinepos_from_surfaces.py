import os
from unittest import TestCase
import nibabel as nib
import numpy as np

from streamlinepos_from_surfaces import process_streamlinepos_from_surfaces_single_hemi
#from voxeldepths_from_surfaces import load_fs_surf_in_scanner_space, merge_surfaces


class Test(TestCase):
    def setUp(self):
        pass
        # # load all surfaces in scanner space
        # surf_white_lh = load_fs_surf_in_scanner_space(os.path.join('data', 'lh.white'))
        # surf_pial_lh = load_fs_surf_in_scanner_space(os.path.join('data', 'lh.pial'))
        # surf_white_rh = load_fs_surf_in_scanner_space(os.path.join('data', 'rh.white'))
        # surf_pial_rh = load_fs_surf_in_scanner_space(os.path.join('data', 'rh.pial'))
        #
        # # load area fils
        # area_white_lh = nib.freesurfer.read_morph_data(os.path.join('data', 'lh.area'))
        # area_pial_lh = nib.freesurfer.read_morph_data(os.path.join('data', 'lh.area.pial'))
        # area_white_rh = nib.freesurfer.read_morph_data(os.path.join('data', 'rh.area'))
        # area_pial_rh = nib.freesurfer.read_morph_data(os.path.join('data', 'rh.area.pial'))
        #
        # # merge hemispheres
        # surf_white = merge_surfaces(surf_white_lh, surf_white_rh)
        # surf_pial = merge_surfaces(surf_pial_lh, surf_pial_rh)
        # area_white = np.concatenate((area_white_lh, area_white_rh), axis=0)
        # area_pial = np.concatenate((area_pial_lh, area_pial_rh), axis=0)
        #
        # # load streamlines
        # streamlines = nib.streamlines.tck.TckFile.load(os.path.join(
        #     'data',
        #     'sub-04_acq-01_rh_v1v2v3_tracking_tracto_down_mp2rage.tck')).streamlines
        #
        # # load area files
        # self.faces = surf_pial['tris']
        # self.vertices_white = surf_white['vertices']
        # self.vertices_pial = surf_pial['vertices']
        # self.area_white = area_white
        # self.area_pial = area_pial
        # self.streamlines = streamlines
        # self.points = np.concatenate(streamlines)

    def test_process_streamlinepos_from_surfaces_single_hemi(self):
        surf_white_rh_file = os.path.join('data', 'lh.white')
        surf_pial_rh_file = os.path.join('data', 'lh.pial')
        area_white_rh_file = os.path.join('data', 'lh.area')
        area_pial_rh_file = os.path.join('data', 'lh.area.pial')
        streamlines_file = os.path.join('data', 'sub-04_acq-01_rh_v1v2v3_tracking_tracto_down_mp2rage.tck')
        output_fname = 'test.mat'
        n_streamlines = 100
        method = 'equivol'

        process_streamlinepos_from_surfaces_single_hemi(surf_white_rh_file, area_white_rh_file,
                                                        surf_pial_rh_file, area_pial_rh_file,
                                                        streamlines_file, output_fname,
                                                        method=method,
                                                        grid_d=1, n_jobs=4, n_streamlines = n_streamlines)