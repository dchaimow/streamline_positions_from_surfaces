"""
This repurposes the code from the vdfs package to calculate the cortical depth (and gray matter inclusion)
of streamline points. The vdfs code applied to voxel grids given by a nifti calculates the cortical 
depth of all voxels in that voxel grid. It would first calculate for each voxel a list of prisms 
(=volume spanned by corresponding white and pial surface triangles) that could potentially contain
the voxel. Then, in the next step, it would for each voxel test for intersections between its center 
and parameterized intermediate depth triangle planes of each prism it the voxels list. 
The first calculation can be done for arbitrary voxel grids that enclose the points to be tested as long
as we use the list of prisms calculated for that grid voxel that would contain the point. If the resolution
of the grid is too low, we would get a long list of prisms for each voxel, which would slow down the second step.
But if the grid is too fine, the first step would be slow, so there is a tradeoff.
"""

import os

import nibabel as nib
from numba import jit
from numba.typed import List
from joblib import Parallel, delayed, parallel_backend
import vdfs as vdfs
import numpy as np
from hdf5storage import savemat
import sys


def calc_depth_from_surfaces_points(surf_white, area_white, surf_pial, area_pial,
                                    points, method, n_jobs=32):
    """
    Calculate depths for arbitrart sets of points from surfaces given all surface information.
    Assumes that the points (and the surface have been brought into a coordinate system) such that integer steps
    correspond to a grid that will be used to calculate bounding prisms.
    """
    faces = surf_pial['tris']
    vertices_white = surf_white['vertices']
    vertices_pial = surf_pial['vertices']

    n_x = int(np.ceil(np.max(points[:, 0])) + 1)
    n_y = int(np.ceil(np.max(points[:, 1])) + 1)
    n_z = int(np.ceil(np.max(points[:, 2])) + 1)

    # For each voxel, calculate a list of prism indices that potentially contain the voxel
    # so that we only to solve intersection equations for these prisms
    bounding_prisms = vdfs.numba_calc_bounding_prisms(faces,
                                                      vertices_white, vertices_pial,
                                                      n_x, n_y, n_z)

    # calculate depths in parallel by calling calc_depth_from_surfaces_voxelidx for each point
    with parallel_backend('loky', inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(
            delayed(
                lambda point:
                vdfs.calc_depth_from_surfaces_voxelidx(
                    np.round(point).astype(int), faces, vertices_white, vertices_pial,
                    area_white, area_pial, bounding_prisms, p=point, method=method))(point)
            for point in points)
        # TODO: check if np.round(point) is correct

    return results


def define_point_enclosing_grid(points, d):
    """
    Define a grid with spacing d that encloses the points and return the transformation matrix
    that maps the grid to the point space. (interpreted as scanner space)
    """
    # 1st voxel in each dimension should map to corresponding minimum coordinate in points
    # steps are size d, and the number of points if ceil((min_x-max_x)/d):
    # TODO: CHECK, DOES COUNTING START AT 1?
    # x_scanner = min_x(streamline_point) + voxels_idx_x * d
    min_x = np.min(points[:, 0])
    min_y = np.min(points[:, 1])
    min_z = np.min(points[:, 2])

    grid_to_scanner = np.matrix(
        [[d, 0, 0, min_x],
         [0, d, 0, min_y],
         [0, 0, d, min_z],
         [0, 0, 0, 1]])

    return grid_to_scanner


def process_streamlinepos_from_surfaces_single_hemi(surf_white_file, area_white_file,
                                                    surf_pial_file, area_pial_file,
                                                    streamlines_file, output_fname,
                                                    method='equivol',
                                                    grid_d=1, n_jobs=32, n_streamlines='all'):
    """
    This is the main function. It loads the streamlines, creates an enclosing grid with spacing 1 mm,
    loads the surfaces, and calculates the depths of the streamlines in the surfaces. 
    The results are saved in a .mat file.
    """
    # load streamlines and generate set of all points by concatenating all streamlines
    streamlines = nib.streamlines.tck.TckFile.load(streamlines_file).streamlines
    if n_streamlines != 'all':
        streamlines = streamlines[:n_streamlines]
    points_orig_space = np.concatenate(streamlines)

    # set grid that encloses all streamline points and transform points to grid coordinates
    grid_to_scanner = define_point_enclosing_grid(points_orig_space, grid_d)
    n_points = points_orig_space.shape[0]
    points = np.array(grid_to_scanner.I.dot(np.hstack((points_orig_space, np.ones((n_points, 1)))).T).T[:, :3])

    # load all surfaces and transform to grid space
    surf_white = vdfs.load_fs_surf_in_grid(surf_white_file, grid_to_scanner)
    surf_pial = vdfs.load_fs_surf_in_grid(surf_pial_file, grid_to_scanner)

    # load area files
    area_white = nib.freesurfer.read_morph_data(area_white_file)
    area_pial = nib.freesurfer.read_morph_data(area_pial_file)

    # calc voxel depths
    results = calc_depth_from_surfaces_points(surf_white, area_white,
                                              surf_pial, area_pial,
                                              points, method, n_jobs)

    # assemble results in a dictionary and save to .mat file
    gray_matter = []
    depths = []
    faces = []
    vertices = []
    point_idx = 0
    results = np.array(results)
    for streamline in streamlines:
        l = len(streamline)
        streamline_depths = results[point_idx:(point_idx + l), 0]
        streamline_faces = results[point_idx:(point_idx + l), 1]
        streamline_gray_matter = np.isfinite(streamline_depths)
        streamline_vertices = np.ones((len(streamline), 3)) * np.nan
        streamline_vertices[streamline_gray_matter, :] = surf_white['tris'][streamline_faces[streamline_gray_matter].astype(int)]

        gray_matter.append(streamline_gray_matter)
        depths.append(streamline_depths)
        faces.append(streamline_faces)
        vertices.append(streamline_vertices)

        point_idx += l

    data = dict()
    data['sl_gray_matter'] = np.array(gray_matter, dtype=object)
    data['sl_depths'] = np.array(depths, dtype=object)
    data['sl_faces'] = np.array(faces, dtype=object)
    data['sl_vertices'] = np.array(vertices, dtype=object)
    data['sl_coordinates'] = np.array(streamlines, dtype=object)
    savemat(output_fname, data,format=7.3, matlab_compatible=True, compress=False)


if __name__ == '__main__':
    # parse command line arguments and call process_streamlinepos_from_surfaces_single_hemi
    surf_white_file = sys.argv[1]
    area_white_file = sys.argv[2]
    surf_pial_file = sys.argv[3]
    area_pial_file = sys.argv[4]
    streamlines_file = sys.argv[5]
    output_fname = sys.argv[6]
    method = sys.argv[7]
    n_jobs = int(sys.argv[8])

    if len(sys.argv)==10:
        n_streamlines = int(sys.argv[9])
    else:
        n_streamlines = 'all'
    process_streamlinepos_from_surfaces_single_hemi(surf_white_file, area_white_file,
                                                    surf_pial_file, area_pial_file,
                                                    streamlines_file, output_fname,
                                                    method=method,
                                                    grid_d=1, n_jobs=n_jobs, n_streamlines=n_streamlines)
