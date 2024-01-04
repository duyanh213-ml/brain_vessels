# -*- coding: utf-8 -*-

"""
===========================================================================================================

Created on Thu Jun 29 2023
Latest update: 06-29-23 

@author: AnhND
===========================================================================================================
"""

import tifffile as tiff


def generate_non_artery_patches(series_list, mask_list, patch_size, path_patches, path_mask, flag=False):
    idx_nonartery = 0
    half_patch_size = patch_size // 2

    step_size = None

    if flag:
        step_size = patch_size // 4
    else:
        step_size = patch_size


    for series_idx in range(len(series_list)):
        z, y, x = series_list[series_idx].shape

        for pz in range(z):
            for px in range(half_patch_size, x, step_size):
                for py in range(half_patch_size, y, step_size):
                    # (pz, py, px) is the center of the patch
                    if mask_list[series_idx][pz, py, px] == 0 or mask_list[series_idx][pz, py, px] != 0:
                        pyy = py
                        pxx = px

                        if pxx > x - half_patch_size:
                            pxx = x - half_patch_size

                        if pyy > y - half_patch_size:
                            pyy = y - half_patch_size

                        # Get the current patch
                        im_cur = series_list[series_idx][pz, pyy - half_patch_size:pyy + half_patch_size,
                                                pxx - half_patch_size:pxx + half_patch_size]
                        mask_cur = mask_list[series_idx][pz, pyy - half_patch_size:pyy +
                                                half_patch_size, pxx - half_patch_size:pxx + half_patch_size]


                        # Paths to save patch and mask
                        path_patch_zero = f"{path_patches}/slice_patch_{idx_nonartery}.tiff"
                        path_mask_zero = f"{path_mask}/mask_patch_{idx_nonartery}.tiff"

                        # save patch and mask
                        tiff.imsave(path_patch_zero, im_cur)
                        tiff.imsave(path_mask_zero, mask_cur)

                        # increment index
                        idx_nonartery += 1

    print(f"Successfully generate {idx_nonartery} non-artery-patches!!!")
    return idx_nonartery, step_size


'''
Get artery patches
Note that the number of artery patches is approximately equal to the number of non-artery patches
'''
def generate_artery_patches(series_list, mask_list, patch_size, path_patches, path_mask, idx_nonartery, step_size):

    idx_artery = idx_nonartery #initialize artery path index, starting from the last non-artery index

    half_patch_size = patch_size // 2

    for series_idx in range(len(series_list)):
        z, y, x = series_list[series_idx].shape
        n_patches_per_slice = (x // step_size) * (y // step_size)


        for pz in range(z):

            n_artery_pixels_of_slice = 0  # total artery pixels at slice pz
            n_artery_frame = 0  # number of frames having artery pixels
            # artery_pos_of_frame contains the position of artery pixels in frame[i]
            artery_pos_of_frame = []
            n_artery_pixels_of_frame = []
            n_patches_of_slice = n_patches_per_slice  # total artery patches of slice pz

            for px in range(half_patch_size, x, patch_size):
                for py in range(half_patch_size, y, patch_size):
                    pyy = py
                    pxx = px
                    if (pxx > x-half_patch_size):
                        pxx = x-half_patch_size
                    if (pyy > y-half_patch_size):
                        pyy = y-half_patch_size

                    n_artery_pixels_i = 0
                    artery_pos_i = []

                    for ix in range(pxx-half_patch_size, pxx+half_patch_size):
                        for iy in range(pyy-half_patch_size, pyy+half_patch_size):
                            if (mask_list[series_idx][pz, iy, ix] == 1):
                                n_artery_pixels_i += 1
                                artery_pos_i.append([ix, iy])

                    if (n_artery_pixels_i > 0):
                        n_artery_frame += 1
                        n_artery_pixels_of_slice += n_artery_pixels_i
                        artery_pos_of_frame.append(artery_pos_i)
                        n_artery_pixels_of_frame.append(n_artery_pixels_i)

            if (n_artery_pixels_of_slice > 0):

                if (n_artery_pixels_of_slice < n_patches_of_slice):
                    n_patches_of_slice = n_artery_pixels_of_slice

                for i in range(n_artery_frame):

                    n_patches_of_frame = (
                        n_patches_of_slice * n_artery_pixels_of_frame[i]) // n_artery_pixels_of_slice
                    if (n_patches_of_frame == 0):
                        n_patches_of_frame = 1

                    for j in range(0, n_artery_pixels_of_frame[i], n_artery_pixels_of_frame[i] // n_patches_of_frame):
                        ix = artery_pos_of_frame[i][j][0]
                        iy = artery_pos_of_frame[i][j][1]

                        if (iy >= half_patch_size and iy <= y-half_patch_size and ix >= half_patch_size and ix <= x-half_patch_size):

                            # Get the current patch
                            im_cur = series_list[series_idx][pz, iy-half_patch_size:iy+half_patch_size,
                                                    ix-half_patch_size:ix+half_patch_size]
                            mask_cur = mask_list[series_idx][pz, iy-half_patch_size:iy +
                                                    half_patch_size, ix-half_patch_size:ix+half_patch_size]

                            #paths to saved patch and mask
                            path_patch_nonzero = f"{path_patches}/slice_patch_{idx_artery}.tiff"
                            path_mask_nonzero  = f"{path_mask}/mask_patch_{idx_artery}.tiff"

                            #save the path and mask
                            tiff.imsave(path_patch_nonzero, im_cur)
                            tiff.imsave(path_mask_nonzero, mask_cur)

                            # increment index
                            idx_artery += 1
    
    print(f"Successfully generate {idx_artery - idx_nonartery} artery-patches!!!")