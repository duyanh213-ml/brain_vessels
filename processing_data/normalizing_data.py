# -*- coding: utf-8 -*-

"""
===========================================================================================================

Created on Mon Jun 26 2023
Latest update: 06-26-23 

@author: AnhND
===========================================================================================================
"""


import SimpleITK as sitk
import numpy as np
import os
from process_data_config import PATH_MIN_MAX


# We just use matplotlib in main() if you to see the visualization of distribution,
import matplotlib.pyplot as plt


# This min-max normaliztion is applied after z-score
def min_max_normalization(series, max, min, epsilon=1e-9):
    # use epsilon to avoid underflow (divided by 0)
    return (series - min) / (max - min + epsilon)


def z_score_normalization(series, epsilon=1e-9):
    # The use of epsilon is similar to the min_max_normalization
    return (series - np.mean(series)) / (np.std(series) + epsilon)


def normalized_process(path_dicom, path_segment, flag=False):
    # Create empty lists to store series image and mask data
    series_list = []
    mask_list = []

    # Iterate over each folder in the raw data path
    for patient in os.listdir(path=path_dicom):
        # Create the path to the patient folder
        patient_dicom_path = f'{path_dicom}{patient}/'
        patient_segment_path = f'{path_segment}{patient}.nii'

        # Get the slices and convert them to numpy array
        reader = sitk.ImageSeriesReader()
        filenamesDICOM = reader.GetGDCMSeriesFileNames(patient_dicom_path)
        reader.SetFileNames(filenamesDICOM)
        img_original = reader.Execute()
        img = sitk.GetArrayFromImage(img_original)


        # Read the mask image and convert them to numpy array
        mask = sitk.ReadImage(patient_segment_path)
        mask = sitk.GetArrayFromImage(mask)

        # Append the image and mask to their respective lists
        series_list.append(img)
        mask_list.append(mask)

        print(f"Append successfully patient {patient}")

    # Normalizing process:
    # 1. Z-score norm
    for i in range(len(series_list)):
        series_list[i] = z_score_normalization(series_list[i])

    min_value = None
    max_value = None

    if flag:
        # Get the min-max value for min_max_normalization(series, max, min, epsilon=1e-7)
        min_value = float(min([np.min(series) for series in series_list]))
        max_value = float(max([np.max(series) for series in series_list]))

        # Save the min-max value (Only need for training-training)
    
        with open("max_and_min.txt", "w") as f:
            f.write(f"{min_value},{max_value}")
            print("Save the constant of min and max values successfully!!!")
    else:
        # Get the min-max value for min_max_normalization(series, max, min, epsilon=1e-7)
        with open(PATH_MIN_MAX, 'r') as f:
            lst = f.read().split(',')


        min_value = float(lst[0])
        max_value = float(lst[1])

        print(f"Load min = {min_value}, max = {max_value}")

    # 2. Min-max norm
    for i in range(len(series_list)):
        series_list[i] = min_max_normalization(
            series_list[i], max_value, min_value)

    # Wrapping normalized data and corresponding labels in a dictionary
    normalized_data = {
        "series": series_list,
        "masks": mask_list
    }

    print("Completely!! Our data has been normalized!!")

    return normalized_data


# # Just for test!! If you don't wanna run, let's comment
# # ---------------------------------------------------------------------------------------

# if __name__ == "__main__":
#     # 2 lines help us do not need to modify the directory path
#     current_path = os.path.dirname(os.path.abspath(__file__))
#     path_raw = os.path.join(current_path, '..', 'raw_data')


#     # Specify the path to the dicom and segmentation in raw data
#     path_dicom = f'{path_raw}/dicom/'
#     path_segment = f'{path_raw}/segmentation_nii/'

#     data = normalized_process(path_dicom, path_segment)

#     for series in data['series']:
#         print(series.shape)


#     # You can choose one of these visualizations to test

#     # 1. All pixel distribution
#     # ===================================================================================
#     # plt.figure(figsize=(12, 5))
#     # bins = 100

#     # plt.hist(data["series"][0].flatten(), bins=bins, histtype='step')
#     # plt.hist(data["series"][1].flatten(), bins=bins, histtype='step')
#     # plt.hist(data["series"][2].flatten(), bins=bins, histtype='step')
#     # plt.hist(data["series"][3].flatten(), bins=bins, histtype='step')

#     # plt.title("Distribution of all pixel values in a series (Normalized)")
#     # plt.xlim(-0.05, 0.4)

#     # plt.show()
#     # ===================================================================================


#     # 2. non-zero pixel distribution
#     # ===================================================================================
#     label_indices0 = np.nonzero(data['masks'][0])
#     label_indices1 = np.nonzero(data['masks'][1])
#     label_indices2 = np.nonzero(data['masks'][2])
#     label_indices3 = np.nonzero(data['masks'][3])

#     plt.figure(figsize=(10, 6))

#     plt.hist(data["series"][0][label_indices0], bins=25, histtype='step')
#     plt.hist(data["series"][1][label_indices1], bins=25, histtype='step')
#     plt.hist(data["series"][2][label_indices2], bins=25, histtype='step')
#     plt.hist(data["series"][3][label_indices3], bins=25, histtype='step')

#     plt.title("Distribution of non-zero pixel values in a series (normalized)")

#     plt.show()
# #     # ===================================================================================
