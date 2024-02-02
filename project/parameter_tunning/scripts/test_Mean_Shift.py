######################################
###########  REQUIREMENTS  ###########
######################################

# Imports

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import sys

from skimage.filters import threshold_otsu
from collections import Counter
from scipy.spatial import KDTree
from sklearn.cluster import MeanShift
from scipy.spatial.distance import euclidean
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from itertools import product
import threading
from concurrent.futures import ThreadPoolExecutor
import itertools
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import time
import gc

# Getting input from the runner script

if len(sys.argv) != 4:
    print("Usage: python main_script.py <input1> <input2> <input3>")
    sys.exit(1)

input1 = float(sys.argv[1])
input2 = float(sys.argv[2])
input3 = float(sys.argv[3])

# Fixed Parameters

gaussian_parameters = [[0.31327064580350383, -117.0, 3.15],
 [0.20894931421944332, -114.0, 2.5],
 [0.4298328475778421, -109.0, 3.0],
 [0.7048214103030962, -105.5, 3.0],
 [0.7551411564937404, -99.5, 4.0],
 [3.2401527428683927, -80.0, 9.0],
 [0.52, -74.0, 4.0],
 [0.6189744061235801, -67.5, 10.0],
 [0.25, -61.0, 2.0],
 [0.44550269392470787, -56.5, 4.0],
 [0.4016194616369327, -47.5, 4.5],
 [0.5235032811559645, -42.5, 2.5],
 [0.9111461622145292, -39.5, 3.5],
 [0.5224152038057177, -35.0, 4.0],
 [1.5, -19.5, 2.5]]

norm_range = [0, 3.2428885]

min_pixel_sum = -21.900393
max_pixel_sum = 8442.391

velocities = []
for i in range(500):
    velocity = -175. + i * 0.5
    velocities.append(velocity)



######################################
#############  FUNCTIONS  ############
######################################


####### Tester Functions #######
    
# Creates tester params list and call test function
def test_parameter_combinations_dynamic(fits_path, start_velocity, end_velocity, seeds,
                                        use_otsu=True, otsu_mult_factor=1.0,
                                        scaling_method='standard', epsilon=1e-3, max_iter=100,
                                        weighting_method='gaussian_kernel', alpha=0.7, beta=0.2,
                                        group_num=-1, bwrange=[0.1, 0.6, 1.2, 1.8],
                                        vel_importance=[1, 2, 4, 6], bright_importance=[1, 2, 4, 6],
                                        xy_importance=[1, 2, 4, 6],ofile=""):
    

    bandwidth_range = bwrange
    weight_range = vel_importance
    pixel_value_range = bright_importance
    xy_range = xy_importance

    start_time = time.time()  

    scaled_dataset, scaler, original_dataset = prepare_data_for_mean_shift(fits_path, start_velocity, end_velocity, use_otsu, otsu_mult_factor, scaling_method)
    scaled_seeds = scale_seeds(seeds, scaler)

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"Data preperation took {elapsed_time} seconds to run.")

    number_of_seeds = len(seeds) 

    param_combinations = product(bandwidth_range, repeat=number_of_seeds)
    param_combinations_with_others = itertools.product(param_combinations, weight_range, pixel_value_range, xy_range)


    print("data preprocessing is done...")
    
    for i in param_combinations_with_others:
        calculate_params_and_error(i, scaled_dataset, scaled_seeds, original_dataset, group_num, epsilon, max_iter, weighting_method, alpha, beta,seeds,ofile)


#Calls test function for one parameter set and writes results
def calculate_params_and_error(params, scaled_dataset, scaled_seeds, original_dataset, group_num, epsilon, max_iter, weighting_method, alpha, beta,seeds,ofile):
    
    bandwidth_combination, w, v, xy = params

    start_time = time.time()  
    
    C, labels, A, B = multi_label_mean_shift(scaled_dataset, scaled_seeds, list(bandwidth_combination), epsilon, max_iter, weighting_method, alpha, beta, w, v, xy)

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"Clustering took {elapsed_time} seconds to run.")
    start_time = time.time()  
    
    if len(original_dataset) == len(labels):
        result_df = pd.concat([original_dataset, labels], axis=1)
    else:
        print("Number of rows in df1 and df2 do not match.")
        exit

    agg_df, df = post_process_data_optimized(result_df)

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"Data Post-Processing took {elapsed_time} seconds to run.")
    start_time = time.time()  

    error1,error2,error3,error4 = calculate_error(agg_df, df, group_num)

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"Error Calculation took {elapsed_time} seconds to run.")

    
    params = {

        "bandwidths": list(bandwidth_combination),
        "alpha": alpha,
        "beta": beta,
        "method": weighting_method,
        "velocity_weight": w,
        "pixel_value_weight": v,
        "xy_weight": xy,

        "centric_shape_consistency": error1[0],
        "shape_smoothness": error1[1],
        "edge_consistency": error1[2],

        "edge_intensity_error":error2[0],
        "intensity_gradient_error":error2[1],
        "intensity_variance_error": error2[2],
        "peak_intensity_consistency": error2[3],

        "ge_normalized": error3[0],
        "ge": error3[1],

        "spatial_spread_error": error4[0],
        "gap_error": error4[1]
    }

    # File writing without lock
    with open(ofile, 'a') as f:
        f.write(f"{params}\n")
        print(f"{params}\n")

    # Explicitly delete variables that are not needed anymore
    del C, labels, A, B, agg_df, df



####### Data Preprocess Functions #######

def prepare_data_for_mean_shift(fits_path, start_velocity, end_velocity, use_otsu=True, otsu_mult_factor=1.0, scaling_method='standard'):
    """
    Prepare data for mean-shift clustering.

    Parameters:
        fits_path (str): Path to the FITS file containing the image data.
        start_velocity (float): Starting velocity for the region of interest.
        end_velocity (float): Ending velocity for the region of interest.
        use_otsu (bool): Whether to use Otsu thresholding.
        otsu_mult_factor (float): Multiplier for Otsu threshold.
        scaling_method (str): Method for scaling, either 'standard' or 'minmax'.

    Returns:
        scaled_dataset (array): Scaled dataset ready for clustering, shape (n_samples, n_features)
        scaler (object): Scaler object used for scaling.
    """
    hdul = fits.open(fits_path)
    data_list = []

    # Read image data into data_list
    for i in range(500):
        image = hdul[0].data[i]
        data_list.append(image)

    # Compute Otsu threshold if required
    if use_otsu:
        all_pixel_values = []
        for z, image in enumerate(data_list):
            velocity = -175. + z * 0.5
            if start_velocity <= velocity <= end_velocity:
                min_val = np.nanmin(image)
                image_no_nan = np.where(np.isnan(image), min_val, image)
                all_pixel_values.extend(image_no_nan.ravel())
        global_thresh = threshold_otsu(np.array(all_pixel_values))
        thresh = global_thresh * otsu_mult_factor
    else:
        thresh = 1.7  # Default threshold if not using Otsu

    # Create the dataset
    dataset = []
    for z, image in enumerate(data_list):
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                pixel_value = image[x, y]
                if not np.isnan(pixel_value) and (pixel_value > thresh):
                    if start_velocity <= (-175. + z * 0.5) <= end_velocity:
                        dataset.append([x, y, (-175. + z * 0.5), pixel_value])

    dataset = np.array(dataset)

    # Convert dataset to DataFrame
    dataset_df = pd.DataFrame(dataset, columns=['x', 'y', 'velocity', 'pixel_value'])

    # Choose the scaling method
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling_method. Choose either 'standard' or 'minmax'")

    # Scale the dataset
    scaled_dataset = scaler.fit_transform(dataset)

    # Convert scaled dataset to DataFrame
    scaled_df = pd.DataFrame(scaled_dataset, columns=['x', 'y', 'velocity', 'pixel_value'])

    return scaled_df, scaler, dataset_df


def scale_seeds(seeds, scaler):
    """
    Scale the seed values using a given scaler.

    Parameters:
        seeds (array): Seed points as a 2D NumPy array, shape (n_clusters, n_features)
        scaler (object): Scaler object used for scaling.

    Returns:
        scaled_seeds (array): Scaled seed points, shape (n_clusters, n_features)
    """

    return scaler.transform(seeds)




####### Clustering Functions #######

def multi_label_mean_shift(X, seeds, bandwidths, epsilon=1e-3, max_iter=100,
                           weighting_method='inverse_distance', alpha=0.6, beta=0.1, fweight_velocity=1,fweight_pixel_value=1,fweight_xy=1):
    """
    Multi-label Mean Shift clustering with soft assignments.

    Parameters:
        All parameters from mean_shift_clustering
        weighting_method (str): Method for calculating B matrix, one of ['inverse_distance', 'gaussian_kernel', 'softmax']
        alpha (float): Threshold for max probability to assign single label
        beta (float): Threshold for minimum probability to consider for multi-label assignment

    Returns:
        C (array): Cluster centers, shape (n_clusters, n_features)
        labels (list): Cluster labels for each data point, can be multi-label
        A (array): Distance matrix, shape (n_samples, n_clusters)
        B (array): Soft assignment matrix, shape (n_samples, n_clusters)
    """

    C, _, A = mean_shift_clustering(X, seeds, bandwidths, epsilon, max_iter, fweight_velocity, fweight_pixel_value,fweight_xy)

    B = calculate_B_matrix(A, method=weighting_method)

    # Initialize labels list
    labels = []

    # Calculate max probabilities and their indices
    max_probs = np.max(B, axis=1)
    max_indices = np.argmax(B, axis=1)

    # Mask for single label assignment
    single_label_mask = max_probs >= alpha

    # Single label assignment
    single_labels = max_indices[single_label_mask]
    single_labels_list = np.expand_dims(single_labels, axis=1).tolist()
    labels.extend(single_labels_list)

    # Multiple label assignment
    multi_label_rows = B[~single_label_mask, :]
    above_beta_mask = multi_label_rows >= beta
    below_beta_mask = ~above_beta_mask

    sum_below_beta = np.sum(multi_label_rows * below_beta_mask, axis=1, keepdims=True)
    sum_above_beta = np.sum(multi_label_rows * above_beta_mask, axis=1, keepdims=True)
    redistributed_probs = (multi_label_rows * above_beta_mask) + (multi_label_rows * above_beta_mask) / (sum_above_beta + 1e-10) * sum_below_beta

    # Normalizing redistributed probabilities
    total_prob = np.sum(redistributed_probs, axis=1, keepdims=True)
    normalized_probs = redistributed_probs / total_prob

    # Iterating only through the necessary indices and rows for final label assignment
    row_indices, col_indices = np.nonzero(above_beta_mask)
    unique_rows = np.unique(row_indices)

    # Building a mapping from row index to relevant column indices
    row_to_cols_map = {row: col_indices[row_indices == row] for row in unique_rows}

    # Using list comprehension to create labels
    labels.extend([list(zip(row_to_cols_map[row], normalized_probs[row, row_to_cols_map[row]])) for row in unique_rows])

    all_labels = []

    # Iterate through the labels list and concatenate single and multi-labels
    for label in labels:
        if isinstance(label, int):
            all_labels.append([label])
        elif isinstance(label, list):
            all_labels.append(label)

    # Create a DataFrame to store the labels
    labels_df = pd.DataFrame({'label': all_labels})


    return C, labels_df, A, B



def mean_shift_clustering(X, seeds, bandwidths, epsilon=1e-3, max_iter=10, fweight_velocity=1, fweight_pixel_value=1, fweight_xy=1):
    """
    Modified Mean Shift Clustering Algorithm with weighted features.

    Parameters:
        X (array): Data points as a 2D NumPy array, shape (n_samples, n_features)
        seeds (array): Initial seed points as a 2D NumPy array, shape (n_clusters, n_features)
        bandwidths (array): Bandwidths for each seed point as a 1D NumPy array, shape (n_clusters,)
        weight_velocity (float): Weight to apply to the velocity feature
        epsilon (float): Convergence threshold
        max_iter (int): Maximum number of iterations

    Returns:
        C (array): Cluster centers, shape (n_clusters, n_features)
        labels (array): Cluster labels for each data point, shape (n_samples,)
        A (array): Distance matrix, shape (n_samples, n_clusters)
    """

    weight_velocity = fweight_velocity
    weight_pixel_value = fweight_pixel_value
    weight_xy = fweight_xy

    # Weight the features
    X_weighted = np.copy(X)
    seeds_weighted = np.copy(seeds)

    # weighting all 4 features
    X_weighted[:, 0] *= weight_xy
    seeds_weighted[:, 0] *= weight_xy

    X_weighted[:, 1] *= weight_xy
    seeds_weighted[:, 1] *= weight_xy

    X_weighted[:, 2] *= weight_velocity
    seeds_weighted[:, 2] *= weight_velocity

    X_weighted[:, 3] *= weight_pixel_value
    seeds_weighted[:, 3] *= weight_pixel_value


    # Initialize cluster centers as seeds
    C = np.copy(seeds_weighted)

    # Initialize distance matrix A
    n_samples, n_features = X.shape
    n_clusters = len(seeds)
    A = np.zeros((n_samples, n_clusters))


    # Main loop for each seed point
    for i in range(n_clusters):
        s_i = seeds_weighted[i]
        b_i = bandwidths[i]
        
        # Using a NumPy array for efficient distance calculations
        X_weighted_np = np.array(X_weighted)

        for _ in range(max_iter):
            #calculate distances and filter neighbors within bandwidth
            distances = distance.cdist([s_i], X_weighted_np, 'euclidean').flatten()
            neighbors_mask = distances < b_i
            N = X_weighted_np[neighbors_mask]

            if len(N) == 0:
                break
            
            # Calculate the shift vector M using vectorized operations
            weights = np.exp(-np.sum((N - s_i)**2, axis=1) / (2 * b_i**2))
            M = np.sum(weights[:, None] * N, axis=0) / np.sum(weights)

            # Check for convergence using efficient distance calculation
            if distance.euclidean(s_i, M) < epsilon:
                break

            # Update the seed point
            s_i = M

        # Update cluster center
        C[i] = s_i


    # Fill distance matrix A and assign clusters
    labels = np.zeros(n_samples, dtype=int)
    # compute all distances using vectorized operations
    A = distance.cdist(X_weighted, C, 'euclidean')
    # Assign clusters by finding the index of the minimum distance for each sample
    labels = np.argmin(A, axis=1)


    # Revert the weighting on the final cluster centers for interpretability
    C[:, 0] /= weight_xy
    C[:, 1] /= weight_xy
    C[:, 2] /= weight_velocity
    C[:, 3] /= weight_pixel_value


    return C, labels, A


def calculate_B_matrix(A, method='inverse_distance'):
    """
    Calculate the B matrix from the A distance matrix using different methods.

    Parameters:
        A (array): Distance matrix, shape (n_samples, n_clusters)
        method (str): Method for calculating B, one of ['inverse_distance', 'gaussian_kernel', 'softmax']

    Returns:
        B (array): Matrix with percentage of each data point belonging to each cluster, shape (n_samples, n_clusters)
    """
    n_samples, n_clusters = A.shape
    B = np.zeros((n_samples, n_clusters))

    if method == 'inverse_distance':
        B = 1 / A
        row_sums = B.sum(axis=1, keepdims=True)
        B /= row_sums

    elif method == 'gaussian_kernel':
        B = np.exp(-A ** 2)
        row_sums = B.sum(axis=1, keepdims=True)
        zero_sum_mask = (row_sums == 0).flatten()
        row_sums[zero_sum_mask] = 1  # Prevent division by zero
        B /= row_sums
        B[zero_sum_mask, :] = 1 / n_clusters  # Uniform distribution for zero-sum rows

    elif method == 'softmax':
        B = np.exp(-A)
        row_sums = B.sum(axis=1, keepdims=True)
        B /= row_sums

    else:
        raise ValueError("Invalid method. Supported methods are 'inverse_distance', 'gaussian_kernel', 'softmax'")

    return B


def post_process_data_optimized(df):
    # Convert relevant columns to numpy arrays for faster processing
    labels = df['label'].to_numpy()
    pixel_values = df['pixel_value'].to_numpy()

    rows = []
    for i in range(len(labels)):
        label = labels[i]
        pixel_value = pixel_values[i]

        if isinstance(label, (int, np.integer)):
            rows.append(df.iloc[i])
        elif isinstance(label, list):
            if len(label) == 1 and isinstance(label[0], (int, np.integer)):
                new_row = df.iloc[i].copy()
                new_row['label'] = label[0]
                rows.append(new_row)
            else:
                if all(isinstance(item, tuple) and len(item) == 2 for item in label):
                    label_probs = label
                    new_labels, probs = zip(*label_probs)
                    new_pixel_values = np.multiply(pixel_value, probs)

                    for j in range(len(new_labels)):
                        new_row = df.iloc[i].copy()
                        new_row['label'] = new_labels[j]
                        new_row['pixel_value'] = new_pixel_values[j]
                        rows.append(new_row)
                else:
                    print(f"Warning: Label at index {i} is not in the proper format: {label}")
        else:
            print(f"Warning: Label at index {i} is not in the proper format: {label}")

    redistributed_df = pd.DataFrame(rows)

    # Using groupby and sum with numpy for faster processing
    pixel_sums_df = redistributed_df.groupby(['label', 'velocity'], as_index=False).agg({'pixel_value': np.sum})

    # Vectorized normalization with numpy
    min_pixel_sum = pixel_sums_df['pixel_value'].min()
    max_pixel_sum = pixel_sums_df['pixel_value'].max()
    norm_range = [0, 3.2428885]
    pixel_sums_df['normalized_pixel_sum'] = norm_range[0] + np.divide(np.multiply(np.subtract(pixel_sums_df['pixel_value'], min_pixel_sum), (norm_range[1] - norm_range[0])), (max_pixel_sum - min_pixel_sum))

    return pixel_sums_df, redistributed_df



####### Error Functions #######

# Calculate all errors
def calculate_error(agg_df,df, group_num):

    start_time = time.time() 

    error_shape = shape_errors(df)
    
    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"\t\t Shape Error Calculation took {elapsed_time} seconds to run.")
    start_time = time.time() 


    error_intensity = intensity_errors(df)

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"\t\t Intensity Error Calculation took {elapsed_time} seconds to run.")
    start_time = time.time() 


    error_gaussian = gaussian_errors(agg_df, group_num)

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"\t\t Gaussian Error Calculation took {elapsed_time} seconds to run.")
    start_time = time.time() 


    error_spatial = spatial_errors(df)

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"\t\t Spatial Error Calculation took {elapsed_time} seconds to run.")


    return (error_shape,error_intensity,error_gaussian,error_spatial)

# Shape Errors
def shape_errors(df):

    error1 = centric_shape_consistency(df)
    error2 = shape_smoothness(df)
    error3 = edge_consistency(df)
    
    return (error1,error2,error3)

def centric_shape_consistency(df):

    error = 0
    cloud_num = 0

    for label, group in df.groupby('label'):

        cloud_num += 1
        temp_error = 0
        center_velocity = group['velocity'].median()
        center_area = len(group[group['velocity'] == center_velocity])
        areas = group.groupby('velocity').size()
        size = len(areas)

        for i in areas:
            temp_error += (center_area-i)**2

        error += temp_error/size

    return error/cloud_num

def shape_smoothness(df):

    error = 0
    all_images = generate_images(df)
    num=0

    for label, images in all_images.items():
        unique_velocities = sorted(images.keys())
        num+=1
        for v in unique_velocities:
            if v + 0.5 in images and v - 0.5 in images:
                error += image_difference(images[v], images[v + 0.5])
                error += image_difference(images[v], images[v - 0.5])
        error/=len(unique_velocities)
    error/=num

    return error

def image_difference(img1, img2):
    # sum of squared differences
    return np.sum((img1 - img2) ** 2)

def generate_images(df):
    image_dict = {}

    # Group dataframe by label and iterate over groups
    for label, group in df.groupby('label'):
        label_images = {}
        # Filter velocities within group's max and min, iterate over unique velocities
        for v in np.arange(group['velocity'].min(), group['velocity'].max() + 1, 0.5):
            image = np.zeros((55, 55))
            # Select rows within the group that match the current velocity
            subset = group[group['velocity'] == v]
            # Use loc to efficiently select and assign pixel values in the image
            image[subset['x'].astype(int).values, subset['y'].astype(int).values] = subset['pixel_value'].values
            label_images[v] = image
        image_dict[label] = label_images

    return image_dict

def edge_consistency(df):

    error = 0
    cloud_num = 0

    for label, group in df.groupby('label'):

        cloud_num += 1
        temp_error = 0
        center_velocity = group['velocity'].median()
        center_ratio = eigen_ratio(group[group['velocity'] == center_velocity])
        ratios = df.groupby('velocity', group_keys=False).apply(eigen_ratio)
        size = len(ratios)

        for i in ratios:
            temp_error += (center_ratio-i)**2
        
        error += temp_error/size

    return error/cloud_num

def eigen_ratio(df_velocity):

    if len(df_velocity) == 0 or len(df_velocity) == 1:
        return 1

    if df_velocity[['x', 'y']].isnull().any().any() or np.isinf(df_velocity[['x', 'y']]).any().any():
        print("=============================================")
        print("=======ERROR IN EIGENRETIO FUNCTION==========")
        print("=============================================")
        return 1

    # Compute the covariance matrix
    covariance_matrix = np.cov(df_velocity['x'], df_velocity['y'])
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(covariance_matrix)
    
    # To prevent division by zero
    min_eigenvalue = min(eigenvalues)
    if min_eigenvalue == 0:
        min_eigenvalue = 1

    # Return the eigen ratio
    return max(eigenvalues) / min_eigenvalue



# Intensity Errors
def intensity_errors(df):
    error1 = edge_intensity_error(df)
    error2 = intensity_gradient_error(df)
    error3 = intensity_variance_error(df)
    error4 = peak_intensity_consistency(df)

    return (error1,error2,error3,error4)

def edge_intensity_error(df):
    error = 0
    cloud_num = 0

    for label, group in df.groupby('label'):

        cloud_num += 1
        center_x = group['x'].mean()
        center_y = group['y'].mean()
        center_velocity = group['velocity'].mean()
        group['distance'] = np.sqrt((group['x'] - center_x)**2 + (group['y'] - center_y)**2 + (group['velocity'] - center_velocity)**2)

        max_distance = group['distance'].max()

        # Extract outer and inner regions
        outer_region = group[group['distance'] >= 0.7 * max_distance]
        inner_region = group[group['distance'] < 0.7 * max_distance]

        # Compute average intensity for the outer and inner regions
        I_outer = outer_region['pixel_value'].mean()
        I_inner = inner_region['pixel_value'].mean()

        error += max(I_outer - I_inner, 0)

    return error/cloud_num

def intensity_gradient_error(df):
    error = 0
    cloud_num = 0

    for label, group in df.groupby('label'):
        cloud_num += 1
        brightest_velocity = group[group['pixel_value'] == group['pixel_value'].max()]['velocity'].values[0]

        # Extract data for both sides of the brightest region
        left_side = group[group['velocity'] <= brightest_velocity]
        right_side = group[group['velocity'] >= brightest_velocity]

        # Check the size and compute gradient for both sides
        if len(left_side) <= 1:
            error_left = 0
        else:
            G_left = np.gradient(left_side['pixel_value'].values)
            error_left = sum(g for g in G_left if g > 0) / max(1, sum(1 for g in G_left if g > 0))
            
        if len(right_side) <= 1:
            error_right = 0
        else:
            G_right = np.gradient(right_side['pixel_value'].values)
            error_right = sum(g for g in G_right if g > 0) / max(1, sum(1 for g in G_right if g > 0))

        error += error_left + error_right

    return error/cloud_num

def intensity_variance_error(df):
    error = 0
    cloud_num = 0

    for label, group in df.groupby('label'):
        cloud_num+=1
        variance = group['pixel_value'].var()
        error += variance / group['pixel_value'].mean()

    return error/cloud_num

def peak_intensity_consistency(df):
    error = 0
    cloud_num = 0

    for label, group in df.groupby('label'):
        cloud_num+=1
        brightest_region = group[group['pixel_value'] == group['pixel_value'].max()]
        peak_x = brightest_region['x'].values[0]
        peak_y = brightest_region['y'].values[0]
        peak_velocity = brightest_region['velocity'].values[0]

        # Compute the geometric center of the cloud
        center_x = group['x'].mean()
        center_y = group['y'].mean()
        avg_velocity = df['velocity'].mean()

        # Compute the distance between the geometric center and the peak intensity point
        distance = np.sqrt((peak_x - center_x)**2 + (peak_y - center_y)**2 + (peak_velocity - avg_velocity)**2)
        error += distance

    return error/cloud_num


# Gaussian Fit Error

def gaussian_calc(velocities, gaussian_params):
    result = []

    for amplitude, center, FWHM in gaussian_params:
        sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
        y_values = amplitude * np.exp(-(velocities - center)**2 / (2 * sigma**2))
        result.append(y_values)

    return result

def gaussian_errors(agg_df, group_num):

    velocities = agg_df['velocity'].unique()
    error_norm_pixel_sum = 0
    error_pixel_sum = 0

    if group_num == 0:
        gaussian_params = [[1.5, -19.5, 2.5]]
    elif group_num == 1:
        gaussian_params = [[0.31327064580350383, -117.0, 3.15],
                           [0.20894931421944332, -114.0, 2.5]]
    elif group_num == 2:
        gaussian_params = [[0.4298328475778421, -109.0, 3.0],
                           [0.7048214103030962, -105.5, 3.0],
                           [0.7551411564937404, -99.5, 4.0]]
    elif group_num == 3:
        gaussian_params = [[3.2401527428683927, -80.0, 9.0],
                           [0.52, -74.0, 4.0],
                           [0.6189744061235801, -67.5, 10.0],
                           [0.25, -61.0, 2.0],
                           [0.44550269392470787, -56.5, 4.0]]
    elif group_num == 4:
        gaussian_params = [[0.4016194616369327, -47.5, 4.5],
                           [0.5235032811559645, -42.5, 2.5],
                           [0.9111461622145292, -39.5, 3.5],
                           [0.5224152038057177, -35.0, 4.0]]
        
    expected_values_list = gaussian_calc(velocities, gaussian_params)

    for i in range(len(expected_values_list)):
        cluster_values = agg_df[agg_df['label'] == i]

        # Initialize arrays for storing differences
        diff_norm_pixel_sum = np.zeros(len(velocities))
        diff_pixel_sum = np.zeros(len(velocities))

        for j, velocity in enumerate(velocities):
            if velocity in cluster_values['velocity'].values:
                # Get the cluster value for the corresponding velocity
                cluster_value_norm_pixel_sum = cluster_values[cluster_values['velocity'] == velocity]['normalized_pixel_sum'].values[0]
                cluster_value_pixel_sum = cluster_values[cluster_values['velocity'] == velocity]['pixel_value'].values[0]

                # Calculate and update differences
                diff_norm_pixel_sum[j] = np.abs(cluster_value_norm_pixel_sum - expected_values_list[i][j])
                diff_pixel_sum[j] = np.abs((cluster_value_pixel_sum - expected_values_list[i][j])**2)

        # Calculate the average difference for this cluster
        error_norm_pixel_sum += np.mean(diff_norm_pixel_sum)
        error_pixel_sum += np.mean(diff_pixel_sum)

    # Calculate the average error
    num_clusters = len(expected_values_list)
    error_norm_pixel_sum /= num_clusters
    error_pixel_sum /= num_clusters

    return (error_norm_pixel_sum, error_pixel_sum)



# Spatial Errors
def spatial_errors(df):

    error1 = spatial_spread_error(df)
    error2 = gap_error(df)

    return (error1,error2)

def spatial_spread_error(df):
    # Compute centroids
    centroids = df.groupby('label').mean()[['x', 'y', 'velocity']]

    # Efficiently broadcast subtraction of coordinates between the dataframe and centroids
    diffs = df.set_index('label')[['x', 'y', 'velocity']] - centroids
    df['distance_to_centroid'] = np.linalg.norm(diffs, axis=1)

    # Average these distances to get the error
    error = df.groupby('label')['distance_to_centroid'].mean().mean()

    return error


def gap_error(df):

    # Helper function to compute gap for each group
    def compute_group_gap(group):
        x_range = group['x'].max() - group['x'].min() + 1  # +1 to include both boundaries
        y_range = group['y'].max() - group['y'].min() + 1  # +1 to include both boundaries
        velocity_range = (group['velocity'].max() - group['velocity'].min()) / 0.5 + 1  # +1 to include both boundaries

        total_possible_points = x_range * y_range * velocity_range
        actual_points = len(group)

        return (total_possible_points - actual_points) / total_possible_points

    # Compute gaps for each cluster and take the mean
    # Compute gaps for each cluster and take the mean
    average_gap = df.groupby('label', group_keys=False).apply(compute_group_gap).mean()

    return average_gap

######################################
#############  MAIN PART  ############
######################################
    

fits_path = "4U1630_12CO.fits"


seeds_list = [ [[ 2, 18, -20, 16.267826 ]],
              
               [[36, 28, -117, 4.000822],
                [3, 34, -114, 4.590847	]],
               
                [[2, 35, -100,   5.597802],
                 [34, 22, -105,  6.21070 ],
                 [24, 2, -109,  6.593767 ]],
               
               [[5, 2, -80, 11.789562	  ],
                [18, 95, -73, 3.884139	  ],
                [26, 40, -68, 10.8933067 ],
                [12, 14, -61, 4.530908		],
                [6, 13, -56, 4.385234    ]],
              
               [[13, 42, -47, 4.081348 ],
                [11, 43, -42, 6.984776	],
                [20, 27, -39, 2.355332 ],
                [12, 1, -35, 2.021111  ]]]

intervals_list = [[-25, -15],[-121, -111.5],[-112, -95],[-95, -52,],[-53, -30]]

otsu_factor_list = [0.2,0.3,0.5,0.25,0.6]

file_lock = threading.Lock()


### ONLY CHANGE HERE (for testing) (starts) ###

gnum = 3

bandwiths = [0.1]

if input1 < 8:
    velocity_imp = np.concatenate([
            np.arange(0.5+input1, input1+1, 0.5),
            #np.arange(8, 16, 1),
        ])
else:
    velocity_imp = np.concatenate([
            #np.arange(0.5+input1, input1, 0.5),
            np.arange(input1, input1+1, 1),
        ])


if input2 == 0:
    brightness_imp = np.concatenate([
            np.arange(0.5, 4, 0.5),
            #np.arange(4, 8, 0.5),
            #np.arange(8, 16, 1),
        ])
elif input2 == 1:
    brightness_imp = np.concatenate([
            #np.arange(0.5, 4, 0.5),
            np.arange(4, 8, 0.5),
            #np.arange(8, 16, 1),
        ])
elif input2 == 2:
    brightness_imp = np.concatenate([
            #np.arange(0.5, 4, 0.5),
            #np.arange(4, 8, 0.5),
            np.arange(8, 12, 1),
        ])
    
elif input2 == 3:
    brightness_imp = np.concatenate([
            #np.arange(0.5, 4, 0.5),
            #np.arange(4, 8, 0.5),
            np.arange(12, 16, 1),
        ])

if input3 == 0:
    xy_imp = np.concatenate([
            np.arange(0.5, 8, 0.5),
            #np.arange(8, 16, 1),
        ])

elif input3 == 1:
    xy_imp = np.concatenate([
            #np.arange(0.5, 8, 0.5),
            np.arange(8, 16, 1),
        ])

### ONLY CHANGE HERE (for testing)  (ends) ###
    
suffix_list = ["20","117-114","109-100","80-56","47-35"]
output_file_name = "output_MS_"+ suffix_list[gnum] +".txt"


test_parameter_combinations_dynamic(fits_path, intervals_list[gnum][0],
                                    intervals_list[gnum][1],seeds_list[gnum],
                                    otsu_mult_factor = otsu_factor_list[gnum],
                                    group_num=gnum,bwrange = bandwiths,
                                    vel_importance = velocity_imp,
                                    bright_importance = brightness_imp,
                                    xy_importance = xy_imp,ofile=output_file_name)

print("=====FINISHED=====")
