import os
import nibabel as nib
from nilearn import datasets, image, regions, plotting
from nilearn.image import resample_to_img
from typing import List, Dict, Tuple
import logging
import numpy as np
import matplotlib.pyplot as plt


def visualize_registration(
    mri_path: str,
    atlas_name: str = 'harvard_oxford',
    output_dir: str = './visualization_outputs',
    sample_regions: List[str] = None
):
    """
    Visualize the registration of MRI data to an atlas space.
    
    Args:
        mri_path: Path to the subject's MRI file (NIfTI format)
        atlas_name: Name of the atlas ('harvard_oxford', 'aal', or 'destrieux')
        output_dir: Directory for saving visualization outputs
        sample_regions: List of specific regions to visualize (optional)
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("registration_visualization")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load atlas
    logger.info(f"Loading {atlas_name} atlas...")
    if atlas_name == 'harvard_oxford':
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    elif atlas_name == 'aal':
        atlas = datasets.fetch_atlas_aal()
    elif atlas_name == 'destrieux':
        atlas = datasets.fetch_atlas_destrieux_2009()
    else:
        raise ValueError(f"Unsupported atlas: {atlas_name}")
    
    atlas_img = atlas.maps
    atlas_labels = atlas.labels if isinstance(atlas.labels, list) else atlas.labels.tolist()
    
    # Load MRI
    logger.info(f"Loading MRI from {mri_path}...")
    try:
        mri_img = nib.load(mri_path)
    except Exception as e:
        logger.error(f"Failed to load MRI: {str(e)}")
        raise
    
    # Step 1: Visualize original MRI
    logger.info("Generating original MRI visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plotting.plot_anat(mri_img, title='Original MRI', axes=ax)
    plt.savefig(os.path.join(output_dir, 'original_mri.png'), dpi=300)
    plt.close(fig)
    
    # Step 2: Visualize atlas
    logger.info("Generating atlas visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plotting.plot_roi(atlas_img, title=f'{atlas_name.capitalize()} Atlas', axes=ax)
    plt.savefig(os.path.join(output_dir, 'atlas.png'), dpi=300)
    plt.close(fig)
    
    # Step 3: Register MRI to atlas space
    logger.info("Registering MRI to atlas space...")
    try:
        registered_mri = resample_to_img(
            mri_img,
            atlas_img,
            interpolation='linear'
        )
        
        # Save registered image for inspection
        nib.save(registered_mri, os.path.join(output_dir, 'registered_mri.nii.gz'))
        
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise
    
    # Step 4: Visualize registered MRI
    logger.info("Generating registered MRI visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plotting.plot_anat(registered_mri, title='Registered MRI', axes=ax)
    plt.savefig(os.path.join(output_dir, 'registered_mri.png'), dpi=300)
    plt.close(fig)
    
    # Step 5: Overlay registered MRI with atlas
    logger.info("Generating overlay visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    display = plotting.plot_anat(registered_mri, title='MRI Registered to Atlas Space', axes=ax)
    display.add_contours(atlas_img, levels=np.unique(atlas_img.get_fdata())[1:], 
                         colors='r', linewidths=0.5, alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'registered_mri_with_atlas_contours.png'), dpi=300)
    plt.close(fig)
    
    # Step 6: Detailed visualization of specific regions if requested
    if sample_regions:
        atlas_data = atlas_img.get_fdata()
        for region_name in sample_regions:
            if region_name in atlas_labels:
                logger.info(f"Visualizing region: {region_name}")
                region_idx = atlas_labels.index(region_name)
                region_mask = atlas_data == region_idx
                
                # Create region mask image
                region_mask_img = nib.Nifti1Image(region_mask.astype(np.int16), 
                                                 atlas_img.affine, atlas_img.header)
                
                # Plot region mask overlaid on registered MRI
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Sagittal view
                plotting.plot_roi(region_mask_img, bg_img=registered_mri, 
                                 display_mode='y', cut_coords=1, title=f'{region_name} (Sagittal)',
                                 axes=axes[0])
                
                # Coronal view
                plotting.plot_roi(region_mask_img, bg_img=registered_mri, 
                                 display_mode='x', cut_coords=1, title=f'{region_name} (Coronal)',
                                 axes=axes[1])
                
                # Axial view
                plotting.plot_roi(region_mask_img, bg_img=registered_mri, 
                                 display_mode='z', cut_coords=1, title=f'{region_name} (Axial)',
                                 axes=axes[2])
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'region_{region_name.replace(" ", "_")}.png'), dpi=300)
                plt.close(fig)
    
    # Step 7: Generate a composite image showing registration steps
    logger.info("Generating composite visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original MRI
    plotting.plot_anat(mri_img, title='1. Original MRI', axes=axes[0, 0])
    
    # Atlas
    plotting.plot_roi(atlas_img, title=f'2. {atlas_name.capitalize()} Atlas', axes=axes[0, 1])
    
    # Registered MRI
    plotting.plot_anat(registered_mri, title='3. Registered MRI', axes=axes[1, 0])
    
    # Overlay
    display = plotting.plot_anat(registered_mri, title='4. Registration Overlay', axes=axes[1, 1])
    display.add_contours(atlas_img, levels=[1], colors='r', linewidths=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'registration_process_summary.png'), dpi=300)
    plt.close(fig)
    
    # Step 8: Create a detailed registration quality check
    logger.info("Generating registration quality check...")
    # Extract slices from atlas and registered MRI
    atlas_data = atlas_img.get_fdata()
    registered_data = registered_mri.get_fdata()
    
    # Find meaningful slices (where atlas has labels)
    nonzero_indices = np.where(atlas_data > 0)
    if len(nonzero_indices[0]) > 0:
        mid_x = int(np.median(nonzero_indices[0]))
        mid_y = int(np.median(nonzero_indices[1]))
        mid_z = int(np.median(nonzero_indices[2]))
        
        # Create quality check visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # Row 1: Atlas slices
        axes[0, 0].imshow(atlas_data[mid_x, :, :], cmap='nipy_spectral')
        axes[0, 0].set_title(f'Atlas (Sagittal, x={mid_x})')
        axes[0, 1].imshow(atlas_data[:, mid_y, :], cmap='nipy_spectral')
        axes[0, 1].set_title(f'Atlas (Coronal, y={mid_y})')
        axes[0, 2].imshow(atlas_data[:, :, mid_z], cmap='nipy_spectral')
        axes[0, 2].set_title(f'Atlas (Axial, z={mid_z})')
        
        # Row 2: Registered MRI slices
        axes[1, 0].imshow(registered_data[mid_x, :, :], cmap='gray')
        axes[1, 0].set_title(f'Registered MRI (Sagittal, x={mid_x})')
        axes[1, 1].imshow(registered_data[:, mid_y, :], cmap='gray')
        axes[1, 1].set_title(f'Registered MRI (Coronal, y={mid_y})')
        axes[1, 2].imshow(registered_data[:, :, mid_z], cmap='gray')
        axes[1, 2].set_title(f'Registered MRI (Axial, z={mid_z})')
        
        # Row 3: Overlay
        # Create custom colormap for overlay
        from matplotlib.colors import LinearSegmentedColormap
        atlas_cmap = LinearSegmentedColormap.from_list('atlas_transparent', 
                                                       [(0, (0, 0, 0, 0)), 
                                                        (1, (1, 0, 0, 0.5))])
        
        # Sagittal overlay
        axes[2, 0].imshow(registered_data[mid_x, :, :], cmap='gray')
        atlas_overlay = np.ma.masked_where(atlas_data[mid_x, :, :] == 0, atlas_data[mid_x, :, :])
        axes[2, 0].imshow(atlas_overlay, cmap=atlas_cmap, alpha=0.7)
        axes[2, 0].set_title(f'Registration Overlay (Sagittal)')
        
        # Coronal overlay
        axes[2, 1].imshow(registered_data[:, mid_y, :], cmap='gray')
        atlas_overlay = np.ma.masked_where(atlas_data[:, mid_y, :] == 0, atlas_data[:, mid_y, :])
        axes[2, 1].imshow(atlas_overlay, cmap=atlas_cmap, alpha=0.7)
        axes[2, 1].set_title(f'Registration Overlay (Coronal)')
        
        # Axial overlay
        axes[2, 2].imshow(registered_data[:, :, mid_z], cmap='gray')
        atlas_overlay = np.ma.masked_where(atlas_data[:, :, mid_z] == 0, atlas_data[:, :, mid_z])
        axes[2, 2].imshow(atlas_overlay, cmap=atlas_cmap, alpha=0.7)
        axes[2, 2].set_title(f'Registration Overlay (Axial)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'registration_quality_check.png'), dpi=300)
        plt.close(fig)
        
    # Log completion
    logger.info(f"Visualization complete. Results saved to {output_dir}")
    return output_dir


# Example usage:
# output_path = visualize_registration('path/to/mri.nii.gz', 
#                                      atlas_name='harvard_oxford',
#                                      sample_regions=['Frontal Pole', 'Temporal Pole'])
# metrics = evaluate_registration_quality('path/to/mri.nii.gz')


def evaluate_registration_quality(
    mri_path: str,
    atlas_name: str = 'harvard_oxford',
    output_dir: str = './evaluation_outputs'
):
    """
    Evaluate the quality of MRI to atlas registration and output metrics.
    
    Args:
        mri_path: Path to the subject's MRI file
        atlas_name: Name of the atlas
        output_dir: Directory for saving evaluation outputs
    
    Returns:
        Dictionary containing registration quality metrics
    """
    # Set up logging and output directory
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("registration_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load atlas
    logger.info(f"Loading {atlas_name} atlas for evaluation...")
    if atlas_name == 'harvard_oxford':
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    elif atlas_name == 'aal':
        atlas = datasets.fetch_atlas_aal()
    elif atlas_name == 'destrieux':
        atlas = datasets.fetch_atlas_destrieux_2009()
    else:
        raise ValueError(f"Unsupported atlas: {atlas_name}")
    
    atlas_img = atlas.maps
    
    # Load MRI
    logger.info(f"Loading MRI from {mri_path}...")
    mri_img = nib.load(mri_path)
    
    # Register MRI to atlas space
    logger.info("Registering MRI to atlas space for evaluation...")
    registered_mri = resample_to_img(
        mri_img,
        atlas_img,
        interpolation='linear'
    )
    
    # Extract data
    atlas_data = atlas_img.get_fdata()
    registered_data = registered_mri.get_fdata()
    
    # 1. Check alignment of brain boundaries
    logger.info("Evaluating brain boundary alignment...")
    
    # Create binary masks
    atlas_mask = atlas_data > 0
    # Threshold the MRI data to create a binary mask
    # using Otsu's method to find optimal threshold
    from skimage.filters import threshold_otsu
    try:
        thresh = threshold_otsu(registered_data)
        mri_mask = registered_data > thresh
    except Exception as e:
        logger.warning(f"Could not use Otsu thresholding: {str(e)}. Using simple thresholding.")
        # Simple thresholding as fallback
        mri_mask = registered_data > np.mean(registered_data)
    
    # Calculate Dice coefficient for overlap
    intersection = np.logical_and(atlas_mask, mri_mask)
    dice_coefficient = 2 * np.sum(intersection) / (np.sum(atlas_mask) + np.sum(mri_mask))
    
    # 2. Check intensity distributions in atlas regions
    logger.info("Evaluating intensity distributions in atlas regions...")
    
    # Get unique atlas regions
    unique_regions = np.unique(atlas_data)
    unique_regions = unique_regions[unique_regions > 0]  # Skip background
    
    region_stats = {}
    total_intensity_variance = 0
    
    for region_id in unique_regions:
        region_mask = atlas_data == region_id
        region_values = registered_data[region_mask]
        
        if len(region_values) > 0:
            # Calculate statistics
            region_stats[int(region_id)] = {
                'mean': float(np.mean(region_values)),
                'std': float(np.std(region_values)),
                'min': float(np.min(region_values)),
                'max': float(np.max(region_values)),
                'voxel_count': int(np.sum(region_mask))
            }
            
            # Add to total variance calculation (weighted by region size)
            total_intensity_variance += region_stats[int(region_id)]['std'] * region_stats[int(region_id)]['voxel_count']
    
    # Normalize total variance
    total_voxels = np.sum([stats['voxel_count'] for stats in region_stats.values()])
    avg_weighted_variance = total_intensity_variance / total_voxels if total_voxels > 0 else 0
    
    # 3. Calculate mutual information between atlas and registered MRI
    logger.info("Calculating mutual information...")
    
    from sklearn.metrics import mutual_info_score
    
    # Bin the data for mutual information calculation
    bins = 32
    atlas_hist = np.histogram(atlas_data.flatten(), bins=bins)[0]
    mri_hist = np.histogram(registered_data.flatten(), bins=bins)[0]
    
    # Normalize histograms
    atlas_hist = atlas_hist / np.sum(atlas_hist)
    mri_hist = mri_hist / np.sum(mri_hist)
    
    # Calculate 2D histogram
    h, _, _ = np.histogram2d(
        atlas_data.flatten(), 
        registered_data.flatten(), 
        bins=bins
    )
    h = h / np.sum(h)
    
    # Calculate mutual information
    mutual_info = 0
    for i in range(bins):
        for j in range(bins):
            if h[i, j] > 0:
                mutual_info += h[i, j] * np.log2(h[i, j] / (atlas_hist[i] * mri_hist[j]))
    
    # 4. Visualization of evaluation results
    logger.info("Generating evaluation visualizations...")
    
    # 4.1 Plot region intensity distributions
    plt.figure(figsize=(15, 10))
    region_means = [stats['mean'] for stats in region_stats.values()]
    region_stds = [stats['std'] for stats in region_stats.values()]
    region_ids = list(region_stats.keys())
    
    plt.bar(range(len(region_ids)), region_means, yerr=region_stds, alpha=0.7)
    plt.xlabel('Atlas Region ID')
    plt.ylabel('Mean Intensity (with std. dev.)')
    plt.title('Intensity Distribution by Atlas Region')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'region_intensity_distribution.png'), dpi=300)
    plt.close()
    
    # 4.2 Plot alignment visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Find the middle slices
    x_mid = atlas_data.shape[0] // 2
    y_mid = atlas_data.shape[1] // 2
    z_mid = atlas_data.shape[2] // 2
    
    # Plot alignment in axial view
    axes[0].imshow(mri_mask[:, :, z_mid], cmap='Blues', alpha=0.5)
    axes[0].imshow(atlas_mask[:, :, z_mid], cmap='Reds', alpha=0.5)
    axes[0].set_title(f'Alignment (Axial, z={z_mid})')
    
    # Plot alignment in coronal view
    axes[1].imshow(mri_mask[:, y_mid, :], cmap='Blues', alpha=0.5)
    axes[1].imshow(atlas_mask[:, y_mid, :], cmap='Reds', alpha=0.5)
    axes[1].set_title(f'Alignment (Coronal, y={y_mid})')
    
    # Plot alignment in sagittal view
    axes[2].imshow(mri_mask[x_mid, :, :], cmap='Blues', alpha=0.5)
    axes[2].imshow(atlas_mask[x_mid, :, :], cmap='Reds', alpha=0.5)
    axes[2].set_title(f'Alignment (Sagittal, x={x_mid})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'registration_alignment.png'), dpi=300)
    plt.close()
    
    # Compile evaluation metrics
    metrics = {
        'dice_coefficient': float(dice_coefficient),
        'mutual_information': float(mutual_info),
        'avg_weighted_variance': float(avg_weighted_variance),
        'region_coverage': len(region_stats) / len(unique_regions) if len(unique_regions) > 0 else 0,
        'total_regions_detected': len(region_stats),
        'evaluation_timestamp': str(np.datetime64('now')),
        'region_stats': region_stats
    }
    
    # Save metrics to file
    import json
    with open(os.path.join(output_dir, 'registration_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    return metrics
