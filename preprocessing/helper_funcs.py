import os
import nibabel as nib
import nilearn
from nilearn.image import resample_to_img, smooth_img, load_img
import logging
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
from nilearn.plotting import plot_anat, plot_img
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import torch
import torch.nn.functional as F

def identify_nifti_files(files):
    """
    Identify and categorize NIfTI files for a patient
    """
    #nifti_files = [f for f in os.listdir(patient_folder) if f.endswith('.nii.gz')]
    
    # Categorize files
    file_info = []
    for filename in files:
        #full_path = os.path.join(patient_folder, filename)
        
        # Load NIfTI header to get metadata
        #nifti_img = nib.load(full_path)
        nifti_img = nib.load(filename)
        header = nifti_img.header
        
        file_details = {
            'filename': filename,
            # 'full_path': full_path,
            'dimensions': header['dim'],
            'voxel_size': header['pixdim'][1:4],
            'datatype': header['datatype']
        }
        
        # Try to infer scan type from filename
        filename_lower = filename.lower()
        if 't1' in filename_lower:
            file_details['type'] = 'T1'
        elif 't2' in filename_lower:
            file_details['type'] = 'T2'
        elif 'flair' in filename_lower:
            file_details['type'] = 'FLAIR'
        else:
            file_details['type'] = 'Unknown'
        
        file_info.append(file_details)
    
    return file_info

def select_best_nifti_file(file_info):
    """
    Select the most appropriate NIfTI file for analysis
    """
    # Prioritization strategy
    priority_order = ['T1', 'FLAIR', 'T2', 'Unknown']
    
    # First, filter for files with meaningful dimensions
    valid_files = [
        f for f in file_info 
        if len(f['dimensions']) > 3 and f['dimensions'][3] > 1
    ]
    
    # Sort by type priority
    sorted_files = sorted(
        valid_files, 
        key=lambda x: priority_order.index(x['type']) if x['type'] in priority_order else len(priority_order)
    )
    
    # Return the top choice
    return sorted_files[0] if sorted_files else None

# Example usage
def process_patient_mri(files):
    # Identify all NIfTI files
    file_info = identify_nifti_files(files)
    
    # Print out file information
    # for file in file_info:
    #     print(f"Filename: {file['filename']}")
    #     print(f"Type: {file['type']}")
    #     print(f"Dimensions: {file['dimensions']}")
    #     print(f"Voxel Size: {file['voxel_size']}")
    #     print("---")
    
    # Select best file
    best_file = select_best_nifti_file(file_info)
    
    if best_file:
        #print(f"Recommended file: {best_file['filename']}")
        return best_file['filename']
    else:
        print("No suitable NIfTI file found.")
        return None
    
def enhance_registration(mri_path: str, atlas_img, target_resolution_mm=2):
    """
    Enhanced MRI registration pipeline with robust preprocessing
    
    Args:
        mri_path: Path to input MRI
        atlas_img: Target atlas image
        target_resolution_mm: Target voxel resolution in mm
    """
    # 1. Load MRI
    mri_img = load_img(mri_path)
    
    # 2. Intensity normalization
    mri_data = mri_img.get_fdata()
    brain_mask = mri_data > np.percentile(mri_data, 10)
    brain_mean = np.mean(mri_data[brain_mask])
    brain_std = np.std(mri_data[brain_mask])
    mri_norm_data = (mri_data - brain_mean) / (brain_std + 1e-8)
    
    # Create normalized image
    mri_norm = nib.Nifti1Image(mri_norm_data, mri_img.affine)
    
    # 3. Resample to target resolution
    target_affine = np.diag([target_resolution_mm] * 3 + [1])
    mri_resampled = resample_to_img(
        mri_norm,
        atlas_img,
        interpolation='continuous'
    )
    
    # 4. Noise reduction
    mri_smooth = smooth_img(mri_resampled, fwhm=2)
    
    # 5. Enhanced registration using DIPY
    try:
        from dipy.align.imaffine import (transform_centers_of_mass,
                                        AffineMap,
                                        MutualInformationMetric,
                                        AffineRegistration)
        from dipy.align.transforms import (TranslationTransform3D,
                                         RigidTransform3D,
                                         AffineTransform3D)
        
        # Convert to dipy-compatible format
        moving_data = np.asarray(mri_smooth.get_fdata(), dtype=np.float64)
        static_data = np.asarray(atlas_img.get_fdata(), dtype=np.float64)
        
        # Ensure arrays are 3D
        if moving_data.ndim > 3:
            moving_data = moving_data[:,:,:,0]
        if static_data.ndim > 3:
            static_data = static_data[:,:,:,0]
            
        # Get affine transforms
        moving_affine = mri_smooth.affine
        static_affine = atlas_img.affine
        
        # Setup registration parameters
        nbins = 32
        sampling_prop = None
        metric = MutualInformationMetric(nbins, sampling_prop)
        
        # Increase iterations and add more levels
        level_iters = [10000, 1000, 100]
        sigmas = [3.0, 1.0, 0.0]
        factors = [4, 2, 1]
        
        # Initialize AffineRegistration object
        affreg = AffineRegistration(metric=metric,
                                   level_iters=level_iters,
                                   sigmas=sigmas,
                                   factors=factors)
        
        # Get the center of mass transform
        c_of_mass = transform_centers_of_mass(static_data,
                                            static_affine,
                                            moving_data,
                                            moving_affine)
        
        # Translation
        transform = TranslationTransform3D()
        translation = affreg.optimize(static_data,
                                    moving_data,
                                    transform,
                                    None,
                                    static_affine,
                                    moving_affine,
                                    starting_affine=c_of_mass.affine)
        
        # Rigid
        transform = RigidTransform3D()
        rigid = affreg.optimize(static_data,
                               moving_data,
                               transform,
                               None,
                               static_affine,
                               moving_affine,
                               starting_affine=translation.affine)
        
        # Affine
        transform = AffineTransform3D()
        affine = affreg.optimize(static_data,
                                moving_data,
                                transform,
                                None,
                                static_affine,
                                moving_affine,
                                starting_affine=rigid.affine)
        
        # Apply final transformation
        affine_map = AffineMap(affine.affine,
                              static_data.shape, static_affine,
                              moving_data.shape, moving_affine)
        
        registered_data = affine_map.transform(moving_data)
        final_img = nib.Nifti1Image(registered_data, static_affine)
        
    except Exception as e:
        print(f"DIPY registration failed with error: {str(e)}")
        print("Falling back to simple registration...")
        final_img = resample_to_img(
            mri_smooth,
            atlas_img,
            interpolation='continuous'
        )
    
    return final_img

def verify_registration(registered_img, atlas_img, output_dir='./registration_check1'):
    """
    Verify registration quality with enhanced metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    reg_data = registered_img.get_fdata()
    atlas_data = atlas_img.get_fdata()
    
    # Ensure arrays are 3D
    if reg_data.ndim > 3:
        reg_data = reg_data[:,:,:,0]
    if atlas_data.ndim > 3:
        atlas_data = atlas_data[:,:,:,0]
    
    # Normalize data for comparison
    reg_data_norm = (reg_data - reg_data.mean()) / (reg_data.std() + 1e-8)
    atlas_data_norm = (atlas_data - atlas_data.mean()) / (atlas_data.std() + 1e-8)
    
    # Calculate mutual information
    from sklearn.metrics import mutual_info_score
    hist_2d, _, _ = np.histogram2d(reg_data_norm.ravel(),
                                  atlas_data_norm.ravel(),
                                  bins=20)
    mi = mutual_info_score(None, None, contingency=hist_2d)
    
    # Calculate correlation
    from scipy.stats import pearsonr
    corr, _ = pearsonr(reg_data_norm.ravel(), atlas_data_norm.ravel())
    
    # Create visualization
    from nilearn import plotting
    import matplotlib.pyplot as plt
    
    # Multi-view comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get middle slices
    x, y, z = [s//2 for s in reg_data.shape]
    
    # Plot registered image
    axes[0,0].imshow(reg_data[x,:,:], cmap='gray')
    axes[0,0].set_title('Registered (Sagittal)')
    axes[0,1].imshow(reg_data[:,y,:], cmap='gray')
    axes[0,1].set_title('Registered (Coronal)')
    axes[0,2].imshow(reg_data[:,:,z], cmap='gray')
    axes[0,2].set_title('Registered (Axial)')
    
    # Plot atlas
    axes[1,0].imshow(atlas_data[x,:,:], cmap='gray')
    axes[1,0].set_title('Atlas (Sagittal)')
    axes[1,1].imshow(atlas_data[:,y,:], cmap='gray')
    axes[1,1].set_title('Atlas (Coronal)')
    axes[1,2].imshow(atlas_data[:,:,z], cmap='gray')
    axes[1,2].set_title('Atlas (Axial)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'registration_comparison.png'))
    plt.close()
    
    # Create edge-based visualization
    from scipy.ndimage import sobel
    reg_edges = sobel(reg_data_norm)
    atlas_edges = sobel(atlas_data_norm)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    views = [(x, ':', ':', 'Sagittal'),
             (':', y, ':', 'Coronal'),
             (':', ':', z, 'Axial')]
    
    for i, (slice_x, slice_y, slice_z, title) in enumerate(views):
        if slice_x == ':':
            reg_slice = eval(f'reg_edges[{slice_x},{slice_y},{slice_z}]')
            atlas_slice = eval(f'atlas_edges[{slice_x},{slice_y},{slice_z}]')
        else:
            reg_slice = eval(f'reg_edges[{slice_x},{slice_y},{slice_z}]')
            atlas_slice = eval(f'atlas_edges[{slice_x},{slice_y},{slice_z}]')
            
        axes[i].imshow(reg_slice, cmap='Reds', alpha=0.5)
        axes[i].imshow(atlas_slice, cmap='Blues', alpha=0.5)
        axes[i].set_title(f'Edge Overlay ({title})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'registration_overlay.png'))
    plt.close()
    
    # Calculate edge overlap score
    edge_overlap = np.sum(reg_edges * atlas_edges) / np.sqrt(np.sum(reg_edges**2) * np.sum(atlas_edges**2))
    
    metrics = {
        'mutual_information': float(mi),
        'correlation': float(corr),
        'intensity_range_ratio': float(np.ptp(reg_data) / np.ptp(atlas_data)),
        'mean_intensity_ratio': float(np.mean(reg_data) / np.mean(atlas_data)),
        'edge_overlap_score': float(edge_overlap)
    }
    
    return metrics

def plot_training_curves(train_losses, train_accuracies, val_accuracies):
    """Plot training and validation curves"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.show()

def plot_confusion_matrix(cm):
    """Plot confusion matrix as a heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

def plot_roc_curve(y_true, y_scores):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    plt.show()
    
    # Print AUC score
    print(f"ROC AUC Score: {roc_auc:.4f}")

def plot_precision_recall_curve(y_true, y_scores):
    """Plot precision-recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=300)
    plt.show()
    
    # Print PR AUC score
    print(f"Precision-Recall AUC Score: {pr_auc:.4f}")

def feature_importance_analysis(model, test_loader, device):
    """
    Analyze feature importance using a simple perturbation-based approach
    Note: This is a basic implementation and works best with GCN models
    """
    model.eval()
    
    # Get a batch of data
    batch = next(iter(test_loader)).to(device)
    base_features = batch.x.clone()
    num_features = base_features.shape[1]
    
    # Get baseline predictions
    with torch.no_grad():
        base_output = model(batch)
        base_preds = torch.argmax(base_output, dim=1)
    
    # Measure impact of perturbing each feature
    importance_scores = []
    for feat_idx in range(num_features):
        # Perturb one feature
        perturbed_features = base_features.clone()
        perturbed_features[:, feat_idx] = 0  # Zero out the feature
        
        # Replace features in the batch
        perturbed_batch = batch.clone()
        perturbed_batch.x = perturbed_features
        
        # Get predictions with perturbed feature
        with torch.no_grad():
            perturbed_output = model(perturbed_batch)
            perturbed_preds = torch.argmax(perturbed_output, dim=1)
        
        # Calculate impact as prediction difference
        impact = (base_preds != perturbed_preds).float().mean().item()
        importance_scores.append(impact)
    
    return importance_scores

def plot_feature_importance(importance_scores, feature_names=None):
    """Plot feature importance scores"""
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance_scores))]
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance (Impact on Predictions)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()
