import mne
import torch
import os
import nibabel as nib
from nilearn import datasets, image, regions, plotting
from nilearn.image import resample_to_img
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
import numpy as np
import traceback
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


#MRI-sEEG graph processor 
class GraphProcessor:
    def __init__(self,
                 mni_coordinates: dict,
                 seeg_channels: list,
                 region_code_map: dict,
                 target_length: int = 50000,
                 temporal_window: int = 100):
        """
        Initialize simplified graph processor without atlas-based region mapping.
        
        Args:
            mni_coordinates: Dictionary mapping regions to MNI coordinates
            seeg_channels: List of standard sEEG channel names
            region_code_map: Dictionary mapping region codes to region names
            target_length: Fixed length for sEEG time series
            temporal_window: Window size for temporal features
        """
        self.target_length = target_length
        self.mni_coordinates = mni_coordinates
        self.seeg_channels = seeg_channels
        self.temporal_window = temporal_window
        self.region_code_map = region_code_map
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create the channel to MNI mapping
        self.channel_mni_mapping = self._create_channel_mni_mapping()

    def _create_channel_mni_mapping(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Create mapping from sEEG channel names to MNI coordinates.
        Uses the standard channel list from self.seeg_channels.
        
        Returns:
            Dictionary mapping channel names to MNI coordinates
        """
        channel_mni_mapping = {}

        for channel in self.seeg_channels:
            parts = channel.split('-')
            region_code1 = parts[0]  # Extract first part of the channel name
            region_prefix1 = ''.join([c for c in region_code1 if not c.isdigit()])  # Remove digits
            
            region_code2 = parts[1] if len(parts) > 1 else region_code1
            region_prefix2 = ''.join([c for c in region_code2 if not c.isdigit()])
            
            # Check if either prefix is in the MNI coordinates
            if region_prefix1 in self.mni_coordinates:
                channel_mni_mapping[channel] = self.mni_coordinates[region_prefix1]
            elif region_prefix2 in self.mni_coordinates:
                channel_mni_mapping[channel] = self.mni_coordinates[region_prefix2]
        
        self.logger.info(f"Created mapping for {len(channel_mni_mapping)} channels to MNI coordinates")
        return channel_mni_mapping
    
    def _map_regions_to_standard_channels(self) -> Dict[str, List[str]]:
        """
        Map regions to standard channel names.
        
        Returns:
            Dictionary mapping region names to lists of standard channel names
        """
        # Initialize atlas to channels mapping
        region_to_channels = {region: [] for region in self.mni_coordinates.keys()}
        
        # Map each standard channel to its region
        for channel in self.seeg_channels:
            parts = channel.split('-')
            
            # Try to match based on the first part
            prefix1 = ''.join([c for c in parts[0] if not c.isdigit()])
            if prefix1 in region_to_channels:
                region_to_channels[prefix1].append(channel)
                continue
                
            # If that fails, try the second part
            if len(parts) > 1:
                prefix2 = ''.join([c for c in parts[1] if not c.isdigit()])
                if prefix2 in region_to_channels:
                    region_to_channels[prefix2].append(channel)
        
        # Log the mapping results
        for region, channels in region_to_channels.items():
            if channels:
                self.logger.info(f"Region {region} mapped to {len(channels)} standard channels: {channels[:3] if len(channels) > 3 else channels}")
        
        return region_to_channels

    def _create_channel_mask(self, data, affine, mni_coord, radius=5):
        """
        Create a spherical mask centered at the given MNI coordinates.
        
        Args:
            data: MRI data array
            affine: MRI affine transformation matrix
            mni_coord: MNI coordinates (x, y, z)
            radius: Radius of the sphere in voxels
            
        Returns:
            Boolean mask of the same shape as data
        """
        # Convert MNI coordinate to voxel coordinate
        voxel_coord = np.linalg.inv(affine).dot(np.append(mni_coord, 1))[:3]
        voxel_coord = np.round(voxel_coord).astype(int)
        
        # Create a mask centered around this coordinate
        mask = np.zeros_like(data, dtype=bool)
        
        # Ensure the coordinate is within bounds
        x, y, z = voxel_coord
        shape = data.shape
        if (0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]):
            # Create a sphere with specified radius
            for i in range(max(0, x-radius), min(shape[0], x+radius)):
                for j in range(max(0, y-radius), min(shape[1], y+radius)):
                    for k in range(max(0, z-radius), min(shape[2], z+radius)):
                        if np.sqrt((i-x)**2 + (j-y)**2 + (k-z)**2) <= radius:
                            mask[i, j, k] = True
        
        return mask

    def preprocess_mri(self, mri_img, return_path=False):
        """
        Process already MNI-registered MRI data to create region masks.
        
        Args:
            mri_img: Either a path to MRI file or a nibabel image object
                that is already registered to MNI152 space
            return_path: If True, return the path along with data
                
        Returns:
            Tuple of (registered MRI data, region masks dictionary)
        """
        try:
            # Handle both filepath and nibabel image object inputs
            if isinstance(mri_img, str):
                self.logger.info(f"Loading MRI from {mri_img}")
                img = nib.load(mri_img)
                img_path = mri_img
            else:
                self.logger.info("Using provided nibabel image object")
                img = mri_img
                img_path = None
            
            # Get MRI data
            mri_data = img.get_fdata()
            mri_affine = img.affine
            
            self.logger.info(f"MRI data shape: {mri_data.shape}")
            self.logger.info(f"MRI is in MNI152 space")
            
            # Create region masks based on MNI coordinates
            region_masks = {}
            for region_name, mni_coord in self.mni_coordinates.items():
                mask = self._create_channel_mask(mri_data, mri_affine, mni_coord)
                region_masks[region_name] = mask
                self.logger.info(f"Created coordinate-based mask for region {region_name} with {np.sum(mask)} voxels")
            
            if return_path:
                return mri_data, region_masks, img_path
            return mri_data, region_masks
            
        except Exception as e:
            self.logger.error(f"Error processing MRI: {str(e)}")
            raise

    def match_seeg_to_regions(self, channel_names: List[str]) -> Dict[str, List[str]]:
        """
        Match sEEG channels to regions based on name prefixes and MNI coordinates.
        
        Args:
            channel_names: List of sEEG channel names
                
        Returns:
            Dictionary mapping regions to lists of channel names
        """
        try:
            # Dictionary to map standardized channel names (without EEG prefix) to original names
            standardized_to_original = {}
            
            # First, standardize channel names
            standardized_channels = []
            for channel in channel_names:
                # Remove "EEG " prefix and spaces
                standardized = channel.replace('EEG ', '').replace(' ', '')
                standardized_channels.append(standardized)
                standardized_to_original[standardized] = channel
            
            self.logger.info(f"Standardized channel examples: {standardized_channels[:5]}")

            # Create a mapping from region prefixes to full region names
            region_prefixes = {}
            for region in self.mni_coordinates.keys():
                region_prefixes[region] = region
            
            self.logger.info(f"Region prefixes: {region_prefixes}")

            channel_coords = self.channel_mni_mapping[std_channel]
            matched_region = self.assign_channel_to_atlas_region(channel_coords)
            self.logger.info(f"Matched {orig_channel} to region {matched_region} via MNI coordinates")
    
            # Initialize region to channels mapping
            region_to_channels = {region: [] for region in self.mni_coordinates.keys()}
            
            # Process each standardized channel
            for std_channel, orig_channel in zip(standardized_channels, channel_names):
                # Extract parts from the channel name
                parts = std_channel.split('-')
                
                # Create a list of potential prefixes from the channel name parts
                channel_prefixes = []
                for part in parts:
                    prefix = ''.join([c for c in part if not c.isdigit()])
                    if prefix:  # Only add non-empty prefixes
                        channel_prefixes.append(prefix)
                
                # Find matching regions based on prefix
                matched_region = None
                for prefix in channel_prefixes:
                    if prefix in self.mni_coordinates:
                        matched_region = prefix
                        break
                
                # If no direct match, look for the best partial match
                if matched_region is None:
                    best_match = None
                    best_score = 0
                    
                    for prefix in channel_prefixes:
                        for region in self.mni_coordinates.keys():
                            # Calculate longest common prefix
                            common_len = 0
                            for i in range(min(len(prefix), len(region))):
                                if prefix[i].upper() == region[i].upper():
                                    common_len += 1
                                else:
                                    break
                            
                            # Calculate match score
                            if common_len > 0:
                                score = common_len / max(len(prefix), len(region))
                                if score > best_score and score > 0.5:  # At least 50% match
                                    best_score = score
                                    best_match = region
                    
                    matched_region = best_match
                    if matched_region:
                        self.logger.info(f"Fuzzy matched {std_channel} to region {matched_region} with score {best_score:.2f}")
                
                # Add the original channel to the matched region
                if matched_region:
                    region_to_channels[matched_region].append(orig_channel)
                    self.logger.info(f"Mapped channel {orig_channel} to region {matched_region}")
            
            # Log mapping summary
            for region, channels in region_to_channels.items():
                self.logger.info(f"Region {region} has {len(channels)} channels")
                if channels:
                    self.logger.info(f"Sample channels: {channels[:min(3, len(channels))]}")
            
            return region_to_channels
            
        except Exception as e:
            self.logger.error(f"Error matching sEEG to regions: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def extract_region_features(self, 
                              mri_data: np.ndarray,
                              region_masks: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract MRI features for each region.
        
        Args:
            mri_data: MRI data array
            region_masks: Dictionary of region masks
                
        Returns:
            Tuple of (feature array, region names)
        """
        try:
            features = []
            region_names = []
            
            self.logger.info(f"MRI data shape: {mri_data.shape}")
            self.logger.info("Starting feature extraction for each region...")
            
            for region, mask in region_masks.items():
                region_data = mri_data[mask]
                
                if region_data.size == 0:
                    self.logger.warning(f"Empty region data for {region}")
                    # Use default values instead of skipping
                    region_features = [0, 0, 0, 0]
                else:
                    # Extract features with safety checks
                    mean_val = float(np.mean(region_data))
                    std_val = float(np.std(region_data)) if region_data.size > 1 else 0
                    percentile_val = float(np.percentile(region_data, 90))
                    above_mean = float(np.sum(region_data > mean_val))
                    
                    region_features = [mean_val, std_val, percentile_val, above_mean]
                
                features.append(region_features)
                region_names.append(region)
            
            if not features:
                raise ValueError("No features were extracted from any region")
            
            feature_array = np.array(features)
            self.logger.info(f"Final feature array shape: {feature_array.shape}")
            return feature_array, region_names
            
        except Exception as e:
            self.logger.error(f"Error extracting region features: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}: {str(e)}")
            raise

    def preprocess_seeg(self, 
                   file_path: str,
                   epsilon: float = 1e-10) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess sEEG data.
        
        Args:
            file_path: Path to sEEG file
            epsilon: Small value to avoid division by zero
            
        Returns:
            Tuple of (sEEG features array, list of region names)
        """
        try:
            # Load sEEG data
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            raw.resample(128, npad="auto")
            data, _ = raw[:, :]
            file_channel_names = raw.ch_names
            
            self.logger.info(f"Loaded sEEG data with shape: {data.shape}")
            self.logger.info(f"File has {len(file_channel_names)} channels, we have {len(self.seeg_channels)} standard channels")

            n_channels = len(self.seeg_channels)
            index_to_standard = {i: self.seeg_channels[i] for i in range(n_channels)}
            
            # Normalize the data
            std_dev = np.std(data, axis=1, keepdims=True)
            data = (data - np.mean(data, axis=1, keepdims=True)) / (std_dev + epsilon)

            # Handle the case where the data length is less than target_length 
            orig_length = data.shape[1]
            if orig_length < self.target_length:
                self.logger.info(f"Data length {orig_length} is less than target length {self.target_length}, padding with zeros")
                # Create padded data array
                padded_data = np.zeros((data.shape[0], self.target_length))
                padded_data[:, :orig_length] = data
                data = padded_data
                self.logger.info(f"Padded data shape: {data.shape}")
            # Downsample when data is longer than target length
            elif orig_length > self.target_length:
                window_size = orig_length // self.target_length
                if window_size > 1:
                    data_reshaped = data[:, :(window_size * self.target_length)]
                    data_reshaped = data_reshaped.reshape(data.shape[0], self.target_length, window_size)
                    data = np.mean(data_reshaped, axis=2)
                    self.logger.info(f"Downsampled from {orig_length} to {self.target_length} timepoints")
            
            # Map channels to regions
            #region_to_channels = self.match_seeg_to_regions(file_channel_names)
            channel_to_region = {}
            for std_channel in self.seeg_channels:
                # Extract prefix (first part without numbers)
                prefix = std_channel.split('-')[0]
                prefix = ''.join([c for c in prefix if not c.isdigit()])
                channel_to_region[std_channel] = prefix
            
            # Aggregate by region
            region_features = []
            region_names = []
            
            for region in self.mni_coordinates.keys():
                # Find all channels for this region using our standard channel list
                channel_indices = []
                for idx, std_channel in index_to_standard.items():
                    std_prefix = channel_to_region.get(std_channel, '')
                    if std_prefix == region:
                        channel_indices.append(idx)
                
                if channel_indices:
                    region_data = np.mean(data[channel_indices], axis=0)
                    region_features.append(region_data)
                    region_names.append(region)
                    self.logger.info(f"Aggregated {len(channel_indices)} channels for region {region}")
                else:
                    self.logger.warning(f"No channels found for region {region}")
            
            if not region_features:
                raise ValueError("No valid region features extracted. Check channel mappings.")

            region_features_array = np.array(region_features)
            self.logger.info(f"Final region features shape: {region_features_array.shape}")
            self.logger.info(f"Regions with data: {region_names}")
            
            return region_features_array, region_names
        
        except Exception as e:
            self.logger.error(f"Error processing sEEG file {file_path}: {str(e)}")
            raise

    def identify_important_channels(self, seeg_features: np.ndarray, region_names: List[str], top_n: int = 5) -> List[Dict]:
        """
        Identify the most important channels based on their connectivity.
        
        Args:
            seeg_features: Array of sEEG features per region
            region_names: List of region names corresponding to features
            top_n: Number of top channels to return
            
        Returns:
            List of dictionaries with region name and importance score
        """
        try:
            self.logger.info(f"Identifying top {top_n} important channels...")
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(seeg_features)
            np.fill_diagonal(correlation_matrix, 0)  # Remove self-correlations
            
            # Calculate connectivity score (sum of absolute correlations)
            connectivity_scores = np.sum(np.abs(correlation_matrix), axis=1)
            
            # Create list of (region, score) tuples
            region_scores = [(region, score) for region, score in zip(region_names, connectivity_scores)]
            
            # Sort by score in descending order
            region_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top N
            top_regions = region_scores[:top_n]
            
            # Convert to list of dictionaries
            result = [{'region': region, 'score': float(score)} for region, score in top_regions]
            
            self.logger.info(f"Top regions: {[r['region'] for r in result]}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error identifying important channels: {str(e)}")
            return []  # Return empty list on error

    def create_temporal_edges(self, 
                         seeg_features: np.ndarray,
                         region_names: List[str],
                         correlation_threshold: float = 0.2,
                         min_connections: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edges between regions based on temporal correlations in sEEG data.
        
        Args:
            seeg_features: Array of shape (n_regions, n_timepoints)
            region_names: List of region names
            correlation_threshold: Minimum correlation to create an edge
            min_connections: Minimum connections per node
        
        Returns:
            edge_index: Tensor of shape (2, n_edges)
            edge_weights: Tensor of shape (n_edges,)
        """
        try:
            self.logger.info("Creating edges based on temporal correlations...")
            self.logger.info(f"sEEG features shape: {seeg_features.shape}")
            
            # Calculate correlation matrix with safety check for all-zero padded regions
            correlation_matrix = np.zeros((len(region_names), len(region_names)))
            valid_regions = np.std(seeg_features, axis=1) > 1e-10
            
            if np.any(valid_regions):
                valid_features = seeg_features[valid_regions]
                valid_corr = np.corrcoef(valid_features)
                
                # Fill correlation matrix with valid correlations
                valid_indices = np.where(valid_regions)[0]
                for i, vi in enumerate(valid_indices):
                    for j, vj in enumerate(valid_indices):
                        correlation_matrix[vi, vj] = valid_corr[i, j]
            else:
                self.logger.warning("No valid regions with non-zero variance found")
            
            # Replace any remaining NaN values with zeros
            correlation_matrix = np.nan_to_num(correlation_matrix)
            
            self.logger.info(f"Correlation matrix shape: {correlation_matrix.shape}")
            
            edges = []
            weights = []
            
            # Create edges for correlations above threshold
            n_regions = len(region_names)
            for i in range(n_regions):
                for j in range(i + 1, n_regions):  # Upper triangle only
                    correlation = correlation_matrix[i, j]
                    if abs(correlation) >= correlation_threshold:
                        # Add bidirectional edges
                        edges.extend([[i, j], [j, i]])
                        # Use absolute correlation as weight
                        weights.extend([abs(correlation), abs(correlation)])
                        self.logger.info(f"Added temporal edge between {region_names[i]} and {region_names[j]} "
                                    f"with correlation {correlation:.3f}")
            
            # Track connections for each node
            node_connections = {i: set() for i in range(n_regions)}
            for edge in edges:
                i, j = edge
                node_connections[i].add(j)
            
            # Ensure each node has at least min_connections
            for i in range(n_regions):
                if len(node_connections[i]) < min_connections:
                    # Sort other nodes by correlation strength (descending)
                    correlations = [(j, abs(correlation_matrix[i, j])) for j in range(n_regions) if j != i and j not in node_connections[i]]
                    correlations.sort(key=lambda x: x[1], reverse=True)
                    
                    # Add edges to the strongest correlated nodes
                    needed = min_connections - len(node_connections[i])
                    for j, corr in correlations[:needed]:
                        edges.extend([[i, j], [j, i]])
                        weights.extend([corr, corr])
                        node_connections[i].add(j)
                        node_connections[j].add(i)
                        self.logger.info(f"Added minimum connection edge between {region_names[i]} and {region_names[j]} "
                                    f"with correlation {corr:.3f}")
            
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_weights = torch.tensor(weights, dtype=torch.float)
            
            self.logger.info(f"Created {len(edges) // 2} unique temporal edges ({len(edges)} directed edges)")
            self.logger.info(f"Edge index shape: {edge_index.shape}")
            self.logger.info(f"Edge weights shape: {edge_weights.shape}")
            
            return edge_index, edge_weights
            
        except Exception as e:
            self.logger.error(f"Error in create_temporal_edges: {str(e)}")
            raise

    def create_spatial_edges(self,
                           region_names: List[str],
                           distance_threshold: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edges between regions based on spatial proximity in MNI space.
        
        Args:
            region_names: List of region names
            distance_threshold: Maximum distance (in mm) to create an edge
            
        Returns:
            edge_index: Tensor of shape (2, n_edges)
            edge_weights: Tensor of shape (n_edges,)
        """
        try:
            self.logger.info("Creating edges based on spatial proximity...")
            
            # Get MNI coordinates for each region
            coords = []
            valid_regions = []
            
            for region in region_names:
                if region in self.mni_coordinates:
                    coords.append(self.mni_coordinates[region])
                    valid_regions.append(region)
            
            if not coords:
                raise ValueError("No valid MNI coordinates found for regions")
            
            # Convert to numpy array
            coords = np.array(coords)
            
            # Calculate distance matrix
            dist_matrix = cdist(coords, coords, metric='euclidean')
            
            edges = []
            weights = []
            
            # Create edges for points within distance threshold
            n_regions = len(valid_regions)
            for i in range(n_regions):
                for j in range(i + 1, n_regions):  # Upper triangle only
                    distance = dist_matrix[i, j]
                    if distance <= distance_threshold:
                        # Add bidirectional edges
                        edges.extend([[i, j], [j, i]])
                        # Use inverse distance as weight (closer = stronger connection)
                        weight = 1.0 / (distance + 1.0)  # Add 1 to avoid division by zero
                        weights.extend([weight, weight])
                        self.logger.info(f"Added spatial edge between {valid_regions[i]} and {valid_regions[j]} "
                                       f"with distance {distance:.3f} mm")
            
            # If no edges were found, create edges with nearest neighbors
            if not edges:
                self.logger.warning("No regions within distance threshold, creating nearest neighbor edges...")
                k = min(3, n_regions - 1)  # Connect to 3 nearest neighbors or all if less
                for i in range(n_regions):
                    # Get k closest regions
                    distances = dist_matrix[i]
                    # Exclude self and get k closest
                    neighbor_indices = np.argsort(distances)[1:k+1]
                    for j in neighbor_indices:
                        edges.extend([[i, j], [j, i]])
                        weight = 1.0 / (dist_matrix[i, j] + 1.0)
                        weights.extend([weight, weight])
                        self.logger.info(f"Added nearest neighbor edge between {valid_regions[i]} and "
                                       f"{valid_regions[j]} with distance {dist_matrix[i, j]:.3f} mm")
            
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_weights = torch.tensor(weights, dtype=torch.float)
            
            self.logger.info(f"Created {len(edges) // 2} unique spatial edges ({len(edges)} directed edges)")
            self.logger.info(f"Edge index shape: {edge_index.shape}")
            self.logger.info(f"Edge weights shape: {edge_weights.shape}")
            self.logger.info(f"Edge weight range: [{edge_weights.min():.3f}, {edge_weights.max():.3f}]")
            
            return edge_index, edge_weights
            
        except Exception as e:
            self.logger.error(f"Error in create_spatial_edges: {str(e)}")
            raise

    def _extract_temporal_features(self, seeg_features: np.ndarray) -> np.ndarray:
        """
        Extract temporal features from sEEG data.
        
        Args:
            seeg_features: Array of shape (n_regions, n_timepoints)
            
        Returns:
            Array of shape (n_regions, n_features)
        """
        try:
            # Get basic time-domain features
            time_features = np.concatenate([
                np.mean(seeg_features, axis=1, keepdims=True),
                np.std(seeg_features, axis=1, keepdims=True),
                np.maximum.reduce(seeg_features, axis=1, keepdims=True, initial=-np.inf),
                np.minimum.reduce(seeg_features, axis=1, keepdims=True, initial=np.inf)
            ], axis=1)
            
            # Calculate frequency domain features
            freq_features = []
            from scipy import signal
            
            for region_data in seeg_features:
                # Calculate power spectral density
                f, psd = signal.welch(region_data, fs=128, nperseg=1024)
                
                # Calculate band powers (assuming 128 Hz sampling rate)
                delta_idx = np.logical_and(f >= 0.5, f <= 4)
                theta_idx = np.logical_and(f > 4, f <= 8)
                alpha_idx = np.logical_and(f > 8, f <= 13)
                beta_idx = np.logical_and(f > 13, f <= 30)
                gamma_idx = np.logical_and(f > 30, f <= 80)
                
                # Extract band power features
                delta_power = np.mean(psd[delta_idx]) if np.any(delta_idx) else 0
                theta_power = np.mean(psd[theta_idx]) if np.any(theta_idx) else 0
                alpha_power = np.mean(psd[alpha_idx]) if np.any(alpha_idx) else 0
                beta_power = np.mean(psd[beta_idx]) if np.any(beta_idx) else 0
                gamma_power = np.mean(psd[gamma_idx]) if np.any(gamma_idx) else 0
                
                # Add band power ratio features
                theta_alpha_ratio = theta_power / alpha_power if alpha_power > 0 else 0
                delta_theta_ratio = delta_power / theta_power if theta_power > 0 else 0
                
                # Combine all frequency features
                freq_features.append([
                    delta_power, theta_power, alpha_power, beta_power, gamma_power,
                    theta_alpha_ratio, delta_theta_ratio
                ])
            
            # Combine time and frequency domain features
            return np.concatenate([time_features, np.array(freq_features)], axis=1)
            
        except Exception as e:
            self.logger.error(f"Error extracting temporal features: {str(e)}")
            # Fallback to basic features if something goes wrong
            return time_features

    def create_graph(self,
               seeg_features: np.ndarray,
               region_names: List[str],
               mri_features: Optional[np.ndarray] = None,
               label: int = 0) -> Data:
        """
        Create a graph from sEEG and optional MRI features.
        
        Args:
            seeg_features: Array of sEEG features
            region_names: List of region names
            mri_features: Optional array of MRI features
            label: Graph label
            
        Returns:
            PyTorch Geometric Data object
        """
        try:
            self.logger.info("Creating graph...")
            self.logger.info(f"sEEG features shape: {seeg_features.shape}")
            
            # Create temporal edges using correlations
            temporal_edge_index, temporal_edge_weights = self.create_temporal_edges(
                seeg_features=seeg_features,
                region_names=region_names,
                correlation_threshold=0.1,
                min_connections=2
            )
            
            # Create spatial edges using MNI coordinates
            spatial_edge_index, spatial_edge_weights = self.create_spatial_edges(
                region_names=region_names,
                distance_threshold=25.0
            )
            
            # Combine edges
            edge_index = torch.cat([temporal_edge_index, spatial_edge_index], dim=1)
            edge_weights = torch.cat([temporal_edge_weights, spatial_edge_weights])
            
            # Extract temporal features
            temporal_features = self._extract_temporal_features(seeg_features)
            self.logger.info(f"Temporal features shape: {temporal_features.shape}")
            
            # Combine features if MRI features are available
            if mri_features is not None:
                self.logger.info(f"MRI features shape: {mri_features.shape}")
                
                # Make sure feature count matches
                if mri_features.shape[0] != temporal_features.shape[0]:
                    self.logger.warning(f"MRI features count {mri_features.shape[0]} doesn't match temporal features count {temporal_features.shape[0]}")
                    min_regions = min(mri_features.shape[0], temporal_features.shape[0])
                    mri_features = mri_features[:min_regions]
                    temporal_features = temporal_features[:min_regions]
                
                # Combine features
                node_features = np.concatenate([temporal_features, mri_features], axis=1)
            else:
                self.logger.info("No MRI features provided, using only temporal features")
                node_features = temporal_features
            
            self.logger.info(f"Final node features shape: {node_features.shape}")
            
            # Create graph
            graph_data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_weights,
                y=torch.tensor(label, dtype=torch.long)
            )
            
            # Store metadata
            graph_data.region_names = region_names
            
            # Add edge type (0 for temporal, 1 for spatial)
            n_temporal = temporal_edge_index.shape[1]
            n_spatial = spatial_edge_index.shape[1]
            edge_type = torch.cat([
                torch.zeros(n_temporal, dtype=torch.long),
                torch.ones(n_spatial, dtype=torch.long)
            ])
            graph_data.edge_type = edge_type
            
            return graph_data
            
        except Exception as e:
            self.logger.error(f"Error creating graph: {str(e)}")
            raise

    
    def process_dataset(self, dataset: List[Dict], use_mri: bool = True) -> List[Data]:
        """
        Process entire dataset by combining all processing steps into a single pipeline.
        
        Args:
            dataset: List of dictionaries where each dictionary has keys:
                - 'reg_img': nibabel image object or path to MRI file (if use_mri=True)
                - 'seeg_path': path to sEEG file
                - 'label': class label (optional, defaults to 0)
            use_mri: Whether to use MRI data or not
                
        Returns:
            List of PyTorch Geometric Data objects
        """
        graph_data_list = []
        
        for i, item in enumerate(dataset):
            try:
                self.logger.info(f"Processing item {i+1}/{len(dataset)}")
                
                # Extract common values
                label = item.get('label', 0)
                seeg_path = item.get('seeg_path')
                
                if not seeg_path:
                    self.logger.error(f"Missing sEEG path for item {i}")
                    continue
                
                # Step 1: Process sEEG data
                self.logger.info(f"Step 1: Processing sEEG data from {seeg_path}")
                seeg_features, region_names = self.preprocess_seeg(seeg_path)
                
                # Step 2: Identify important channels based on connectivity
                self.logger.info("Step 2: Identifying important channels")
                important_channels = self.identify_important_channels(seeg_features, region_names, top_n=5)
                self.logger.info(f"Important channels: {[ch['region'] for ch in important_channels]}")
                
                mri_features = None
                
                # Step 3: Process MRI data if requested
                if use_mri:
                    reg_img = item.get('reg_img')
                    
                    if reg_img is None:
                        self.logger.warning(f"No MRI data found for item {i}, proceeding with sEEG data only")
                    else:
                        try:
                            self.logger.info("Step 3: Processing MRI data")
                            
                            # Step 3.1: Process MRI and create region masks
                            mri_data, region_masks = self.preprocess_mri(reg_img)
                            
                            # Step 3.2: Extract MRI features for each region
                            mri_features, mri_region_names = self.extract_region_features(
                                mri_data,
                                region_masks
                            )
                            
                            # Step 3.3: Align sEEG and MRI region names
                            if set(region_names) != set(mri_region_names):
                                self.logger.warning(f"Region names mismatch between sEEG and MRI")
                                
                                # Use only common regions
                                common_regions = list(set(region_names).intersection(set(mri_region_names)))
                                self.logger.info(f"Using {len(common_regions)} common regions: {common_regions}")
                                
                                if common_regions:
                                    # Filter features to common regions
                                    seeg_indices = [region_names.index(r) for r in common_regions]
                                    mri_indices = [mri_region_names.index(r) for r in common_regions]
                                    
                                    seeg_features = seeg_features[seeg_indices]
                                    mri_features = mri_features[mri_indices]
                                    region_names = common_regions
                                else:
                                    self.logger.warning(f"No common regions found, using sEEG data only")
                                    mri_features = None
                        except Exception as mri_error:
                            self.logger.error(f"Error processing MRI data: {str(mri_error)}")
                            self.logger.info("Proceeding with sEEG data only")
                            mri_features = None
                
                # Step 4: Create temporal edges based on sEEG correlations
                self.logger.info("Step 4: Creating temporal edges")
                temporal_edge_index, temporal_edge_weights = self.create_temporal_edges(
                    seeg_features=seeg_features,
                    region_names=region_names,
                    correlation_threshold=0.1,
                    min_connections=2
                )
                
                # Step 5: Create spatial edges based on MNI coordinates
                self.logger.info("Step 5: Creating spatial edges")
                spatial_edge_index, spatial_edge_weights = self.create_spatial_edges(
                    region_names=region_names,
                    distance_threshold=25.0
                )
                
                # Step 6: Extract enhanced temporal features
                self.logger.info("Step 6: Extracting temporal features")
                temporal_features = self._extract_temporal_features(seeg_features)
                
                # Step 7: Create the graph
                self.logger.info("Step 7: Creating graph")
                graph_data = self.create_graph(
                    seeg_features=seeg_features,
                    region_names=region_names,
                    mri_features=mri_features,
                    label=label
                )
                
                # Store the important channels in the graph data
                graph_data.important_channels = important_channels
                
                # Add original data sources for reference
                graph_data.seeg_path = seeg_path
                graph_data.reg_img_path = item.get('reg_img') if isinstance(item.get('reg_img'), str) else None
                
                self.logger.info(f"Completed processing item {i+1}")
                graph_data_list.append(graph_data)
                
            except Exception as e:
                self.logger.error(f"Error processing item {i}: {str(e)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                continue
                
        self.logger.info(f"Dataset processing complete. Created {len(graph_data_list)} graphs.")
        return graph_data_list
        

#sEEG Graph Processor
class SEEGGraphProcessor:
    def __init__(self,
                 mni_coordinates: dict,
                 seeg_channels: list,
                 region_code_map: dict,
                 target_length: int = 50000,
                 temporal_window: int = 100):
        """
        Initialize SEEG-only graph processor.
        
        Args:
            mni_coordinates: Dictionary mapping regions to MNI coordinates
            seeg_channels: List of standard sEEG channel names
            region_code_map: Dictionary mapping region codes to region names
            target_length: Fixed length for sEEG time series
            temporal_window: Window size for temporal features
        """
        self.target_length = target_length
        self.mni_coordinates = mni_coordinates
        self.seeg_channels = seeg_channels
        self.temporal_window = temporal_window
        self.region_code_map = region_code_map
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create the channel to MNI mapping
        self.channel_mni_mapping = self._create_channel_mni_mapping()
        
    def _create_channel_mni_mapping(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Create mapping from sEEG channel names to MNI coordinates.
        Uses the standard channel list from self.seeg_channels.
        
        Returns:
            Dictionary mapping channel names to MNI coordinates
        """
        channel_mni_mapping = {}

        for channel in self.seeg_channels:
            parts = channel.split('-')
            region_code1 = parts[0]  # Extract first part of the channel name
            region_prefix1 = ''.join([c for c in region_code1 if not c.isdigit()])  # Remove digits
            
            region_code2 = parts[1] if len(parts) > 1 else region_code1
            region_prefix2 = ''.join([c for c in region_code2 if not c.isdigit()])
            
            # Check if either prefix is in the MNI coordinates
            if region_prefix1 in self.mni_coordinates:
                channel_mni_mapping[channel] = self.mni_coordinates[region_prefix1]
            elif region_prefix2 in self.mni_coordinates:
                channel_mni_mapping[channel] = self.mni_coordinates[region_prefix2]
        
        self.logger.info(f"Created mapping for {len(channel_mni_mapping)} channels to MNI coordinates")
        return channel_mni_mapping
    
    def _map_regions_to_standard_channels(self) -> Dict[str, List[str]]:
        """
        Map regions to standard channel names.
        
        Returns:
            Dictionary mapping region names to lists of standard channel names
        """
        # Initialize region to channels mapping
        region_to_channels = {region: [] for region in self.mni_coordinates.keys()}
        
        # Map each standard channel to its region
        for channel in self.seeg_channels:
            parts = channel.split('-')
            
            # Try to match based on the first part
            prefix1 = ''.join([c for c in parts[0] if not c.isdigit()])
            if prefix1 in region_to_channels:
                region_to_channels[prefix1].append(channel)
                continue
                
            # If that fails, try the second part
            if len(parts) > 1:
                prefix2 = ''.join([c for c in parts[1] if not c.isdigit()])
                if prefix2 in region_to_channels:
                    region_to_channels[prefix2].append(channel)
        
        # Log the mapping results
        for region, channels in region_to_channels.items():
            if channels:
                self.logger.info(f"Region {region} mapped to {len(channels)} standard channels: {channels[:3]}...")
        
        return region_to_channels
    
    def preprocess_seeg(self, 
                        file_path: str,
                        epsilon: float = 1e-10) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess sEEG data and aggregate by regions.
        
        Args:
            file_path: Path to sEEG file
            epsilon: Small value to avoid division by zero
            
        Returns:
            Tuple of (array of shape [n_regions, n_timepoints], list of region names)
        """
        try:
            # Load sEEG data
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            raw.resample(128, npad="auto")
            data, _ = raw[:, :]
            file_channel_names = raw.ch_names
            
            self.logger.info(f"Loaded sEEG data with shape: {data.shape}")
            self.logger.info(f"File has {len(file_channel_names)} channels, we have {len(self.seeg_channels)} standard channels")
            
            # Check if the number of channels matches our standard list
            if len(file_channel_names) != len(self.seeg_channels):
                self.logger.warning(f"Channel count mismatch: file has {len(file_channel_names)}, expected {len(self.seeg_channels)}")
                # If lengths don't match, we'll use the minimum length to avoid index errors
                n_channels = min(len(file_channel_names), len(self.seeg_channels))
                self.logger.info(f"Using only the first {n_channels} channels")
                
                # Truncate the data if necessary
                if data.shape[0] > n_channels:
                    data = data[:n_channels]
            else:
                n_channels = len(self.seeg_channels)
            
            # Create a channel mapping from indices to standard names
            # We're assuming the channels in the file correspond 1:1 with our standard list
            index_to_standard = {i: self.seeg_channels[i] for i in range(n_channels)}
            
            self.logger.info(f"Created 1:1 index mapping between file channels and standard channels")
            
            # Normalize the data
            std_dev = np.std(data, axis=1, keepdims=True)
            data = (data - np.mean(data, axis=1, keepdims=True)) / (std_dev + epsilon)

            # Handle the case where the data length is less than target_length
            orig_length = data.shape[1]
            if orig_length < self.target_length:
                self.logger.info(f"Data length {orig_length} is less than target length {self.target_length}, padding with zeros")
                # Create padded data array
                padded_data = np.zeros((data.shape[0], self.target_length))
                padded_data[:, :orig_length] = data
                data = padded_data
                self.logger.info(f"Padded data shape: {data.shape}")
            # Downsample when data is longer than target length
            elif orig_length > self.target_length:
                window_size = orig_length // self.target_length
                if window_size > 1:
                    data_reshaped = data[:, :(window_size * self.target_length)]
                    data_reshaped = data_reshaped.reshape(data.shape[0], self.target_length, window_size)
                    data = np.mean(data_reshaped, axis=2)
                    self.logger.info(f"Downsampled from {orig_length} to {self.target_length} timepoints")
            
            # Create a mapping from standard channel names to region prefixes
            channel_to_region = {}
            for std_channel in self.seeg_channels:
                # Extract prefix (first part without numbers)
                prefix = std_channel.split('-')[0]
                prefix = ''.join([c for c in prefix if not c.isdigit()])
                channel_to_region[std_channel] = prefix
            
            # Aggregate by region
            region_features = []
            region_names = []
            
            for region in self.mni_coordinates.keys():
                # Find all channels for this region using our standard channel list
                channel_indices = []
                for idx, std_channel in index_to_standard.items():
                    std_prefix = channel_to_region.get(std_channel, '')
                    if std_prefix == region:
                        channel_indices.append(idx)
                
                if channel_indices:
                    region_data = np.mean(data[channel_indices], axis=0)
                    region_features.append(region_data)
                    region_names.append(region)
                    self.logger.info(f"Aggregated {len(channel_indices)} channels for region {region}")
                else:
                    self.logger.warning(f"No channels found for region {region}")
            
            if not region_features:
                raise ValueError("No valid region features extracted. Check channel mappings.")

            region_features_array = np.array(region_features)
            self.logger.info(f"Final region features shape: {region_features_array.shape}")
            self.logger.info(f"Regions with data: {region_names}")
            
            return region_features_array, region_names
        
        except Exception as e:
            self.logger.error(f"Error processing sEEG file {file_path}: {str(e)}")
            raise

    def create_temporal_edges(self, 
                             region_names: List[str], 
                             seeg_features: np.ndarray,
                             correlation_threshold: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edges between regions based on temporal correlations in sEEG data.
        
        Args:
            region_names: List of region names
            seeg_features: Array of shape (n_regions, n_timepoints)
            correlation_threshold: Minimum correlation to create an edge
        
        Returns:
            edge_index: Tensor of shape (2, n_edges)
            edge_weights: Tensor of shape (n_edges,)
        """
        try:
            self.logger.info("Creating edges based on temporal correlations...")
            self.logger.info(f"sEEG features shape: {seeg_features.shape}")
            
            # Calculate correlation matrix with safety check for all-zero padded regions
            correlation_matrix = np.zeros((len(region_names), len(region_names)))
            valid_regions = np.std(seeg_features, axis=1) > 1e-10
            
            if np.any(valid_regions):
                valid_features = seeg_features[valid_regions]
                valid_corr = np.corrcoef(valid_features)
                
                # Fill correlation matrix with valid correlations
                valid_indices = np.where(valid_regions)[0]
                for i, vi in enumerate(valid_indices):
                    for j, vj in enumerate(valid_indices):
                        correlation_matrix[vi, vj] = valid_corr[i, j]
            else:
                self.logger.warning("No valid regions with non-zero variance found")
            
            # Replace any remaining NaN values with zeros
            correlation_matrix = np.nan_to_num(correlation_matrix)
            
            self.logger.info(f"Correlation matrix shape: {correlation_matrix.shape}")
            
            edges = []
            weights = []
            
            # Create edges for correlations above threshold
            n_regions = len(region_names)
            for i in range(n_regions):
                for j in range(i + 1, n_regions):  # Upper triangle only
                    correlation = correlation_matrix[i, j]
                    if abs(correlation) >= correlation_threshold:
                        # Add bidirectional edges
                        edges.extend([[i, j], [j, i]])
                        # Use absolute correlation as weight
                        weights.extend([abs(correlation), abs(correlation)])
                        self.logger.info(f"Added temporal edge between {region_names[i]} and {region_names[j]} "
                                    f"with correlation {correlation:.3f}")
            
            # If no edges were found, create edges with nearest neighbors
            if not edges:
                self.logger.warning("No correlations above threshold, creating nearest neighbor edges...")
                k = min(3, n_regions - 1)  # Connect to 3 nearest neighbors or all if less
                for i in range(n_regions):
                    # Get k highest correlations for this region
                    correlations = correlation_matrix[i]
                    # Exclude self-correlation and take k highest
                    neighbor_indices = np.argsort(np.abs(correlations))[::-1][1:k+1]
                    for j in neighbor_indices:
                        edges.extend([[i, j], [j, i]])
                        weights.extend([abs(correlation_matrix[i, j]), abs(correlation_matrix[i, j])])
                        self.logger.info(f"Added nearest neighbor edge between {region_names[i]} and "
                                    f"{region_names[j]} with correlation {correlation_matrix[i, j]:.3f}")
                        
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_weights = torch.tensor(weights, dtype=torch.float)
            
            self.logger.info(f"Created {len(edges) // 2} unique temporal edges ({len(edges)} directed edges)")
            self.logger.info(f"Edge index shape: {edge_index.shape}")
            self.logger.info(f"Edge weights shape: {edge_weights.shape}")
            
            return edge_index, edge_weights
            
        except Exception as e:
            self.logger.error(f"Error in create_temporal_edges: {str(e)}")
            raise

    def _extract_temporal_features(self, seeg_features: np.ndarray) -> np.ndarray:
        """
        Extract temporal features from sEEG data.
        
        Args:
            seeg_features: Array of shape (n_regions, n_timepoints)
            
        Returns:
            Array of shape (n_regions, n_features)
        """
        try:
            # Handle case when data is shorter than temporal window
            if seeg_features.shape[1] < self.temporal_window:
                self.logger.warning(f"sEEG data length {seeg_features.shape[1]} is less than temporal window {self.temporal_window}")
                # Use simple statistics of the entire sequence instead of windows
                features = np.concatenate([
                    np.mean(seeg_features, axis=1, keepdims=True),
                    np.std(seeg_features, axis=1, keepdims=True),
                    np.maximum.reduce(seeg_features, axis=1, keepdims=True, initial=-np.inf),
                    np.minimum.reduce(seeg_features, axis=1, keepdims=True, initial=np.inf)
                ], axis=1)
                return features
            else:
                windows_features = []
                for t in range(seeg_features.shape[1] - self.temporal_window + 1):
                    window = seeg_features[:, t:t+self.temporal_window]
                    features = np.concatenate([
                        np.mean(window, axis=1, keepdims=True),
                        np.std(window, axis=1, keepdims=True),
                        np.maximum.reduce(window, axis=1, keepdims=True, initial=-np.inf),
                        np.minimum.reduce(window, axis=1, keepdims=True, initial=np.inf)
                    ], axis=1)
                    windows_features.append(features)
                
                if windows_features:
                    return np.mean(windows_features, axis=0)
                else:
                    # Fallback if no valid windows
                    self.logger.warning("No valid temporal windows, using basic statistics")
                    return np.concatenate([
                        np.mean(seeg_features, axis=1, keepdims=True),
                        np.std(seeg_features, axis=1, keepdims=True),
                        np.maximum.reduce(seeg_features, axis=1, keepdims=True, initial=-np.inf),
                        np.minimum.reduce(seeg_features, axis=1, keepdims=True, initial=np.inf)
                    ], axis=1)
        except Exception as e:
            self.logger.error(f"Error extracting temporal features: {str(e)}")
            raise

    def create_graph(self, seeg_features: np.ndarray, region_names: List[str], label: int = 0) -> Data:
        """
        Create graph with regions as nodes and temporal edges.
        
        Args:
            seeg_features: sEEG features of shape (n_regions, n_timepoints)
            region_names: List of region names
            label: Graph label
            
        Returns:
            PyTorch Geometric Data object
        """
        try:
            self.logger.info("Creating graph...")
            self.logger.info(f"sEEG features shape: {seeg_features.shape}")
            self.logger.info(f"Number of regions: {len(region_names)}")
            
            # Create temporal edges using correlations
            edge_index, edge_weights = self.create_temporal_edges(
                region_names=region_names,
                seeg_features=seeg_features
            )
            
            # Extract temporal features
            node_features = self._extract_temporal_features(seeg_features)
            self.logger.info(f"Temporal features shape: {node_features.shape}")
            
            # Create graph
            graph_data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_weights,
                y=torch.tensor(label, dtype=torch.long)
            )
            
            # Store metadata
            graph_data.region_names = region_names
            
            # Add edge type (all temporal: 0)
            graph_data.edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
            
            return graph_data
            
        except Exception as e:
            self.logger.error(f"Error creating graph: {str(e)}")
            raise
        
    def process_dataset(self, dataset: List[Dict]) -> List[Data]:
        """
        Process entire dataset using only sEEG data.
        
        Args:
            dataset: List of dictionaries where each dictionary has keys:
                - 'seeg_path': path to sEEG file
                - 'label': class label
                
        Returns:
            List of PyTorch Geometric Data objects
        """
        graph_data_list = []
        
        for i, item in enumerate(dataset):
            try:
                self.logger.info(f"Processing item {i+1}/{len(dataset)}")
                
                # Extract common values
                label = item.get('label', 0)
                seeg_path = item.get('seeg_path')
                
                if not seeg_path:
                    self.logger.error(f"Missing sEEG path for item {i}")
                    continue
                
                # Process sEEG data
                seeg_features, region_names = self.preprocess_seeg(seeg_path)
                
                # Create graph with temporal features and edges only
                graph_data = self.create_graph(
                    seeg_features=seeg_features,
                    region_names=region_names,
                    label=label
                )
                
                graph_data_list.append(graph_data)
                
            except Exception as e:
                self.logger.error(f"Error processing item {i}: {str(e)}")
                self.logger.error(f"Error details: {type(e).__name__}: {str(e)}")
                continue
                
        return graph_data_list
    

data = np.load('patient_data.npy', allow_pickle=True)
mni_coordinates = np.load('mni_coordinates.npy', allow_pickle=True)
eeg_channels = np.load('eeg_channels.npy', allow_pickle=True)
region_names = np.load('region_names.npy', allow_pickle=True)
combined_labels = np.load('combined_labels.npy', allow_pickle=True)

processor = GraphProcessor(mni_coordinates, eeg_channels, region_names)
graph_data_list = processor.process_dataset(data, use_mri = True)

# seeg_processor = SEEGGraphProcessor(mni_coordinates, eeg_channels, region_names)
# graph_data_list = seeg_processor.process_dataset(data)

class_distribution = np.unique(combined_labels, return_counts=True)
print("Class distribution (before split):", class_distribution)

train_indices, test_indices = train_test_split(range(len(combined_labels)), test_size=0.4, stratify=combined_labels)
train_data_list = [graph_data_list[i] for i in train_indices]
test_data_list = [graph_data_list[i] for i in test_indices]

train_loader = DataLoader(train_data_list, batch_size=1, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data_list, batch_size=1, shuffle=False, pin_memory=True)

torch.save(train_loader, 'train_loader.pt')
torch.save(test_loader, 'test_loader.pt')
