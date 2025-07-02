import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, datasets
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch_geometric.data import Data
import logging
from typing import List, Dict, Optional, Tuple
import mne

class BrainNetworkVisualizer:
    def __init__(self, 
                 atlas_name: str = 'harvard_oxford',
                 node_size_scale: float = 70,
                 edge_threshold: float = 0.3,
                 edge_alpha_scale: float = 0.7,
                 colormap: str = 'viridis'):
        """
        Initialize 3D glass brain network visualizer.
        
        Args:
            atlas_name: Name of the atlas to use (harvard_oxford, aal, or destrieux)
            node_size_scale: Scaling factor for node sizes
            edge_threshold: Minimum edge weight to visualize
            edge_alpha_scale: Scaling factor for edge transparency
            colormap: Colormap to use for nodes and edges
        """
        self.node_size_scale = node_size_scale
        self.edge_threshold = edge_threshold
        self.edge_alpha_scale = edge_alpha_scale
        self.colormap = colormap
        self.logger = logging.getLogger(__name__)
        
        # Load the atlas
        self.atlas = self._load_atlas(atlas_name)
        self.atlas_name = atlas_name
        
    def _load_atlas(self, atlas_name: str) -> Dict:
        """Load brain atlas and return atlas data."""
        try:
            if atlas_name == 'harvard_oxford':
                atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            elif atlas_name == 'aal':
                atlas = datasets.fetch_atlas_aal()
            elif atlas_name == 'destrieux':
                atlas = datasets.fetch_atlas_destrieux_2009()
            else:
                raise ValueError(f"Unsupported atlas: {atlas_name}")
            
            return {
                'img': atlas.maps,
                'labels': atlas.labels if isinstance(atlas.labels, list) else atlas.labels.tolist(),
                'maps': atlas.maps
            }
        except Exception as e:
            self.logger.error(f"Error loading atlas: {str(e)}")
            raise
            
    def _get_region_coordinates(self, region_names: List[str]) -> np.ndarray:
        """
        Get 3D coordinates for each brain region.
        
        Args:
            region_names: List of region names
            
        Returns:
            Array of shape (n_regions, 3) with x, y, z coordinates
        """
        try:
            # Load atlas data
            atlas_data = self.atlas['img'].get_fdata()
            affine = self.atlas['img'].affine
            
            # Get coordinates for each region
            coords = []
            for region in region_names:
                # Find region index in atlas labels
                if region in self.atlas['labels']:
                    region_idx = self.atlas['labels'].index(region)
                    # Get voxels corresponding to this region
                    region_mask = atlas_data == region_idx
                    if np.any(region_mask):
                        # Find center of mass
                        voxel_coords = np.array(np.where(region_mask)).mean(axis=1)
                        # Convert to MNI coordinates using affine transformation
                        mni_coords = nib.affines.apply_affine(affine, voxel_coords)
                        coords.append(mni_coords)
                    else:
                        # If region not found, use a default position with warning
                        self.logger.warning(f"Region {region} not found in atlas mask, using default position")
                        coords.append(np.zeros(3))
                else:
                    # If region name not in atlas, use a default position with warning
                    self.logger.warning(f"Region {region} not in atlas labels, using default position")
                    coords.append(np.zeros(3))
                        
            return np.array(coords)
            
        except Exception as e:
            self.logger.error(f"Error getting region coordinates: {str(e)}")
            raise
            
    def _prepare_network_data(self, graph: Data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Extract network data from PyG Data object.
        
        Args:
            graph: PyG Data object containing graph structure
            
        Returns:
            Tuple of node coordinates, node sizes, edge weights, and region names
        """
        # Get region names
        region_names = graph.region_names
        
        # Get node coordinates
        node_coords = self._get_region_coordinates(region_names)
        
        # Use node features for node sizes (use mean of temporal features)
        if hasattr(graph, 'x') and graph.x is not None:
            # Calculate node importance based on features
            node_importances = graph.x.abs().mean(dim=1).numpy()
            node_sizes = self.node_size_scale * (node_importances / node_importances.max())
        else:
            node_sizes = np.ones(len(region_names)) * self.node_size_scale
            
        # Convert edge index and attributes to numpy
        edge_index = graph.edge_index.numpy()
        edge_weights = graph.edge_attr.numpy() if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
        
        # Filter edges by threshold if weights are available
        if edge_weights is not None:
            mask = edge_weights >= self.edge_threshold
            edge_index = edge_index[:, mask]
            edge_weights = edge_weights[mask]
            
        return node_coords, node_sizes, edge_weights, region_names
        
    def visualize_3d_network(self, 
                           graph: Data, 
                           title: str = "Brain Network Visualization",
                           show_labels: bool = True,
                           save_path: Optional[str] = None,
                           azimuth: int = -100,
                           elevation: int = 10,
                           alpha: float = 0.7) -> None:
        """
        Create 3D visualization of brain network.
        
        Args:
            graph: PyG Data object
            title: Plot title
            show_labels: Whether to show region labels
            save_path: Path to save the figure (None for display only)
            azimuth: Horizontal viewing angle
            elevation: Vertical viewing angle
            alpha: Transparency of brain surface
        """
        try:
            # Prepare data
            node_coords, node_sizes, edge_weights, region_names = self._prepare_network_data(graph)
            
            # Create figure
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot nodes
            scatter = ax.scatter(
                node_coords[:, 0], 
                node_coords[:, 1], 
                node_coords[:, 2],
                s=node_sizes,
                c=np.arange(len(node_coords)),
                cmap=self.colormap,
                alpha=0.8,
                edgecolors='w'
            )
            
            # Plot edges
            if edge_weights is not None:
                edge_count = edge_weights.shape[0]
                for i in range(edge_count):
                    start_idx = graph.edge_index[0, i]
                    end_idx = graph.edge_index[1, i]
                    
                    weight = edge_weights[i]
                    width = weight * 2  # Scale width by weight
                    alpha = min(1.0, weight * self.edge_alpha_scale)
                    
                    start = node_coords[start_idx]
                    end = node_coords[end_idx]
                    
                    ax.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        alpha=alpha,
                        linewidth=width,
                        color='gray'
                    )
            
            # Add labels if requested
            if show_labels:
                for i, (coord, name) in enumerate(zip(node_coords, region_names)):
                    # Truncate long names
                    short_name = name[:15] + '...' if len(name) > 15 else name
                    ax.text(
                        coord[0], coord[1], coord[2],
                        short_name,
                        size=8,
                        zorder=100,
                        color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                    )
            
            # Set view angle
            ax.view_init(elev=elevation, azim=azimuth)
            
            # Add a glass brain surface
            self._add_glass_brain_surface(ax, alpha=alpha)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
            cbar.set_label('Node Importance')
            
            # Set title and labels
            ax.set_title(title, size=16, pad=20)
            ax.set_xlabel('X (mm)', labelpad=10)
            ax.set_ylabel('Y (mm)', labelpad=10)
            ax.set_zlabel('Z (mm)', labelpad=10)
            
            # Set axis limits to focus on brain
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            ax.set_zlim(-60, 100)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved visualization to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error in visualize_3d_network: {str(e)}")
            raise
            
    def _add_glass_brain_surface(self, ax, alpha=0.2):
        """Add a transparent brain surface."""
        try:
            # Fetch a template brain surface
            fsaverage = datasets.fetch_surf_fsaverage()
            
            # Load vertices and faces
            mesh = nib.freesurfer.read_geometry(fsaverage['pial_left'])
            vertices, triangles = mesh
            
            # Scale and center the brain mesh
            vertices = vertices / 5  # Scale to match MNI coordinates
            
            # Add the surface mesh
            ax.plot_trisurf(
                vertices[:, 0], 
                vertices[:, 1], 
                vertices[:, 2],
                triangles=triangles,
                alpha=alpha,
                color='gray',
                shade=True,
                edgecolor=None
            )
            
        except Exception as e:
            self.logger.warning(f"Could not add brain surface: {str(e)}")
            # Continue without the surface
            pass
            
    def visualize_nilearn_connectome(self,
                                    graph: Data,
                                    title: str = "Brain Connectome",
                                    display_mode: str = 'ortho',
                                    save_path: Optional[str] = None,
                                    view_type: str = 'connectome') -> None:
        """
        Create a 2D connectome visualization using nilearn plotting.
        
        Args:
            graph: PyG Data object
            title: Plot title
            display_mode: Display mode ('ortho', 'x', 'y', 'z', 'l', 'r')
            save_path: Path to save the figure
            view_type: Type of view ('connectome', 'glass_brain', 'stat_map')
        """
        try:
            # Prepare data
            node_coords, node_sizes, edge_weights, region_names = self._prepare_network_data(graph)
            
            # Create adjacency matrix from edge index and weights
            n_nodes = len(region_names)
            adjacency = np.zeros((n_nodes, n_nodes))
            
            edge_index = graph.edge_index.numpy()
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                edge_weights = graph.edge_attr.numpy()
                for i in range(edge_index.shape[1]):
                    src, dst = edge_index[0, i], edge_index[1, i]
                    adjacency[src, dst] = edge_weights[i]
            else:
                # Binary adjacency if no weights
                for i in range(edge_index.shape[1]):
                    src, dst = edge_index[0, i], edge_index[1, i]
                    adjacency[src, dst] = 1
            
            # Normalize node sizes for plotting
            sizes = node_sizes / node_sizes.max() * 50
            
            if view_type == 'connectome':
                # Plot connectome
                plotting.plot_connectome(
                    adjacency,
                    node_coords,
                    node_size=sizes,
                    title=title,
                    display_mode=display_mode,
                    edge_threshold=self.edge_threshold,
                    edge_cmap=self.colormap,
                    colorbar=True
                )
            elif view_type == 'glass_brain':
                # Create a fake stat map for node importance
                stat_map = np.zeros(self.atlas['img'].shape)
                atlas_data = self.atlas['img'].get_fdata()
                
                # Fill stat map based on node importance
                for i, region in enumerate(region_names):
                    if region in self.atlas['labels']:
                        region_idx = self.atlas['labels'].index(region)
                        region_mask = atlas_data == region_idx
                        if np.any(region_mask):
                            # Use node size (importance) as intensity
                            stat_map[region_mask] = node_sizes[i] / self.node_size_scale
                
                # Create nifti image from stat map
                stat_img = nib.Nifti1Image(stat_map, self.atlas['img'].affine)
                
                # Plot glass brain with connections
                display = plotting.plot_glass_brain(
                    stat_img,
                    display_mode=display_mode,
                    title=title,
                    colorbar=True,
                    plot_abs=False,
                    cmap=self.colormap
                )
                
                # Add edges
                display.add_graph(
                    adjacency, 
                    node_coords,
                    edge_threshold=self.edge_threshold,
                    edge_cmap=self.colormap,
                    edge_vmax=adjacency.max(),
                    node_size=30
                )
                
            elif view_type == 'stat_map':
                # Create node importance map
                stat_map = np.zeros(self.atlas['img'].shape)
                atlas_data = self.atlas['img'].get_fdata()
                
                for i, region in enumerate(region_names):
                    if region in self.atlas['labels']:
                        region_idx = self.atlas['labels'].index(region)
                        region_mask = atlas_data == region_idx
                        if np.any(region_mask):
                            stat_map[region_mask] = node_sizes[i] / self.node_size_scale
                
                # Create nifti image
                stat_img = nib.Nifti1Image(stat_map, self.atlas['img'].affine)
                
                # Plot statistical map
                plotting.plot_stat_map(
                    stat_img,
                    display_mode=display_mode,
                    title=title,
                    colorbar=True,
                    cmap=self.colormap
                )
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            plt.tight_layout()
            
        except Exception as e:
            self.logger.error(f"Error in visualize_nilearn_connectome: {str(e)}")
            raise

    def create_visualization_panel(self,
                                 graph: Data,
                                 base_title: str = "Brain Network Analysis",
                                 save_path: Optional[str] = None) -> None:
        """
        Create a complete visualization panel with multiple views.
        
        Args:
            graph: PyG Data object
            base_title: Base title for the visualization
            save_path: Path to save the complete figure
        """
        try:
            plt.figure(figsize=(20, 15))
            
            # 3D Network visualization (larger)
            plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
            self.visualize_3d_network(
                graph,
                title=f"{base_title} - 3D Network",
                show_labels=True
            )
            
            # Connectome view (top right)
            plt.subplot2grid((2, 3), (0, 2))
            self.visualize_nilearn_connectome(
                graph,
                title="Horizontal View",
                display_mode='z',
                view_type='connectome'
            )
            
            # Glass brain view (bottom right)
            plt.subplot2grid((2, 3), (1, 2))
            self.visualize_nilearn_connectome(
                graph,
                title="Activation Map",
                display_mode='ortho',
                view_type='glass_brain'
            )
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
        except Exception as e:
            self.logger.error(f"Error creating visualization panel: {str(e)}")
            raise
            
    def visualize_multiple_subjects(self,
                                  graphs: List[Data],
                                  subject_ids: List[str],
                                  grid_size: Tuple[int, int] = None,
                                  save_path: Optional[str] = None) -> None:
        """
        Create comparative visualization for multiple subjects.
        
        Args:
            graphs: List of PyG Data objects
            subject_ids: List of subject identifiers
            grid_size: Tuple of (rows, cols) for subplot grid
            save_path: Path to save the figure
        """
        num_subjects = len(graphs)
        
        # Determine grid size if not provided
        if grid_size is None:
            cols = min(3, num_subjects)
            rows = (num_subjects + cols - 1) // cols  # Ceiling division
            grid_size = (rows, cols)
        
        # Create figure
        fig = plt.figure(figsize=(6*grid_size[1], 5*grid_size[0]))
        
        for i, (graph, subject_id) in enumerate(zip(graphs, subject_ids)):
            if i < grid_size[0] * grid_size[1]:
                ax = fig.add_subplot(grid_size[0], grid_size[1], i+1, projection='3d')
                
                # Prepare data
                node_coords, node_sizes, edge_weights, region_names = self._prepare_network_data(graph)
                
                # Plot nodes
                scatter = ax.scatter(
                    node_coords[:, 0], 
                    node_coords[:, 1], 
                    node_coords[:, 2],
                    s=node_sizes * 0.7,  # Smaller nodes for comparison view
                    c=np.arange(len(node_coords)),
                    cmap=self.colormap,
                    alpha=0.8
                )
                
                # Plot edges (simplified for comparison)
                if edge_weights is not None and graph.edge_index is not None:
                    edge_index = graph.edge_index.numpy()
                    for j in range(min(100, edge_weights.shape[0])):  # Limit edges for clarity
                        if edge_weights[j] >= self.edge_threshold:
                            start_idx = edge_index[0, j]
                            end_idx = edge_index[1, j]
                            
                            start = node_coords[start_idx]
                            end = node_coords[end_idx]
                            
                            ax.plot(
                                [start[0], end[0]],
                                [start[1], end[1]],
                                [start[2], end[2]],
                                alpha=min(0.7, edge_weights[j]),
                                linewidth=edge_weights[j],
                                color='gray'
                            )
                
                # Add transparent brain outline
                self._add_glass_brain_surface(ax, alpha=0.15)
                
                # Set consistent view
                ax.view_init(elev=20, azim=-70)
                ax.set_title(f"Subject: {subject_id}", size=12)
                
                # Standard axis limits
                ax.set_xlim(-100, 100)
                ax.set_ylim(-100, 100)
                ax.set_zlim(-60, 100)
                
                # Minimal labels for comparison
                if i % grid_size[1] == 0:  # First column
                    ax.set_ylabel('Y (mm)')
                if i >= (grid_size[0]-1) * grid_size[1]:  # Last row
                    ax.set_xlabel('X (mm)')
                    
        # Add overall title
        plt.suptitle("Comparative Brain Network Analysis", size=20, y=0.98)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Node Importance')
        
        # Save or show
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

# Example usage function
def visualize_brain_network(graph_processor, sample_data, output_path=None):
    """
    Create brain network visualization from processed graph data.
    
    Args:
        graph_processor: AtlasRegionGraphProcessor instance  
        sample_data: Sample data dictionary with paths to MRI and sEEG
        output_path: Path to save visualization results
    """
    try:
        # Process sample data to create graph
        mri_data, region_masks = graph_processor.register_mri_to_atlas(sample_data['mri_path'])
        
        # Get channel names
        raw = mne.io.read_raw_fif(sample_data['seeg_path'], preload=False)
        channel_names = raw.ch_names
        
        # Match channels to regions
        atlas_to_channels = graph_processor.match_seeg_to_atlas_regions(
            channel_names,
            region_masks
        )
        
        # Extract features
        mri_features, region_names = graph_processor.extract_region_features(
            mri_data,
            region_masks,
            atlas_to_channels
        )
        
        # Process sEEG
        seeg_features = graph_processor.preprocess_seeg(
            sample_data['seeg_path'],
            atlas_to_channels
        )
        
        # Create graph
        graph_data = graph_processor.create_graph(
            seeg_features=seeg_features,
            mri_features=mri_features,
            region_names=region_names,
            label=sample_data['label']
        )
        
        # Initialize visualizer with same atlas
        visualizer = BrainNetworkVisualizer(
            atlas_name=graph_processor.atlas['name'] if hasattr(graph_processor.atlas, 'name') else 'harvard_oxford',
            edge_threshold=0.3  # Adjust as needed
        )
        
        # Create visualization panel
        if output_path:
            base_output = output_path.split('.')[0]
            panel_path = f"{base_output}_panel.png"
            single_path = f"{base_output}_3d.png"
        else:
            panel_path = None
            single_path = None
            
        # Visualize in multiple ways
        visualizer.create_visualization_panel(
            graph_data,
            base_title=f"Subject Analysis (Label: {sample_data['label']})",
            save_path=panel_path
        )
        
        # Create detailed 3D visualization
        visualizer.visualize_3d_network(
            graph_data,
            title=f"Detailed Brain Network (Label: {sample_data['label']})",
            save_path=single_path,
            show_labels=True
        )
        
        logging.info(f"Visualization complete. Files saved to: {output_path}")
        return graph_data
        
    except Exception as e:
        logging.error(f"Visualization failed: {str(e)}")
        raise