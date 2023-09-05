import json
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor


class Field:
    def __init__(self, data_dir: Path, timestep: int):
        """
        Initializes the Data class.

        Parameters:
        - data_dir (Path): The directory containing the data files.
        - timestep (int): The timestep for which data needs to be loaded.
        """
        metadata_path = Path(data_dir, "metadata.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with metadata_path.open("r") as file:
            self.metadata = json.load(file)

        field_path = Path(data_dir, "field.npy")
        if not field_path.exists():
            raise FileNotFoundError(f"Field file not found at {field_path}")

        self.vf = torch.from_numpy(np.load(field_path))[:, timestep]
        self.vf = self.vf.view(2, -1)
        self.grid = self._compute_grid()
        self.coord_samples, self.boundaries = self._compute_boundaries()

    def _compute_grid(self) -> Tensor:
        """
        Computes the grid based on metadata resolution.

        Returns:
        - Tensor: Computed grid.
        """
        dim = np.array(self.metadata["res"][1:])
        x_grid = torch.linspace(0, dim[0] - 1, dim[0])
        y_grid = torch.linspace(0, dim[1] - 1, dim[1])

        return torch.stack(torch.meshgrid(x_grid, y_grid), dim=0).view(2, -1)

    def _compute_boundaries(self) -> Tuple[Tensor, list]:
        """
        Computes the boundaries based on metadata resolution.

        Returns:
        - Tuple[Tensor, list]: Coordinate samples and boundary coordinates.
        """
        dim = np.array(self.metadata["res"][1:])
        x_coords = torch.linspace(0, 1, dim[0])
        y_coords = torch.linspace(0, 1, dim[1])
        coord_grid = torch.stack(torch.meshgrid(x_coords, y_coords), dim=0)
        coord_samples = coord_grid.view(2, -1)

        boundaries = [
            torch.tensor([[x_coords[0], y_coords[0]], [x_coords[-1], y_coords[0]]]),
            torch.tensor([[x_coords[0], y_coords[-1]], [x_coords[-1], y_coords[-1]]]),
            torch.tensor([[x_coords[0], y_coords[0]], [x_coords[0], y_coords[-1]]]),
            torch.tensor([[x_coords[-1], y_coords[0]], [x_coords[-1], y_coords[-1]]]),
        ]
        return coord_samples, boundaries

    def sample_boundary_conditions(self, n_points: int) -> Tuple[Tensor, Tensor]:
        """
        Samples boundary conditions based on the given number of points.

        Parameters:
        - n_points (int): Number of points for sampling.

        Returns:
        - Tuple[Tensor, Tensor]: Sampled boundary coordinates and normals.
        """
        boundary_lengths = torch.tensor([torch.norm(b[0] - b[1]) for b in self.boundaries])
        total_boundary_length = boundary_lengths.sum()
        rot90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])

        boundary_probs = boundary_lengths / total_boundary_length
        all_boundary_coords = []
        all_boundary_normals = []
        for bdx, boundary in enumerate(self.boundaries):
            n_boundary_samples = int(n_points * boundary_probs[bdx])

            boundary_tangent = boundary[1] - boundary[0]
            boundary_tangent /= boundary_tangent.norm()
            rot_tangent = rot90.mm(boundary_tangent.unsqueeze(1)).squeeze()
            rep_normals = torch.stack([rot_tangent for _ in torch.arange(n_boundary_samples)], dim=0)
            all_boundary_normals.append(rep_normals)

            alphas = torch.rand(n_boundary_samples).unsqueeze(1)
            boundary_samples = boundary[0].unsqueeze(0) * (1 - alphas) + boundary[1].unsqueeze(0) * (alphas)
            all_boundary_coords.append(boundary_samples)

        return torch.vstack(all_boundary_coords), torch.vstack(all_boundary_normals)
