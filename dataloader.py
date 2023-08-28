import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class VFDataset(Dataset):
    def __init__(self, data_dir: Path, metadata: json, boundary_points: int) -> None:
        self.data_dir = data_dir
        self.metadata = metadata
        self.grid, self.coord_samples, self.boundaries = self.get_grid()
        self.n_points = boundary_points
        self.vf = torch.from_numpy(np.load(Path(self.data_dir, "gt.npy")))

    def get_grid(self) -> Union[Tensor, Tensor, Tensor]:
        dim = np.array(self.metadata["grid_dim"][1:])
        print(dim)
        min_x, max_x = (
            self.metadata["min_simulation_domain"][1],
            self.metadata["max_simulation_domain"][1],
        )
        min_y, max_y = (
            self.metadata["min_simulation_domain"][2],
            self.metadata["max_simulation_domain"][2],
        )
        max_siren_domain = dim / max(dim)
        min_siren_domain = -dim / max(dim)

        x_grid = torch.linspace(min_x, max_x, dim[0])
        y_grid = torch.linspace(min_y, max_y, dim[1])

        grid = torch.stack(torch.meshgrid(x_grid, y_grid), dim=0)
        grid = grid.view(2, -1)

        x_coords = torch.linspace(min_siren_domain[0], max_siren_domain[0], dim[0])
        y_coords = torch.linspace(min_siren_domain[1], max_siren_domain[1], dim[1])
        coord_grid = torch.stack(torch.meshgrid(x_coords, y_coords), dim=0)
        coord_samples = coord_grid.view(2, -1)

        boundaries = []
        boundaries.append(torch.tensor([[x_coords[0], y_coords[0]], [x_coords[-1], y_coords[0]]]))
        boundaries.append(torch.tensor([[x_coords[0], y_coords[-1]], [x_coords[-1], y_coords[-1]]]))
        boundaries.append(torch.tensor([[x_coords[0], y_coords[0]], [x_coords[0], y_coords[-1]]]))
        boundaries.append(torch.tensor([[x_coords[-1], y_coords[0]], [x_coords[-1], y_coords[-1]]]))

        return grid, coord_samples, boundaries

    def sample_boundary_conditions(self) -> Union[Tensor, Tensor]:
        boundary_lengths = torch.tensor([torch.norm(b[0] - b[1]) for b in self.boundaries])
        total_boundary_length = boundary_lengths.sum()
        rot90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])

        boundary_probs = boundary_lengths / total_boundary_length
        all_boundary_coords = []
        all_boundary_normals = []
        for bdx, boundary in enumerate(self.boundaries):
            n_boundary_samples = int(self.n_points * boundary_probs[bdx])

            boundary_tangent = boundary[1] - boundary[0]
            boundary_tangent /= boundary_tangent.norm()
            rot_tangent = rot90.mm(boundary_tangent.unsqueeze(1)).squeeze()
            rep_normals = torch.stack([rot_tangent for _ in torch.arange(n_boundary_samples)], dim=0)
            all_boundary_normals.append(rep_normals)

            alphas = torch.rand(n_boundary_samples).unsqueeze(1)
            boundary_samples = boundary[0].unsqueeze(0) * (1 - alphas) + boundary[1].unsqueeze(0) * (alphas)
            all_boundary_coords.append(boundary_samples)

        return torch.vstack(all_boundary_coords), torch.vstack(all_boundary_normals)

    def get_vector(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = x.to(torch.long), y.to(torch.long)
        vf_sample = self.vf[:, x, y]

        return vf_sample

    def __getitem__(self, index) -> Dict:
        x, y = self.grid[:, index]
        coords = self.coord_samples[:, index]
        vf_sample = self.get_vector(x, y)
        boundary_coords, boundary_normals = self.sample_boundary_conditions()

        return {
            "coord_samples": coords,
            "target": vf_sample,
            "boundary_samples": boundary_coords,
            "boundary_normals": boundary_normals,
        }

    def __len__(self):
        return self.vf.shape[1] * self.vf.shape[2]
