import argparse
import json

import torch
import torch.optim as optim
from loguru import logger

from data import Field
from grad_net import GradSiren

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="path to vector field data")
parser.add_argument("--in_features", type=int, default=2, help="space dimension")
parser.add_argument("--out_features", type=int, default=1, help="scalar field")
parser.add_argument("--n_layers", type=int, default=5, help="number of hidden layers in the network")
parser.add_argument("--hidden_features", type=int, default=128, help="number of hidden features in hidden layers")
parser.add_argument("--batch_size", type=int, default=10000)
parser.add_argument("--n_iters", type=int, default=1000, help="number of iterations")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--device", default="cuda", help="cuda or cpu")

opt = parser.parse_args()


if __name__ == "__main__":
    curl_free_net = GradSiren(opt.in_features, opt.out_features, opt.hidden_features, opt.n_layers).to(opt.device)
    div_free_net = GradSiren(opt.in_features, opt.out_features, opt.hidden_features, opt.n_layers).to(opt.device)

    curl_free_net.train()
    div_free_net.train()

    field = Field(opt.data_dir, 50)
    grid = field.grid
    vf = field.vf
    coord_samples = field.coord_samples
    boundary_coords, boundary_normals = field.sample_boundary_conditions(5000)

    print(grid.shape, grid.min(), grid.max())
    print(vf.shape, vf.min(), vf.max())
    print(boundary_coords.shape, boundary_coords.min(), boundary_coords.max())
    print(boundary_normals.shape, boundary_normals.min(), boundary_normals.max())
    print(coord_samples.shape, coord_samples.min(), coord_samples.max())

    optimizer = optim.Adam(list(curl_free_net.parameters()) + list(div_free_net.parameters()), lr=opt.lr)
    criterion = torch.nn.MSELoss().to(opt.device)
    for idx in range(opt.n_iters):
        rand_idx = torch.randint(0, grid.shape[1], (opt.batch_size,))
        rand_samples, target_vf = (
            coord_samples[:, rand_idx].to(opt.device),
            vf[:, rand_idx].to(opt.device),
        )

        optimizer.zero_grad()
        curl_free_vf = curl_free_net(rand_samples.T)
        div_free_vf_prime = div_free_net(rand_samples.T)
        div_free_vf = torch.zeros_like(div_free_vf_prime)
        div_free_vf[:, 0] = -div_free_vf_prime[:, 1]
        div_free_vf[:, 1] = div_free_vf_prime[:, 0]

        predicted_vf = curl_free_vf + div_free_vf

        prediction_loss = criterion(predicted_vf, target_vf.T)

        prediction_loss.backward()
        optimizer.step()
        logger.info(f"Prediction Loss : {prediction_loss.item()}")
