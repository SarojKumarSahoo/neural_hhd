import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import VFDataset
from grad_net import GradSiren
from loguru import logger


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="path to vector field data")
parser.add_argument("--metadata", required=True, help="path to metadata")
parser.add_argument("--in_features", type=int, default=2, help="space dimension")
parser.add_argument("--out_features", type=int, default=1, help="scalar field")
parser.add_argument("--n_layers", type=int, default=5, help="number of hidden layers in the network")
parser.add_argument("--hidden_features", type=int, default=128, help="number of hidden features in hidden layers")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--device", default="cuda", help="cuda or cpu")

opt = parser.parse_args()


if __name__ == "__main__":
    metadata = json.load(open(opt.metadata))
    curl_free_net = GradSiren(opt.in_features, opt.out_features, opt.hidden_features, opt.n_layers).to(opt.device)
    div_free_net = GradSiren(opt.in_features, opt.out_features, opt.hidden_features, opt.n_layers).to(opt.device)

    curl_free_net.train()
    div_free_net.train()

    optimizer = optim.Adam(list(curl_free_net.parameters()) + list(div_free_net.parameters()), lr=opt.lr)
    vf_dataset = VFDataset(opt.data_dir, metadata, 10)
    dataloader = DataLoader(vf_dataset, batch_size=opt.batch_size, shuffle=True)

    criterion = torch.nn.MSELoss().to(opt.device)
    for epoch in range(opt.n_epochs):
        for idx, batch in enumerate(dataloader):
            coords_samples, target_vf, boundary_samples, boundary_normals = (
                batch["coord_samples"].to(opt.device),
                batch["target"].to(opt.device),
                batch["boundary_samples"].to(opt.device),
                batch["boundary_normals"].to(opt.device),
            )

            optimizer.zero_grad()
            curl_free_vf = curl_free_net(coords_samples)
            div_free_vf_prime = div_free_net(coords_samples)
            div_free_vf = torch.zeros_like(div_free_vf_prime)
            div_free_vf[:, 0] = -div_free_vf_prime[:, 1]
            div_free_vf[:, 1] = div_free_vf_prime[:, 0]

            predicted_vf = curl_free_vf + div_free_vf
            logger.debug(f"{coords_samples.shape}, {target_vf.shape}, {predicted_vf.shape}")

            prediction_loss = criterion(predicted_vf, target_vf)

            prediction_loss.backward()
            optimizer.step()
            logger.info(f"Prediction Loss : {prediction_loss.item()}")
