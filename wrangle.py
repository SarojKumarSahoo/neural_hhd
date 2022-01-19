import numpy as np
import argparse
import netCDF4 as nc
import json
from pathlib import Path


def netcdf_to_numpy(dataset_name: str, nc_file: Path, out_dir: Path, timestep: int) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    f = nc.Dataset(nc_file)
    u = f["u"]
    v = f["v"]
    xdim = f["xdim"][:].shape[0]
    ydim = f["ydim"][:].shape[0]
    tdim = f["tdim"][:].shape[0]

    min_sim_x = float(f["xdim"][0].data)
    max_sim_x = float(f["xdim"][-1].data)

    min_sim_y = float(f["ydim"][0].data)
    max_sim_y = float(f["ydim"][-1].data)

    min_sim_t = float(f["tdim"][0].data)
    max_sim_t = float(f["tdim"][-1].data)

    u_vec = np.array(u[timestep, ...]).T
    v_vec = np.array(v[timestep, ...]).T

    gt = np.stack((u_vec, v_vec), axis=0)
    np.save(Path(out_dir, "gt.npy"), gt)

    metadata = {
        "dataset_name": dataset_name,
        "grid_dim": [tdim, xdim, ydim],
        "min_simulation_domain": [min_sim_t, min_sim_x, min_sim_y],
        "max_simulation_domain": [max_sim_t, max_sim_x, max_sim_y],
    }

    print(metadata)
    with open(Path(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name")
    parser.add_argument("--in_file")
    parser.add_argument("--out_dir")
    parser.add_argument("--timestep", type=int)

    opt = parser.parse_args()

    netcdf_to_numpy(opt.dataset_name, opt.in_file, opt.out_dir, opt.timestep)
