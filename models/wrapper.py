import pyngp as ngp
import torch
import numpy as np


class NGPSDFWrapper:
    def __init__(self, snapshot_path, device="cuda"):
        self.device = device
        self.testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

        print(f"[NGP] Loading snapshot: {snapshot_path}")
        self.testbed.load_snapshot(snapshot_path)
        self.testbed.shall_train = False

        self.res = 256
        self.sdf_values = self.testbed.query_sdf_on_grid([self.res, self.res, self.res])
        assert self.sdf_values.shape[0] == self.res * self.res * self.res

        self.sdf_values = torch.from_numpy(self.sdf_values).to(self.device)

    def sdf(self, x):
        x = (x + 1.0) * 0.5  # → [0,1]
    
        coords = x * (self.res - 1)
    
        x0 = torch.floor(coords).long()
        x1 = (x0 + 1).clamp(0, self.res - 1)
    
        xd = coords - x0.float()
    
        x0 = x0.clamp(0, self.res - 1)
    
        x0x, x0y, x0z = x0[:,0], x0[:,1], x0[:,2]
        x1x, x1y, x1z = x1[:,0], x1[:,1], x1[:,2]
    
        def idx(ix, iy, iz):
            return ix + iy * self.res + iz * self.res * self.res
    
        sdf = self.sdf_values  # now torch tensor
    
        # 8 corners (ALL torch now)
        c000 = sdf[idx(x0x, x0y, x0z)]
        c100 = sdf[idx(x1x, x0y, x0z)]
        c010 = sdf[idx(x0x, x1y, x0z)]
        c110 = sdf[idx(x1x, x1y, x0z)]
        c001 = sdf[idx(x0x, x0y, x1z)]
        c101 = sdf[idx(x1x, x0y, x1z)]
        c011 = sdf[idx(x0x, x1y, x1z)]
        c111 = sdf[idx(x1x, x1y, x1z)]
    
        xd, yd, zd = xd[:,0], xd[:,1], xd[:,2]
    
        # trilinear interpolation (all torch)
        c00 = c000*(1-xd) + c100*xd
        c01 = c001*(1-xd) + c101*xd
        c10 = c010*(1-xd) + c110*xd
        c11 = c011*(1-xd) + c111*xd
    
        c0 = c00*(1-yd) + c10*yd
        c1 = c01*(1-yd) + c11*yd
    
        c = c0*(1-zd) + c1*zd
    
        return c.unsqueeze(-1)