
import torch

class DisjointSetUnion:
    def __init__(self, n: int, device="cuda"):
        # parent[i] = parent of node i
        self.parent = torch.arange(n, device=device)

    def find(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized find with path compression.
        x: torch.Tensor of indices
        """
        shape = x.shape
        x = x.flatten()
        while True:
            px = self.parent[x]
            # if we converged on parents
            if (px == x).all():
                break
            
            # Path compression: one step climb
            self.parent[x] = px
            x=px
        return self.parent[x].reshape(shape)

    def union(self, x: torch.Tensor, y: torch.Tensor):
        """
        Vectorized union using union-by-size.
        Always attaches smaller set under larger set.
        """
        x = x.flatten()
        y = y.flatten()
        root_x = self.find(x)
        root_y = self.find(y)

        # Only merge different sets
        mask = root_x != root_y
        if not mask.any():
            return False

        root_x = root_x[mask]
        root_y = root_y[mask]
        self.parent[torch.minimum(root_x,root_y)]=torch.maximum(root_x,root_y)
        
        return True
    
import math
import torch

def find_clusters_circular(A,device=None):
    if device is None:
        if torch.cuda.is_available():
            device='cuda'
        else: device='cpu'
    
    A=A.to(device)
    A_shape = A.shape
    pos = torch.arange(math.prod(A_shape),device=device).view(A.shape)
    dsu = DisjointSetUnion(math.prod(A_shape),device)

    unions = []
    ind_x = torch.arange(A_shape[0])
    ind_x1 = (ind_x+1)%A_shape[0]
    is_same = (A[ind_x]==A[ind_x1]) & (A[ind_x]==1)
    x_same = pos[ind_x][is_same],pos[ind_x1][is_same]
    unions.append(x_same)
    
    if A.ndim>1:
        ind_y = torch.arange(A_shape[1])
        ind_y1 = (ind_y+1)%A_shape[1]
        is_same = (A[:,ind_y]==A[:,ind_y1]) & (A[:,ind_y]==1)
        y_same=pos[is_same],pos[:,ind_y1][is_same]
        unions.append(y_same)
    
    if A.ndim>2:
        ind_z = torch.arange(A_shape[2])
        ind_z1 = (ind_z+1)%A_shape[2]
        is_same = (A[:,:,ind_z]==A[:,:,ind_z1]) & (A[:,:,ind_z]==1)
        z_same=pos[is_same],pos[:,:,ind_y1][is_same]
        unions.append(z_same)

    for i in range(3):
        for same in unions:
            dsu.union(*same)
            dsu.union(*same)
    
    clusters = dsu.find(pos)
    clusters[A==0]=-1
    
    uniq = clusters.unique()
    
    for i,c in enumerate(uniq):
        if c==-1: continue
        clusters[clusters==c]=i+1
    
    return clusters


def find_clusters_bounded(A,device=None):
    if device is None:
        if torch.cuda.is_available():
            device='cuda'
        else: device='cpu'
    
    A=A.to(device)
    A_shape = A.shape
    pos = torch.arange(math.prod(A_shape),device=device).view(A.shape)
    dsu = DisjointSetUnion(math.prod(A_shape),device)


    unions = []
    ind_x = torch.arange(A_shape[0])
    ind_x1 = ind_x*1
    ind_x1[1:]=ind_x[:-1]
    is_same = (A[ind_x]==A[ind_x1]) & (A[ind_x]==1)
    x_same = pos[ind_x][is_same],pos[ind_x1][is_same]
    unions.append(x_same)
    
    if A.ndim>1:
        ind_y = torch.arange(A_shape[1])
        ind_y1 = ind_y*1
        ind_y1[1:]=ind_y[:-1]
        is_same = (A[:,ind_y]==A[:,ind_y1]) & (A[:,ind_y]==1)
        y_same=pos[is_same],pos[:,ind_y1][is_same]
        unions.append(y_same)
    
    if A.ndim>2:
        ind_z = torch.arange(A_shape[2])
        ind_z1 = ind_z*1
        ind_z1[1:]=ind_z[:-1]
        is_same = (A[:,:,ind_z]==A[:,:,ind_z1]) & (A[:,:,ind_z]==1)
        z_same=pos[is_same],pos[:,:,ind_y1][is_same]
        unions.append(z_same)

    for i in range(3):
        for same in unions:
            dsu.union(*same)
            dsu.union(*same)
        
    clusters = dsu.find(pos)
    clusters[A==0]=-1
    uniq = clusters.unique()
    for i,c in enumerate(uniq):
        if c==-1: continue
        clusters[clusters==c]=i+1
    
    return clusters
