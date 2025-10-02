
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
    
    # def find(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Vectorized find with path compression, masked iteration.
    #     """
    #     shape = x.shape
    #     x = x.flatten()
    #     # Track active elements (those whose parent != themselves)
    #     active = torch.ones_like(x, dtype=torch.bool)
    #     while active.any():
    #         px = self.parent[x[active]]
    #         # Find which ones are done
    #         done_mask = px == x[active]
    #         # Update parent only for active elements
    #         self.parent[x[active]] = px
    #         # Deactivate finished paths
    #         active_indices = torch.nonzero(active).squeeze(-1)
    #         active[active_indices[done_mask]] = False
    #         # Continue climbing only unfinished ones
    #         x[active] = px[~done_mask]
    #     return self.parent[x].reshape(shape)

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
        z_same=pos[is_same],pos[:,:,ind_z1][is_same]
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
        clusters[clusters==c]=i
    
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
        z_same=pos[is_same],pos[:,:,ind_z1][is_same]
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
        clusters[clusters==c]=i
    
    return clusters


def get_percolation_clusters(clusters):
    """
    Identify percolating clusters that touch all boundary corners 
    of a 1D, 2D, or 3D labeled tensor.

    A cluster is considered "percolating" if its label appears on 
    all sides (corners for 1D, edges for 2D, and faces for 3D) 
    of the input tensor. This function checks cluster labels at 
    the boundaries of the tensor and determines which clusters 
    span across the entire domain.

    Parameters
    ----------
    clusters : torch.Tensor
        An integer-valued tensor of dimension 1, 2, or 3 where each 
        element represents the cluster label at that position. 

    Returns
    -------
    percolation_cluster_ids : list of int
        A list of cluster IDs that percolate (i.e., touch all corners/faces).
    sizes : list of int
        The sizes (number of elements) of the corresponding percolating clusters.

    Raises
    ------
    AttributeError
        If `clusters.ndim` is not 1, 2, or 3.

    Notes
    -----
    - For 1D tensors: the function checks the first and last elements.
    - For 2D tensors: the function checks the top row, bottom row, 
      left column, and right column.
    - For 3D tensors: the function checks the six faces of the cube.
    - The function uses `torch.isin` and `torch.unique` to efficiently 
      determine label overlap across corners/faces.

    Examples
    --------
    >>> import torch
    >>> clusters = torch.tensor([
    ...     [0, 0, 1],
    ...     [2, 1, 1],
    ...     [2, 2, 1]
    ... ])
    >>> get_percolation_clusters(clusters)
    ([1], [4])   # cluster label 1 percolates and has size 4
    """
    if clusters.ndim==1:
        corners = [clusters[0],clusters[-1]]
    elif clusters.ndim==2:
        corners = [clusters[0,:],clusters[:,0],clusters[-1,:],clusters[:,-1]]
    elif clusters.ndim==3:
        corners = [
            clusters[0,:,:],
            clusters[:,0,:],
            clusters[:,:,0],
            clusters[-1,:,:],
            clusters[:,-1,:],
            clusters[:,:,-1],
        ]
    else:
        raise AttributeError("clusters ndim must be in range 1-3")
    
    corners = [c.unique() for c in corners]
    all_corner_labels = torch.concat(corners).unique()
    corner_labels_count=torch.zeros_like(all_corner_labels)

    for c in corners:
        corner_labels_count += torch.isin(all_corner_labels,c)
    percolation_cluster_ids = all_corner_labels[corner_labels_count==4].tolist()
    
    # skip background
    sizes = [(clusters==c).sum() for c in percolation_cluster_ids]
    return percolation_cluster_ids,sizes