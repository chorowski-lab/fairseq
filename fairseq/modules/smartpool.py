import torch
import torch.nn as nn
import torch.nn.functional as F

class Smartpool(nn.Module):
    def __init__(
        self,
        factor,
        search_perc
    ):
        """Smart pooling algorithm

        Args:
            factor: factor by which the sequence's length will be reduced
            search_perc: percentage of length of sequence after smartpooling to search for border. Ideally the border is located somewhere in +-search_perc
        """
        super().__init__()

        self.search_perc = search_perc
        self.factor = factor
        self.register_buffer("filters", torch.FloatTensor([[[[-1,1],[1,-1]]]]), persistent=False)

    def warp(self, X, new_lens):
        new_lens_cs = new_lens.cumsum(1)
        # This really searches for the low boundary of each new pixel
        pixel_contributions = new_lens_cs.view(1, -1, 1) - torch.arange(torch.round(new_lens_cs[0, -1]).item(), device=X.device).view(1, 1, -1)
        pixel_contributions = pixel_contributions.view(X.size(0), X.size(1), pixel_contributions.size(2))
        # Zero out the negative contributions, i.e. pixels which come before each row                              
        pixel_contributions = torch.max(torch.tensor(0.0, device=X.device), pixel_contributions)       
        
        # # This contains the cumulated pixel lengths for all pixels in each 
        # pixel_contributions
    
        pixel_contributions = pixel_contributions.unsqueeze(1)
        interp_weights = F.conv2d(pixel_contributions, self.filters, padding=1)
        interp_weights = interp_weights[:,:,:-1,1:] # Removing padding
        interp_weights = interp_weights.squeeze(1)

        # # Each column corresponds to a new element. Its values are the 
        # # weights associated with the original data.
        # interp_weights

        interp_weights = interp_weights.transpose(1, 2)
        Xnew = interp_weights @ X
        return Xnew, interp_weights

    def nonzero_interval_length(self, x, dim):
        nonz = (x > 0)
        _, low = ((nonz.cumsum(dim) == 1) & nonz).max(dim, keepdim=True)
        rev_cumsum = nonz.long().flip(dim).cumsum(dim).flip(dim)
        _, high = ((rev_cumsum == 1) & nonz).max(dim, keepdim=True)
        
        return high - low + 1

    def forward(self, features, padding_mask):
        B,T,C = features.size()

        padding_per_batch = (padding_mask > 0).sum(1)
        total_T = padding_mask.numel() - padding_per_batch.sum()
        features_together = torch.cat([features[i,:T-x] for i,x in enumerate(padding_per_batch)]).unsqueeze(0)

        features_tmp = F.pad(features, (0,0,1,0), value=features_together.mean().item())
        features_tmp = features_tmp.view(1, B * (T+1), C)

        # We have to remove 1 front padding and X_i back paddings from each batch. X_i can be arbitrary
        # but we have to append factors zeros so that there is one on the 
        # border between batches in resulting reduced sequence 
        # BATCH_1 000 BATCH_2 000 BATCH_3 -> REDUCED_1 0 REDUCED_2 0 REDUCED_3
        new_lens = (features_tmp[:,1:,:] - features_tmp[:,:-1,:]).abs().sum(dim=2).squeeze(0)
        new_lens = F.pad(new_lens, (1,0), value=0)
        new_lens = torch.cat([torch.cat([new_lens[i*(T+1)+1:(i+1)*(T+1)-x], torch.zeros(3*int(self.factor), device=new_lens.device)]) for i,x in enumerate(padding_per_batch)]).unsqueeze(0)
        new_lens = new_lens / new_lens.sum(1, keepdim=True) * ((total_T / self.factor) + B) # Reducing the original length T by some factor

        features = torch.cat([torch.cat([features[i,:T-x], torch.zeros(3*int(self.factor), C, device=new_lens.device)]) for i,x in enumerate(padding_per_batch)]).unsqueeze(0)
        features, interp_weights = self.warp(features, new_lens)
        
        # The idea is to remove B-1 the longest spanning intervals
        # which contain several zeros we added earlier
        
        # Get the indices to remove
        lengths_nonzero = self.nonzero_interval_length(interp_weights, 2)
        theor_lengths = ((T - padding_per_batch) // int(self.factor) + 1).view(-1)
        theor_cumsum = theor_lengths.cumsum(0)
        theor_lengths = (theor_lengths.float() * self.search_perc).long()
        to_remove = torch.cat(
            [torch.argmax(
                lengths_nonzero[:, theor_cumsum[i] - theor_lengths[i] : theor_cumsum[i] + theor_lengths[i], :]).view(1) 
                + theor_cumsum[i] - theor_lengths[i] for i in range(0,B-1)])
        
        indices = torch.arange(lengths_nonzero.size(1), device=lengths_nonzero.device)
        to_remove = torch.cat([to_remove.view(-1), indices[-1].view(1)])

        # Remove indices
        mask = torch.ones_like(features, dtype=torch.bool, device=features.device).view(1, -1, C)
        mask[0, to_remove, :] = False
        features = features[mask].view(-1,C)

        # Compute new features with padding
        start_idx, _ = torch.sort(to_remove)
        start_idx = start_idx - torch.arange(B, device=features.device)
        start_idx = F.pad(start_idx, [1,0])
        sizes = start_idx[1:] - start_idx[:-1]
        new_T = torch.max(sizes)
        sizes = new_T - sizes

        features = torch.cat([torch.cat([features[start_idx[i-1]:start_idx[i]], torch.zeros(sizes[i-1], C, device=features.device)]) for i in range(1,B+1)])
        features = features.view(B, new_T, C)

        # Compute new mask padding mask
        if padding_mask is not None:
            padding_mask = torch.zeros(B, new_T, dtype=torch.bool, device=features.device)
            for i,x in enumerate(sizes):
                padding_mask[i, new_T-x:] = True 

        return features, padding_mask


