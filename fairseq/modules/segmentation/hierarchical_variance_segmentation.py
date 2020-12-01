
import numpy as np
from .segment_dict import *
from heapq import *


def variance(linearSum, squaresSum, size):
    return np.sum((squaresSum / size) - np.square(linearSum / size))  # sum of "variance vector"


# lines is a tensor or array of tensors?
def hierarchicalVarianceSegmentation(linesGPU, k=None):  # k is per line (total num of segments to be made is k*numLines)
    
    # tensor to CPU  (don't really need copy, will just need to put tensors in segmentsDict)
    lines = linesGPU.detach().to('cpu').numpy()  
    # https://discuss.pytorch.org/t/cant-convert-cuda-tensor-to-numpy-use-tensor-cpu-to-copy-the-tensor-to-host-memory-first/38301 ,
    # https://discuss.pytorch.org/t/what-is-the-cpu-in-pytorch/15007/3
    
   # TODO check if tensor parts correctly taken etc. [!]
    segmentsDict = SegmentDict(lines)
    
    # maybe will need to change this to arrays or so instead of dicts for efficiency
    
    # q for ranges to merge
    q = []
    
    # every pair added only one time; after merges will need to add both to right and to left
    for segm in segmentsDict.getSegments():
        segmRight = segmentsDict.getSegmentRight(segm)
        if segmRight is not None:
            linSum1, sqSum1 = segmentsDict.getSegmentSums(segm)
            linSum2, sqSum2 = segmentsDict.getSegmentSums(segmRight)
            line1, left1, right1 = segm
            line2, left2, right2 = segmRight
            oldVar1 = variance(linSum1, sqSum1, right1 - left1 + 1)
            oldVar2 = variance(linSum2, sqSum2, right2 - left2 + 1)
            mergedVariance = variance(linSum1 + linSum2, sqSum1 + sqSum2, right2 - left1 + 1)
            heappush(q, (mergedVariance - oldVar1 - oldVar2, segm, segmRight))
       
    varChanges = []
    merges = []
    
    while len(q) and (k is None or segmentsDict.numSegments() > k * lines.shape[0]):
    
        varChange, left, right = heappop(q)
        merged = segmentsDict.mergeSegments(left, right)  # checks if merge is valid
        
        if merged is None:  # old merge possibility, now impossible
            continue
        
        varChanges.append(varChange)
        merges.append((left, right))
        
        toLeft = segmentsDict.getSegmentLeft(merged)
        toRight = segmentsDict.getSegmentRight(merged)
        linSumMerged, sqSumMerged = segmentsDict.getSegmentSums(merged)
        lineMerged, leftMerged, rightMerged = merged
        varMerged = variance(linSumMerged, sqSumMerged, rightMerged - leftMerged + 1)
        
        if toLeft is not None:
            linSum2, sqSum2 = segmentsDict.getSegmentSums(toLeft)
            line2, left2, right2 = toLeft
            oldVar2 = variance(linSum2, sqSum2, right2 - left2 + 1)
            mergedVariance = variance(linSumMerged + linSum2, sqSumMerged + sqSum2, rightMerged - left2 + 1)
            heappush(q, (mergedVariance - varMerged - oldVar2, toLeft, merged))
            
        if toRight is not None:
            linSum2, sqSum2 = segmentsDict.getSegmentSums(toRight)
            line2, left2, right2 = toRight
            oldVar2 = variance(linSum2, sqSum2, right2 - left2 + 1)
            mergedVariance = variance(linSumMerged + linSum2, sqSumMerged + sqSum2, right2 - leftMerged + 1)
            heappush(q, (mergedVariance - varMerged - oldVar2, merged, toRight))
            
    return varChanges, merges



if __name__ == '__main__':
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7309))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()

    # run from .. with python -m segmentation.hierarchical_variance_segmentation

    import torch

    tensor = torch.tensor([[[1,2],[1,2],[3,4],[3,4],[3,4],[8,9]], [[1,2],[1,2],[3,4],[3,4],[3,4],[8,9]]], dtype=torch.float64)
    print(tensor[0][1])
    print(hierarchicalVarianceSegmentation(tensor, 2))  # pre-last merge in each line (merging (0,1) and (2,4)) should be 1.92 if summing 'variance vectors'