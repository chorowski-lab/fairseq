
import numpy as np
from .segment_dict import *
from heapq import *
from torch.autograd import Function

def variance(linearSum, squaresSum, size):
    return np.sum((squaresSum / size) - np.square(linearSum / size))  # sum of "variance vector"


# lines is a tensor or array of tensors?
# won't modify lines; assuming lines in on CPU if a tensor
def hierarchicalVarianceSegmentation(lines, padMask=None, k=None):  # k is per line (total num of segments to be made is k*numLines)
    
   # TODO check if tensor parts correctly taken etc. [!]
    segmentsDict = SegmentDict(lines, padMask=padMask)
    
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

class HierarchicalVarianceSegmentationLayer(Function):

    @staticmethod
    def flatten(x):
        s = x.shape()
        if len(s) < 3:
            return x
        if len(s) == 3:
            return x.view(-1, s[2])
        assert False

    # perhaps that ^ is not needed, and restore_shapes also

    @staticmethod
    def forward(ctx, inputGPU, padMask=None, k=None, allowKrange=None):  # k for strict num of segments, allowKrange for range and choosing 'best' split point

        assert k is None or allowKrange is None  # mutually exclusive options

        # TODO if input only 2-dim, add another dimension possibly (W x H -> 1 x W x H, consistent with B x W x H - later assuming that in some places)

        wasInputOnGPU = inputGPU.is_cuda
        wasPadMaskOnGPU = padMask.is_cuda if padMask is not None else False

        # tensor to CPU  (don't really need copy, will just need to put tensors in segmentsDict)
        input = inputGPU.detach().to('cpu').numpy()  
        # https://discuss.pytorch.org/t/cant-convert-cuda-tensor-to-numpy-use-tensor-cpu-to-copy-the-tensor-to-host-memory-first/38301 ,
        # https://discuss.pytorch.org/t/what-is-the-cpu-in-pytorch/15007/3

        varChanges, merges = hierarchicalVarianceSegmentation(input, padMask=padMask, k=k)  # won't modify input
        if allowKrange:  # full merge done above, k=None
            begin, end = allowKrange
            assert begin <= end
            beginIdx = len(varChanges) - end  # max allowed num of segments, smallest num of merges
            endIdx = len(varChanges) - begin  # min allowed num of segments, biggest num of merges
            prefSums = []
            s = 0.
            for chng in varChanges:
                s += chng
                prefSums.append(chng)
            best = -1
            where = -1
            for i in range(beginIdx, min(endIdx+1, len(varChanges))):
                sufSum = s - prefSums[i]  # sum after this index
                prefSum = prefSums[i] if prefSums[i] > 0. else 1.  # don't div by 0
                # v the bigger the better split point; suffix div by prefix averages of variance change
                here = (sufSum / (len(varChanges)-i))  /  (prefSum / (i+1.))  
                if here > best:
                    best = here
                    where = i
            if where == -1:
                print("WARNING: problems choosing best num segments")
                where = int((beginIdx + endIdx) // 2)
            varChanges = varChanges[:where+1]  # this one is not really needed
            merges = merges[:where+1]
            
        finalSegments, segmentNumsInLines = SegmentDict.getFinalSegments(merges)

        maxSegments = max(segmentNumsInLines)
        paddingMaskOut = np.full((input.shape[0], maxSegments), False)  #torch.BoolTensor(size=(input.shape[0], maxSegments)).fill_(False)
        for i, n in enumerate(segmentNumsInLines):
            paddingMaskOut[i][n:] = True
        
        segmented = np.full((input.shape[0], maxSegments, input.shape[2]), 0.)  #torch.tensor(size=(input.shape[0], maxSegments, input.shape[2])).fill_(0.)
        for line, idxInLine in finalSegments.keySet():
            line, begin, end = finalSegments[(line, idxInLine)]
            segmented[line][idxInLine] = np.avg(input[line][begin:(end+1)])  #torch.mean(input[line][begin:(end+1)])

        resOutput = torch.tensor(segmented).to('cuda') if wasInputOnGPU else torch.tensor(segmented)
        resPadMask = torch.BoolTensor(paddingMaskOut).to('cuda') if wasPadMaskOnGPU else torch.BoolTensor(paddingMaskOut)

        ctx.save_for_backward(finalSegments, segmentNumsInLines, paddingMask, paddingMaskOut, input.shape)
        ctx.mark_non_differentiable(resPadMask, finalSegments, segmentNumsInLines)  # TODO those/some are not tensors, not sure if should do that

        return resOutput, resPadMask, finalSegments, segmentNumsInLines

    @staticmethod
    def backward(ctx, dxThrough, outPadMask=None, finalSegments=None, segmentNumsInLines=None):

        finalSegments, segmentNumsInLines, paddingMask, paddingMaskOut, inputShape = ctx.saved_tensors

        dx = torch.tensor(size=inputShape).fill_(0.)
        for line, idxInLine in finalSegments.keySet():
            line, begin, end = finalSegments[(line, idxInLine)]
            dx[line][begin:(end+1)] = dxThrough[line][idxInLine] / (end - begin + 1)

        return dx, None, None, None


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