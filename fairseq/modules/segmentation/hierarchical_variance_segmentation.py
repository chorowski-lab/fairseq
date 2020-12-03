
import torch
import numpy as np
from .segment_dict import *
from heapq import *
from torch.autograd import Function, Variable

def variance(linearSum, squaresSum, size):
    return np.sum((squaresSum / size) - np.square(linearSum / size))  # sum of "variance vector"


# [!] lines has to be a numpy array, np.sum() crashes if done on tensor
def hierarchicalVarianceSegmentation(lines, padMask=None, k=None, minSegmsPerLine=None):  # k is sum of number of segments for all lines
    
   # TODO check if tensor parts correctly taken etc. [!]
    segmentsDict = SegmentDict(lines, padMask=padMask, minSegmsPerLine=minSegmsPerLine)
    
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
    
    while len(q) and (k is None or segmentsDict.numSegments() > k):  # will stop merging before k reached if minSegmsPerLine reached
    
        varChange, left, right = heappop(q)
        merged = segmentsDict.mergeSegments(left, right)  # checks if merge is valid
        
        if merged is None:  # old merge possibility, now impossible (or minSegmsPerLine reached for this line)
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
            
    return varChanges, merges, segmentsDict

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
    def forward(ctx, inputGPU, padMask=None, k=None, allowKsumRange=None, minSegmsPerLine=None): 
    # k for strict num of segments (SUM FOR ALL LINES), allowKsumRange for range OF SUM OF SEGMENTS IN ALL LINES and choosing 'best' split point
    # min and max number of merges adjusted to what is possible - e.g. because of minSegmsPerLine

        assert k is None or allowKsumRange is None  # mutually exclusive options

        # TODO if input only 2-dim, add another dimension possibly (W x H -> 1 x W x H, consistent with B x W x H - later assuming that in some places)

        inputDevice = inputGPU.device
        padMaskInputDevice = padMask.device if padMask is not None else False

        # tensor to CPU  (don't really need copy, will just need to put tensors in segmentsDict)
        input = inputGPU.detach().to('cpu').numpy()  
        # https://discuss.pytorch.org/t/cant-convert-cuda-tensor-to-numpy-use-tensor-cpu-to-copy-the-tensor-to-host-memory-first/38301 ,
        # https://discuss.pytorch.org/t/what-is-the-cpu-in-pytorch/15007/3

        varChanges, merges, segmentsDict = hierarchicalVarianceSegmentation(input, padMask=padMask, k=k, minSegmsPerLine=minSegmsPerLine)  # won't modify input
        #print("MERGES0: ", merges)
        if allowKsumRange:  # full merge done above, k=None, so each line now has minSegmsPerLine, but can also just get it from SegmDict - cleaner
            begin, end = allowKsumRange
            assert begin <= end
            # [!] min and max number of merges adjusted to what is possible - e.g. because of minSegmsPerLine
            beginIdx = max(0, min(len(varChanges) - 1, (segmentsDict.numSegments() + (len(varChanges) - 1) - end)))  # max allowed num of segments, smallest num of merges; input.shape[0] is num of segments if all merges done
            endIdx = max(0, min(len(varChanges) - 1, (segmentsDict.numSegments() + (len(varChanges) - 1) - begin)))  # min allowed num of segments, biggest num of merges; input.shape[0] is num of segments if all merges done
            #print("::::::::::", beginIdx, endIdx)
            prefSums = []
            s = 0.
            for chng in varChanges:
                s += chng
                prefSums.append(s)
            best = -1
            where = -1
            #print("PREFSUMS: ", prefSums)
            for i in range(beginIdx, min(endIdx+1, len(varChanges))):
                sufSum = s - prefSums[i]  # sum after this index
                prefSum = prefSums[i] if prefSums[i] > 0. else .0000001  # don't div by 0
                # v the bigger the better split point; suffix div by prefix averages of variance change
                here = (sufSum / (len(varChanges)-i))  /  (prefSum / (i+1.))  
                #print("!", i, ":", prefSum ,sufSum, here)
                
                if here > best:
                    best = here
                    where = i
            if where == -1:
                print("WARNING: problems choosing best num segments")
                where = int((beginIdx + endIdx) // 2)
            varChanges = varChanges[:where+1]  # this one is not really needed
            merges = merges[:where+1]
            
        finalSegments, segmentNumsInLines = SegmentDict.getFinalSegments(merges, input.shape[:2], padMask=padMask)
        #print("MERGES: ", merges)
        #print("FINAL SEGMENTS: ", finalSegments)

        maxSegments = max(segmentNumsInLines)
        paddingMaskOut = np.full((input.shape[0], maxSegments), False)  #torch.BoolTensor(size=(input.shape[0], maxSegments)).fill_(False)
        for i, n in enumerate(segmentNumsInLines):
            paddingMaskOut[i][n:] = True
        
        segmented = np.full((input.shape[0], maxSegments, input.shape[2]), 0.)  #torch.tensor(size=(input.shape[0], maxSegments, input.shape[2])).fill_(0.)
        for line, idxInLine in finalSegments.keys():
            line, begin, end = finalSegments[(line, idxInLine)]
            segmented[line][idxInLine] = np.mean(input[line][begin:(end+1)], axis=0)  #torch.mean(input[line][begin:(end+1)])

        resOutput = torch.tensor(segmented, dtype=inputGPU.dtype).to(inputDevice)   #if wasInputOnGPU else torch.tensor(segmented)  #.requires_grad_(True)
        resPadMask = torch.BoolTensor(paddingMaskOut).to(padMaskInputDevice)   #if wasPadMaskOnGPU else torch.BoolTensor(paddingMaskOut)

        #print("********************", dir(ctx))
        ctx.save_for_backward(padMask, resPadMask)
        # save_for_backward is only for tensors / variables / stuff
        ctx.finalSegments = finalSegments
        ctx.segmentNumsInLines = segmentNumsInLines
        ctx.inputShape = input.shape
        ctx.mark_non_differentiable(resPadMask)  # can only pass torch variables here and only that makes sense

        #print("FINAL SEGMENTS: ", finalSegments, segmentNumsInLines)
        return resOutput, resPadMask  #, finalSegments, segmentNumsInLines can only return torch variables... TODO maybe check how to fetch this info, but not sure if needed

    @staticmethod
    def backward(ctx, dxThrough, outPadMask=None):  #, finalSegments=None, segmentNumsInLines=None):

        dxThroughDevice = dxThrough.device

        paddingMask, paddingMaskOut = ctx.saved_tensors
        dx = torch.empty(size=ctx.inputShape, dtype=dxThrough.dtype).fill_(0.).to('cpu')

        for line, idxInLine in ctx.finalSegments.keys():
            line, begin, end = ctx.finalSegments[(line, idxInLine)]
            dx[line][begin:(end+1)] = dxThrough[line][idxInLine] / (end - begin + 1)

        dx = dx.to(dxThroughDevice)

        return dx, None, None, None, None


if __name__ == '__main__':
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7309))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()

    # run from .. with python -m segmentation.hierarchical_variance_segmentation

    tensor = torch.tensor([[[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]], [[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]]], dtype=torch.float64).requires_grad_(True)
    print(tensor[0][1])
    print(hierarchicalVarianceSegmentation(tensor.detach().numpy(), padMask=None, k=4, minSegmsPerLine=None))  # pre-last merge in each line (merging (0,1) and (2,4)) should be 1.92 if summing 'variance vectors'
    print(hierarchicalVarianceSegmentation(tensor.detach().numpy(), padMask=None, k=2, minSegmsPerLine=None))  # pre-last merge in each line (merging (0,1) and (2,4)) should be 1.92 if summing 'variance vectors'

    print("-------------------------- torch ---------------------------")
    # (tensor, padMask, k, kSumRange)
    resOutput, resPadMask = HierarchicalVarianceSegmentationLayer.apply(tensor, torch.tensor([[True, False, False, False, False, False, False], [False, False, False, False, False, False, True]]), None, (2,5), None)  #(2, 5))  # can;t have keyword args for torch Functions...
    print(resOutput)
    print(resPadMask)
    #print(finalSegments)
    #print(segmentNumsInLines)
    #loss = Variable(resOutput, requires_grad=True)
    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)

    print("-------------------------- torch2 ---------------------------")
    # (tensor, padMask, k, kSumRange)
    tensor.grad.data.zero_()
    resOutput, resPadMask = HierarchicalVarianceSegmentationLayer.apply(tensor, torch.tensor([[True, False, False, False, False, False, False], [False, False, False, False, False, False, True]]), 3, None, None)  #(2, 5))  # can;t have keyword args for torch Functions...
    print(resOutput)
    print(resPadMask)
    #print(finalSegments)
    #print(segmentNumsInLines)
    #loss = Variable(resOutput, requires_grad=True)
    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)

    print("-------------------------- torch3 ---------------------------")
    # (tensor, padMask, k, kSumRange)
    tensor.grad.data.zero_()
    resOutput, resPadMask = HierarchicalVarianceSegmentationLayer.apply(tensor, torch.tensor([[True, False, False, False, False, False, False], [False, False, False, False, False, False, True]]), 3, None, 2)  #(2, 5))  # can;t have keyword args for torch Functions...
    print(resOutput)
    print(resPadMask)
    # [!] here will return 4 segments instead of specified 3, because of specified minSegmsPerLine

    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)