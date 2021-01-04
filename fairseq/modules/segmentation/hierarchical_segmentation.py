
import torch
import numpy as np
from .segment_dict import *
from heapq import *
from torch.autograd import Function, Variable

def variance(linearSum, squaresSum, size):
    return np.sum((squaresSum / size) - np.square(linearSum / size))  # sum of "variance mse vector"

def se(linearSum, squaresSum, size):  # square error
    return np.sum(squaresSum - np.square(linearSum) / size)  # sum of "se vector"

def varianceDiff(linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2):
    return variance(linearSum1 + linearSum2, squaresSum1 + squaresSum2, size1 + size2) - variance(linearSum1, squaresSum1, size1) - variance(linearSum2, squaresSum2, size2)

def seDiff(linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2):
    return se(linearSum1 + linearSum2, squaresSum1 + squaresSum2, size1 + size2) - se(linearSum1, squaresSum1, size1) - se(linearSum2, squaresSum2, size2)

def cosDist(linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2):  # cosine distance
    unscaledSim = np.dot(linearSum1, linearSum2) / (np.sqrt(np.dot(linearSum1, linearSum1)) * np.sqrt(np.dot(linearSum2, linearSum2)))
    unscaledAsDist = -unscaledSim + 1.  # change from similarity to distance; we mainly care about order for priority queue, for that any mapping reversing order is ok (low similarity = high distance)
    # ^ here we have a change from [-1, 1] to [0, 2]; standard "cosine distance"
    return unscaledAsDist * (size1 + size2)  
    # scaling so that big nonsense averaged almost-random segments don't appear as similar (randomnoise1 ~= randomnoise2)
    # this is where changing form similarity to distance mapping can make a difference, but linear one seems ok
    # this scaling is similar to the sum of distances of all elements to the average of the another segment and vice versa (can use sums instead of averages for cosine sim; 
    # but that's perhaps not exactly this sum as cosine_similarity ( (sum_i a_i) , x ) is not the same as (sum_i cosine_similarity ( a_i , x )) )
    # but the other one would be more expensive to compute

# [!] lines has to be a numpy array, np.sum() crashes if done on tensor
def hierarchicalSegmentation(lines, padMask=None, k=None, minSegmsPerLine=None, mergePriority="mse"):  # k is sum of number of segments for all lines
    
    if mergePriority == "se":  # var not divided by size, square error
        costFun = seDiff  #lambda linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2: seDiff(linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2)
    elif mergePriority == "var":  # var is mse
        costFun = varianceDiff  #lambda linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2: varianceDiff(linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2)
    elif mergePriority == "cos":
        costFun = cosDist  #lambda linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2: cos(linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2)
    else:
        assert False

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
            #oldVar1 = costFun(linSum1, sqSum1, right1 - left1 + 1)
            #oldVar2 = costFun(linSum2, sqSum2, right2 - left2 + 1)
            #mergedVariance = costFun(linSum1 + linSum2, sqSum1 + sqSum2, right2 - left1 + 1)
            size1 = right1 - left1 + 1
            size2 = right2 - left2 + 1
            costDiff = costFun(linSum1, sqSum1, size1, linSum2, sqSum2, size2)
            heappush(q, (costDiff, segm, segmRight))
       
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
        sizeMerged = rightMerged - leftMerged + 1
        #varMerged = costFun(linSumMerged, sqSumMerged, rightMerged - leftMerged + 1)
        
        if toLeft is not None:
            linSum2, sqSum2 = segmentsDict.getSegmentSums(toLeft)
            line2, left2, right2 = toLeft
            size2 = right2 - left2 + 1
            #oldVar2 = costFun(linSum2, sqSum2, right2 - left2 + 1)
            #mergedVariance = costFun(linSumMerged + linSum2, sqSumMerged + sqSum2, rightMerged - left2 + 1)
            costDiff = costFun(linSumMerged, sqSumMerged, sizeMerged, linSum2, sqSum2, size2)
            heappush(q, (costDiff, toLeft, merged))
            
        if toRight is not None:
            linSum2, sqSum2 = segmentsDict.getSegmentSums(toRight)
            line2, left2, right2 = toRight
            size2 = right2 - left2 + 1
            #oldVar2 = costFun(linSum2, sqSum2, right2 - left2 + 1)
            #mergedVariance = costFun(linSumMerged + linSum2, sqSumMerged + sqSum2, right2 - leftMerged + 1)
            costDiff = costFun(linSumMerged, sqSumMerged, sizeMerged, linSum2, sqSum2, size2)
            heappush(q, (costDiff, merged, toRight))
            
    return varChanges, merges, segmentsDict

class HierarchicalSegmentationLayer(Function):

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
    def forward(ctx, inputGPU, padMask=None, k=None, allowKsumRange=None, minSegmsPerLine=None, mergePriority="se", shorteningPolicy="orig_len"): 
    # k for strict num of segments (SUM FOR ALL LINES), allowKsumRange for range OF SUM OF SEGMENTS IN ALL LINES and choosing 'best' split point
    # min and max number of merges adjusted to what is possible - e.g. because of minSegmsPerLine

        assert k is None or allowKsumRange is None  # mutually exclusive options
        assert shorteningPolicy in ("shorten", "orig_len")  # orig_len&guess_orig is only at the higher level

        # TODO if input only 2-dim, add another dimension possibly (W x H -> 1 x W x H, consistent with B x W x H - later assuming that in some places)

        inputDevice = inputGPU.device
        padMaskInputDevice = padMask.device if padMask is not None else False

        # TODO TODO add shortening policy stuff !

        # tensor to CPU  (don't really need copy, will just need to put tensors in segmentsDict)
        input = inputGPU.detach().to('cpu').numpy()  
        # https://discuss.pytorch.org/t/cant-convert-cuda-tensor-to-numpy-use-tensor-cpu-to-copy-the-tensor-to-host-memory-first/38301 ,
        # https://discuss.pytorch.org/t/what-is-the-cpu-in-pytorch/15007/3

        varChanges, merges, segmentsDict = hierarchicalSegmentation(input, padMask=padMask, k=k, minSegmsPerLine=minSegmsPerLine, mergePriority=mergePriority, shorteningPolicy=shorteningPolicy)  # won't modify input
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

        # TODO change from here (and also change backward) depending on shortening policy [!]

        maxSegments = max(segmentNumsInLines)
        paddingMaskOut = np.full((input.shape[0], maxSegments), False)  #torch.BoolTensor(size=(input.shape[0], maxSegments)).fill_(False)
        for i, n in enumerate(segmentNumsInLines):
            paddingMaskOut[i][n:] = True
        
        segmented = np.full((input.shape[0], maxSegments, input.shape[2]), 0.)  #torch.tensor(size=(input.shape[0], maxSegments, input.shape[2])).fill_(0.)
        # can perhaps return a tensor with 1 at the beginning of the segments, -1 at the end, 0s elsewhere
        segmentBorders = np.zeros((input.shape[0], input.shape[1]), dtype=np.int8)
        for line, idxInLine in finalSegments.keys():
            line, begin, end = finalSegments[(line, idxInLine)]
            if shorteningPolicy == "shorten":
                segmented[line][idxInLine] = np.mean(input[line][begin:(end+1)], axis=0)  #torch.mean(input[line][begin:(end+1)])
            else:
                segmented[line][begin:(end+1)] = np.mean(input[line][begin:(end+1)], axis=0)
            segmentBorders[line][end] = -1  
            segmentBorders[line][begin] = 1  # [!] can be e.g. [...0, 0, 1, 1, ...] with segment of length 1 
            # - marking begins when length 1 as * scaling doesn't need + (scale-1) there if logging only begins

        resOutput = torch.tensor(segmented, dtype=inputGPU.dtype).to(inputDevice)   #if wasInputOnGPU else torch.tensor(segmented)  #.requires_grad_(True)
        resPadMask = torch.BoolTensor(paddingMaskOut).to(padMaskInputDevice)   #if wasPadMaskOnGPU else torch.BoolTensor(paddingMaskOut)
        segmentBorders = torch.IntTensor(segmentBorders).to(inputDevice)

        #print("********************", dir(ctx))
        #[not really needed] ctx.save_for_backward(padMask, resPadMask)
        # save_for_backward is only for tensors / variables / stuff
        if shorteningPolicy == "shorten":
            ctx.shortened = True
        else:
            ctx.shortened = False
        ctx.finalSegments = finalSegments
        ctx.segmentNumsInLines = segmentNumsInLines
        ctx.inputShape = input.shape
        ctx.mark_non_differentiable(resPadMask)  # can only pass torch variables here and only that makes sense

        #print("FINAL SEGMENTS: ", finalSegments, segmentNumsInLines)

        return resOutput, resPadMask, segmentBorders  #, finalSegments, segmentNumsInLines can only return torch variables... TODO maybe check how to fetch this info, but not sure if needed

    @staticmethod
    def backward(ctx, dxThrough, outPadMask=None, segmentBorders=None):  #, finalSegments=None, segmentNumsInLines=None):

        dxThroughDevice = dxThrough.device

        #[not really needed] paddingMask, paddingMaskOut = ctx.saved_tensors
        dx = torch.empty(size=ctx.inputShape, dtype=dxThrough.dtype).fill_(0.).to('cpu')

        wasShortened = ctx.shortened

        for line, idxInLine in ctx.finalSegments.keys():
            line, begin, end = ctx.finalSegments[(line, idxInLine)]
            if wasShortened:
                dx[line][begin:(end+1)] = dxThrough[line][idxInLine] / (end - begin + 1)
            else:
                dx[line][begin:(end+1)] = (dxThrough[line][begin:(end+1)].sum(dim=0)) / (end - begin + 1)

        dx = dx.to(dxThroughDevice)

        return dx, None, None, None, None, None, None


if __name__ == '__main__':
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7309))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()

    # run from .. with python -m segmentation.hierarchical_variance_segmentation

    tensor = torch.tensor([[[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]], [[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]]], dtype=torch.float64).requires_grad_(True)
    print(tensor[0][1])
    print(hierarchicalSegmentation(tensor.detach().numpy(), padMask=None, k=4, minSegmsPerLine=None, mergePriority="se"))  # pre-last merge in each line (merging (0,1) and (2,4)) should be 1.92 if summing 'variance vectors'
    print(hierarchicalSegmentation(tensor.detach().numpy(), padMask=None, k=2, minSegmsPerLine=None, mergePriority="var"))  # pre-last merge in each line (merging (0,1) and (2,4)) should be 1.92 if summing 'variance vectors'

    print("-------------------------- torch ---------------------------")
    # (tensor, padMask, k, kSumRange)
    resOutput, resPadMask, borders = HierarchicalSegmentationLayer.apply(tensor, torch.tensor([[True, False, False, False, False, False, False], [False, False, False, False, False, False, True]]), None, (2,5), None, "var", "shorten")  #(2, 5))  # can;t have keyword args for torch Functions...
    print(resOutput)
    print(resPadMask)
    print(borders)
    #print(finalSegments)
    #print(segmentNumsInLines)
    #loss = Variable(resOutput, requires_grad=True)
    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)

    print("-------------------------- torch2 ---------------------------")
    # (tensor, padMask, k, kSumRange)
    tensor.grad.data.zero_()
    resOutput, resPadMask, borders = HierarchicalSegmentationLayer.apply(tensor, torch.tensor([[True, False, False, False, False, False, False], [False, False, False, False, False, False, True]]), 3, None, None, "se", "shorten")  #(2, 5))  # can;t have keyword args for torch Functions...
    print(resOutput)
    print(resPadMask)
    print(borders)
    #print(finalSegments)
    #print(segmentNumsInLines)
    #loss = Variable(resOutput, requires_grad=True)
    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)

    print("-------------------------- torch3 ---------------------------")
    # (tensor, padMask, k, kSumRange)
    tensor.grad.data.zero_()
    resOutput, resPadMask, borders = HierarchicalSegmentationLayer.apply(tensor, torch.tensor([[True, False, False, False, False, False, False], [False, False, False, False, False, False, True]]), 3, None, 2, "se", "shorten")  #(2, 5))  # can;t have keyword args for torch Functions...
    print(resOutput)
    print(resPadMask)
    print(borders)
    # [!] here will return 4 segments instead of specified 3, because of specified minSegmsPerLine

    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)