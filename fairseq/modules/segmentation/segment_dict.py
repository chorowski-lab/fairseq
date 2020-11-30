
import numpy as np

class SegmentDict:
    
    def __init__(self, lines):  # lines assumed to be of shape [#lines x line_len * k]
        # (line#, place in line): (begin in line, end in line, sum(x), sum(x^2)) ; sums are possibly vectors
        self._dct = {(i, j): (j, j, lines[i][j], np.square(lines[i][j])) for i in range(len(lines)) for j in range(len(lines[i]))}
        
    # there is a 'segment' implicit format (tuple) used: (line#, leftIndex(begin), rightIndex(end))
        
    def segmentInDict(self, segment):
        line, leftIdx, rightIdx = segment
        if (line, leftIdx) not in self._dct:
            return False
        _, rightIdxFromDict, _, _ = self._dct[(line, leftIdx)]
        return rightIdx == rightIdxFromDict  # left already checked by key
        
    def removeSegment(self, segment):
        line, leftIdx, rightIdx = segment
        if (line, leftIdx) in self._dct:
            del self._dct[(line, leftIdx)]
        if (line, rightIdx) in self._dct:
            del self._dct[(line, rightIdx)]
            
    def mergeSegments(self, segment1, segment2):
        line1, left1, right1 = segment1
        line2, left2, right2 = segment2
        if not self.segmentInDict(segment1) or not self.segmentInDict(segment2) \
           or line1 != line2 or right1 + 1 != left2:  # not subsequent
            return None
        linearSum1, squaresSum1 = self.getSegmentSums(segment1)
        linearSum2, squaresSum2 = self.getSegmentSums(segment2)
        # remove old segments
        self.removeSegment(segment1)
        self.removeSegment(segment2)
        # add a new merged one
        self._dct[(line1, left1)] = (left1, right2, linearSum1 + linearSum2, squaresSum1 + squaresSum2)
        self._dct[(line1, right2)] = (left1, right2, linearSum1 + linearSum2, squaresSum1 + squaresSum2)
        return (line1, left1, right2)
            
    def getSegments(self):
        res = []
        for (line, leftIdx) in self._dct.keys():
            begin, end, _, _ = self._dct[(line, leftIdx)]
            res.append((line, begin, end))
        return res
        
    def getSegmentLeft(self, segment):
        if not self.segmentInDict(segment):
            return None
        line, left, right = segment
        if (line, left - 1) not in self._dct:
            return None
        segmLeft, segmRight, _, _ = self._dct[(line, left - 1)]
        return (line, segmLeft, segmRight)
    
    def getSegmentRight(self, segment):
        if not self.segmentInDict(segment):
            return None
        line, left, right = segment
        if (line, right + 1) not in self._dct:
            return None
        segmLeft, segmRight, _, _ = self._dct[(line, right + 1)]
        return (line, segmLeft, segmRight)
    
    def getSegmentSums(self, segment):
        if not self.segmentInDict(segment):
            return None
        line, left, _ = segment
        _, _, linearSum, squaresSum = self._dct[(line, left)]
        return (linearSum, squaresSum)