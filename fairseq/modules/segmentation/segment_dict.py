
import numpy as np

class SegmentDict:
    
    def __init__(self, lines, padMask=None):  # lines assumed to be of shape [#lines x line_len * k]
        # (line#, place in line): (begin in line, end in line, sum(x), sum(x^2)) ; sums are possibly vectors
        self._dct = {(i, j): (j, j, lines[i][j], np.square(lines[i][j])) for i in range(len(lines)) for j in range(len(lines[i])) if padMask is None or not padMask[i][j]}
        self._size = len(self._dct)  # sometimes 1 (now all segments have 1 entry), sometimes later 2 entries per segment - better keep a counter
        
    # there is a 'segment' implicit format (tuple) used: (line#, leftIndex(begin), rightIndex(end))
        
    def numSegments(self):
        return self._size
    
    def segmentInDict(self, segment):
        line, leftIdx, rightIdx = segment
        if (line, leftIdx) not in self._dct:
            return False
        _, rightIdxFromDict, _, _ = self._dct[(line, leftIdx)]
        return rightIdx == rightIdxFromDict  # left already checked by key
        
    def removeSegment(self, segment):
        line, leftIdx, rightIdx = segment
        wasThere = False
        if (line, leftIdx) in self._dct:
            del self._dct[(line, leftIdx)]
            wasThere = True
        if (line, rightIdx) in self._dct:
            del self._dct[(line, rightIdx)]
            wasThere = True
        if wasThere:
            self._size -= 1
            
    def mergeSegments(self, segment1, segment2):
        line1, left1, right1 = segment1
        line2, left2, right2 = segment2
        if not self.segmentInDict(segment1) or not self.segmentInDict(segment2) \
           or line1 != line2 or right1 + 1 != left2:  # not subsequent
            return None
        linearSum1, squaresSum1 = self.getSegmentSums(segment1)
        linearSum2, squaresSum2 = self.getSegmentSums(segment2)
        # remove old segments; will update _size
        self.removeSegment(segment1)
        self.removeSegment(segment2)
        # add a new merged one; need to update _size by hand
        self._dct[(line1, left1)] = (left1, right2, linearSum1 + linearSum2, squaresSum1 + squaresSum2)
        self._dct[(line1, right2)] = (left1, right2, linearSum1 + linearSum2, squaresSum1 + squaresSum2)
        self._size += 1
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

    @staticmethod
    def getFinalSegments(merges, shape, padMask=None):  # shape needs to be B x W, without height!

        visited = np.zeros(shape, dtype=np.int32)
        finalSegments = []
        for i in range(len(merges)-1,-1,-1):
            leftSegm, rightSegm = merges[i]
            line, beginLeft, endLeft = leftSegm
            if visited[line][beginLeft] != 0:
                continue  # merge already seen
            _, beginRight, endRight = rightSegm
            finalSegments.append((line, beginLeft, endRight))
            visited[line][beginLeft:(endRight+1)] = 1

        # add length-1 segments that are there not padded but were not a part of any merge
        for i in range(visited.shape[0]):
            for j in range(visited.shape[1]):
                if not visited[i][j] and (padMask is None or not padMask[i][j]):
                    finalSegments.append((i, j, j))

        lineCounter = 0
        prevLine = 0  # don't append useless 0 at the beginning
        res = {}  # {(line, #ofSegmentInLine): (line, beginIdx, endIdx)}
        segmentsInLines = []  # numbers of segments in lines
        for line, begin, end in sorted(finalSegments):
            if line != prevLine:
                prevLine = line
                segmentsInLines.append(lineCounter)
                lineCounter = 0
            res[(line, lineCounter)] = (line, begin, end)
            lineCounter += 1
        segmentsInLines.append(lineCounter)

        return res, segmentsInLines  # there will be always at least 1 segment in a line