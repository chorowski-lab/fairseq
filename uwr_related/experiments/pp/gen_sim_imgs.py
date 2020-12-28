
import sys
import os
import numpy as np
from PIL import Image, ImageDraw
#import matplotlib.pyplot as plt

assert len(sys.argv) == 2 or len(sys.argv) == 3  # pass directory with data as an arg and also (default 50) for how many pixels there should be a helper grid line
#print(sys.argv[0], sys.argv[1])

# TODO a mode with grid on segment borders
gridHz = int(sys.argv[2]) if len(sys.argv) == 3 else 50

# this should map for whole array, can e.g. use numpy etc. then
def mapSimFromDist(numArr, minVal=0., maxVal=1.):
    # this one only makes sense with >= 0 values
    # maps to one colour
    # TODO logarithmic option??
    #minRev = 1./maxVal
    #maxRev = 1./maxVal
    #mapped = (1./num - minRev) / maxRev
    if maxVal == 0:
        maxVal = 1
    return (float(maxVal) - (numArr - minVal)) / (maxVal - minVal)

dct = {}
for f in os.scandir(sys.argv[1]):
    els = f.name.split("_")
    t = els[0]
    rest = "_".join(els[1:])
    dct[(t,rest)] = f.name

print(dct)

print("===================", mapSimFromDist(0.), mapSimFromDist(1.))

for t, rest in dct:
    if t != "input":
        continue
    if ("features", rest) not in dct:
        continue
    inputFile = sys.argv[1] + "/" + t + "_" + rest
    featuresFile = sys.argv[1] + "/" + "features" + "_" + rest
    print("processing", inputFile, featuresFile)
    inputArr = np.load(inputFile)
    featuresArr = np.load(featuresFile).T

    # calculating array of distances between the representations; TODO add some params for different options, also similarity without distance in between
    inputSq = np.square(inputArr).sum(axis=0)
    featuresSq = np.square(featuresArr).sum(axis=0)
    #print(inputArr.shape, inputSq.shape, featuresArr.shape, featuresArr.T.shape, featuresSq.shape, featuresSq[np.newaxis,:].T.shape)
    distArr = featuresSq + featuresSq[np.newaxis,:].T - 2. * np.matmul(featuresArr.T, featuresArr)
    #print(distArr.shape)

    bigArr = np.zeros((inputArr.shape[0] + 3 + inputArr.shape[1], inputArr.shape[0] + 3 + inputArr.shape[1], 3))
    #print(bigArr.shape, inputArr.shape)

    # adding input images on the top and on the left and blue lines to an image
    bigArr[:inputArr.shape[0], (inputArr.shape[0]+3):(inputArr.shape[0]+3+inputArr.shape[1]), :] = inputArr[:,:,np.newaxis]
    bigArr[(inputArr.shape[0]+3):(inputArr.shape[0]+3+inputArr.shape[1]), :inputArr.shape[0], :] = np.flip(inputArr, axis=0).T[:,:,np.newaxis]
    bigArr[inputArr.shape[0]:(inputArr.shape[0] + 3), (inputArr.shape[0]):(inputArr.shape[0]+3+inputArr.shape[1]), 2] = 1.
    bigArr[(inputArr.shape[0]):(inputArr.shape[0]+3+inputArr.shape[1]), inputArr.shape[0]:(inputArr.shape[0]  +3), 2] = 1.
    a1 = ((inputArr.shape[0], inputArr.shape[0] + 3), (inputArr.shape[0], inputArr.shape[0]+3+inputArr.shape[1]))
    a2 = ((inputArr.shape[0], inputArr.shape[0]+3+inputArr.shape[1]), (inputArr.shape[0], inputArr.shape[0]  +3))
    #print("!!!", distArr[:2,:2])
    #sim = plt.imshow(distArr, cmap='viridis', )
    scaleFloat = float(inputArr.shape[1]) / float(distArr.shape[0])
    #print("---->", scaleFloat, inputArr.shape[1], distArr.shape[0])

    # creating similarity array from distance array; TODO as mentioned where creating distArr, add some params for different options, also without dist, like e.g. cosine sim
    minVal = distArr.min()
    maxVal = distArr.max()
    simArr = mapSimFromDist(distArr, minVal, maxVal)

    # converting stuff to 0-255 (but not ints where not needed yet) and increasing size 2x
    bigArr = bigArr * 255.  # scaling here, as similarity scaled separately below
    #print("!!!", bigArr.shape, np.ones((2,2)).shape)
    bigArr = bigArr.repeat(2, axis=0).repeat(2, axis=1)
    #print("!!!", bigArr.shape)
    bigArr[2*(inputArr.shape[0] + 3):, 2*(inputArr.shape[0] + 3):, 0] = np.asarray(Image.fromarray(np.array(simArr*255., dtype=np.int8)).resize((inputArr.shape[1]*2, inputArr.shape[1]*2), resample=Image.NEAREST))

    # from here stuff is 2x bigger (as grid lines were otherwise too big)

    # choosing helper grid positions to plot
    if ("segmentborders", rest) in dct and len(sys.argv) == 2:  # ONLY plot as segments if no grid density specified
        bordersFile = featuresFile = sys.argv[1] + "/" + "segmentborders" + "_" + rest
        bordersArr = np.load(bordersFile)
        #print("--------", bordersArr)
        gridBorders = [ 2*(inputArr.shape[0] + 3) + 2*int(round(j*scaleFloat)) for j, k in enumerate(bordersArr) if k == 1]
        #print("[][][][]", gridBorders)
    else:
        gridBorders = range(2*(inputArr.shape[0]+3), 2*(inputArr.shape[0]+3+inputArr.shape[1]), gridHz*2)

    # plotting helper grid
    for i in gridBorders:
        # grid helper
        bigArr[i, 2*(inputArr.shape[0]+3):, :] = 255
        bigArr[2*(inputArr.shape[0]+3):, i, :] = 255
    # for i in range(inputArr.shape[1]):
    #     for j in range(inputArr.shape[1]):
    #         bigArr[inputArr.shape[0] + 3 + i, inputArr.shape[0] + 3 + j][0] = mapSimFromDist(distArr[int(i / scaleFloat)][int(j / scaleFloat)], minVal, maxVal)
    
    # saving image from array
    img = Image.fromarray(np.array(bigArr, dtype=np.int8), 'RGB')  #.resize((bigArr.shape[0]*2, bigArr.shape[1]*2))  # PIL needs EXPLICIT int8, won't understand that int32 of values <256 is int8
    img.save(sys.argv[1] + "/" + "visualization" + "_" + rest.split(".")[0] + ".png")
    #img.show()
