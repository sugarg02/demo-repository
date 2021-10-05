import preprocess2 as pp2
from preprocess3 import PyCNN
import easygui as eg
import cv2
import adaptive_median as amf
import average_filter as af
import BilateralFilter as bf
import NLMFilter as NLMf
import crop as crp
import wiener
import gaussianFilter
import findParams as fp
import matplotlib.pyplot as plt
import numpy as np

file = eg.fileopenbox();
algos = []
psnrs = []
psnrsED = []
psnrsMO = []

crp.cropp(file, 'images/output_crop.png')

amf.adaptiveMF(file, 'images/output_amf.png')
p = fp.findParams(file, 'images/output_amf.png', 'AMF')
algos.append('AMF')
psnrs.append(p)

af.avgFilter(file, 'images/output_af.png')
p = fp.findParams(file, 'images/output_af.png', 'Avg Filter')
algos.append('Avg Filter')
psnrs.append(p)

wiener.applyAdaptiveWiener(file, 'images/output_wiener.png')
p = fp.findParams(file, 'images/output_wiener.png', 'Wiener')
algos.append('Wiener')
psnrs.append(p)

gaussianFilter.applyGaussianFilter(file, 'images/output_gaussian.png')
p = fp.findParams(file, 'images/output_gaussian.png', 'Gaussian')
algos.append('Gaussian')
psnrs.append(p)

bf.BilateralFilter(file, 'images/output_bf.png')
p = fp.findParams(file, 'images/output_bf.png', 'Bilateral Filter')
algos.append('Bilateral Filter')
psnrs.append(p)

NLMf.NLMFilter(file, 'images/output_NLMf.png')
p = fp.findParams(file, 'images/output_NLMf.png', 'NLM Filter')
algos.append('NLM Filter')
psnrs.append(p)

gray_orig = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
gray = cv2.bitwise_not(gray_orig)

# Initialize object
preprocess3 = PyCNN()

# Perform respective image processing techniques on the given image
preprocess3.edgeDetection(file, 'images/output1.png')
p = fp.findParams(file, 'images/output1.png', 'Edge Detection')
algos.append('Edge')
psnrsED.append(p)

preprocess3.grayScaleEdgeDetection(file, 'images/output2.png')
p = fp.findParams(file, 'images/output2.png', 'Gray Scale Edge Detection')
algos.append('GS Edge')
psnrsED.append(p)

preprocess3.optimalEdgeDetection(file, 'images/output6.png')
p = fp.findParams(file, 'images/output6.png', 'Ensemble Edge Detection')
algos.append('Ensemble Edge')
psnrsED.append(p)

proc1 = pp2.remove_isolated_pixels(gray)
cv2.imwrite("images/isolated_pixels.png", proc1)
p = fp.findParams(file, 'images/isolated_pixels.png', 'Isolated Pixels Detection')
algos.append('Isolated Pixels')
psnrsED.append(p)

proc2 = pp2.erode(gray)
proc2 = pp2.remove_isolated_pixels(proc2)
proc2 = pp2.dilate(proc2)
cv2.imwrite("images/hard_open.png", proc2)
p = fp.findParams(file, 'images/hard_open.png', 'Hard Open')
algos.append('Hard Open')
psnrsMO.append(p)

img = cv2.imread(file,0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('images/hist_eq.png',res)
p = fp.findParams(file, 'images/hist_eq.png', 'Histogram Equalization')
algos.append('Hist Eq.')
psnrsMO.append(p)

proc3 = pp2.dilate(gray)
cv2.imwrite("images/dilated.png", proc3)
p = fp.findParams(file, 'images/dilated.png', 'Dilation')
algos.append('Dilation')
psnrsMO.append(p)

proc4 = pp2.erode(gray)
cv2.imwrite("images/eroded.png", proc4)
p = fp.findParams(file, 'images/eroded.png', 'Erosion')
algos.append('Erosion')
psnrsMO.append(p)

proc5 = pp2.median_blur(gray)
cv2.imwrite("images/median_blur.png", proc5)
p = fp.findParams(file, 'images/median_blur.png', 'Median Blur Detection')
algos.append('Blur')
psnrs.append(p)


print('Images stored in images/ folder')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(algos,psnrs)
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(algos,psnrsED)
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(algos,psnrsMO)
plt.show()