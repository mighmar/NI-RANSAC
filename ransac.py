import numpy as np
from numpy import random as rnd
from skimage.feature import match_template
from PIL import Image
from matplotlib import pyplot as plt

NUM_SAMPLES = 40 
SAMPLE_SIZE = (80,80)
NUM_ITER = 30
ERROR_THRESHOLD = 1e0
NUM_INLIERS = 5

image1 = Image.open('wall/img1.ppm').convert('L')
image1 = np.array(image1) / 255

image2 = Image.open('wall/img2.ppm').convert('L')
image2 = np.array(image2) / 255


samples = np.zeros((3, NUM_SAMPLES))
targets = np.zeros((3, NUM_SAMPLES))

xlimit = image1.shape[0] - SAMPLE_SIZE[0]
ylimit = image1.shape[1] - SAMPLE_SIZE[1]
for i, (x,y) in enumerate(zip(rnd.randint(xlimit, size=NUM_SAMPLES),
               rnd.randint(ylimit, size=NUM_SAMPLES))):
    sample = image1[x : x+SAMPLE_SIZE[0], y : y+SAMPLE_SIZE[1]]
    samples[:,i] = np.array([x,y,1])
    
    corr = match_template(image2, sample)
    tx, ty = np.unravel_index(corr.argmax(), corr.shape) 
    targets[:,i] = np.array([tx,ty,1])

def get_transformation(samples, targets, index):
    A = []
    for i, (sample, target) in enumerate(zip(samples.T[index],
                                          targets.T[index])):
        A.append([sample[0], sample[1], 1, 0, 0, 0, -target[0]*sample[0],
                  -target[0]*sample[1], -target[0]])
        A.append([0, 0, 0, sample[0], sample[1], 1,  -target[1]*sample[0],
                  -target[1]*sample[1], -target[1]])
    A = np.asarray(A)
    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    return L.reshape(3,3)

min_error = float('inf')
best_transformation = np.array([[1,0,0],[0,1,0],[0,0,1]])
for _ in range(NUM_ITER):
    points = rnd.choice(range(NUM_SAMPLES), replace=False, size=4)
    transformation = get_transformation(samples, targets, points)


    predicted = (transformation @ samples)[:-1]
    errors = predicted - targets[:-1]
    errors = (errors*errors).sum(axis=0)
    inliers = np.where(errors < ERROR_THRESHOLD) 
    if len(inliers) >= NUM_INLIERS:
       transformation = get_transformation(samples, targets, inliers) 
       predicted = (transformation @ samples)[:-1,inliers]
       errors = predicted - targets[:-1,inliers]
       errors = (errors*errors).sum(axis=0) / 2 
       error = np.mean(errors)
       if error < min_error:
          min_error = error
          best_transformation = transformation

       
