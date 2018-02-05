# Vehicle Detection
## Introduction
This project is about detecting vehicles in an image. The detector is an SVM classifier trained using a 64 * 64 images. The HOGs for YCrCb channels are extracted, as well as a 32 * 32 bits were used as the features for each image. The detection is done as a sliding window on the image in different scales. A false positives filtering is done by thresholding the detections.

## Data
The data is separated into two categories, the car images, and the non car images. The car images contains images for cars or part of cars. The non car images are images from the road that are not cars, like the street, and the lane lines ... These images are fed into the classifier after some processing.

## Image features
The features used are the HOG extracted from each channel, and the bytes of the image resized to 32 * 32. The color space used to extract the features is YCrCb. Some experiments were done using RGB, and included histograms of the image's channels, but it was useless, produced overfitting, as it behaved poorly on the project video.

The HOG properties used were the same introduced in the lectures, and after testing, it appears that these values works just fine.

```python
orientations = 9
pix_per_cell = 8
cell_per_block = 2
```

![alt text](/resources/HOG.PNG "HOG Features")

Same works for the binary image data size.

This is how these features got extracted:

```Python
#HOG:
hog_features = []
for channel in range(feature_image.shape[2]):
  hog_features.append(get_hog_features(feature_image[:,:,channel], 
                      orient, pix_per_cell, cell_per_block, 
                      vis=False, feature_vec=True))
  hog_features = np.ravel(hog_features)


...

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features
#BS:
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
```

## Classification
Linear SVM is used to classify the images. Some parameters were testedusing Grid Search, to find that when C is 1, the model performs best, with accuracy around 99%. As the accuracy was that high, I did not test any other method.

```python

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

parameters = {'C':[1, 10, 100]}
svr = LinearSVC(verbose=100)
clf = GridSearchCV(svr, parameters, n_jobs=4,verbose=100)
clf.fit(X_train_scaled, y_train)

```

## Sliding window search
In order to find cars, a window sized 64 * 64 slides the image. The image got scaled to several scales, to ensure capturing close and far cars from the camera.

```python
bboxess = []
for scale in [.9,1,1.2,1.5,1.7]:
    output, bboxes = find_cars(imageRGB, ystart, ystop, scale, clf, X_scaler, orientations, pix_per_cell, cell_per_block)
    bboxess += bboxes
```

## False Positives
As any classifier, there were some false positives and some false negatives. The false negatives were reduced by passing on the same area several times with different scales, while the false positives were reduced using the heatmap thresholding.


## Result
The final result is the following.

![alt text](/resources/final.JPG "Pipeline result")

## Conclusions
This project can be enhanced by:
1. Adding a memory tracking. There is currently several mature tracking algorithms, that can be used to keep tracking of the objects location and speed, and predict where they will be in future.
2. Sometimes the pipeline could not identify the full body of the car, wich means that there might be some false negatives, more training data and deep nural network model might fit perfectly. check YOLO.

