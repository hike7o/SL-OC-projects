""" Library of useful functions for the notebook...
"""

# ! usr/bin/env python 3
# coding: utf-8

# Importing librairies

import os
import time
import warnings
import numpy as np               # numerical data processing
import pandas as pd              # numerical tables & time series
import scipy as sp
import scipy.stats as st         # statistical functions
import seaborn as sns            # statistical data visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
import plotly.express as px

# Clustering library
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from IPython.display import HTML
from IPython.display import Image
import re

# metrics and scoring
from sklearn import cluster, metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import accuracy_score
# Image processing
import PIL
from PIL import Image, ImageOps, ImageFilter
import cv2
from scipy.spatial import distance
from tensorflow import keras 
from tensorflow.keras.preprocessing import image

palette = sns.color_palette("bright", 10)
plot_kwds = {'alpha' : 1, 's' : 60, 'linewidths':0}
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
# -- List of functions
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# -- Function 1 
# --------------------------------------------------------------------

def display_pixelhist(image, title):
    '''
    Display histograms of the desired item
    Parameters
    ----------
    image : item to choose
    title: display what you want to write down
    -------
    None.
    '''
    plt.figure(figsize=(40, 10))
    plt.subplot(131)
    plt.title(title, fontsize=30)
    plt.imshow(image, cmap='Greys_r') 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(132)
    plt.title('Pixels distribution', fontsize=30)
    hist, bins = np.histogram(np.array(image).flatten(), bins=256)
    plt.bar(range(len(hist[0:255])), hist[0:255])
    plt.xlabel('Grayscale', fontsize=30)
    plt.ylabel('Pixels count', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(133)
    plt.title('Pixels Cumulative Histogram', fontsize=30)
    plt.hist(np.array(image).flatten(), bins=range(256), cumulative=True)
    plt.xlabel('Grayscale', fontsize=30)
    plt.ylabel('Pixels cumulative frequency', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()

# --------------------------------------------------------------------
# -- Function 2
# --------------------------------------------------------------------
    
def preprocessed_image(image):
    '''
    
    - Brightness correction (autocontrast).
    - Contrast correction OpenCV CLAHE.
    - Noise reduction Non-local Means Denoising d'OpenCV.
    - Gray conversion .
    - Dimension reduction OpenCV (resize & interpolation INTER_AREA).
    Parameters
    ----------
    image : -.
    Returns
    -------
    None
    '''
    # Dimensions needed
    dim = (224, 224)

    # Image name
    file_dir = os.path.split(image)

    # Loading image
    img = Image.open(image)
    
    # Brightness correction
    img = ImageOps.autocontrast(img, 1)

     # Gray conversion
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    
    # Contrast correction with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    img = clahe.apply(img)

    # Noise reduction
    img = cv2.fastNlMeansDenoising(img, None, 5, 7, 21)
    
    # Resizing to square without distorsion

    def redim(img, size):
        # get image dimensions
        h, w = img.shape[:2]

        # dif = max (height, width)
        dif = h if h > w else w

        # define interpolation for zooming and shrinkage
        interpolation = cv2.INTER_AREA if dif > size else cv2.INTER_CUBIC    

        # for square images
        if h == w: 
            return cv2.resize(img, (size, size), interpolation)

        # for non square images
        x_pos = (dif - w)//2
        y_pos = (dif - h)//2

        # define mask for both color and back and white images
        if len(img.shape) == 2:
            mask = np.full((dif, dif), 255, dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
        else:
            mask = np.full((dif, dif, img.shape[2]), 255, dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

        return cv2.resize(mask, (size, size), interpolation)
    # Redimensionning
    img = redim(np.array(img), 224)
        
        # img = cv2.resize(np.array(img), dim, interpolation=cv2.INTER_AREA)

    cv2.imwrite('Images_processed/' + file_dir[1], img)

    return  'Images_processed/' + file_dir[1]

# --------------------------------------------------------------------
# -- Function 3 
# --------------------------------------------------------------------


# Resizing to square without distorsion

def redim(img, size):
    # get image dimensions
    h, w = img.shape[:2]
    
    # dif = max (height, width)
    dif = h if h > w else w
    
    # define interpolation for zooming and shrinkage
    interpolation = cv2.INTER_AREA if dif > size else cv2.INTER_CUBIC    
    
    # for square images
    if h == w: 
        return cv2.resize(img, (size, size), interpolation)
    
    # for non square images
    x_pos = (dif - w)//2
    y_pos = (dif - h)//2
    
    # define mask for both color and back and white images
    if len(img.shape) == 2:
        mask = np.full((dif, dif), 255, dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.full((dif, dif, img.shape[2]), 255, dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    
    return cv2.resize(mask, (size, size), interpolation)

# --------------------------------------------------------------------
# -- Function 4
# --------------------------------------------------------------------


def visual_words_display(image, keypoints):
    '''
    Display few Visual Words
    Parameters
    ----------
    image : image considered
    keypoints : Visual Words.
    Returns
    -------
    None.
    '''
    plt.figure(figsize=(10, 10))
    plt.title('Visual Words of the first descriptors')
    for i, kp in enumerate(keypoints[0:12]):
        # Get the coordinates of the center and size
        plt.subplot(3, 4, i + 1)
        x_center = kp.pt[0]
        y_center = kp.pt[1]
        size = kp.size

        # Set the border limits
        left = np.ceil(x_center - size / 2)
        upper = np.ceil(y_center + size / 2)
        right = np.ceil(x_center + size / 2)
        lower = np.ceil(y_center - size / 2)

        # Crop the image and show the parts
        cropped_np = np.array(image)[
            int(lower):int(upper), int(left):int(right)]
        plt.imshow(cropped_np, cmap='Greys_r')
    plt.show()
    
# --------------------------------------------------------------------
# -- Function 4
# --------------------------------------------------------------------

def cluster_optimal(data):

    ssel, dbl, chl = [], [], []
    nb_clusters=[100, 300, 500, 1000, 1500, 2000, 2500, 3000]

    for k in nb_clusters:
        kmean = MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=50, random_state=10)
        start_time = time.time()
        kmean.fit(data)
        cluster_labels = kmean.labels_
        
        # Elbow method
        ssel.append(kmean.inertia_)

        # Davies-Bouldin index
        db_avg = davies_bouldin_score(data, cluster_labels)
        dbl.append(db_avg)

        # Calinski-Harabasz
        ch_avg = calinski_harabasz_score(data, cluster_labels)
        chl.append(ch_avg)

        # Sihouette profile not run for images as computational time 
        # too important
#         silhouette_avg = silhouette_score(data, cluster_labels)
#         slcl.append(silhouette_avg)
        end_time = time.time() - start_time

    print('Running time (s) of clusters selection:', end_time)

# --------------------------------------------------------------------
# -- Function 5
# --------------------------------------------------------------------   
    
def cluster_selection(n_clusters, metrics):
    """
    Display 4 plots: Elbow method, Davies-Bouldin Index
    Calinski-Harabsz & Silhouette score
    vs number of clusters

    Parameters:
    metrics: Dict of metrics results
    n_clusters: Number of clusters observed
    """

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle("Selection of optimal number of clusters",
                 fontsize=18, weight='bold', size=20)

    # Display inertia v clusters number
    axs[0, 0].plot(n_clusters, metrics["ssel"])
    axs[0, 0].set_title('Elbow graph', weight='bold', size=14)
    axs[0, 0].set(xlabel='Clusters number', ylabel='Inertia')

    # Display of Davies-Bouldin index v clusters number
    axs[0, 1].plot(n_clusters, metrics["dbl"])
    axs[0, 1].set_title("Davies-Bouldin Index", weight='bold', size=14)
    axs[0, 1].set(xlabel='Clusters number', ylabel='Davies-Bouldin Index')

    # Display of Calinski-Harabasz index v clusters number
    # plt.subplot(132)
    axs[1, 0].plot(n_clusters, metrics["chl"])
    axs[1, 0].set_title("Calinski-Harabasz Index", weight='bold', size=14)
    axs[1, 0].set(xlabel='Clusters number', ylabel='Calinski-Harabasz Index')

#     # Display silhouette score v clusters number
#     axs[1, 1].plot(n_clusters, metrics["slcl"])
#     axs[1, 1].set_title("Silhouette profile", weight='bold', size=14)
#     axs[1, 1].set(xlabel="Clusters number", ylabel="Silhouette Score")

    plt.show()

# --------------------------------------------------------------------
# -- Function 6
# --------------------------------------------------------------------   

def sift_orb_kmeans(k, descriptor_list):
    """
    KMeans clustering

    Parameters:
    k: Number of clusters observed
    descriptor_list: List of descriptors
    Return: An array that holds central points
    """
    kmeans = MiniBatchKMeans(n_clusters = k, init='k-means++', max_iter=50, random_state=10)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    return visual_words    
    
# --------------------------------------------------------------------
# -- Function 6
# --------------------------------------------------------------------   

def load_image(dir):
    # Load processed images
    images = {}
    for filename in os.listdir(dir):
        path = dir + "/" + filename
        img = cv2.imread(path, 0)
        images[filename] = img
    return images


# --------------------------------------------------------------------
# -- Function 7
# --------------------------------------------------------------------   

def sift_features_out(images):
    '''
    Descriptors & keypoints extraction with SIFT.
    Parameters
    ----------
    images : images we want to study
    
    Returns
    -------
    List of descriptors and vectors obtained with SIFT
    Array whose first index holds the decriptor_list without an order
    The second index holds the sift_vectors dictionary which holds the descriptors.
    '''
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.SIFT_create()
    for key, value in images.items():
        features = []
        kp, des = sift.detectAndCompute(value, None)
        descriptor_list.extend(des)
        # in case no descriptor
        des = [np.zeros((128,))] if des is None else des
        features.append(des)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]

# --------------------------------------------------------------------
# -- Function 7
# --------------------------------------------------------------------   

def orb_features_out(images):
    '''
    Descriptors & keypoints extraction with ORB.
    Parameters
    ----------
    images : images we want to study
    
    Returns
    -------
    List of descriptors and vectors obtained with ORB.
    '''
    orb_vectors = {}
    descriptor_list = []
    orb = cv2.ORB_create()
    for key, value in images.items():
        features = []
        kp, des = orb.detectAndCompute(value, None)
        # descriptor_list.extend(des)
        # in case no descriptor
        des = [np.zeros((128,))] if des is None else des
        features.append(des)
        orb_vectors[key] = features
    return [descriptor_list, orb_vectors]

# --------------------------------------------------------------------
# -- Function 8
# --------------------------------------------------------------------   

def image_class(all_bovw, centers):
    '''
    Histogram creation 
    Parameters
    ----------
    all_bovw : dictionary that holds the descriptors that are separated class by class 
    centers: array that holds the central points (visual words) of the k means clustering
    Returns
    -------
    dictionary that holds the histograms for each images that are separated class by class. 
    '''
    dict_feature = {}
    for key,value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature

# --------------------------------------------------------------------
# -- Function 9
# --------------------------------------------------------------------  
 
def find_index(image, center):
    '''
    Find the index of the closest central point to the each sift descriptor.
    Parameters
    ----------
    image: sift descriptor 
    center: array of central points in k means
    Returns
    -------
    index of the closest central point.
    '''
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
            count = distance.euclidean(image, center[i])
            #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind

# --------------------------------------------------------------------
# -- Function 14 
# --------------------------------------------------------------------

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(True)
    frame.axes.get_yaxis().set_visible(True)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)),
              weight='bold', fontsize=14)
    plt.text(10, 35, 'Clustering time to run {:.2f} s'.
             format(end_time - start_time), fontsize=14)
    
# --------------------------------------------------------------------
# -- Function 10
# --------------------------------------------------------------------  

def tsne_visualisation(data, title):
    
    plt.figure(figsize=[8, 6])
    sns.scatterplot(x=0, y=1,
                    hue='Category',
                    data=data,
                    sizes=2)
    data1 = data.groupby('Category').mean()
    sns.scatterplot(x=0, y=1,
                    data=data1,
                    # hue='Category',
                    marker='*',
                    s=400,
                    color='k',
                    legend=False)
    plt.title(title, fontsize=12, weight='bold')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('t-SNE1', size=14)
    plt.ylabel('t-SNE2', size=14)
    
    plt.show()

# --------------------------------------------------------------------
# -- Function 11
# --------------------------------------------------------------------  
    
def display_clusters_tsne(X_projected, cluster_labels, centres_reduced, title):
    '''
    Display clusters visualization on scatterplot.
    
    '''
    plt.figure(figsize=[12, 8])

    sns.scatterplot(X_projected[:, 0],
                    X_projected[:, 1],
                    hue=cluster_labels,
                    s=70,
                    alpha=1,
                    palette='tab10')
    
    sns.scatterplot(centres_reduced[:, 0],
                    centres_reduced[:, 1],
                    #hue='cluster',
                    marker='*',
                    s=400,
                    linewidths=3,
                    color='k',
                    zorder=20)
    
    plt.title(title, fontsize=20)
    plt.legend(fontsize=15) #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("t-SNE1", weight='bold', size=14)
    plt.ylabel("t-SNE2", weight='bold', size=14)
    plt.grid(True)
    plt.show()
    
# --------------------------------------------------------------------
# -- Function 12
# --------------------------------------------------------------------  
    
def cluster_distribution(data):

    fig = plt.figure(figsize=(8, 6))
    ax = sns.countplot(y=data['cluster'],
                   palette='bright')
    plt.title('Distribution of products per cluster', weight='bold', size=14)
    plt.ylabel('Cluster')
    plt.xlabel('Count', weight='bold', size=12) 
    print("Number of products per cluster:\n{}".\
      format(data['cluster'].value_counts()))

# --------------------------------------------------------------------
# -- Function 11 
# --------------------------------------------------------------------

def metrics_clusters(dataframe, data_type):

    labels_true = dataframe['Category']
    labels_pred = dataframe['cluster']
    
    ARI = adjusted_rand_score(labels_true, labels_pred)
    homogeneity = homogeneity_score(labels_true, labels_pred)
    completeness = completeness_score(labels_true, labels_pred)
    vmeasure = v_measure_score(labels_true, labels_pred)
    AMI = adjusted_mutual_info_score(labels_true, labels_pred)
    Fowlkes_Mallows = fowlkes_mallows_score(labels_true, labels_pred)
    
    results = pd.DataFrame({
        'Methods': [data_type],
        'ARI': [ARI],
        'Homogeneity': [homogeneity],
        'Completness': [completeness],
        'V-Measure': [vmeasure],
        'AMI': [AMI],
        'Fowlkes-Mallows': [Fowlkes_Mallows]})
    
    return results

plt.show()

# --------------------------------------------------------------------
# -- Function 10
# --------------------------------------------------------------------

def threed_clustering(data):

    fig = px.scatter_3d(data,
                        x=0,
                        y=1,
                        z=2,
                        color='cluster',
                        labels={'x':'t-SNE1', 'y':'t-SNE2', 'z': 't-SNE3'})
    fig.update_traces(marker=dict(size=4,
                                  line=dict(width=2,
                                            color='DarkSlateGrey'),
                                  colorscale='viridis'),
                      selector=dict(mode='markers'))
    
    return fig