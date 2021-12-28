""" Library of useful functions for the notebook...
"""

# ! usr/bin/env python 3
# coding: utf-8

# Importing librairies

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
from sklearn.cluster import DBSCAN
import hdbscan

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

from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation, NMF

# USE
import tensorflow as tf
import tensorflow_hub as hub

# BERT
import transformers
from sentence_transformers import SentenceTransformer

# Text cleaning tools
import nltk
import gensim
# nltk.download()
from collections import Counter
from nltk import word_tokenize, FreqDist
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import string
import multiprocessing

from IPython.display import HTML
from IPython.display import Image
import re

palette = sns.color_palette("bright", 10)
plot_kwds = {'alpha' : 1, 's' : 60, 'linewidths':0}
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
# -- List of functions
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# -- Function 1 
# --------------------------------------------------------------------
def tsne_visualisation(data, title):
    
    plt.figure(figsize=[8, 6])
    sns.scatterplot(x=0, y=1, hue='Category', data=data, sizes=2)
    plt.title(title, fontsize=12, weight='bold')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('t-SNE1', size=14)
    plt.ylabel('t-SNE2', size=14)
    
    plt.show()

# --------------------------------------------------------------------
# -- Function 2
# --------------------------------------------------------------------

def display_factorial_planes(X_projected, n_comp, pca,
                             axis_ranks, labels=None,
                             alpha=1, illustrative_var=None):
    '''Display a scatter plot on a factorial plane,
    one for each factorial plane'''

    # For each factorial plane
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            # Initialise the matplotlib figure
            fig = plt.figure(figsize=(7, 6))

            # Display the points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1],
                            X_projected[:, d2],
                            alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1],
                                X_projected[selected, d2],
                                alpha=alpha, #label=value,
                                hue='Category', s=100)
                plt.legend()

            # Display the labels on the points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    plt.text(x, y, labels[i],
                             fontsize='14', ha='center', va='center')

            # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
            plt.xlim([-boundary, boundary])
            plt.ylim([-boundary, boundary])

            # Display grid lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.
                       format(d1+1,
                              round(100*pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('PC{} ({}%)'.
                       format(d2+1,
                              round(100*pca.explained_variance_ratio_[d2], 1)))

            plt.title("Projection of points (on PC{} and PC{})".
                      format(d1+1, d2+1))

# --------------------------------------------------------------------
# -- Function 3
# --------------------------------------------------------------------

def pca_visualization(dataframe,
                      X_projection,
                      x_label,
                      y_label,
                      title):

    # Dataframe creation
    dataframe_work = pd.DataFrame()
    dataframe_work['PC1'] = X_projection[:, 0]
    dataframe_work['PC2'] = X_projection[:, 1]
    dataframe_work['Category'] = dataframe['prod_category1']

    # First 2 components visualisation
    plt.figure(figsize=[25, 15])

    sns.set_palette('tab10')
    sns.scatterplot(x='PC1', y='PC2', data=dataframe_work, hue='Category',
                    s=100, alpha=1)
    plt.title(title, fontsize=40)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)
    plt.xlabel(x_label, fontsize=34)
    plt.ylabel(y_label, fontsize=34)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(False)
    plt.show()

# --------------------------------------------------------------------
# -- Function 4 
# --------------------------------------------------------------------

def display_scree_plot(pca):
    '''Display a scree plot for the pca'''

    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(), c="red", marker='o')
    plt.xlabel("Number of principal components")
    plt.ylabel("Percentage explained variance")
    plt.title("Scree plot")
    plt.show(block=False)

# --------------------------------------------------------------------
# -- Function 5 
# --------------------------------------------------------------------

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # 3 first factorial plans displayed i.e the first 6 componants
        if d2 < n_comp:

            # plot initialisation 
            fig, ax = plt.subplots(figsize=(7,6))

            # Limits of the plot fixed
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # Arrows display
            # If more than 30 arrows, triangle not displayed
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # Variable names displayed  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # Circle display
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # Plot limits definition
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # Horizontal and vertical lines display
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("PCA Correlation circle (F{} & F{})".format(d1+1, d2+1))
            plt.show(block=False)

# --------------------------------------------------------------------
# -- Function 6
# --------------------------------------------------------------------

def affiche_correlation_circle(pcs, pca, labels, axis_ranks=[(0, 1)],
                               long=6, larg=6):
    ''' Affiche les graphiques de cercle de corrélation de l'ACP pour les
        différents plans factoriels.
        Parameters
        ----------------
        pcs : PCA composants, obligatoire.
        labels : nom des différentes composantes, obligatoire.
        axis_ranks : liste de tuple de plan factoriel (0, 1) par défaut.
        long : longueur de la figure, facultatif (8 par défaut).
        larg : largeur de la figure, facultatif (8 par défaut).
        Returns
        ---------------
        None
    '''
    for i, (d1, d2) in enumerate(axis_ranks):

        fig, axes = plt.subplots(figsize=(long, larg))

        for i, (x_value, y_value) in enumerate(zip(pcs[d1, :], pcs[d2, :])):
            if(x_value > 0.2 or y_value > 0.2):
                plt.plot([0, x_value], [0, y_value], color='k')
                plt.text(x_value, y_value, labels[i], fontsize='14')

        circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='k')
        axes.set_aspect(1)
        axes.add_artist(circle)

        plt.plot([-1, 1], [0, 0], color='grey', ls='--')
        plt.plot([0, 0], [-1, 1], color='grey', ls='--')

        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

        # nom des axes, avec le pourcentage d'inertie expliqué
        axes.set_xlabel(
            'PC{} ({}%)'.format(
                d1 +
                1,
                round(
                    100 *
                    pca.explained_variance_ratio_[d1],
                    1)),
            fontsize=16)
        axes.set_ylabel(
            'PC{} ({}%)'.format(
                d2 +
                1,
                round(
                    100 *
                    pca.explained_variance_ratio_[d2],
                    1)),
            fontsize=16)
        axes.set_title('PCA correlation circle (PC{} and PC{})'.format(
            d1 + 1, d2 + 1), fontsize=18)

# --------------------------------------------------------------------
# -- Function 7
# --------------------------------------------------------------------

def threed_pca(data):
    fig = px.scatter_3d(data,
                        x='PC1',
                        y='PC2',
                        z='PC3',
                        color='Category',
                        labels={'x':'PC1', 'y':'PC2', 'z': 'PC3'})
    fig.update_traces(marker=dict(size=4,
                                  line=dict(width=2,
                                            color='DarkSlateGrey'),
                                  colorscale='viridis'),
                      selector=dict(mode='markers'))
    
    return fig
    
# --------------------------------------------------------------------
# -- Function 9
# --------------------------------------------------------------------

def display_clusters(X_projected, cluster_labels, centres_reduced, title):
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
                    marker='x',
                    s=150,
                    linewidths=3,
                    color='k',
                    zorder=20)
    
    plt.title(title, fontsize=20)
    plt.legend(fontsize=15) #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("PC1", weight='bold', size=14)
    plt.ylabel("PC2", weight='bold', size=14)
    plt.grid(True)
    plt.show()
    
# --------------------------------------------------------------------
# -- Function 10
# --------------------------------------------------------------------

def threed_clustering(data):

    fig = px.scatter_3d(data,
                        x='0',
                        y='1',
                        z='2',
                        color='cluster',
                        labels={'x':'0', 'y':'1', 'z': '2'})
    fig.update_traces(marker=dict(size=4,
                                  line=dict(width=2,
                                            color='DarkSlateGrey'),
                                  colorscale='viridis'),
                      selector=dict(mode='markers'))
    
    return fig

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

# --------------------------------------------------------------------
# -- Function 12 
# --------------------------------------------------------------------

def metrics_clusters_lda(dataframe, data_type):

    labels_true = dataframe['Category']
    labels_pred = dataframe['Topics']
    
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

# --------------------------------------------------------------------
# -- Function 13
# --------------------------------------------------------------------

def cluster_distribution(data):

    fig = plt.figure(figsize=(8, 6))
    ax = sns.countplot(y=data['cluster'],
                   palette='viridis_r')
    plt.title('Distribution of products per cluster', weight='bold', size=14)
    plt.ylabel('Cluster')
    plt.xlabel('Count', weight='bold', size=12) 
    print("Number of products per cluster:\n{}".\
      format(data['cluster'].value_counts()))
    
    plt.show()

# --------------------------------------------------------------------
# -- Function 14
# --------------------------------------------------------------------
def top_words_display(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 4, figsize=(20, 12), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_idx]
        weights = topic[top_features_idx]

        ax = axes[topic_idx]
        ax.barh(top_features, weights)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 20})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=20, weight='bold')

    plt.show()

# --------------------------------------------------------------------
# -- Function 15 
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
    plt.text(12.5, 4.5, 'Clustering time to run {:.2f} s'.
             format(end_time - start_time), fontsize=14)

# --------------------------------------------------------------------
# -- Function 17
# --------------------------------------------------------------------

def cluster_optimal(data):

    ssel, dbl, chl, slcl = [], [], [], []
    for k in range(2, 15):
        kmean = KMeans(n_clusters=k, max_iter=50, random_state=10)
        start_time = time.time()
        kmean.fit(data)
        cluster_labels = kmean.labels_
        time.time() - start_time

        # Elbow method
        ssel.append(kmean.inertia_)

        # Davies-Bouldin index
        db_avg = davies_bouldin_score(data, cluster_labels)
        dbl.append(db_avg)

        # Calinski-Harabasz
        ch_avg = calinski_harabasz_score(data, cluster_labels)
        chl.append(ch_avg)

        # Silhouette profile
        silhouette_avg = silhouette_score(data, cluster_labels)
        slcl.append(silhouette_avg)

    # Creation of a dict to keep the different metrics
    metrics = {"ssel": ssel,
               "dbl": dbl,
               "chl": chl,
               "slcl": slcl}

# --------------------------------------------------------------------
# -- Function 18
# --------------------------------------------------------------------   
    
def cluster_selection(range_n_clusters, metrics):
    """
    Display 4 plots: Elbow method, Davies-Bouldin Index
    Calinski-Harabsz & Silhouette score
    vs number of clusters

    Parameters:
    metrics: Dict of metrics results
    range_n_clusters: Number of clusters observed
    """

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle("Selection of optimal number of clusters",
                 fontsize=18, weight='bold', size=20)

    # Display inertia v clusters number
    axs[0, 0].plot(range_n_clusters, metrics["ssel"])
    axs[0, 0].set_title('Elbow graph', weight='bold', size=14)
    axs[0, 0].set(xlabel='Clusters number', ylabel='Inertia')

    # Display of Davies-Bouldin index v clusters number
    axs[0, 1].plot(range_n_clusters, metrics["dbl"])
    axs[0, 1].set_title("Davies-Bouldin Index", weight='bold', size=14)
    axs[0, 1].set(xlabel='Clusters number', ylabel='Davies-Bouldin Index')

    # Display of Calinski-Harabasz index v clusters number
    # plt.subplot(132)
    axs[1, 0].plot(range_n_clusters, metrics["chl"])
    axs[1, 0].set_title("Calinski-Harabasz Index", weight='bold', size=14)
    axs[1, 0].set(xlabel='Clusters number', ylabel='Calinski-Harabasz Index')

    # Display silhouette score v clusters number
    axs[1, 1].plot(range_n_clusters, metrics["slcl"])
    axs[1, 1].set_title("Silhouette profile", weight='bold', size=14)
    axs[1, 1].set(xlabel="Clusters number", ylabel="Silhouette Score")

    plt.show()
    
 
    
