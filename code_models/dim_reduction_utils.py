import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
# from skimage.transform import resize
import tensorflow as tf



def single_img_svd(img,n_component):
    
    
     u,s,vh=np.linalg.svd(img)
     
     img_rec=u[:,0:n_component]@np.diag(s[0:n_component])@vh[0:n_component,:]
     return img_rec
 
def pca_transform(images,n_component):
    
    pca=PCA(n_component).fit(images) #learning the manifold
    coeffs=pca.transform(images) # projection of high res samples onto the low-dim basis vectors
    rec_imgs=pca.inverse_transform(coeffs)

    # plt.plot(np.cumsum(pca.explained_variance_ratio_),'--')
    # plt.xlabel('number of components')
    # plt.ylabel('cumulative explained variance');
    return rec_imgs
    

def tSNE_transform(features,n_components,early_exaggeration=None,perplexity=30):
    
    features_embedded = TSNE(n_components=n_components,early_exaggeration=early_exaggeration,
                             perplexity=perplexity).fit_transform(features)

    return features_embedded




