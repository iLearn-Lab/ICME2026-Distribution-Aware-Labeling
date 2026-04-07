import json
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture

def read_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
    return data_list

emb_refcoco = read_data('/path_to_emb')
emb_refcocoplus = read_data('/path_to_emb')
emb_refcocog = read_data('/path_to_emb')

emb_refcoco = np.array(emb_refcoco['mp'])
emb_refcocoplus = np.array(emb_refcocoplus['mp'])
emb_refcocog = np.array(emb_refcocog['mp'])

# fit GMM
K = 10  # 高斯分量数，可调
gmm_refcoco = GaussianMixture(n_components=K, covariance_type='full', random_state=42)
gmm_refcoco.fit(emb_refcoco)
joblib.dump(gmm_refcoco, "gmm_refcoco.pkl")

gmm_refcocoplus = GaussianMixture(n_components=K, covariance_type='full', random_state=42)
gmm_refcocoplus.fit(emb_refcocoplus)
joblib.dump(gmm_refcocoplus, "gmm_refcocoplus.pkl")

gmm_refcocog= GaussianMixture(n_components=K, covariance_type='full', random_state=42)
gmm_refcocog.fit(emb_refcocog)
joblib.dump(gmm_refcocog, "gmm_refcocog.pkl")