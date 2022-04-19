from dataMinimizer import DatasetMinimizer
from featureExtractor import FeatureExtractor
from modelFitter import ModelFitter
from recommender import Recommender
import copy
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering

datasetObj = DatasetMinimizer("../datasets")
artDF, transDF = datasetObj.getProcessedData()
fObj = FeatureExtractor(artDF, transDF, datasetObj.bestCustomers)
artDF, custDF = fObj.getReadyDF()

# from sklearn.metrics.pairwise import cosine_similarity
# processedCopy = copy.deepcopy(custDF)
# processedCopy.drop(columns=['customer_id'], inplace=True)
# affinity = cosine_similarity(processedCopy)
import pdb
pdb.set_trace()
results = np.zeros((10,10))
for artC in range(11, 21):
    for custC in range(6, 16):
        modelArt = KMeans(n_clusters=artC)
        modelCust = KMeans(n_clusters=custC)
        # modelCust = SpectralClustering(n_clusters=custC, affinity="nearest_neighbors")
        fitObj = ModelFitter(articlesDF=artDF, custDF=custDF, modelArtObj=modelArt, modelCustObj=modelCust)
        recomObj = Recommender()

        recomObj.getRecommendations(datasetObj.bestCustomers, fitObj.custDF, fitObj.articlesDF, fitObj, modelArt, fObj.customer_to_article_map)
        scores = recomObj.getScore(datasetObj.bestCustomers, fObj.customer_to_article_map)
        print("Articles [{}]; Customers[{}]: {}".format(artC, custC, scores))
        results[artC-11][custC-6] = scores
import pdb
pdb.set_trace()
print("Test")
