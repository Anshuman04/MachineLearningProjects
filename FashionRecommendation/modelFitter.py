import copy
import random

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, spectral_clustering

class ModelFitter(object):
    def __init__(self, articlesDF, custDF, modelCustObj, modelArtObj):
        random.seed(21)
        self.modelCustObj = modelCustObj
        self.modelArtObj = modelArtObj
        self.articlesDF = copy.deepcopy(articlesDF)
        self.custDF = copy.deepcopy(custDF)
        self.customerIds = self.custDF['customer_id'].to_list()
        self.readyDFs()
        self.fitModel()
        self._mapper()
        self.custDF['customer_id'] = self.customerIds

    def readyDFs(self):
        # Dropping columns for model training
        self.articlesDF.drop(columns=["detail_desc", "prod_name", "product_type_name", "product_group_name",\
                                      "graphical_appearance_name", "colour_group_name", "perceived_colour_value_name",\
                                      "perceived_colour_master_name", "department_name", "index_name",\
                                      "index_group_name", "section_name", "garment_group_name",\
                                      "index_code"], inplace=True)

        self.custDF.drop(columns=["customer_id"], inplace=True)


        columns = ["article_id", "product_code", "product_type_no", "graphical_appearance_no", "colour_group_code",\
                   "perceived_colour_value_id", "perceived_colour_master_id", "department_no", "index_group_no",\
                   "section_no", "garment_group_no"]

        for col in columns:
            self.articlesDF[col] = self.articlesDF[col].astype('category')

    def fitModel(self):
        self.opArt = self.modelArtObj.fit_predict(self.articlesDF)
        self.opCust = self.modelCustObj.fit_predict(self.custDF)
        self.articlesDF['resCluster'] = self.opArt
        self.custDF['resCluster'] = self.opCust

    def _mapper(self):
        cluster_to_article_map = {}
        article_to_cluster_map = {}

        for idx, row in self.articlesDF.iterrows():
            clusterNo = self.opArt[idx]
            art_id = row['article_id']
            if clusterNo not in cluster_to_article_map:
                cluster_to_article_map[clusterNo] = []
            cluster_to_article_map[clusterNo].append(art_id)
            article_to_cluster_map[art_id] = clusterNo

        cluster_to_customer_map = {}
        customer_to_cluster_map = {}

        for idx, row in self.custDF.iterrows():
            clusterNo = self.opCust[idx]
            cust_id = self.customerIds[idx]
            if clusterNo not in cluster_to_customer_map:
                cluster_to_customer_map[clusterNo] = []
            cluster_to_customer_map[clusterNo].append(cust_id)
            customer_to_cluster_map[cust_id] = clusterNo


        self.cluster_to_article_map = cluster_to_article_map
        self.article_to_cluster_map = article_to_cluster_map
        self.cluster_to_customer_map = cluster_to_customer_map
        self.customer_to_cluster_map = customer_to_cluster_map



