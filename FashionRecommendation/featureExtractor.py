import pandas as pd
import os
import copy

class FeatureExtractor(object):
    def __init__(self, articlesDF, transactionsDF, customerIDs):
        self.articlesDF = copy.deepcopy(articlesDF)
        self.transactionsDF = copy.deepcopy(transactionsDF)
        self.custIDs = customerIDs
        self.assignArtCtg()
        self.createArticleMapping()
        self.featureHash = self.extractFeatures()
        self.createCustomerDF()
        self.customer_to_article_map = self.getCustToArtMap()

    def assignArtCtg(self):
        index_code = {"A": 14, "B": 10,"G": 11, "F": 12,"C": 13, "S": 5,"H": 6, "D": 7,"I": 8, "J": 9}
        group_no = {1: 1, 4: 4, 3: 3, 26: 0, 2: 2}
        self.articlesDF.replace({"index_code":index_code},inplace=True)
        self.articlesDF.replace({"index_group_no":group_no},inplace=True)

    def createArticleMapping(self):
        self.artMapping = {}
        for idx, row in self.articlesDF.iterrows():
            articleId = row['article_id']
            c1, c2 = row['index_group_no'], row['index_code']
            self.artMapping[articleId] = [c1, c2]

    def createCustomerDF(self):
        self.customerDF = pd.DataFrame({'customer_id': self.custIDs})
        self.customerDF['customer_id'] = self.customerDF['customer_id'].astype(dtype=pd.StringDtype())
        for i in range(15):
            self.customerDF["cat_{}".format(i)] = 0.0

        # Updating new dataframe
        for idx, row in self.customerDF.iterrows():
            cust_id = row['customer_id']
            for i in range(15):
                self.customerDF.at[idx, 'cat_{}'.format(i)] = self.featureHash[cust_id][i]

    def extractFeatures(self):
        featureHash = {}
        for item in self.custIDs:
            majCategory = [0 for i in range(5)]
            subCategory = [0 for i in range(10)]
            featureHash[item] = [majCategory, subCategory]

        for idx, row in self.transactionsDF.iterrows():
            articleId = row['article_id']
            assert articleId in self.artMapping,  "ArticleID [{}] not found".format(articleId)
            cust_id = row['customer_id']
            assert cust_id in featureHash,  "CustID [{}] not found".format(cust_id)
            cat1, cat2 = self.artMapping[articleId]
            assert cat1 < 5, "cat1 failed: {}".format(cat1)
            assert 4 < cat2 < 15, "cat2 failed: {}".format(cat2)
            featureHash[cust_id][0][cat1] += 1
            featureHash[cust_id][1][cat2 - 5] += 1

        for item in featureHash:
            newVec = []
            majSum = float(sum(featureHash[item][0]))
            minSum = float(sum(featureHash[item][1]))
            newVec.extend([x/majSum for x in featureHash[item][0]])
            newVec.extend([x/minSum for x in featureHash[item][1]])
            featureHash[item] = newVec
        return featureHash

    def getCustToArtMap(self):
        customer_to_article_map = {}
        for idx, row in self.transactionsDF.iterrows():
            cust_id = row['customer_id']
            art_id = row['article_id']
            if cust_id not in customer_to_article_map:
                customer_to_article_map[cust_id] = [art_id]
            else:
                customer_to_article_map[cust_id].append(art_id)
        return customer_to_article_map

    def getReadyDF(self):
        return self.articlesDF, self.customerDF





