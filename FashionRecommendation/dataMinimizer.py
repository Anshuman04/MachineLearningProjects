import pandas as pd
import os

class DatasetMinimizer():
    def __init__(self, rootPath, articleThresh=20000, customerThresh=500):
        self.articleThresh=articleThresh
        self.customerThresh = customerThresh
        assert os.path.exists(rootPath), "Unable to find rootPath: {}".format(rootPath)
        self.data = self.loadData(rootPath)
        self.minimizeData()
        # self.createCustomerDF()

    def loadData(self, baseDir):
        articlePath = os.path.join(baseDir, "articles.csv")
        customerPath = os.path.join(baseDir, "transactions_train.csv")
        self.articlesDF = pd.read_csv(articlePath)
        self.transactionsDF = pd.read_csv(customerPath)

    def minimizeData(self):
        self.bestArticles = self._minimizeArticles()
        self.transactionsDF = self.transactionsDF[self.transactionsDF["article_id"].isin(self.bestArticles)]
        self.bestCustomers = self._minimizeCustomers()
        self.transactionsDF = self.transactionsDF[self.transactionsDF["customer_id"].isin(self.bestCustomers)]
        self.articlesDF = self.articlesDF[self.articlesDF["article_id"].isin(self.bestArticles)]
        self.articlesDF = self.articlesDF.reset_index(drop=True)
        self.transactionsDF = self.transactionsDF.reset_index(drop=True)

    def _minimizeCustomers(self):
        bestCustHash = {}
        for cid in self.transactionsDF["customer_id"]:
            if cid in bestCustHash:
                bestCustHash[cid] += 1
            else:
                bestCustHash[cid] = 1
        bestCustList = [[item, bestCustHash[item]] for item in bestCustHash]
        bestCustList.sort(key=lambda x: x[1], reverse=True)
        # bestCustCount = [item[1] for item in bestCustList]
        bestCustIDs = [item[0] for item in bestCustList]
        return bestCustIDs[:self.customerThresh]


    def _minimizeArticles(self):
        bestArticleHash = {}
        for aid in self.transactionsDF["article_id"]:
            if aid in bestArticleHash:
                bestArticleHash[aid] += 1
            else:
                bestArticleHash[aid] = 1
        bestArticleList = [[item, bestArticleHash[item]] for item in bestArticleHash]
        bestArticleList.sort(key=lambda x: x[1], reverse=True)
        # bestArticleCount = [item[1] for item in bestArticleList]
        bestArticleIDs = [item[0] for item in bestArticleList]
        return bestArticleIDs[:self.articleThresh]

    def getProcessedData(self):
        return self.articlesDF, self.transactionsDF

