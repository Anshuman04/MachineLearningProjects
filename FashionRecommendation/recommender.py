from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


"""
from sklearn.metrics.pairwise import cosine_similarity
n_recom = 100
n_cust = 100  # <= total best customers
recommendations = []
for idx in range(n_cust):
    testCust = best_500_id[idx]
    # print("Test Cust: ", testCust)
    targetClusterCust = int(processedCust[processedCust['customer_id'] == testCust]['resCluster'])
    similarCustomers = cluster_to_customer_map[targetClusterCust]
    # print("Similar Customers: ", similarCustomers)
    unionArticles = []
    for tempCust in similarCustomers:
        unionArticles.extend(customer_to_article_map[tempCust])
    # print(len(unionArticles))
    unionArticles = list(set(unionArticles))
    # print(len(unionArticles))
    # print("Union: ", unionArticles)

    # get best article cluster to start searching
    voteMap = {}
    for testArt in unionArticles:
        art_cluster = article_to_cluster_map[testArt]
        if art_cluster in voteMap:
            voteMap[art_cluster] += 1
        else:
            voteMap[art_cluster] = 1
    # print(voteMap)
    voteList = [[item, voteMap[item]] for item in voteMap]
    voteList.sort(key=lambda x: x[1], reverse=True)
    bestArtCluster = voteList[0][0]
    # print(bestArtCluster)

    searchOrigin = cluster_center_articles[bestArtCluster]
    distances = []
    for testSearch in cluster_to_article_map[bestArtCluster]:
        testVec = processArticles[processArticles['article_id'] == testSearch]
        dist = cosine_similarity([searchOrigin], testVec)
        distances.append([testSearch, dist])

    distances.sort(key=lambda x:x[1], reverse=True)
    recommend = [item[0] for item in distances[:min(n_recom, len(distances))]]



    # distances = km_model.transform(processArticles)[:, bestArtCluster]
    # bestIndexes = np.argsort(distances)[::][:n_recom]
    # # print(bestIndexes)
    #
    # recommend = []
    # for rIdx in bestIndexes:
    #     # print ("Recommend idx: ", rIdx)
    #     recommend.append(articlesCopy.loc[rIdx]['article_id'])

    recommendations.append(recommend)

print(recommendations)

"""
class Recommender(object):
    def getRecommendations(self, custIds, processedCustDF, processedArtDF, mapObj, modelArtObj,\
                           customer_to_article_map, nCust=100, nRecom=12):
        self.recommendations = []
        maxRecom = nRecom
        custLimit = nCust
        for idx in range(custLimit):
            testCust = custIds[idx]
            targetClusterCust = int(processedCustDF[processedCustDF['customer_id'] == testCust]['resCluster'])
            similarCustomers = mapObj.cluster_to_customer_map[targetClusterCust]
            unionArticles = []
            for tempCust in similarCustomers:
                unionArticles.extend(customer_to_article_map[tempCust])
            unionArticles = list(set(unionArticles))

            # get best article cluster to start searching
            voteMap = {}
            for testArt in unionArticles:
                art_cluster = mapObj.article_to_cluster_map[testArt]
                if art_cluster in voteMap:
                    voteMap[art_cluster] += 1
                else:
                    voteMap[art_cluster] = 1

            voteList = [[item, voteMap[item]] for item in voteMap]
            voteList.sort(key=lambda x: x[1], reverse=True)
            bestArtCluster = voteList[0][0]

            cluster_center_articles = modelArtObj.cluster_centers_
            searchOrigin = cluster_center_articles[bestArtCluster]
            distances = []
            # import pdb
            # pdb.set_trace()
            resClusterOp = processedArtDF['resCluster']
            processedArtDF.drop(columns=['resCluster'], inplace=True)
            # for testSearch in mapObj.cluster_to_article_map[bestArtCluster]:
            #     testVec = processedArtDF[processedArtDF['article_id'] == testSearch]
            #     dist = cosine_similarity([searchOrigin], testVec)
            #     distances.append([testSearch, dist])
            #
            # distances.sort(key=lambda x:x[1], reverse=True)
            # recommend = [item[0] for item in distances[:min(maxRecom, len(distances))]]

            distances = modelArtObj.transform(processedArtDF)[:, bestArtCluster]
            bestIndexes = np.argsort(distances)[::][:maxRecom]
            # print(bestIndexes)

            recommend = []
            for rIdx in bestIndexes:
                # print ("Recommend idx: ", rIdx)
                recommend.append(processedArtDF.loc[rIdx]['article_id'])

            processedArtDF['resCluster'] = resClusterOp

            self.recommendations.append(recommend)

    def getScore(self, custIds, customer_to_article_map, nCust=100, nRecom=12):
        results = []
        totalList = []
        custLimit = nCust
        recomLimit = nRecom
        for idx in range(custLimit):
            targetCust = custIds[idx]
            allArt = customer_to_article_map[targetCust]
            tot = 0.0
            tp = 0.0
            for item in self.recommendations[idx]:
                tot += 1
                if item in allArt:
                    tp += 1
            results.append(tp)
            totalList.append(tot)



        indPrecision = [results[idx] / totalList[idx] for idx in range(len(results))]
        # import pdb
        # pdb.set_trace()
        finalScore = sum(indPrecision) / nCust
        return finalScore
    # [0.027777777777777776]












