import ast
import math
import pickle
import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from kneed import KneeLocator
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
# print("Processing data")


# UTILITY FUNCTION
def extract_keys(attr, key):
    if attr is None:
        return "{}"
    if key in attr:
        return attr.pop(key)


# UTILITY FUNCTION
def str_to_dict(attr):
    if attr is not None:
        return ast.literal_eval(attr)
    else:
        return ast.literal_eval("{}")


# UTILITY FUNCTION
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# LOAD DATA
def load_restaurant_data():
    try:
        businesses = pd.read_json(
            "./data/yelp_academic_dataset_business.json", lines=True,
            orient='columns', chunksize=150346)
    except:
        businesses = pd.read_json(
            "../data/yelp_academic_dataset_business.json", lines=True,
            orient='columns', chunksize=150346)

    for business in businesses:
        subset_business = business
        break

    city = subset_business[(subset_business['city'] == 'Kenner') & (subset_business['is_open'] == 1)]
    kenner_city = city[['business_id', 'name', 'address', 'categories', 'attributes', 'stars']]

    rest = kenner_city[kenner_city['categories'].str.contains('Restaurant.*') == True].reset_index(drop=True)

    rest['BusinessParking'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'BusinessParking')),
                                         axis=1)
    rest['Ambience'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Ambience')), axis=1)
    rest['GoodForMeal'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'GoodForMeal')), axis=1)
    rest['Dietary'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Dietary')), axis=1)
    rest['Music'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Music')), axis=1)

    df_attr = pd.concat([rest['attributes'].apply(pd.Series), rest['BusinessParking'].apply(pd.Series),
                         rest['Ambience'].apply(pd.Series), rest['GoodForMeal'].apply(pd.Series),
                         rest['Dietary'].apply(pd.Series)], axis=1)

    df_attr_dummies = pd.get_dummies(df_attr)

    df_categories_dummies = pd.Series(rest['categories']).str.get_dummies(',')

    result = rest[['name', 'stars']]

    df_final = pd.concat([df_attr_dummies, df_categories_dummies, result], axis=1)

    df_final.drop('Restaurants', inplace=True, axis=1)

    mapper = {1.0: 1, 1.5: 2, 2.0: 2, 2.5: 3, 3.0: 3, 3.5: 4, 4.0: 4, 4.5: 5, 5.0: 5}
    df_final['stars'] = df_final['stars'].map(mapper)

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(df_final.drop(['name', 'stars'], axis=1))

    final_df = pd.DataFrame(data=principalComponents, columns=['x', 'y'])

    x = np.array(final_df['x'].values)
    y = np.array(final_df['y'].values)

    norm_x = NormalizeData(x)
    norm_y = NormalizeData(y)
    return df_final, final_df, norm_x, norm_y


# KNN EXECUTION
def execute_knn(df_final):
    """# KNN clustering"""
    print("KNN clustering execution")
    knn = KNeighborsClassifier(n_neighbors=40)

    test = df_final[df_final['name'] == "Lil Doug's Cajun Market"]

    test_set = test.iloc[-1:, :]
    test_set = test_set.drop('name', axis=1)

    X_val = df_final.iloc[:-1, :]
    X_val = X_val.drop('name', axis=1)

    y_val = df_final['stars'].iloc[:-1]

    # # Create color maps
    # cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
    # cmap_bold = ["darkorange", "c", "darkblue"]

    # for weights in ["uniform", "distance"]:
    #     # we create an instance of Neighbours Classifier and fit the data.
    #     clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    #     clf.fit(X, y)

    #     _, ax = plt.subplots()
    #     DecisionBoundaryDisplay.from_estimator(
    #         clf,
    #         X,
    #         cmap=cmap_light,
    #         ax=ax,
    #         response_method="predict",
    #         plot_method="pcolormesh",
    #         xlabel=iris.feature_names[0],
    #         ylabel=iris.feature_names[1],
    #         shading="auto",
    #     )

    #     # Plot also the training points
    #     sns.scatterplot(
    #         x=X[:, 0],
    #         y=X[:, 1],
    #         hue=iris.target_names[y],
    #         palette=cmap_bold,
    #         alpha=1.0,
    #         edgecolor="black",
    #     )
    #     plt.title(
    #         "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    #     )

    # plt.show()

    n_knn = knn.fit(X_val, y_val)

    accuracy_train = n_knn.score(X_val, y_val)

    knn_cl = n_knn.predict(X_val)

    print(f"KNN Silhouette score: {metrics.silhouette_score(X_val, knn_cl)}")
    df_sil = pd.DataFrame([metrics.silhouette_score(X_val, knn_cl)], columns=['silhouette'])

    final_table = pd.DataFrame(n_knn.kneighbors(test_set)[0][0], columns=['distance'])

    final_table['index'] = n_knn.kneighbors(test_set)[1][0]

    final_table.set_index('index')

    result = final_table.join(df_final, on='index')
    return result, df_sil


# ELBOW COMPUTING
def elbow_method(final_df, norm_x, norm_y):
    """# ELBOW Method computing"""
    print("Elbow method computing")

    final_dst = []
    size = len(final_df)
    for idx, val in enumerate(norm_x):
        if idx < size:
            old_dst = 1
            dst = []
            for idx2, val2 in enumerate(norm_x):
                dist = math.sqrt(pow(norm_x[idx2] - norm_x[idx], 2) + pow(norm_y[idx2] - norm_y[idx], 2))
                if idx != idx2 and dist < old_dst:
                    dst.append(dist)
                    old_dst = dist
            dst.sort()
            final_dst.append(dst[0])
            print("\rProgress ---> " + str(idx) + "/" + str(size), end="")

    final_dst.sort()

    x_for_plot = np.arange(0, len(final_dst), dtype=int)

    kneedle = KneeLocator(x_for_plot, final_dst, S=1.0, curve="convex", direction="increasing")
    return final_dst, kneedle.elbow


# DBSCAN EXECUTION
def execute_dbscan(df_final, final_dst, elbow, norm_x, norm_y, df_sil):
    """# DBSCAN Clustering"""
    print("DBSCAN Clustering execution")
    norm_x = norm_x.reshape(-1, 1)
    norm_y = norm_y.reshape(-1, 1)

    lim_n_x = np.take(norm_x, np.arange(0, len(final_dst), dtype=int))
    lim_n_y = np.take(norm_y, np.arange(0, len(final_dst), dtype=int))
    lim_n_x = lim_n_x.reshape(-1, 1)
    lim_n_y = lim_n_y.reshape(-1, 1)

    n_list = []
    for idx, val in enumerate(lim_n_x):
        n_list.append([val[0], lim_n_y[idx][0]])

    X = np.array(n_list)

    dbscan = DBSCAN(eps=final_dst[elbow], min_samples=4)
    db = dbscan.fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("DBSCAN Estimated number of clusters: %d" % n_clusters_)
    print("DBSCAN Estimated number of noise points: %d" % n_noise_)
    print("DBSCAN Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
    df_sil = df_sil.append({'silhouette': metrics.silhouette_score(X, labels)}, ignore_index=True)

    """# Plotting DBSCAN results"""

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(16, 10))
    counter = 0
    n_cl = 0
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=10,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=1,
        )
        counter += 1
    n_cl = n_clusters_
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    try:
        plt.savefig("./plots/task3/task3_dbscan_res.png")
    except:
        plt.savefig("../plots/task3/task3_dbscan_res.png")
    plt.show()

    db = dbscan.fit_predict(X)

    df_res_dbs = df_final
    df_res_dbs['cluster'] = db.tolist()

    df_res_dbs = df_res_dbs
    df_res_2_dbs = df_res_dbs[['name', 'stars', 'cluster']]
    return df_res_2_dbs, df_sil, db


# KMEANS BENCHMARK FUNCTION
def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    sil = metrics.silhouette_score(
        data,
        estimator[-1].labels_,
        metric="euclidean",
        sample_size=300,
    )

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))

    return sil


# KMEANS EXECUTION
def execute_kmeans(df_final, final_df, norm_x, norm_y, df_sil, db):
    """# KMEANS Clustering"""
    print("KMEANS clustering execution")
    lim_n_x = np.take(norm_x, np.arange(0, len(final_df), dtype=int))
    lim_n_y = np.take(norm_y, np.arange(0, len(final_df), dtype=int))

    lim_n_x = lim_n_x.reshape(-1, 1)
    lim_n_y = lim_n_y.reshape(-1, 1)

    n_list = []
    km_test = np.array(1)
    for idx, val in enumerate(lim_n_x):
        if idx < len(lim_n_x) - 1:
            n_list.append([val[0], lim_n_y[idx][0]])
        else:
            km_test = np.array([list([val[0], lim_n_y[idx][0]])])

    X = np.array(n_list)

    kmeans = KMeans(init="k-means++", n_clusters=5, n_init=4)

    km_res = kmeans.fit(X)

    try:
        with open("./saved_models/kmeans_model.pkl", "wb") as f:
            pickle.dump(kmeans, f)
    except:
        with open("../saved_models/kmeans_model.pkl", "wb") as f:
            pickle.dump(kmeans, f)

    print(82 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

    # benchmark rispetto alle labels risultato di DBSCAN perché il dataset è unsupervised
    sil_score = bench_k_means(kmeans=km_res, name="PCA-based", data=X, labels=db[:-1])

    print(82 * "_")

    df_sil = df_sil.append({'silhouette': sil_score}, ignore_index=True)

    plt.figure(figsize=(16, 10))

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.001  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    plt.plot(X[:, 0], X[:, 1], "k.", markersize=3)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=180,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering\n"
        "Centroids are marked with white cross"
    )

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xticks(())
    plt.yticks(())
    try:
        plt.savefig("./plots/task3/task3_kmeans_res.png")
    except:
        plt.savefig("../plots/task3/task3_kmeans_res.png")
    plt.show()

    km_clusters = kmeans.fit_predict(X)

    df_res_km = df_final.iloc[:-1, :]
    df_res_km['cluster'] = km_clusters.tolist()

    df_res_km_2 = df_res_km[['name', 'stars', 'cluster']]

    km_pred = kmeans.predict(km_test)
    return df_res_km_2, km_pred, df_sil


# KMEANS FOR DEMO
def execute_kmeans_demo(df_final, final_df, norm_x, norm_y):
    """# KMEANS Clustering"""
    # print("KMEANS clustering execution")
    lim_n_x = np.take(norm_x, np.arange(0, len(final_df), dtype=int))
    lim_n_y = np.take(norm_y, np.arange(0, len(final_df), dtype=int))

    lim_n_x = lim_n_x.reshape(-1, 1)
    lim_n_y = lim_n_y.reshape(-1, 1)

    n_list = []
    km_test = np.array(1)
    for idx, val in enumerate(lim_n_x):
        if idx < len(lim_n_x) - 1:
            n_list.append([val[0], lim_n_y[idx][0]])
        else:
            km_test = np.array([list([val[0], lim_n_y[idx][0]])])

    X = np.array(n_list)

    kmeans = KMeans(init="k-means++", n_clusters=5, n_init=4)

    kmeans.fit(X)

    km_clusters = kmeans.fit_predict(X)

    df_res_km = df_final.iloc[:-1, :]
    df_res_km['cluster'] = km_clusters.tolist()

    df_res_km_2 = df_res_km[['name', 'stars', 'cluster']]

    # km_pred = kmeans.predict(km_test)
    return df_res_km_2, kmeans


def print_all_results(df_res_km_2, km_pred, result, df_res_2_dbs, df_sil):
    df = df_res_km_2[df_res_km_2['cluster'] == km_pred[0]].drop(['cluster'], axis=1).reset_index(drop=True)
    print("\nKMEANS result")
    print(df)

    df = result[['name', 'stars']]
    print("\nKNN result")
    print(df)

    df = df_res_2_dbs[
        df_res_2_dbs['cluster'] == df_res_2_dbs[df_res_2_dbs['name'] == "Lil Doug's Cajun Market"]['cluster'].values[
            0]][
        ['name', 'stars']].reset_index(drop=True)
    print("\nDBSCAN result")
    print(df)

    df_sil_aux = df_sil

    df_sil_aux = df_sil_aux.reindex(['KNN', 'DBSCAN', 'KMEANS'])
    df_sil_aux['silhouette'] = [df_sil['silhouette'].iloc[0], df_sil['silhouette'].iloc[1],
                                df_sil['silhouette'].iloc[2]]
    fig = df_sil_aux.plot.bar(title="Silhouette score comparison").get_figure()
    try:
        fig.savefig("./plots/task3/task3_silhouette.png")
    except:
        fig.savefig("../plots/task3/task3_silhouette.png")
    fig.show()


# RUN KMEANS FOR DEMO
def run_kmeans_restaurant_demo(x_test):
    df_final, final_df, norm_x, norm_y = load_restaurant_data()
    res_km, model = execute_kmeans_demo(df_final, final_df, norm_x, norm_y)
    km_pred = model.predict(x_test)
    df = res_km[res_km['cluster'] == km_pred[0]].drop(['cluster'], axis=1).reset_index(drop=True)
    print("\nRestaurant similar with respect to chosen features")
    print(df)


# RUN ALL
def execute_restaurant_comp():
    df_fin, fin_df, nrm_x, nrm_y = load_restaurant_data()
    res, sil = execute_knn(df_fin)
    dst, elb = elbow_method(fin_df, nrm_x, nrm_y)
    dbs_res, dbs_sil, lbl = execute_dbscan(df_fin, dst, elb, nrm_x, nrm_y, sil)
    km_res, km_pred, km_sil = execute_kmeans(df_fin, fin_df, nrm_x, nrm_y, dbs_sil, lbl)
    print_all_results(km_res, km_pred, res, dbs_res, km_sil)
