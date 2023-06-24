from time import time

import matplotlib.pyplot as plt
# Importo un elenco di stop-words.
import nltk
import math
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
# print("Processing data")

cluster_tags = {}


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


# FUNZIONE PER NORMALIZZARE DATI
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# CARICAMENTO DATI UTENTI
def load_user_data():
    try:
        file = pd.read_csv("./data/user.csv")
    except:
        file = pd.read_csv("../data/user.csv")
    df = pd.DataFrame(file)

    pca = PCA(n_components=2)

    df_dropped = df.drop(['name', 'yelping_since', 'elite', 'friends', 'Unnamed: 0', 'user_id'], axis=1)
    # df_dropped_1 = df_dropped[['review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars']]
    # df_dropped_x = df_dropped[['review_count', 'useful', 'funny', 'cool', 'fans']]
    # df_dropped_y = df_dropped['average_stars']

    principalComponents = pca.fit_transform(df_dropped)

    final_df = pd.DataFrame(data=principalComponents, columns=['x', 'y'])

    final_df.plot(kind='scatter', x=0, y=1, s=1)

    x = np.array(final_df['x'].values)
    y = np.array(final_df['y'].values)

    norm_x = NormalizeData(x)
    norm_y = NormalizeData(y)
    norm_x = norm_x.reshape(-1, 1)
    norm_y = norm_y.reshape(-1, 1)
    return norm_x, norm_y, df


# CALCOLO ELBOW METHOD
def elbow_method(norm_x, norm_y):
    final_dst = []
    try:
        try:
            df_dist = pd.read_csv('./data/distances_IA.csv', header=None)
        except:
            df_dist = pd.read_csv('../data/distances_IA.csv', header=None)
        split = df_dist.head(30100)
        final_dst = split[0].values.tolist()
    except:
        size = 10000
        for idx, val in enumerate(norm_x):
            if idx < size:
                old_dst = 0.1
                dst = []
                for idx2, val2 in enumerate(norm_x):
                    dist = math.sqrt(pow(norm_x[idx2] - norm_x[idx], 2) + pow(norm_y[idx2] - norm_y[idx], 2))
                    if idx != idx2 and dist < old_dst:
                        dst.append(dist)
                        old_dst = dist
                dst.sort()
                final_dst.append(dst[0])
                # print("\rProgress ---> " + str(idx) + "/" + str(size), end="")
    final_dst.sort()
    # plt.plot(final_dst)
    """# Elbow method"""
    # print("Elbow method computing")
    x_for_plot = np.arange(0, len(final_dst), dtype=int)
    kneedle = KneeLocator(x_for_plot, final_dst, S=1.0, curve="convex", direction="increasing")
    # kneedle.plot_knee()
    lim_n_x = np.take(norm_x, np.arange(0, len(final_dst), dtype=int))
    lim_n_y = np.take(norm_y, np.arange(0, len(final_dst), dtype=int))

    lim_n_x = lim_n_x.reshape(-1, 1)
    lim_n_y = lim_n_y.reshape(-1, 1)

    n_list = []
    for idx, val in enumerate(lim_n_x):
        n_list.append([val[0], lim_n_y[idx][0]])

    X = np.array(n_list)
    return X, final_dst, kneedle.elbow


# ESECUZIONE DBSCAN
def execute_dbscan(X, final_dst, elbow, df):
    """# DBSCAN Clustering"""
    print("DBSCAN clustering")

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
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print(
    #    "Adjusted Mutual Information: %0.3f"
    #    % metrics.adjusted_mutual_info_score(labels_true, labels)
    # )
    print("DBSCAN Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
    df_sil = pd.DataFrame([metrics.silhouette_score(X, labels)], columns=['silhouette'])

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
        plt.savefig("./plots/task2/task2_dbscan_res.png")
    except:
        plt.savefig("../plots/task2/task2_dbscan_res.png")
    plt.show()

    db = dbscan.fit_predict(X)

    df_res = df.head(30100)
    df_res['cluster'] = db.tolist()

    df_res = df_res.drop('Unnamed: 0', axis=1)
    df_res_2 = df_res.drop(['user_id', 'name', 'elite', 'friends', 'yelping_since'], axis=1)

    df_data_final = pd.DataFrame()
    for i in range(-1, n_cl):
        df_descr = df_res_2[df_res_2['cluster'] == i].describe()
        df_descr = df_descr.drop('cluster', axis=1)
        # print(df_descr)
        # print("@@@@@@@@@@@@@@@@@@  MEDIANA  @@@@@@@@@@@@@@@@@@@")
        df_median = df_res_2[df_res_2['cluster'] == i].median()
        df_median = df_median.drop('cluster')
        indexes = df_median.index.values.tolist()
        # print(indexes)
        # input()
        val_dict = {}
        for col in indexes:
            val_dict.update({col: [df_median[col]]})
        # print(val_dict)
        df_med_def = pd.DataFrame(val_dict, index=['median'])
        df_merged = pd.concat([df_descr, df_med_def])
        cluster_lst = [i] * 9
        df_merged['cluster'] = cluster_lst
        df_data_final = pd.concat([df_data_final, df_merged])

    for i in range(-1, n_cl):
        df_to_plot = df_data_final[df_data_final['cluster'] == i]
        df_to_plot = df_to_plot.drop('cluster', axis=1)
        df_to_plot = df_to_plot.drop('count', axis=0)
        title_text = ""
        if i == -1:
            title_text = "DBSCAN - Characteristics of outliers"
        else:
            title_text = "DBSCAN - Characteristics of cluster number: " + str(i)

        fig = df_to_plot.plot.bar(title=title_text, figsize=(25, 10)).get_figure()
        try:
            fig.savefig("./plots/task2/task2_dbscan_cl_" + str(i) + ".png")
        except:
            fig.savefig("../plots/task2/task2_dbscan_cl_" + str(i) + ".png")
    return db, df_sil


#ESECUZIONE KMEANS
def execute_kmeans(X, db, df_sil, df):
    """# K-Means Clustering

    Benchmark function definition
    """
    print("KMEANS clustering")

    kmeans = KMeans(init="k-means++", n_clusters=6, n_init=4)
    # silhouette 984 -- 8

    km_res = kmeans.fit(X)

    print(82 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

    # benchmark rispetto alle labels risultato di DBSCAN perché il dataset è unsupervised
    sil_score = bench_k_means(kmeans=km_res, name="PCA-based", data=X, labels=db)

    print(82 * "_")

    df_sil = df_sil.append({'silhouette': sil_score}, ignore_index=True)
    df_sil_aux = df_sil

    df_sil_aux = df_sil_aux.reindex(['DBSCAN', 'KMEANS'])
    df_sil_aux['silhouette'] = [df_sil['silhouette'].iloc[0], df_sil['silhouette'].iloc[1]]

    """# Plotting KMeans results"""

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

    plt.xlim(-0.02, 1.05)
    plt.ylim(-0.03, 0.85)
    plt.xticks(())
    plt.yticks(())
    try:
        plt.savefig("./plots/task2/task2_kmeans_res.png")
    except:
        plt.savefig("../plots/task2/task2_kmeans_res.png")
    plt.show()

    km_clusters = kmeans.fit_predict(X)

    df_res_km = df.head(30100)
    df_res_km['cluster'] = km_clusters.tolist()

    df_res_km = df_res_km.drop('Unnamed: 0', axis=1)
    df_res_km_2 = df_res_km.drop(['user_id', 'name', 'elite', 'friends', 'yelping_since'], axis=1)

    df_data_final_km = pd.DataFrame()
    for i in range(6):
        df_descr_km = df_res_km_2[df_res_km_2['cluster'] == i].describe()
        df_descr_km = df_descr_km.drop('cluster', axis=1)
        df_median_km = df_res_km_2[df_res_km_2['cluster'] == i].median()
        df_median_km = df_median_km.drop('cluster')
        indexes_km = df_median_km.index.values.tolist()
        val_dict = {}
        for col in indexes_km:
            val_dict.update({col: [df_median_km[col]]})
        df_med_def_km = pd.DataFrame(val_dict, index=['median'])
        df_merged_km = pd.concat([df_descr_km, df_med_def_km])
        cluster_lst_km = [i] * 9
        df_merged_km['cluster'] = cluster_lst_km
        df_data_final_km = pd.concat([df_data_final_km, df_merged_km])

    for i in range(6):
        df_to_plot_km = df_data_final_km[df_data_final_km['cluster'] == i]
        df_to_plot_km = df_to_plot_km.drop('cluster', axis=1)
        df_to_plot_km = df_to_plot_km.drop('count', axis=0)
        title_text_km = ""
        if i == -1:
            title_text_km = "Characteristics of outliers"
        else:
            title_text_km = "KMEANS - Characteristics of cluster number: " + str(i)

        fig = df_to_plot_km.plot.bar(title=title_text_km, figsize=(25, 10)).get_figure()
        try:
            fig.savefig("./plots/task2/task2_kmeans_cl_" + str(i) + ".png")
        except:
            fig.savefig("../plots/task2/task2_kmeans_cl_" + str(i) + ".png")

    fig2 = df_sil_aux.plot.bar(title="Silhouette score comparison").get_figure()
    try:
        fig2.savefig("./plots/task2/task2_silhouette.png")
    except:
        fig2.savefig("../plots/task2/task2_silhouette.png")


# ESECUZIONE KMEANS PER DEMO
def execute_kmeans_demo():
    norm_x, norm_y, df = load_user_data()
    X, _, _ = elbow_method(norm_x, norm_y)
    """# K-Means Clustering

    Benchmark function definition
    """
    # print("KMEANS clustering")

    kmeans = KMeans(init="k-means++", n_clusters=6, n_init=4)
    # silhouette 984 -- 8

    kmeans.fit(X)

    km_clusters = kmeans.fit_predict(X)

    df_res_km = df.head(30100)
    df_res_km['cluster'] = km_clusters.tolist()

    df_res_km = df_res_km.drop('Unnamed: 0', axis=1)
    # 'name',
    df_res_km_2 = df_res_km.drop(['user_id', 'elite', 'friends', 'yelping_since'], axis=1)

    cluster_tags_aux = {0: 'New or not active users', 1: 'Users with a huge number of votes sent (up to 200000)',
                        2: 'Users with medium number of votes sent (up to 45000)',
                        3: 'Users with a high number of votes sent (up to 100000) and good number of votes receipt',
                        4: 'Users with low number of votes sent (up to 17500)',
                        5: 'Users with medium number of votes sent (up to 45000) an high number of votes receipt'}

    for i in range(6):
        df_descr_km = df_res_km_2[df_res_km_2['cluster'] == i].head(20)
        df_descr_km = df_descr_km.drop(['cluster',
                                        'review_count',
                                        'funny',
                                        'cool',
                                        'compliment_more',
                                        'compliment_profile',
                                        'compliment_cute',
                                        'compliment_list',
                                        'compliment_note',
                                        'compliment_plain',
                                        'compliment_cool',
                                        'compliment_funny',
                                        'compliment_writer',
                                        'compliment_photos'], axis=1)
        if df_descr_km['useful'].max() > 107000.0:
            cluster_tags.update({i: cluster_tags_aux[1]})
        elif 45000.0 < df_descr_km['useful'].max() <= 107000.0:
            cluster_tags.update({i: cluster_tags_aux[3]})
        elif 40000.0 < df_descr_km['useful'].max() <= 45000.0:
            cluster_tags.update({i: cluster_tags_aux[2]})
        elif 28000.0 < df_descr_km['useful'].max() <= 40000.0:
            cluster_tags.update({i: cluster_tags_aux[5]})
        elif 6500.0 < df_descr_km['useful'].max() <= 28000.0:
            cluster_tags.update({i: cluster_tags_aux[4]})
        elif df_descr_km['useful'].max() <= 6500.0:
            cluster_tags.update({i: cluster_tags_aux[0]})
        # print(cluster_tags)
        # print(df_descr_km['useful'].max())
        print("Sample of ", cluster_tags[i])
        print(df_descr_km)


# ESECUZIONE COMPARAZIONE ALGORITMI
def execute_comp():
    n_x, n_y, df_comp = load_user_data()
    X_in, f_dst, elbow = elbow_method(n_x, n_y)
    lbl, sil = execute_dbscan(X_in, f_dst, elbow, df_comp)
    execute_kmeans(X_in, lbl, sil, df_comp)

# execute_kmeans_demo()
