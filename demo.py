import os
import pandas as pd
import numpy as np
import ast
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from prettytable import PrettyTable
from subprocess import Popen as start
from subprocess import call
import time
import traceback

from modules.city_restaurant_grouping import run_kmeans_restaurant_demo
from modules.city_restaurant_grouping import execute_restaurant_comp
from modules.give_stars_to_review import stars_prediction_demo
from modules.give_stars_to_review import execute_stars_comp
from modules.positive_or_negative_review import bin_prediction_demo
from modules.positive_or_negative_review import execute_bin_comp
from modules.user_grouping_task import execute_kmeans_demo
from modules.user_grouping_task import execute_comp

# from compare.give_stars_to_review import stars_prediction_demo

user = str(input("Enter your username: "))
print("Welcome ", user)
time.sleep(0.5)

# Scrivo codice per eseguire dei task in base a cosa sceglie l'utente
# 1 per il primo task, 2 per il secodo, ecc...

time.sleep(0.5)


def Application(num):
    business = pd.read_csv("./data/business.csv")

    if num == 0:
        print("MAIN MENÙ")
        print("Choose option:\n"
              "1) - New Review\n"
              "2) - Restaurants Visualization\n"
              "3) - User Grouping\n"
              "4) - Methods Comparison\n"
              "0) - End Program")
        time.sleep(0.5)
        while True:
            try:
                num = int(input("Choose operation: "))
                if num < 0 or num >= 5:
                    print("Input Error! you need input 0 < num < 4")
                elif num == 1:
                    print("NEW REVIEW")
                    print(business['city'], "\n")
                    time.sleep(0.5)
                    while True:
                        try:
                            input_city = str(input("Choose a city from those listed:")).title()
                            print("\nList of Restaurant in ", input_city)
                            if input_city in list(business['city'].values):
                                print(business.loc[business.city == input_city].name)
                                time.sleep(0.5)
                                break
                            else:
                                print("The city chosen dosen't exists!\nPlease chose another city!")
                                time.sleep(0.5)
                        except Exception as ex:
                            print(ex)
                    while True:
                        try:
                            time.sleep(0.5)
                            print("Choose a restaurant in ", input_city)
                            bus_input = str(input("Enter the name of the restaurant: "))
                            time.sleep(0.5)
                            if bus_input in list(business['name'].values):
                                print("You chose, ", bus_input)
                                time.sleep(0.5)
                                while True:
                                    try:
                                        input_rev = str(input("\n\nWrite your review:\n")).capitalize()
                                        if input_rev[0].isdigit():
                                            print("Error! Don't start reviewing with a number")
                                        else:
                                            print("Prediction on review")
                                            bin_prediction_demo(input_rev)
                                            time.sleep(0.5)
                                            print("\nSuggested stars")
                                            stars_prediction_demo(input_rev)
                                            time.sleep(0.5)
                                            Application(0)
                                    except Exception as ex:
                                        print(ex)
                                break
                            else:
                                print("The restaurant doesn't exist!\nChoose a restaurant of", input_city,
                                      "from those listed")
                        except Exception as ex:
                            print("Input Error! Try Again with type string")
                            print(traceback.format_exc())
                            print(ex)
                    # time.sleep(0.5)
                    # print("1) - Back")
                    # print("0) - End Program")
                    # num = int(input("\nChoose operation: "))
                    # if num == 1:
                    #     Application(0)
                    # elif num == 0:
                    #     exit()

                elif num == 2:
                    print("RESTAURANTS VISUALIZATION")
                    print("\nBUSINESS AND CITY")
                    print(business['city'], "\n")
                    while True:
                        try:
                            input_city = str(input("Choose a city from those listed:")).title()
                            # print("\nList of Restaurants in ", input_city)
                            if input_city in list(business['city'].values):
                                # print(business.loc[business.city == input_city].name)
                                # time.sleep(0.5)
                                break
                            else:
                                print("The city does not exist!\nChoose another city")
                                # time.sleep(0.5)
                        except Exception as ex:
                            print(ex)
                    while True:
                        try:
                            businesses = pd.read_json(
                                "./data/yelp_academic_dataset_business.json", lines=True,
                                orient='columns',
                                chunksize=100000)
                            for business in businesses:
                                subset_business = business
                                break
                            city = subset_business[
                                (subset_business['city'] == input_city) & (subset_business['is_open'] == 1)]
                            city_chosed = city[
                                ['business_id', 'name', 'address', 'categories', 'attributes', 'stars']]

                            rest = city_chosed[
                                city_chosed['categories'].str.contains('Restaurant.*') == True].reset_index(
                                drop=True)

                            def extract_keys(attr, key):
                                if attr is None:
                                    return "{}"
                                if key in attr:
                                    return attr.pop(key)

                            def str_to_dict(attr):
                                if attr is not None:
                                    return ast.literal_eval(attr)
                                else:
                                    return ast.literal_eval("{}")

                            rest['BusinessParking'] = rest.apply(
                                lambda x: str_to_dict(extract_keys(x['attributes'], 'BusinessParking')), axis=1)
                            rest['Ambience'] = rest.apply(
                                lambda x: str_to_dict(extract_keys(x['attributes'], 'Ambience')),
                                axis=1)
                            rest['GoodForMeal'] = rest.apply(
                                lambda x: str_to_dict(extract_keys(x['attributes'], 'GoodForMeal')), axis=1)
                            rest['Dietary'] = rest.apply(
                                lambda x: str_to_dict(extract_keys(x['attributes'], 'Dietary')),
                                axis=1)
                            rest['Music'] = rest.apply(
                                lambda x: str_to_dict(extract_keys(x['attributes'], 'Music')), axis=1)

                            df_attr = pd.concat(
                                [rest['attributes'].apply(pd.Series), rest['BusinessParking'].apply(pd.Series),
                                 rest['Ambience'].apply(pd.Series), rest['GoodForMeal'].apply(pd.Series),
                                 rest['Dietary'].apply(pd.Series)], axis=1)

                            city = subset_business[
                                (subset_business['city'] == input_city) & (subset_business['is_open'] == 1)]
                            k_city = city[['business_id', 'name', 'address', 'categories', 'attributes', 'stars']]

                            rest = k_city[k_city['categories'].str.contains('Restaurant.*') == True].reset_index(
                                drop=True)
                            # rest = k_city[k_city['categories'] == Fdf].reset_index(drop=True)

                            df_attr_dummies = pd.get_dummies(df_attr)
                            df_categories_dummies = pd.Series(rest['categories']).str.get_dummies(',')
                            result = rest[['name', 'stars']]
                            df_final = pd.concat([df_attr_dummies, df_categories_dummies, result], axis=1)

                            print("Select Features\n")
                            pT = PrettyTable()
                            pT.field_names = ['Index', 'FEATURES']
                            list_features = list(df_final.columns)
                            list_features.remove('name')
                            list_features.remove('stars')
                            for i, u in enumerate(list_features):
                                pT.add_row([i, u])
                            print(pT)

                            while True:
                                try:
                                    features_input = input("\nChoose features from those listed: ")
                                    Flist = features_input.split(',')
                                    for el in Flist:
                                        if int(el) < 0:
                                            print(
                                                "Input Error: The number of features cannot be less than or equal to 0")
                                        elif int(el) >= len(list_features):
                                            print(
                                                "Input Error: Do not exceed the maximum number of features listed")
                                        else:
                                            Fdict = {}
                                            for i, u in enumerate(list_features):
                                                if str(i) in Flist:
                                                    # l'indice della features è in Flist, gli assegno 1 (True), altrimenti assegno 0 (False)
                                                    Fdict.update({u: 1})
                                                else:
                                                    Fdict.update({u: 0})
                                        break
                                    Fdf = pd.DataFrame(Fdict, index=[0])
                                    pd.set_option('display.max_columns', None)
                                    # print("\n\nFeatures Select\n", Fdf)
                                    break
                                except Exception as ex:
                                    print(
                                        "Input Error:  The feature doesn't exist! Choose one of the listed features")
                                    print(ex)
                                    print(traceback.format_exc())

                        except Exception as ex:
                            print(ex)
                            print(traceback.format_exc())

                        time.sleep(1)

                        # cludstering qui
                        # print("\n\nFDF\n", Fdf)
                        # input()
                        # print("\n\nCITY\n", city)
                        # input()
                        # print("\n\nK_CITY\n", k_city)
                        # input()

                        # print("\n\nDF_FINAL\n", df_final)
                        # input()

                        # df_final.drop('Restaurants', inplace=True, axis=1)
                        mapper = {1.0: 1, 1.5: 2, 2.0: 2, 2.5: 3, 3.0: 3, 3.5: 4, 4.0: 4, 4.5: 5, 5.0: 5}
                        df_final['stars'] = df_final['stars'].map(mapper)

                        """# KMeans Method computing"""
                        pca = PCA(n_components=2)
                        principalComponents = pca.fit_transform(
                            pd.concat([df_final.drop(['name', 'stars'], axis=1), Fdf], axis=0))
                        final_df = pd.DataFrame(data=principalComponents, columns=['x', 'y'])

                        x = np.array(final_df['x'].values)
                        y = np.array(final_df['y'].values)

                        def NormalizeData(data):
                            return (data - np.min(data)) / (np.max(data) - np.min(data))

                        norm_x = NormalizeData(x)
                        norm_y = NormalizeData(y)

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
                        run_kmeans_restaurant_demo(km_test)
                        # break
                        time.sleep(0.5)
                        Application(0)
                    # break
                    # time.sleep(0.5)
                    # print("1) - Back")
                    # print("0) - End Program")
                    # num = int(input("\nChoose operation: "))
                    # if num == 1:
                    #     Application(0)
                    # elif num == 0:
                    #     exit()

                elif num == 3:
                    print("USER GROUPING")
                    time.sleep(1)
                    execute_kmeans_demo()
                    time.sleep(0.5)
                    Application(0)

                elif num == 4:
                    print("CHOOSE TASK")
                    time.sleep(0.5)
                    print("1) - Positive or negative review")
                    print("2) - User grouping")
                    print("3) - Restaurant grouping")
                    print("4) - Give stars to a review")
                    print("5) - Back")
                    print("0) - End program")
                    num = int(input("\nChoose operation: "))
                    if num == 1:
                        execute_bin_comp()
                    elif num == 2:
                        execute_comp()
                    elif num == 3:
                        execute_restaurant_comp()
                    elif num == 4:
                        execute_stars_comp()
                    elif num == 5:
                        Application(0)
                    elif num == 6:
                        exit()
                    Application(0)
                elif num == 0:
                    exit()

            except ValueError:
                print("Input Error! Try with number")
                print(traceback.format_exc())


Application(0)
