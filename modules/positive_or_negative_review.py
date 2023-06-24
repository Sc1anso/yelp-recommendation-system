import re
import string
import warnings

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from keras.layers import Dense, Input, LSTM, Embedding, SpatialDropout1D, GRU
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from textblob import Word
from transformers import AutoTokenizer
from transformers import TFAutoModel
from xgboost import XGBClassifier
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

SEQ_LEN = 50
embed_size = 200
# max number of unique words
max_features = 20000
# max number of words from review to use
maxlen = 200


# UTILITY FUNCTION
def remove_stopwords(s):
    filtered_words = [word.lower() for word in s.split() if word.lower() not in set(stopwords.words('english'))]
    return " ".join(filtered_words)


# UTILITY FUNCTION FOR CUSTOM TOKENIZER
def custom_tokenizer(sentence):
    listofwords = sentence.strip().split()
    listof_words = []
    ENGLISH_STOP_WORDS = stopwords.words('english')
    for word in listofwords:
        if not word in ENGLISH_STOP_WORDS:
            lemm_word = WordNetLemmatizer().lemmatize(word)
            for punctuation_mark in string.punctuation:
                word = word.replace(punctuation_mark, '').lower()
            if len(word) > 0:
                listof_words.append(word)
    return (listof_words)


# REMOVE STOPWORDS FROM TEXT
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # removing the URL Http
    # Removal of mentions
    text = re.sub("@[^\s]*", "", text)
    # Removal of hashtags
    text = re.sub("#[^\s]*", "", text)  # converting a word to its base form
    # Removal of numbers
    text = re.sub('[0-9]*[+-:]*[0-9]+', '', text)
    text = re.sub("'s", "", text)
    text = str(text)
    return text


# LOAD REVIEWS DATA
def load_rev_bin_data():
    print("Processing data")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    try:
        df_review = pd.read_csv("./data/review.csv")
    except:
        df_review = pd.read_csv("../data/review.csv")

    df_review['date'] = df_review['date'].str.replace("\n", "")

    values, counts = np.unique(df_review['stars'], return_counts=True)

    df_review['is_positive'] = np.where(df_review['stars'] >= 3, 1, 0)

    df_review_tr = df_review
    df_review = df_review.head(100000)

    df_review_final = df_review[['text', 'is_positive']]
    df_review_final_tr = df_review_tr[['text', 'is_positive']]

    print("Cleaning text")
    df_review_final['text'] = df_review_final['text'].apply(lambda x: " ".join(word.lower() for word in str(x).split()))
    df_review_final['text'] = df_review_final['text'].str.replace('[^\w\s]', ' ')

    stop_words = ['english']
    df_review_final['text'] = df_review_final['text'].apply(
        lambda x: "  ".join(word for word in str(x).split() if word not in stop_words))
    other_stop_words = ['get', 'told', 'would', 'week', 'us', 'test', 'right', 'left', 'one', 'even',
                        'also', 'go', 'asked']
    df_review_final['text'] = df_review_final['text'].apply(
        lambda x: " ".join(word for word in str(x).split() if word not in other_stop_words))
    df_review_final['text'] = df_review_final['text'].apply(
        lambda x: " ".join(Word(word).lemmatize() for word in str(x).split()))

    print("Just cleaning...")
    df_review_final_tr['text'] = df_review_final_tr['text'].apply(
        lambda x: "  ".join(word for word in str(x).split() if word not in stop_words))
    df_review_final_tr['text'] = df_review_final_tr['text'].apply(
        lambda x: " ".join(word for word in str(x).split() if word not in other_stop_words))
    df_review_final_tr['text'] = df_review_final_tr['text'].apply(
        lambda x: " ".join(Word(word).lemmatize() for word in str(x).split()))

    # applying the cleaning function to text column
    df_review_final['text'] = df_review_final.text.map(clean_text)
    X = df_review_final["text"]
    y = df_review_final["is_positive"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    vectorizer = TfidfVectorizer(min_df=100, tokenizer=custom_tokenizer, stop_words={'english'},
                                 ngram_range=(1, 3)).fit(
        x_train)
    x_train_vectorized = vectorizer.transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)
    # SMOTE the training data
    sm = SMOTE(random_state=1)
    X_bal, y_bal = sm.fit_resample(x_train_vectorized, y_train)
    return X_bal, y_bal, x_train_vectorized, x_test_vectorized, y_train, y_test, df_review_final_tr, vectorizer


# LOGISTIC REGRESSION EXECUTION
def execute_logreg(X_bal, y_bal, x_train_vectorized, x_test_vectorized, y_train, y_test):
    print("Preprocessing data for Logistic Regression")

    # x_train_vectorized

    # new_df_words = pd.DataFrame(columns=vectorizer.get_feature_names(), data=x_train_vectorized.toarray())
    #
    # # counting the most repetitive words
    # word_counts = np.array(np.sum(x_train_vectorized, axis=0)).reshape((-1,))
    # words = np.array(vectorizer.get_feature_names())
    # words_df = pd.DataFrame({"word": words, "count": word_counts})

    logreg = LogisticRegression(C=10, class_weight=None, dual=False,
                                fit_intercept=True, intercept_scaling=1,
                                l1_ratio=None, max_iter=100,
                                multi_class='auto', n_jobs=None,
                                penalty='l2', random_state=1,
                                solver='lbfgs', tol=0.0001, verbose=0,
                                warm_start=False)
    print("Fitting LogReg model")
    logreg.fit(X_bal, y_bal)

    # Predicting the test set results
    print("Predicting...")
    y_pred_logreg = logreg.predict(x_test_vectorized)

    # Training score
    print(f"Logistic Regresscion Score on training set: {logreg.score(x_train_vectorized, y_train)}")
    print(f"Logistic Regresscion Score on test set: {logreg.score(x_test_vectorized, y_test)}")

    # Creating confusion matrix
    confunsion_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
    df_confusion_matrix_logreg = pd.DataFrame(confunsion_matrix_logreg, columns=['Predicted 0', 'Predicted 1'],
                                              index=['True 0', 'True 1'])
    print('The Classification report')
    report = classification_report(y_test, y_pred_logreg, output_dict=True)
    df_logreg_report = pd.DataFrame(report).transpose()
    return df_logreg_report, logreg


# RANDOM FOREST EXECUTION
def execute_rf(X_bal, y_bal, x_test_vectorized, y_test):
    print("Random Forest execution")
    random_forest = RandomForestClassifier(random_state=1)
    random_forest.fit(X_bal, y_bal)
    print(f"Random Forest training score: {random_forest.score(X_bal, y_bal)}")
    print(f"Random Forest test score: {random_forest.score(x_test_vectorized, y_test)}")

    print("Random Forest prediction")
    y_pred_rand = random_forest.predict(x_test_vectorized)

    # Creating confusion matrix/ dataFrame
    # confusion_matrix_rf = confusion_matrix(y_test, y_pred_rand)
    # df_confusion_matrix_rf = pd.DataFrame(confusion_matrix_rf, columns=['Predicted 0', 'Predicted 1'],
    #                                       index=['True 0', 'True 1'])

    report = classification_report(y_test, y_pred_rand, output_dict=True)
    df_rf_report = pd.DataFrame(report).transpose()
    return df_rf_report, random_forest


# XGBOOST EXECUTION
def execute_xgb(X_bal, y_bal, x_test_vectorized, y_test):
    print("XGBoost training")
    xgb_model = XGBClassifier(random_state=1)
    xgb_model.fit(X_bal, y_bal)

    print("XGBoost predicting")
    y_predict_xgb = xgb_model.predict(x_test_vectorized)

    print(f"XG Boost train score: {xgb_model.score(X_bal, y_bal)}")
    print(f"XG Boost test score: {xgb_model.score(x_test_vectorized, y_test)}")

    report = classification_report(y_test, y_predict_xgb, output_dict=True)
    df_xgb_report = pd.DataFrame(report).transpose()
    return df_xgb_report, xgb_model


# read in embeddings
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


# WORD2VEC NN EXECUTION
def execute_bin_nn(df_review_final_tr):
    """# Neural Network Model"""
    print("Neural Network data preprocessing")
    # we are using glove word vectors to get pretrained word embeddings due to huge time required instead
    try:
        embedding_file = "./data/glove.6B.200d.txt"
    except:
        embedding_file = "../data/glove.6B.200d.txt"

    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file, encoding='utf-8'))

    rev_train, rev_test = train_test_split(df_review_final_tr, test_size=0.2, random_state=4)

    rev_train = pd.get_dummies(rev_train, columns=['is_positive'])

    rev_test = pd.get_dummies(rev_test, columns=['is_positive'])

    train_samp = rev_train.sample(frac=.1, random_state=42)
    test_samp = rev_test.sample(frac=.1, random_state=42)

    class_names = ['is_positive_0', 'is_positive_1']

    y = train_samp[class_names].values
    train_samp.to_csv("./saved_models/train_samp.csv", index=False)
    print(type(train_samp))
    #input()
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_samp['text'].values))
    X_train = tokenizer.texts_to_sequences(train_samp['text'].values)
    X_test = tokenizer.texts_to_sequences(test_samp['text'].values)
    x_train = pad_sequences(X_train, maxlen=maxlen)
    x_test = pad_sequences(X_test, maxlen=maxlen)

    word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    # create a zeros matrix of the correct dimensions
    embedding_matrix = np.zeros((nb_words, embed_size))

    missed = []
    for word, i in word_index.items():
        if i >= max_features: break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            missed.append(word)

    try:
        np.save("./saved_models/emb_matrix.npy", embedding_matrix)
    except:
        np.save("../saved_models/emb_matrix.npy", embedding_matrix)
    inp = Input(shape=(maxlen,))

    # Embedding trasforma gli interi positivi (indici)i n vettori di dimensione fissa
    # Tra gli argomenti della funzione, ci sono:
    # - la dimensione dell'input (max_features);
    # - la dimensione dell'output (embed_size);
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(LSTM(40, return_sequences=True))(x)
    x = Bidirectional(GRU(40, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(2, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=outp)
    # patience is how many epochs to wait to see if val_loss will improve again.
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
    checkpoint = ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath='yelp_lstm_gru_weights.hdf5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Neural Network execution")
    try:
        try:
            model.load_weights("./saved_models/NN_bin_weights.h5")
        except:
            model.load_weights("../saved_models/NN_bin_weights.h5")
    except:
        model.fit(x_train, y, batch_size=512, epochs=20, validation_split=.1,
                  callbacks=[earlystop, checkpoint])
        try:
            model.save_weights("./saved_models/NN_bin_weights.h5")
        except:
            model.save_weights("../saved_models/NN_bin_weights.h5")
    print("Predicting on test set")
    y_test = model.predict([x_test], batch_size=1024, verbose=1)
    print("Metrics evaluation")
    model.evaluate(x_test, test_samp[class_names].values, verbose=1, batch_size=1024)

    v = metrics.classification_report(np.argmax(test_samp[class_names].values, axis=1), np.argmax(y_test, axis=1))

    report = metrics.classification_report(np.argmax(test_samp[class_names].values, axis=1), np.argmax(y_test, axis=1),
                                           output_dict=True)
    df_nn_report = pd.DataFrame(report).transpose()
    return df_nn_report, tokenizer, model


# UTILITY FUNCTION
def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels


# UTILITY FUNCTION
def map_func_test(input_ids, masks):
    return {'input_ids': input_ids, 'attention_mask': masks}


# BERT MODEL EXECUTION
def execute_bin_tr():
    """# Transformer Model"""
    print("BERT Transformer data preprocessing")
    # rev_aux = df_review_final
    try:
        rev_aux = pd.read_csv("./data/rev_final_10k.csv")
    except:
        rev_aux = pd.read_csv("../data/rev_final_10k.csv")

    tokenizer_tr = AutoTokenizer.from_pretrained('bert-base-cased')

    Xids = np.zeros((len(rev_aux), SEQ_LEN))
    Xmask = np.zeros((len(rev_aux), SEQ_LEN))

    for i, sequence in enumerate(rev_aux['text']):
        try:
            tokens = tokenizer_tr.encode_plus(sequence, max_length=SEQ_LEN,
                                              truncation=True, padding="max_length",
                                              add_special_tokens=True, return_token_type_ids=False,
                                              return_attention_mask=True, return_tensors='tf')
            Xids[i, :], Xmask[i, :] = tokens['input_ids'], tokens['attention_mask']
        except:
            print("FAIL")

    labels = np.zeros((len(rev_aux), 2))

    stars_lst = rev_aux['is_positive'].values
    stars_lst = stars_lst.astype(int)
    len_lst = np.arange(stars_lst.size)
    len_lst = len_lst.astype(int)
    stars_lst = np.subtract(stars_lst, 1)
    labels[len_lst, stars_lst] = 1

    dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

    dataset_aux = dataset
    dataset = dataset.map(map_func)
    dataset_aux = dataset_aux.map(map_func)
    dataset = dataset.shuffle(100000).batch(32)
    dataset_aux = dataset_aux.batch(32)

    DS_LEN = len(list(dataset))
    DS_LEN_aux = len(list(dataset_aux))
    SPLIT = .9

    train = dataset.take(round(DS_LEN * SPLIT))
    test = dataset.skip(round(DS_LEN * SPLIT))
    test_set = dataset_aux.skip(round(DS_LEN * SPLIT))

    m_X_ds = test_set.enumerate()  # Load into memory.

    Xids_test = np.zeros((len(test_set), SEQ_LEN))
    Xmask_test = np.zeros((len(test_set), SEQ_LEN))
    lbls_test = np.zeros((len(test_set), 2))

    for idx, val in m_X_ds:
        if idx == 0:
            Xids_test = list(val)[0]['input_ids']
            Xmask_test = list(val)[0]['attention_mask']
            lbls_test = list(val)[1]
        else:
            Xids_test = np.concatenate((Xids_test, list(val)[0]['input_ids']), axis=0)
            Xmask_test = np.concatenate((Xmask_test, list(val)[0]['attention_mask']), axis=0)
            lbls_test = np.concatenate((lbls_test, list(val)[1]), axis=0)

    print("Obtaining BERT pre-trained model")
    bert = TFAutoModel.from_pretrained('bert-base-cased')

    input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask')

    embeddings = bert(input_ids, attention_mask=mask)[0]

    X = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(128, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.Dense(32, activation='relu')(X)
    # X = tf.keras.layers.Dropout(0.1)(X)
    y = tf.keras.layers.Dense(2, activation='softmax', name='outputs')(X)

    model_tr = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

    model_tr.layers[2].trainable = False

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
    checkpoint = ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath='../yelp_bert_transformer.hdf5')
    model_tr.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("BERT Transformer execution")
    try:
        try:
            model_tr.load_weights("./saved_models/transformer_bin_mod_weights.h5")
        except:
            model_tr.load_weights("../saved_models/transformer_bin_mod_weights.h5")
    except:
        model_tr.fit(
            train,
            validation_data=test,
            epochs=100,
            callbacks=[earlystop, checkpoint]
            # callbacks=[checkpoint]
        )

        try:
            model_tr.save_weights("./saved_models/transformer_bin_mod_weights.h5")
        except:
            model_tr.save_weights("../saved_models/transformer_bin_mod_weights.h5")
    print("Predicting on test set")
    y_test = model_tr.predict(test_set, batch_size=32, verbose=1)
    print("Metrics evaluation")
    model_tr.evaluate(test_set, verbose=1, batch_size=32)

    v = metrics.classification_report(np.argmax(lbls_test, axis=1), np.argmax(y_test, axis=1))

    report = metrics.classification_report(np.argmax(lbls_test, axis=1), np.argmax(y_test, axis=1), output_dict=True)
    df_tr_report = pd.DataFrame(report).transpose()
    return df_tr_report, tokenizer_tr, model_tr


# METRICS EVALUATION
def metrics_eval_bin(df_logreg_report, df_rf_report, df_xgb_report, df_nn_report, df_tr_report):
    """# Evaluation Metrics Comparison"""

    del df_logreg_report['support']
    del df_rf_report['support']
    del df_xgb_report['support']
    del df_nn_report['support']
    del df_tr_report['support']

    print('Logistic Regression report:\n')
    print(df_logreg_report)

    print('\nRandom Forest report:\n')
    print(df_rf_report)

    print('\nXGBoost report:\n')
    print(df_xgb_report)

    print('\n Neural Network report:\n')
    print(df_nn_report)

    print('\n Bert Transformer report:\n')
    print(df_tr_report)

    # Getting scores of classification algorithms

    algorithms_label = ['Logisitc Regression', 'Random Forest', 'XGBoost', 'Neural Network']
    scores_label = ['Precision', 'Recall', 'F1-Score']
    precision_scores = []
    recall_scores = []
    f1_scores = []

    precision_score_logreg = df_logreg_report['precision']['weighted avg']
    precision_score_rf = df_rf_report['precision']['weighted avg']
    precision_score_xgb = df_xgb_report['precision']['weighted avg']
    precision_score_nn = df_nn_report['precision']['weighted avg']
    precision_score_tr = df_tr_report['precision']['weighted avg']
    precision_scores.extend([precision_score_logreg, precision_score_rf, precision_score_xgb, precision_score_nn])

    recall_score_logreg = df_logreg_report['recall']['weighted avg']
    recall_score_rf = df_rf_report['recall']['weighted avg']
    recall_score_xgb = df_xgb_report['recall']['weighted avg']
    recall_score_nn = df_nn_report['recall']['weighted avg']
    recall_score_tr = df_tr_report['recall']['weighted avg']
    recall_scores.extend([recall_score_logreg, recall_score_rf, recall_score_xgb, recall_score_nn])

    f1_score_logreg = df_logreg_report['f1-score']['weighted avg']
    f1_score_rf = df_rf_report['f1-score']['weighted avg']
    f1_score_xgb = df_xgb_report['f1-score']['weighted avg']
    f1_score_nn = df_nn_report['f1-score']['weighted avg']
    f1_score_tr = df_tr_report['f1-score']['weighted avg']
    f1_scores.extend([f1_score_logreg, f1_score_rf, f1_score_xgb, f1_score_nn])
    return precision_scores, recall_scores, f1_scores


# RUN ALL CONFIGURATION
def predict_bin_all(vectorizer, logreg, random_forest, xgb_model, tokenizer, model, tokenizer_tr, model_tr):
    """# Results"""

    input_string = "There was too noises in the resturant and the chef was very rude!!!"
    print("\nPrediction on an input string: " + input_string)
    print("Logistic Regression model:", logreg.predict(vectorizer.transform([clean_text(input_string)]))[0])
    print("Random Forest model      :", random_forest.predict(vectorizer.transform([clean_text(input_string)]))[0])
    print("XGboost model            :", xgb_model.predict(vectorizer.transform([clean_text(input_string)]))[0])

    input_review = 'There was too noises in the resturant and the chef was very rude!!!'

    Xids_in = np.zeros((1, SEQ_LEN))
    Xmask_in = np.zeros((1, SEQ_LEN))

    tokens = tokenizer_tr.encode_plus(input_review, max_length=SEQ_LEN,
                                      truncation=True, padding="max_length",
                                      add_special_tokens=True, return_token_type_ids=False,
                                      return_attention_mask=True, return_tensors='tf')

    Xids_in[0, :], Xmask_in[0, :] = tokens['input_ids'], tokens['attention_mask']

    input_test = tf.data.Dataset.from_tensor_slices((Xids_in, Xmask_in))
    input_test = input_test.map(map_func_test)
    input_test = input_test.batch(32)

    result = model_tr.predict(input_test)

    input_star = result.argmax()

    print('Transformer class.       :', input_star)

    input_review = ['There was too noises in the resturant and the chef was very rude!!!']

    input = tokenizer.texts_to_sequences(input_review)
    input = pad_sequences(input, maxlen=maxlen)
    result = model.predict(input)
    input_star = result.argmax()

    print('NN classification        :', input_star)
    print(82 * "_")
    input_string = "Magnificent place, very good food and friendly workers"
    print("\nPrediction on an input string: " + input_string)
    print("Logistic Regression model:", logreg.predict(vectorizer.transform([clean_text(input_string)]))[0])
    print("Random Forest model      :", random_forest.predict(vectorizer.transform([clean_text(input_string)]))[0])
    print("XGboost model            :", xgb_model.predict(vectorizer.transform([clean_text(input_string)]))[0])

    input_review = 'Magnificent place, very good food and friendly workers'

    Xids_in = np.zeros((1, SEQ_LEN))
    Xmask_in = np.zeros((1, SEQ_LEN))

    tokens = tokenizer_tr.encode_plus(input_review, max_length=SEQ_LEN,
                                      truncation=True, padding="max_length",
                                      add_special_tokens=True, return_token_type_ids=False,
                                      return_attention_mask=True, return_tensors='tf')

    Xids_in[0, :], Xmask_in[0, :] = tokens['input_ids'], tokens['attention_mask']

    input_test = tf.data.Dataset.from_tensor_slices((Xids_in, Xmask_in))
    input_test = input_test.map(map_func_test)
    # input_test = input_test.shuffle(100000).batch(32)
    input_test = input_test.batch(32)

    result = model_tr.predict(input_test)

    input_star = result.argmax()

    print('Transformer class.       :', input_star)

    input_review = ['Magnificent place, very good food and friendly workers']

    input = tokenizer.texts_to_sequences(input_review)
    input = pad_sequences(input, maxlen=maxlen)
    result = model.predict(input)
    input_star = result.argmax()

    print('NN classification        :', input_star)


# RUN BEST CONFIGURATION
def predict_bin_best(tokenizer, model):
    input_review = ['There was too noises in the resturant and the chef was very rude!!!']
    print("\nPrediction on an input string: " + input_review[0])
    input = tokenizer.texts_to_sequences(input_review)
    input = pad_sequences(input, maxlen=maxlen)
    result = model.predict(input)
    input_star = result.argmax()

    print('NN classification        :', input_star)
    input_review = ['Magnificent place, very good food and friendly workers']
    print("\nPrediction on an input string: " + input_review[0])
    input = tokenizer.texts_to_sequences(input_review)
    input = pad_sequences(input, maxlen=maxlen)
    result = model.predict(input)
    input_star = result.argmax()

    print('NN classification        :', input_star)


# METRICS COMPARISON
def metrics_comparison_bin(precision_scores, recall_scores, f1_scores):
    metrics_dict = {'precision': precision_scores, 'recall': recall_scores, 'f1-score': f1_scores}
    df_scores = pd.DataFrame(metrics_dict)
    df_scores_aux = df_scores
    df_scores_aux = df_scores_aux.reindex(['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network'])

    df_scores_aux['precision'] = [df_scores['precision'].iloc[0], df_scores['precision'].iloc[1],
                                  df_scores['precision'].iloc[2], df_scores['precision'].iloc[3]]
    df_scores_aux['recall'] = [df_scores['recall'].iloc[0], df_scores['recall'].iloc[1],
                               df_scores['recall'].iloc[2], df_scores['recall'].iloc[3]]
    df_scores_aux['f1-score'] = [df_scores['f1-score'].iloc[0], df_scores['f1-score'].iloc[1],
                                 df_scores['f1-score'].iloc[2], df_scores['f1-score'].iloc[3]]

    fig1 = df_scores_aux['precision'].plot.bar(title= "Precision score comparison").get_figure()
    fig2 = df_scores_aux['recall'].plot.bar(title= "Recall score comparison").get_figure()
    fig3 = df_scores_aux['f1-score'].plot.bar(title= "F1 score comparison").get_figure()
    try:
        fig1.savefig("./plots/task1/task1_precision.png")
        fig2.savefig("./plots/task1/task1_recall.png")
        fig3.savefig("./plots/task1/task1_f1_score.png")
    except:
        fig1.savefig("../plots/task1/task1_precision.png")
        fig2.savefig("../plots/task1/task1_recall.png")
        fig3.savefig("../plots/task1/task1_f1_score.png")


# EXECUTION FOR DEMO
def bin_prediction_demo(input_review):
    try:
        embedding_matrix = np.load("../saved_models/emb_matrix.npy")
    except:
        embedding_matrix = np.load("./saved_models/emb_matrix.npy")
    try:
        train_samp = pd.read_csv("../saved_models/train_samp.csv")
    except:
        train_samp = pd.read_csv("./saved_models/train_samp.csv")

    train_samp['text'] = train_samp['text'].astype(str)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_samp['text'].values))

    inp = Input(shape=(maxlen,))

    # Embedding trasforma gli interi positivi (indici) in vettori di dimensione fissa
    # Tra gli argomenti della funzione, ci sono:
    # - la dimensione dell'input (max_features);
    # - la dimensione dell'output (embed_size);
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(LSTM(40, return_sequences=True))(x)
    x = Bidirectional(GRU(40, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(2, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=outp)
    # patience is how many epochs to wait to see if val_loss will improve again.
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
    checkpoint = ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath='yelp_lstm_gru_weights.hdf5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    try:
        model.load_weights("./saved_models/NN_bin_weights.h5")
    except:
        model.load_weights("../saved_models/NN_bin_weights.h5")

    input_review = [input_review]
    # print("\nPrediction on an input string: " + input_review[0])
    input_str = tokenizer.texts_to_sequences(input_review)
    input_str = pad_sequences(input_str, maxlen=maxlen)
    result = model.predict(input_str, verbose=0)
    input_star = result.argmax()

    print('Review is : ', "positive" if input_star == 1 else "negative")


# EXECUTE ALL
def execute_bin_comp():
    X_bal, y_bal, x_train_vectorized, x_test_vectorized, y_train, y_test, df_review_final_tr,vectorizer = load_rev_bin_data()
    lg_report, model_lg = execute_logreg(X_bal, y_bal, x_train_vectorized, x_test_vectorized, y_train, y_test)
    rf_report, model_rf = execute_rf(X_bal, y_bal, x_test_vectorized, y_test)
    xgb_report, model_xgb = execute_xgb(X_bal, y_bal, x_test_vectorized, y_test)
    nn_report, tokenizer_nn, model_nn = execute_bin_nn(df_review_final_tr)
    tr_report, tokenizer_tr, model_tr = execute_bin_tr()
    predict_bin_all(vectorizer, model_lg, model_rf, model_xgb, tokenizer_nn, model_nn, tokenizer_tr, model_tr)
    # predict_bin_best(tokenizer_nn, model_nn)
    precision, recall, f1 = metrics_eval_bin(lg_report, rf_report, xgb_report, nn_report, tr_report)
    metrics_comparison_bin(precision, recall, f1)
