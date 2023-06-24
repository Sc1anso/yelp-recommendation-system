import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from keras.layers import Dense, Input, LSTM, Embedding, SpatialDropout1D, GRU
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from scipy import sparse
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import TFAutoModel
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


'''HYPERPARAMETERS'''
# max_features is an upper bound on the number of words in the vocabulary
max_features = 20000
embed_size = 200
# max number of words from review to use
maxlen = 200
SEQ_LEN = 50


# UTILITY FUNCTION
def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels


# UTILITY FUNCTION
def map_func_test(input_ids, masks):
    return {'input_ids': input_ids, 'attention_mask': masks}


# read in embeddings
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


# LOAD REVIEWS DATA
def load_rev_data():
    print("Processing data")
    try:
        df_review = pd.read_csv('./data/review.csv')
        df_business = pd.read_csv('./data/business.csv')
    except:
        df_review = pd.read_csv('../data/review.csv')
        df_business = pd.read_csv('../data/business.csv')

    business = df_business[df_business['categories'].str.contains('Restaurant') == True]

    rev = df_review[df_review.business_id.isin(business['business_id']) == True]

    rev_train, rev_test = train_test_split(rev, test_size=0.2, random_state=4)

    rev_train = rev_train[['text', 'stars']]
    rev_bert = rev[['text', 'stars']].head(10000)

    rev_train = pd.get_dummies(rev_train, columns=['stars'])

    rev_test = rev_test[['text', 'stars']]
    rev_test = pd.get_dummies(rev_test, columns=['stars'])

    return rev_train, rev_test, rev_bert


# Implmenting NB Linear Model (Negative Binomial Regression Model)
class NBFeatures(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha  # Smoothing Parameter

    def preprocess_x(self, x, r):
        return x.multiply(r)

    # to calculate probabilities
    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + self.alpha) / ((y == y_i).sum() + self.alpha)

    # to calculate the log ratio and represent as sparse matrix
    def fit(self, x, y=None):
        self._r = sparse.csr_matrix(np.log(self.pr(x, 1, y) / self.pr(x, 0, y)))
        return self

    # to apply the nb fit to original features x
    def transform(self, x):
        x_nb = self.preprocess_x(x, self._r)
        return x_nb


# NAIVE BAYES MODEL
def create_nb_pipeline():
    # Create pipeline using sklearn pipeline:
    # I basically create my tfidf features which are fed to my NB model
    # for probability calculations. Then those are fed as input to my
    # logistic regression model.
    tfidf = TfidfVectorizer(max_features=max_features)
    lr = LogisticRegression()
    nb = NBFeatures(1)
    p = Pipeline([
        ('tfidf', tfidf),
        ('nb', nb),
        ('lr', lr)
    ])

    return p


def execute_naive_bayes(train_samp, test_samp, p):  # p = create_nb_pipeline
    # set frac = .1 to use the entire sample
    # train_samp, test_samp, _ = load_rev_data()
    train_samp = train_samp.sample(frac=.1, random_state=42)
    test_samp = test_samp.sample(frac=.1, random_state=42)

    class_names = ['stars_1.0', 'stars_2.0', 'stars_3.0', 'stars_4.0', 'stars_5.0']
    scores = []
    predictions = np.zeros((len(test_samp), len(class_names)))
    print("Naive Bayes Classification computing")
    for i, class_name in enumerate(class_names):
        train_target = train_samp[class_name]
        train_target = np.array(train_target)
        cv_score = np.mean(cross_val_score(estimator=p, X=train_samp['text'].values,
                                           y=train_target, cv=3, scoring='accuracy'))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))
        p.fit(train_samp['text'].values, train_target)
        # print(test_samp['text'].values)
        predictions[:, i] = p.predict_proba(test_samp['text'].values)[:, 1]

    report = metrics.classification_report(np.argmax(test_samp[class_names].values, axis=1),
                                           np.argmax(predictions, axis=1), output_dict=True)
    df_nb_report = pd.DataFrame(report).transpose()

    return df_nb_report


# WORD2VEC NN EXECUTION
def execute_word2vec_nn(train_samp, test_samp):
    print("Neural Network data preprocessing")
    # we are using glove word vectors to get pretrained word embeddings due to huge time required instead
    try:
        embedding_file = './data/glove.6B.200d.txt'
    except:
        embedding_file = '../data/glove.6B.200d.txt'

    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file, encoding='utf-8'))

    class_names = ['stars_1.0', 'stars_2.0', 'stars_3.0', 'stars_4.0', 'stars_5.0']
    train_samp = train_samp.sample(frac=.1, random_state=42)
    test_samp = test_samp.sample(frac=.1, random_state=42)
    # Splitting off my y variable
    y = train_samp[class_names].values

    tokenizer_nn = Tokenizer(num_words=max_features)
    tokenizer_nn.fit_on_texts(list(train_samp['text'].values))
    X_train = tokenizer_nn.texts_to_sequences(train_samp['text'].values)
    X_test = tokenizer_nn.texts_to_sequences(test_samp['text'].values)
    x_train = pad_sequences(X_train, maxlen=maxlen)
    x_test = pad_sequences(X_test, maxlen=maxlen)

    word_index = tokenizer_nn.word_index
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

    inp = Input(shape=(maxlen,))

    # Embedding trasforma gli interi positivi (indici)i n vettori di dimensione fissa
    # Tra gli argometni della funzione, ci sono:
    # - la dimensione dell'input (max_features);
    # - la dimensione dell'output (embed_size);
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(LSTM(40, return_sequences=True))(x)
    x = Bidirectional(GRU(40, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(5, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=outp)
    # patience is how many epochs to wait to see if val_loss will improve again.
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
    checkpoint = ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath='yelp_lstm_gru_weights.hdf5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Neural Network execution")
    try:
        try:
            model.load_weights('./saved_models/NN_weights.h5')
        except:
            model.load_weights('../saved_models/NN_weights.h5')
    except:
        model.fit(x_train, y, batch_size=512, epochs=20, validation_split=.1,
                  callbacks=[earlystop, checkpoint])
        try:
            model.save_weights('./saved_models/NN_weights.h5')
        except:
            model.save_weights('../saved_models/NN_weights.h5')

    print("Predicting on test set")
    y_test = model.predict([x_test], batch_size=1024, verbose=1)

    print("Metrics evaluation")
    model.evaluate(x_test, test_samp[class_names].values, verbose=1, batch_size=1024)

    report = metrics.classification_report(np.argmax(test_samp[class_names].values, axis=1), np.argmax(y_test, axis=1),
                                           output_dict=True)
    df_nn_report = pd.DataFrame(report).transpose()

    return df_nn_report, model


# BERT EXECUTION
def execute_bert(rev_bert):
    """# Transformer Model"""
    print("BERT Transformer data preprocessing")
    print(len(rev_bert))
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    Xids = np.zeros((len(rev_bert), SEQ_LEN))
    Xmask = np.zeros((len(rev_bert), SEQ_LEN))

    failed = []
    for i, sequence in enumerate(rev_bert['text']):
        try:
            tokens = tokenizer.encode_plus(sequence, max_length=SEQ_LEN,
                                           truncation=True, padding="max_length",
                                           add_special_tokens=True, return_token_type_ids=False,
                                           return_attention_mask=True, return_tensors='tf')
            Xids[i, :], Xmask[i, :] = tokens['input_ids'], tokens['attention_mask']
        except:
            failed.append(i)
        print("\rProgress: " + str(i + 1) + "/" + str(len(rev_bert)), end="")
    print("\n")

    labels = np.zeros((len(rev_bert), 5))

    stars_lst = rev_bert['stars'].values
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

    DS_LEN = (round(len(list(dataset)) * 80) / 100)
    TS_LEN = len(list(dataset)) - DS_LEN

    SPLIT = .9

    train = dataset.take(round(DS_LEN * SPLIT))
    test = dataset.skip(round(DS_LEN * SPLIT))
    test_set = dataset_aux.skip(round(TS_LEN))

    # TEST SET CREATION
    m_X_ds = test_set.enumerate()
    Xids_test = np.zeros((len(test), SEQ_LEN))
    Xmask_test = np.zeros((len(test), SEQ_LEN))
    lbls_test = np.zeros((len(test), 5))

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
    tf.config.experimental.list_physical_devices('GPU')
    bert = TFAutoModel.from_pretrained('bert-base-cased')

    input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask')

    embeddings = bert(input_ids, attention_mask=mask)[0]

    X = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(128, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.Dense(32, activation='relu')(X)
    y = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(X)

    model_tr = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

    model_tr.layers[2].trainable = False

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
    checkpoint = ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath='../yelp_bert_transformer.hdf5')
    model_tr.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("BERT Transformer execution")
    try:
        try:
            model_tr.load_weights('./saved_models/transformer_weights.h5')
        except:
            model_tr.load_weights('../saved_models/transformer_weights.h5')
    except:
        # model_tr.load_weights('../saved_models/transformer_weights.h5')
        model_tr.fit(
            train,
            validation_data=test,
            epochs=20,
            callbacks=[earlystop, checkpoint]
        )
        try:
            model_tr.save_weights('./saved_models/transformer_weights.h5')
        except:
            model_tr.save_weights('../saved_models/transformer_weights.h5')

    print("Predicting on test set")
    y_test = model_tr.predict(test_set, batch_size=32, verbose=1)
    print("Metrics evaluation")
    model_tr.evaluate(test_set, verbose=1, batch_size=32)

    report = metrics.classification_report(np.argmax(lbls_test, axis=1), np.argmax(y_test, axis=1), output_dict=True)
    df_tr_report = pd.DataFrame(report).transpose()

    return df_tr_report, model_tr


# METRICS COMPARISON
def metrics_comparison(df_nb_report, df_nn_report, df_tr_report):
    """# Evaluation metrics comparison"""
    del df_nb_report['support']
    del df_nn_report['support']
    del df_tr_report['support']

    print('Naive Bayes report:\n')
    print(df_nb_report)

    print('\n Neural Network report:\n')
    print(df_nn_report)

    print('\n Bert Transformer report:\n')
    print(df_tr_report)

    # Getting scores of classification algorithms
    precision_scores = []
    recall_scores = []
    f1_scores = []

    precision_score_nb = df_nb_report['precision']['weighted avg']
    precision_score_nn = df_nn_report['precision']['weighted avg']
    precision_score_tr = df_tr_report['precision']['weighted avg']
    precision_scores.extend([precision_score_nb, precision_score_nn, precision_score_tr])

    recall_score_nb = df_nb_report['recall']['weighted avg']
    recall_score_nn = df_nn_report['recall']['weighted avg']
    recall_score_tr = df_tr_report['recall']['weighted avg']
    recall_scores.extend([recall_score_nb, recall_score_nn, recall_score_tr])

    f1_score_nb = df_nb_report['f1-score']['weighted avg']
    f1_score_nn = df_nn_report['f1-score']['weighted avg']
    f1_score_tr = df_tr_report['f1-score']['weighted avg']
    f1_scores.extend([f1_score_nb, f1_score_nn, f1_score_tr])

    return precision_scores, recall_scores, f1_scores


# RUN ALL
def predict_all(train_samp, p, model, model_tr):
    class_names = ['stars_1.0', 'stars_2.0', 'stars_3.0', 'stars_4.0', 'stars_5.0']
    tokenizer_nn = Tokenizer(num_words=max_features)
    train_samp = train_samp.sample(frac=.1, random_state=42)
    tokenizer_nn.fit_on_texts(list(train_samp['text'].values))
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    input_revs = ['There was too noises in the resturant and the chef was very rude!!!',
                  'fantastic restaurant!!! very good food but services was bad',
                  'fantastic restaurant!!!',
                  'ordinary restaurant, quite good food',
                  'ordinary restaurant, quite good food, no service. fast-food like']
    for input_review in input_revs:
        print("\n\n")
        print(82 * "_")
        # print("\nNaive Bayes prediction")
        # input_review = "There was too noises in the resturant and the chef was very rude!!!"
        # input_review = 'fantastic restaurant!!! very good food but services was bad'
        # input_review = 'fantastic restaurant!!!'
        # input_review = 'ordinary restaurant, quite good food, no service. fast-food like'
        print("\nPrediction on an input string: " + input_review)
        input_ar = np.array([input_review])
        # print(input)
        # predicted = p.predict_proba(input)

        predictions = np.zeros((len(input_ar), len(class_names)))
        for i, class_name in enumerate(class_names):
            train_target = train_samp[class_name]
            train_target = np.array(train_target)
            p.fit(train_samp['text'].values, train_target)
            predictions[:, i] = p.predict_proba(input_ar)[:, 1]

        input_star = np.argmax(predictions) + 1

        print('\nNaive Bayes prediction:', "\u2B50" * (input_star), 'stars')
        print(82 * "_")
        # print("\nNeural Network prediction")
        # input_review = ['There was too noises in the resturant and the chef was very rude!!!']
        # input_review = ['fantastic restaurant!!! very good food but services was bad']
        # input_review = ['fantastic restaurant!!!']
        # print("\nPrediction on an input string: " + input_review)
        input = tokenizer_nn.texts_to_sequences([input_review])
        input = pad_sequences(input, maxlen=maxlen)
        result = model.predict(input)
        input_star = result.argmax() + 1

        print('\nNeural Network prediction:', "\u2B50" * (input_star), 'stars')
        print(82 * "_")
        # print("\nTransformer prediction")
        # input_review = "There was too noises in the resturant and the chef was very rude!!!"
        # input_review = 'fantastic restaurant!!! very good food but services was bad'
        # input_review = 'fantastic restaurant!!!'
        # print("\nPrediction on an input string: " + input_review)
        Xids_in = np.zeros((1, SEQ_LEN))
        Xmask_in = np.zeros((1, SEQ_LEN))

        tokens = tokenizer.encode_plus(input_review, max_length=SEQ_LEN,
                                       truncation=True, padding="max_length",
                                       add_special_tokens=True, return_token_type_ids=False,
                                       return_attention_mask=True, return_tensors='tf')

        Xids_in[0, :], Xmask_in[0, :] = tokens['input_ids'], tokens['attention_mask']

        input_test = tf.data.Dataset.from_tensor_slices((Xids_in, Xmask_in))
        input_test = input_test.map(map_func_test)
        input_test = input_test.batch(32)

        result = model_tr.predict(input_test)

        input_star = result.argmax() + 1

        print('\nTransformer prediction:', "\u2B50" * input_star, 'stars')


# RUN BEST CONFIGURATION
def predict_best(model_tr):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    print(82 * "_")
    print("\nTransformer prediction")
    input_review = 'ordinary restaurant, quite good food, no service. fast-food like'
    # input_review = "There was too noises in the resturant and the chef was very rude!!!"
    # input_review = 'fantastic restaurant!!! very good food but services was bad'
    # input_review = 'fantastic restaurant!!!'
    print("\nPrediction on an input string: " + input_review)
    Xids_in = np.zeros((1, SEQ_LEN))
    Xmask_in = np.zeros((1, SEQ_LEN))

    tokens = tokenizer.encode_plus(input_review, max_length=SEQ_LEN,
                                   truncation=True, padding="max_length",
                                   add_special_tokens=True, return_token_type_ids=False,
                                   return_attention_mask=True, return_tensors='tf')

    Xids_in[0, :], Xmask_in[0, :] = tokens['input_ids'], tokens['attention_mask']

    input_test = tf.data.Dataset.from_tensor_slices((Xids_in, Xmask_in))
    input_test = input_test.map(map_func_test)
    input_test = input_test.batch(32)

    result = model_tr.predict(input_test)

    input_star = result.argmax() + 1

    print('\nYour review should have', "\u2B50" * input_star, 'stars')


# PLOT METRICS
def plot_metrics_comparison(precision_scores, recall_scores, f1_scores):
    metrics_dict = {'precision': precision_scores, 'recall': recall_scores, 'f1-score': f1_scores}
    df_scores = pd.DataFrame(metrics_dict)
    df_scores_aux = df_scores
    df_scores_aux = df_scores_aux.reindex(['Naive Bayes', 'Neural Network', 'Transformer'])
    df_scores_aux['precision'] = [df_scores['precision'].iloc[0], df_scores['precision'].iloc[1],
                                  df_scores['precision'].iloc[2]]
    df_scores_aux['recall'] = [df_scores['recall'].iloc[0], df_scores['recall'].iloc[1], df_scores['recall'].iloc[2]]
    df_scores_aux['f1-score'] = [df_scores['f1-score'].iloc[0], df_scores['f1-score'].iloc[1],
                                 df_scores['f1-score'].iloc[2]]

    fig = df_scores_aux['precision'].plot.bar(title="Precision score comparison").get_figure()
    try:
        fig.savefig("./plots/task4/task4_precision.png")
    except:
        fig.savefig("../plots/task4/task4_precision.png")
    fig2 = df_scores_aux['recall'].plot.bar(title="Recall score comparison").get_figure()
    try:
        fig2.savefig("./plots/task4/task4_recall.png")
    except:
        fig2.savefig("../plots/task4/task4_recall.png")
    fig3 = df_scores_aux['f1-score'].plot.bar(title="F1 score comparison").get_figure()
    try:
        fig3.savefig("./plots/task4/task4_f1_score.png")
    except:
        fig3.savefig("../plots/task4/task4_f1_score.png")


# RUN FOR DEMO
def stars_prediction_demo(input_review):
    # print("Obtaining BERT pre-trained model")
    tf.config.experimental.list_physical_devices('GPU')
    bert = TFAutoModel.from_pretrained('bert-base-cased')

    input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask')

    embeddings = bert(input_ids, attention_mask=mask)[0]

    X = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(128, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.Dense(32, activation='relu')(X)
    y = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(X)

    model_tr = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

    model_tr.layers[2].trainable = False

    # earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
    # checkpoint = ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath='../yelp_bert_transformer.hdf5')
    model_tr.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    try:
        model_tr.load_weights('./saved_models/transformer_weights.h5')
    except:
        model_tr.load_weights('../saved_models/transformer_weights.h5')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    # print(82 * "_")
    # print("\nTransformer prediction")
    # print("\nPrediction on an input string: " + input_review)
    Xids_in = np.zeros((1, SEQ_LEN))
    Xmask_in = np.zeros((1, SEQ_LEN))

    tokens = tokenizer.encode_plus(input_review, max_length=SEQ_LEN,
                                   truncation=True, padding="max_length",
                                   add_special_tokens=True, return_token_type_ids=False,
                                   return_attention_mask=True, return_tensors='tf')

    Xids_in[0, :], Xmask_in[0, :] = tokens['input_ids'], tokens['attention_mask']

    input_test = tf.data.Dataset.from_tensor_slices((Xids_in, Xmask_in))
    input_test = input_test.map(map_func_test)
    input_test = input_test.batch(32)

    result = model_tr.predict(input_test, verbose=0)

    input_star = result.argmax() + 1

    print('Your review should have ', "\u2B50" * input_star, 'star/stars')


def execute_stars_comp():
    train_data, test_data, bert_data = load_rev_data()
    nb_report = execute_naive_bayes(train_data, test_data, create_nb_pipeline())
    nn_report, nn_model = execute_word2vec_nn(train_data, test_data)
    tr_report, tr_model = execute_bert(bert_data)
    predict_all(train_data, create_nb_pipeline(), nn_model, tr_model)
    # predict_best(tr_model)
    precision, recall, f1 = metrics_comparison(nb_report, nn_report, tr_report)
    plot_metrics_comparison(precision, recall, f1)
