from sklearn.feature_selection import f_oneway
import streamlit as st
import pandas as pd
import joblib,os
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score, mean_squared_error
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
import nltk 
from nltk.corpus import stopwords
from sklearn import preprocessing 
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import re
import warnings
import pickle
warnings.filterwarnings("ignore")
# Vectorizer
news_vectorizer = open("models\\Vectorizer", "rb")
news_cv = joblib.load(news_vectorizer)

#Loading Model
def load_prediction_model(model):
    loaded_model = joblib.load(open(os.path.join(model), "rb"))
    return loaded_model

# Get Category from Numeric Value
def get_category(val, dict):
    for key, value in dict.items():
        if val == value:
            return key

def add_parameter_ui(clf_name):
    params={}
    st.sidebar.write("Select values: ")

    if clf_name == "Logistic Regression":
        R = st.sidebar.slider("Regularization",0.1,10.0,step=0.1)
        MI = st.sidebar.slider("max_iter",50,400,step=50)
        params["R"] = R
        params["MI"] = MI

    elif clf_name == "KNN":
        K = st.sidebar.slider("n_neighbors",1,20)
        params["K"] = K

    elif clf_name == "Naive Bayes":
        cp = st.sidebar.selectbox("class_prior",(True, False))
        fp = st.sidebar.selectbox("fit_prior",(True, False))
        params["cp"] = cp
        params["fp"] = fp

    elif clf_name == "SVM":
        C = st.sidebar.slider("Regularization",0.01,10.0,step=0.01)
        kernel = st.sidebar.selectbox("Kernel",("linear", "poly", "rbf", "sigmoid", "precomputed"))
        params["C"] = C
        params["kernel"] = kernel

    elif clf_name == "Decision Tree":
        M = st.sidebar.slider("max_depth", 2, 20)
        C = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        SS = st.sidebar.slider("min_samples_split",1,10)
        params["M"] = M
        params["C"] = C
        params["SS"] = SS

    RS=st.sidebar.slider("Random State",0,100)
    params["RS"] = RS
    return params


def get_classifier(clf_name,params):
    global clf
    if clf_name == "Logistic Regression":
        clf = LogisticRegression(C=params["R"],max_iter=params["MI"])

    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == "SVM":
        clf = SVC(kernel=params["kernel"],C=params["C"])

    elif clf_name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=params["M"],criterion=params["C"],min_impurity_split=params["SS"])

    elif clf_name == "Naive Bayes":
        clf = MultinomialNB(class_prior=params["cp"],fit_prior=params["fp"])

    return clf
def process_text(text):
    text = text.lower().replace('\n',' ').replace('\r','').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]','',text)
    
    
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    
    text = " ".join(filtered_sentence)
    return text

def get_dataset():
    data = pd.read_csv("data\BBC News Train.csv")
    data['News_length'] = data['Text'].str.len()
    data['Text_parsed'] = data['Text'].apply(process_text)
    label_encoder = preprocessing.LabelEncoder() 
    data['Category_target']= label_encoder.fit_transform(data['Category']) 
    return data


#Plot Output
def compute(Y_pred,Y_test):
    # c1, c2 = st.beta_columns((4,3))
    #Confusion Matrix
    st.set_option('deprecation.showPyplotGlobalUse', False)
    cm=confusion_matrix(Y_test,Y_pred)
    class_label = ["business", "entertainment", "politics", "sport","tech"]
    df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
    plt.figure(figsize=(12, 7.5))
    sns.heatmap(df_cm,annot=True,cmap='Pastel1',linewidths=2,fmt='d')
    plt.title("Confusion Matrix",fontsize=15)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot()
    #Calculate Metrics
    acc=accuracy_score(Y_test,Y_pred)
    mse=mean_squared_error(Y_test,Y_pred)
    precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label=1)
    st.subheader("Metrics of the model: ")
    st.text('Precision: {} \nRecall: {} \nF1-Score: {} \nAccuracy: {} %\nMean Squared Error: {}'.format(
        precision,recall,fscore,acc*100, mse))



#Build Model
def model(clf):
    X_train,X_test,Y_train,Y_test=train_test_split(data['Text_parsed'], 
                                                    data['Category_target'],test_size=0.2,random_state=65)
    ngram_range = (1,2)
    min_df = 10
    max_df = 1.
    max_features = 300
    tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
    features_train = tfidf.fit_transform(X_train).toarray()
    labels_train = Y_train
    

    features_test = tfidf.transform(X_test).toarray()
    labels_test = Y_test
    

    clf.fit(features_train, labels_train)
    Y_pred = clf.predict(features_test)
    acc=accuracy_score(labels_test,Y_pred)
    return clf, Y_test, Y_pred

#tokenize for nlp
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def vec_for_learning(model_dbow, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model_dbow.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

data = get_dataset()
X = data['Text_parsed']
Y = data['Category_target']

def main():
    st.title("News Classification ML App")
    st.subheader("ML J Component")

    activities = ["About","Prediction","NLP"]
    choice = st.sidebar.selectbox("Choose Activity", activities)

    if choice=="About":
        st.info("About Us")
    if choice=="Prediction":
        
        st.info("Prediction with ML")
        news_text = st.text_area("Enter Text", "Type Here")
        all_ml_models = ["Logistic Regression", "Naive Bayes", "Decision Tree", "SVM", "KNN"]
        model_choice = st.selectbox("Choose ML Model", all_ml_models)
        prediction_labels = {'business':0, 'tech':1, 'politics':2, 'sport':3, 'entertainment':4}
        params = add_parameter_ui(model_choice)
        if st.button("Classify"):
            st.text("Original test ::\n{}".format(news_text))
            vect_text = news_cv.transform([news_text]).toarray()
            clf = get_classifier(model_choice,params)
            predictor, Y_pred,Y_test = model(clf)
            prediction = predictor.predict(vect_text)
            result = get_category(prediction, prediction_labels)
            st.success(result)
            st.markdown("<hr>",unsafe_allow_html=True)
            st.subheader(f"Classifier Used: {model_choice}")
            compute(Y_pred,Y_test)
    if choice=="NLP":
        st.info("Natural Language Processing")
        df = pd.read_csv("data/BBC_News_Train_Processed.csv")
        train, test = train_test_split(df, test_size = 0.2, random_state=42)
        print(test)
        test_tagged = test.apply(lambda r: TaggedDocument(words=tokenize_text(r['Text']), tags=[r.Category]), axis=1)
        model_dbow = pickle.load(open('models\\nlp_model_dbow.sav', 'rb'))
        model = pickle.load(open('models\\nlp_model.sav', 'rb'))
        Y_test, X_test = vec_for_learning(model_dbow, test_tagged)
        Y_pred = model.predict(X_test)
        compute(Y_pred, Y_test)


        
if __name__ == '__main__':
    main()