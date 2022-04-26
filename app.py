import streamlit as st
import joblib,os
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

# Vectorizer
news_vectorizer = open("c:\\Users\\dell\\OneDrive\\Desktop\\News Classification\\models\\Vectorizer", "rb")
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
            predictor = load_prediction_model(clf)
            prediction = predictor.predict(vect_text)
            result = get_category(prediction, prediction_labels)
            st.success(result)
    if choice=="NLP":
        st.info("Natural Language Processing")
        
if __name__ == '__main__':
    main()