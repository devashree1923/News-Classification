import streamlit as st
import joblib,os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier
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

    elif clf_name == "SVM":
        C = st.sidebar.slider("Regularization",0.01,10.0,step=0.01)
        kernel = st.sidebar.selectbox("Kernel",("linear", "poly", "rbf", "sigmoid", "precomputed"))
        params["C"] = C
        params["kernel"] = kernel

    elif clf_name == "Decision Trees":
        M = st.sidebar.slider("max_depth", 2, 20)
        C = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        SS = st.sidebar.slider("min_samples_split",1,10)
        params["M"] = M
        params["C"] = C
        params["SS"] = SS

    elif clf_name == "Random Forest":
        N = st.sidebar.slider("n_estimators",50,500,step=50,value=100)
        M = st.sidebar.slider("max_depth",2,20)
        C = st.sidebar.selectbox("Criterion",("gini","entropy"))
        params["N"] = N
        params["M"] = M
        params["C"] = C

    elif clf_name == "Gradient Boosting":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50,value=100)
        LR = st.sidebar.slider("Learning Rate", 0.01, 0.5)
        L = st.sidebar.selectbox("Loss", ('deviance', 'exponential'))
        M = st.sidebar.slider("max_depth",2,20)
        params["N"] = N
        params["LR"] = LR
        params["L"] = L
        params["M"] = M

    elif clf_name == "XGBoost":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50, value=50)
        LR = st.sidebar.slider("Learning Rate", 0.01, 0.5,value=0.1)
        O = st.sidebar.selectbox("Objective", ('binary:logistic','reg:logistic','reg:squarederror',"reg:gamma"))
        M = st.sidebar.slider("max_depth", 1, 20,value=6)
        G = st.sidebar.slider("Gamma",0,10,value=5)
        L = st.sidebar.slider("reg_lambda",1.0,5.0,step=0.1)
        A = st.sidebar.slider("reg_alpha",0.0,5.0,step=0.1)
        CS = st.sidebar.slider("colsample_bytree",0.5,1.0,step=0.1)
        params["N"] = N
        params["LR"] = LR
        params["O"] = O
        params["M"] = M
        params["G"] = G
        params["L"] = L
        params["A"] = A
        params["CS"] = CS

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

    elif clf_name == "Decision Trees":
        clf = DecisionTreeClassifier(max_depth=params["M"],criterion=params["C"],min_impurity_split=params["SS"])

    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["N"],max_depth=params["M"],criterion=params["C"])

    elif clf_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=params["N"],learning_rate=params["LR"],loss=params["L"],max_depth=params["M"])

    # elif clf_name == "XGBoost":
    #     clf = XGBClassifier(booster="gbtree",n_estimators=params["N"],max_depth=params["M"],learning_rate=params["LR"],
    #                         objective=params["O"],gamma=params["G"],reg_alpha=params["A"],reg_lambda=params["L"],colsample_bytree=params["CS"])

    return clf

def main():
    st.title("News Classification ML App")
    st.subheader("NLP and ML App with Streamlit")

    activities = ["About","Prediction","NLP"]
    choice = st.sidebar.selectbox("Choose Activity", activities)

    if choice=="About":
        st.info("About Us")
    if choice=="Prediction":
        st.info("Prediction with ML")
        news_text = st.text_area("Enter Text", "Type Here")
        all_ml_models = ["Logistic Regression", "Naive Bayes", "Random Forest", "Decision Tree", "SVM", "KNN", "MLP"]
        model_choice = st.selectbox("Choose ML Model", all_ml_models)
        prediction_labels = {'business':0, 'tech':1, 'politics':2, 'sport':3, 'entertainment':4}
        params = add_parameter_ui(model_choice)
        if st.button("Classify"):
            st.text("Original test ::\n{}".format(news_text))
            vect_text = news_cv.transform([news_text]).toarray()
            if model_choice =="Logistic Regression":
                predictor = load_prediction_model("c:\\Users\\dell\\OneDrive\\Desktop\\News Classification\\models\\Logistic_model")
                prediction = predictor.predict(vect_text)
                result = get_category(prediction, prediction_labels)
                st.success(result)


    if choice=="NLP":
        st.info("Natural Language Processing")
        
if __name__ == '__main__':
    main()