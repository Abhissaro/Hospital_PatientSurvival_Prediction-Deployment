# Patient-Survival-Prediction


During the COVID-19 pandemic as healthcare workers around the world struggle with hospitals overloaded by patients in critical condition. 

Intensive Care Units (ICUs) often lack verified medical histories for incoming patients. A patient in distress or a patient who is brought in confused or unresponsive may not be able to provide information about chronic conditions such as heart disease, injuries, or diabetes. 

Medical records may take days to transfer, especially for a patient from another medical provider or system. Knowledge about chronic conditions can inform clinical decisions about patient care and ultimately improve patient's survival outcomes.

#### Problem Statement:- 

The target feature is hospital_death which is a binary variable. 

The task is to classify this variable based on the other 84 features step-by-step by going through each day's task. 

The scoring metric is Accuracy/Area under ROC curve.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hospital-patient-survival-prediction.onrender.com/)
https://hospital-patient-survival-prediction.onrender.com/
(Go through this link )(Deploy an app using onrender cloud)

### Steps taken to solve the problem:

1) Handled Missing Values using Mean and Mode techniques
2) EDA of the dataset to find distributions and relations of features 
3) Label Encoder for converting target values to numeric forms
4) Feature selection using mutual_info_classif
5) Baseline modeling using neural network - `accuracy: 92.02`
6) Fine tuned neural network - `accuracy: 92.93`
7) Explainable AI usign shap Kernel explainer

### Acknowledgement: TMLC
