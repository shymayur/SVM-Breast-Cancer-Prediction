#importing the Dependencies
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def add_bg_from_url():
    st.markdown(
         f"""
   
         <style>
         .stApp {{
             background-image: url("https://wallpaper.dog/large/5441083.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

image = Image.open("C:/Users/Mayur/Desktop/bcp4.webp")
image=image.resize((700,350))

st.header('Breast Cancer Prediction App')
st.subheader('(Credit: Mayur #ML-LogisticRegression)')

st.image(image)

a=st.number_input('Enter Radius')
b=st.number_input('Enter Texture')
c=st.number_input('Enter Perimeter')
d=st.number_input('Enter Area')
e=st.number_input('Enter Smoothness')
f=st.number_input('Enter Compactness')
g=st.number_input('Enter Concavity')
h=st.number_input('Enter Concave Points')
i=st.number_input('Enter Symmetry')
j=st.number_input('Enter Fractal Dimension')

# Data Collection & Processing

# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

#loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

#data_frame
df=data_frame[['mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension']]

# adding the 'target' column to the data frame
df['label'] = breast_cancer_dataset.target

# checking for missing values
#df.isnull().sum()

#statistical measures about the data
#df.describe()

#checking the distributions of Target Variable
#df['label'].value_counts()

# 1-->Benign
# 0-->Malignant

#df.groupby('label').mean()

#separating the features and target

X = df.drop(columns='label',axis=1)
Y = df['label']

# Splitting the data into training data & Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#shape of the train and test data
#print(X.shape, X_train.shape, X_test.shape)

# Model Training
# Logistic Regression
model = LogisticRegression()

# training the Logistic Regression model using Training data
model.fit(X_train, Y_train)

# Model Evaluation

# Accuracy Score

# accuracy on training data

X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# accuracy on test data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)



#Buliding a Predictive System

input_data=[a,b,c,d,e,f,g,h,i,j]

#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

if st.button('Click Here'):
    st.write('Cancer Test Result:')

    if(prediction[0] == 0):
       st.write(':red[This person has cancer!]')

    else:
        st.write(':green[This person is cancer-free!]')
      

