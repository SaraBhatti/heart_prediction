## early detection of heart failure
## import libraries
import streamlit as st 
import keras 
from PIL import Image
import numpy as np
import pickle

## load logistic regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

## create a function for prediction 
def heart_prediction(input):
    input_array = np.asarray(input)
    input_reshape = input_array.reshape(1,-1)
    prediction = model.predict(input_reshape)
    print(prediction)
    
    if (prediction[0]==0):
        return 'You are likely to die from heart failure, given your health condition'
    else: 
        return 'You are not likely to die of heart failure, given your health condition'

## Page configuration
def main(): 
    st.set_page_config(page_title='Heart Failure Predictor', layout='wide', )
    
    ## add image
    image =Image.open('PredictHF.png')
    st.image(image, use_column_width=False)
    
    ##set title and content
    st.title('Heart Failure preditor using Artificial Neural Network')
    st.write('Enter you personal data to get your heart risk evaluation')
    
    ## Get the input form the user
    age = st.number_input('Age of the patient:',min_value=0, step=1)
    anaemia = st.number_input('Anaemia | Yes or No | yes = 1 and no = 0', min_value=0, max_value=1, step=1)
    creatinine_phosphokinase = st.number_input('Level of the CPK enzyme inb the blood (mcg/L)', min_value=0, step=1)
    diabetes = st.number_input('Diabetes | Yes or No | yes = 1 and no = 0', min_value=0, max_value=1, step=1)
    ejection_fraction = st.number_input('Percentage of blood leaving the heart', min_value=0, max_value=1, step=1)
    high_blood_pressure = st.number_input('Hypertention | Yes or No | yes = 1 and no = 0', min_value=0, max_value=1, step=1)
    platelets = st.number_input('Platelets - count of blood (kiloplatelets/ml)', min_value=0.00, step=1)
    serum_creatinine = st.number_input('Level of serum creatinine in the blood (mg/dl))', min_value=0.00, step=0.01)
    serum_sodium = st.number_input('Level of serum sodium in the blood (mEq/l))', min_value=0, step=1)
    sex = st.number_input('Sex | Male or Female | Female = 1 and Male = 0', min_value=0, max_value=1, step=1)
    smoking = st.number_input('Smoker | Yes or No | yes = 1 and no = 0', min_value=0, max_value=1, step=1)

    
    ## code for prediction
    predict =''
    ## button for prediction
    if st.button('Predict'):
        predict = heart_prediction([age,anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking])
    st.success(predict)
    
if __name__ == '__main__':
        main()
      
##run in terminal
##steamlit run heart-model.py    
##python -m streamlit run heart-model.py
    
    
##save your scikit learn model
#with open('svm.pkl', 'wb') as file:
#pickle.dump(svm_linear, file)