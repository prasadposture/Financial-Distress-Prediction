#Importing the libraries
import streamlit as st
import joblib
import pandas as pd

#setting page configuration
st.set_page_config(page_title='Financial Distress Predictions', page_icon=':book:', layout='wide')


st.write("""# Financial Distress Predictions""")

#taking the inputs
with st.container():
    left,right, p1, p2 = st.columns(4)
    with left:
        NumberOfTime60_89DaysPastDueNotWorse = st.number_input('Number Of Time 60-89 Days Past Due Not Worse', value=0.000000)
        NumberOfTime30_59DaysPastDueNotWorse = st.number_input('Number Of Time 30-59 Days Past Due Not Worse', value=0.000000)
        RevolvingUtilizationOfUnsecuredLines = st.number_input('Revolving Utilization Of UnsecuredLines', value=0.154181)
        NumberOfOpenCreditLinesAndLoans = st.number_input('Number Of Open Credit Lines And Loans', value=8.000000)
        NumberRealEstateLoansOrLines = st.number_input('Number Real Estate Loans Or Lines', value=1.000000)
    with right:
        Age = st.number_input('Age', value=52.000000)
        DebtRatio = st.number_input('Debt Ratio', value=0.366508)
        MonthlyIncome = st.number_input('Monthly Income', value=5.400000e+03)
        NumberOfDependents = st.number_input('Number Of Dependents', value=0.000000)
        NumberOfTimes90DaysLate = st.number_input('Number Of Times 90 Days Late', value=0.000000)
        

#converting inputs into dataframe
new_input = {
    'RevolvingUtilizationOfUnsecuredLines': RevolvingUtilizationOfUnsecuredLines,
    'age': Age,
    'NumberOfTime30-59DaysPastDueNotWorse' : NumberOfTime30_59DaysPastDueNotWorse,
    'DebtRatio' : DebtRatio,
    'MonthlyIncome' : MonthlyIncome,
    'NumberOfOpenCreditLinesAndLoans' : NumberOfOpenCreditLinesAndLoans,
    'NumberOfTimes90DaysLate' : NumberOfTimes90DaysLate,
    'NumberRealEstateLoansOrLines' : NumberRealEstateLoansOrLines,
    'NumberOfTime60-89DaysPastDueNotWorse' : NumberOfTime60_89DaysPastDueNotWorse,
    'NumberOfDependents' : NumberOfDependents
    }


#loading the model
FDP=joblib.load('financial_distress')

#writing a predictor function

def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df = FDP['scaler'].transform(input_df)
    pred = FDP['model'].predict(input_df)[0]
    return pred

st.write("___")

#making predictions
pred = predict_input(new_input)
if pred==1:
    st.write('##### Person Suffers Financial Distress')
else:
    st.write("##### Person Doesn't Suffer Financial Distress")


st.write("___")
st.write('My Profiles')
with st.container():
    left, middle, right = st.columns(3)
    with left:
        st.write('[Kaggle](https://www.kaggle.com/prasadposture121)')
    with middle:
        st.write('[GitHub](https://github.com/prasadposture)')
    with right:
        st.write('[LinkedIn](https://www.linkedin.com/in/prasad-posture-6a3a77215/)')