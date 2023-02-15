#Importing the libraries
pip install joblib
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#setting page configuration
st.set_page_config(page_title='Financial Distress Predictions', page_icon=':chart_with_downwards_trend:', layout='wide')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align:  center; color: blue; text-shadow: 0.1em 0.1em 0.08em #2F4F4F'>Financial Distress Predictions</h1>", unsafe_allow_html=True)
st.write("___")
#taking the inputs
with st.container():
    left, c1, c2, right = st.columns(4)
    with left:
        Age = st.number_input('Age', value=52.000000)
        DebtRatio = st.number_input('Debt Ratio', value=0.366508)
        MonthlyIncome = st.number_input('Monthly Income', value=5.400000e+03)
        NumberRealEstateLoansOrLines = st.number_input('Number Real Estate Loans Or Lines', value=1.000000)
        NumberOfOpenCreditLinesAndLoans = st.number_input('Number Of Open Credit Lines And Loans', value=8.000000)
    with c1:
        NumberOfDependents = st.number_input('Number Of Dependents', value=0.000000)
        RevolvingUtilizationOfUnsecuredLines = st.number_input('Revolving Utilization Of UnsecuredLines', value=0.154181)
        NumberOfTime30_59DaysPastDueNotWorse = st.number_input('Number Of Time 30-59 Days Past Due Not Worse', value=0.000000)
        NumberOfTime60_89DaysPastDueNotWorse = st.number_input('Number Of Time 60-89 Days Past Due Not Worse', value=0.000000)
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
    with c2:
        #making predictions
        st.write("##### Suffers Financial Distress? ")
        st.write('\n')
        pred = predict_input(new_input)
        if pred==1:
            st.write("##### Yes")
            st.write('\n')
            st.image('sad.png')
        else:
            st.write("##### No")
            st.write('\n')
            st.image('happy.png')
    with right:
        features=['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'age', 'MonthlyIncome','NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfOpenCreditLinesAndLoans', 
        'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents', 'NumberRealEstateLoansOrLines' ]
        importance=[0.2715613, 0.1333391, 0.1202234, 0.1141782, 0.0959006, 0.0787225, 0.0786178, 0.0409399,0.033597, 0.032925 ]
        plt.title('Feature Importance')
        sns.barplot(y=importance, x=features)
        plt.xticks(rotation=90)
        st.pyplot(plt.gcf())
        st.write('By Prasad Posture')

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
