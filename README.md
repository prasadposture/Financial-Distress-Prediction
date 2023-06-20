<h1 align='center'>Financial Distress Prediction</h1>

### Introduction
Financial distress can be a major challenge for individuals and families, and it's often difficult to predict when it might occur. However, by analyzing a range of factors related to an individual's past and present financial situation, it may be possible to identify those who are most at risk of facing financial distress in the near future. In this project, I used machine learning algorithms to predict whether someone will face financial distress within the next two years based on age, debt ratio, monthly income, number of open credit lines and loans, number of estate loans, etc. By analyzing these factors and using predictive models, we hope to provide individuals and organizations with valuable insights that can help them make important financial decisions and avoid financial distress. Additionally, I built a Streamlit web application that allows users to input their financial information and receive a prediction of whether they are at risk of facing financial distress in the near future. This application will make it easy for users to access our predictive models and use them to make informed financial decisions.

### Important Links
1. [Data Preparation and Modeling](https://github.com/prasadposture/Financial-Distress-Prediction/blob/main/Financial%20Distress%20Prediction.ipynb)
2. [WebApp](https://prasadposture-financial-distress-prediction-fdp-7abgw6.streamlit.app/)
3. [Source Code](https://github.com/prasadposture/Financial-Distress-Prediction/blob/main/FDP.py)

### Workflow
1. Describing the problem statement and columns
2. Importing libraries and loading the data
3. Exploratory data analysis & Data visualization
4. Normalizing the attributes
5. Data preprocessing
   a. Identifying inputs and target column
   b. Imputing numeric values
   c. Scaling numeric features
6. Predictive modeling using `DecisionTreeClassifier`
   a. Splitting the dataset into training and validation data
   b. Training the model
   c. Evaluating the model
   d. Visualization of tree
   e. Feature importance
   f. Hyper-parameter tuning
   g. Making predictions with best parameter values
7. Repeating the same steps for predictive modeling but with `RandomForestClassifier`
8. Choosing the best model and making predictions on test data for submission
9. Saving the trained and tuned model for future use
10. Adding summary, future work, and references
11. Creating a Streamlit web application using the trained model for making predictions on the user input
