import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()


st.markdown(
    """
    <style>
    .main{
    background-color: #0c0d0c;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache
def get_data(filename):
    zomato_restaurants = pd.read_csv(filename)
    return zomato_restaurants

with header:
    st.title('My first Stream lit project!')   
    st.text('In this project I look into statistics of the Zomato Restaurants')


with dataset:
    st.header('Zomato Restaurants dataset')
    st.text('I found this dataset on Kaggle')

    zomato_restaurants = get_data('data/zomato_restaurants_in_India.csv')
    st.write(zomato_restaurants.head())
    
    
    st.subheader('Restaurant ID distribution on the Zomato Restaurants dataset')
    rest_lcnrtng = pd.DataFrame(zomato_restaurants['res_id'].value_counts()).head(50)
    st.bar_chart(rest_lcnrtng)


with features:
    st.header('The features I created')

    st.markdown('* **first feature:** I created this feature because of this... I calculated this using this logic...')
    st.markdown('* **second feature:** I created this feature because of this... I calculated this using this logic...')



with model_training:
    st.header('Time to train the model!')
    st.text('Here you can choose the hyperparameters of the model and see how the performance changes!!!')
    
    sel_col, disp_col = st.beta_columns(2)
    
    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No limit'], index = 0)

    sel_col.text('Here is the list of features in my data: ')
    sel_col.write(zomato_restaurants.columns)

    input_feature = sel_col.text_input('Which feature should be used as input feature?', 'res_id')


    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = zomato_restaurants[[input_feature]]
    y = zomato_restaurants[['aggregate_rating']]

    regr.fit(X,y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error of the model is: ')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean squared error of the model is: ')
    disp_col.write(mean_squared_error(y, prediction))   

    disp_col.subheader('R squared score of the model is: ')
    disp_col.write(r2_score(y, prediction))

