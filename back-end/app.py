from flask import Flask , render_template,request
import pickle
import joblib
import numpy as np
import pandas as pd 


model=joblib.load(open("DataPreparationModel.pkl",'rb'))
model2=joblib.load(open("RandomForestRegressor.pkl",'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('page2.html')


@app.route('/predict',methods=['POST'])
def home():
    longitude=request.form.get('longitude')
    latitude=request.form.get('latitude')
    housing_median_age=request.form.get('housing_median_age')
    total_rooms=request.form.get('total_rooms')
    total_bedrooms=request.form.get('total_bedrooms')
    population=request.form.get('population')
    households=request.form.get('households')
    median_income=request.form.get('median_income')
    ocean_proximity=request.form['souhait']


    rooms_per_household=float(total_rooms)/float(households)
    bedrooms_per_room=float(total_bedrooms)/float(total_rooms)
    population_per_household=float(population)/float(households)
    

    attributs=np.array([longitude,latitude,housing_median_age,total_rooms,population,households,median_income,rooms_per_household,bedrooms_per_room,population_per_household,ocean_proximity])
    newdataframe= pd.DataFrame(data=[attributs],columns=['longitude','latitude','housing_median_age','total_rooms','population','households','median_income','rooms_per_household','bedrooms_per_room','population_per_household','ocean_proximity'])

    clean_features=model.transform(newdataframe)
    prediction=model2.predict(clean_features)

    return render_template("Untitled.html",prediction_text='FORECAST OF HOUSE PRICE  is :{}'.format(prediction))




if __name__ == "__main__":
    app.run(debug=True)