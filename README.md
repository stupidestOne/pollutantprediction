# pollutant concentration prediction
we are provided air quality data and weather data (observed weather data from 18 weather stations and grid weather data from 651 points in Beijing, China) from January 1, 2017 to April 30, 2018, and we need to predict the concentration levels of PM2.5, PM10 and O3 between May 1 to May 2, 2018 (once an hour, 48 times for 35 station in total).

This project contains four parts:

--Preprocessing.py: Do the preprocessing job. Including data exploration and data cleaning.

--PM10_Stacking.py: Do stacking to predict the concentration of PM10 to fill some of its missing values. (We do that because almost 1/3 values of PM10 are missed, and we find the relationship of PM10 and other pollutant is strong.)

--LSTM.py: To predict the concentration of pollutants by LSTM.

--Final_Stacking.py: To predict the concentration of pollutants by stacking.

