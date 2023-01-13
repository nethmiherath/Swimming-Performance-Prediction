# Swimming-Performance-Prediction
Undergraduate research project

Abstract:
Predicting the performance of individual sports has become a routine measure in sports sciences,
since it helps to plan and improve their training activities. The main objective of this study was to
compare the accuracy of predictive models developed using different statistical and machine learning
techniques to predict the performance time of elite swimmers for the 100m long course swimming
events, using physiological features. A total of four techniques, namely, Multiple Linear Regression
(MLR), Decision Tree Regression (DTR), Random Forest Regression (RFR) and Artificial Neural
Network (ANN) were considered in this study.Performance times of 100m freestyle, backstroke,
breaststroke and butterfly stroke of 2371 swimmers were obtained from the World Olympic games
database. The methodology included three major steps: i) analysing and pre-processing the dataset; ii)
optimizing the models using k-fold cross-validation and hyperparameter tuning; iii) comparing the
performance of different models using accuracy metrics;Mean Absolute Percentage Error (MAPE), R-
squared(R 2 ), Root Mean Square Error (RMSE), Mean Absolute Error(MAE), Median Absolute
Deviation(MAD). All the models were deployed in the same data segmentation for consistency. A
Multi-Layer Perception (MLP)-based ANN was trained to predict the performance times of
swimmers. The baseline model of the study was developed using the MLR technique. Our findings
suggest that the RFR model has the highest accuracy (98.1%) followed by the MLP-based ANN
model (97.8%) and MLR model (97.6%). The accuracy of the DTR model was slightly lower than the
rest of the models. The results also showed that the age, height, weight, reaction time and types of
swimming styles of the swimmers have a significant effect on the performance times of the elite
swimmers. In conclusion, the RFR model outperformed the other models in predicting the
performance times of elite swimmers for all 100 m swimming events.


How to run this:
1). set FLASK_APP = app.py
2). set FLASK_ENV=development
3). flask run
