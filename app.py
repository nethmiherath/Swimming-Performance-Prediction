from __future__ import print_function
import flask
import pickle
import pandas as pd
import sys
import numpy as np
import tensorflow

# Use pickle to load in the pre-trained model.
with open(f'model/swimming_model_MLR.pkl', 'rb') as f:
    modelMLR = pickle.load(f)
with open(f'model/swimming_model_RFR.pkl', 'rb') as j:
    modelRFR = pickle.load(j)
with open(f'model/swimming_model_DTR.pkl', 'rb') as m:
    modelDTR = pickle.load(m)
modelANN = tensorflow.keras.models.load_model('model/swimming_model_ANN.h5')

with open(f'model/MinMaxscalerX.pkl', 'rb') as g:
    scalerMLR_X = pickle.load(g)
with open(f'model/MinMaxscalerY.pkl', 'rb') as h:
    scalerMLR_Y = pickle.load(h)
with open(f'model/MinMaxscalerX_all.pkl', 'rb') as k:
    scalerALL_X = pickle.load(k)
with open(f'model/MinMaxscalerY_all.pkl', 'rb') as l:
    scalerALL_Y = pickle.load(l)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/about')
def about():
    return(flask.render_template('about.html'))

@app.route('/project')
def project():
    return(flask.render_template('project.html'))

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        const = 1
        Reaction_time = flask.request.form['Reaction_time']
        Age = flask.request.form['Age']
        Height = flask.request.form['Height']
        Weight = flask.request.form['Weight']
        Model = flask.request.form['Model']

        Sex = flask.request.form['Sex']
        if Sex == 'Male':
            M = 1
            F = 0
        else:
            M = 0
            F = 1

        EventAll = flask.request.form['EventAll']
        if EventAll == 'Backstroke' and Sex == 'Male':
            Men_Backstroke = 1
            Men_Breaststroke = Men_Butterfly = Men_Freestyle = Women_Backstroke = Women_Breaststroke = Women_Butterfly = Women_Freestyle = 0
        elif EventAll == 'Breaststroke' and Sex == 'Male':
            Men_Breaststroke = 1
            Men_Backstroke = Men_Butterfly = Men_Freestyle = Women_Backstroke = Women_Breaststroke = Women_Butterfly = Women_Freestyle = 0
        elif EventAll == 'Butterfly' and Sex == 'Male':
            Men_Butterfly = 1
            Men_Backstroke = Men_Breaststroke = Men_Freestyle = Women_Backstroke = Women_Breaststroke = Women_Butterfly = Women_Freestyle = 0
        elif EventAll == 'Freestyle' and Sex == 'Male':
            Men_Freestyle = 1
            Men_Backstroke = Men_Butterfly = Men_Breaststroke = Women_Backstroke = Women_Breaststroke = Women_Butterfly = Women_Freestyle = 0
        elif EventAll == 'Backstroke' and Sex == 'Female':
            Women_Backstroke = 1
            Men_Backstroke = Men_Breaststroke = Men_Butterfly = Men_Freestyle = Women_Breaststroke = Women_Butterfly = Women_Freestyle = 0
        elif EventAll == 'Breaststroke' and Sex == 'Female':
            Women_Breaststroke = 1
            Men_Backstroke = Men_Breaststroke = Men_Butterfly = Men_Freestyle = Women_Backstroke = Women_Butterfly = Women_Freestyle = 0
        elif EventAll == 'Butterfly' and Sex == 'Female':
            Women_Butterfly = 1
            Men_Backstroke = Men_Breaststroke = Men_Butterfly = Men_Freestyle = Women_Backstroke = Women_Breaststroke = Women_Freestyle = 0
        else:
            Women_Freestyle = 1
            Men_Backstroke = Men_Breaststroke = Men_Butterfly = Men_Freestyle = Women_Backstroke = Women_Breaststroke = Women_Butterfly = 0

        if Model == 'MLR':
            model = modelMLR
            input_variables = pd.DataFrame([[const, Reaction_time, Age, Height, Weight, Men_Backstroke, Men_Breaststroke, Men_Butterfly, Men_Freestyle, Women_Backstroke, Women_Breaststroke, Women_Butterfly, Women_Freestyle, F, M]],
                                       columns=['const', 'Reaction_time', 'Age', 'Height', 'Weight', 'Men_Backstroke', 'Men_Breaststroke', 'Men_Butterfly', 'Men_Freestyle', 'Women_Backstroke', 'Women_Breaststroke', 'Women_Butterfly', 'Women_Freestyle', 'F', 'M'],
                                       dtype=float)
            scaler_X = scalerMLR_X
            scaler_Y = scalerMLR_Y

            input_variables = input_variables.to_numpy()[0]
            input_variables = input_variables.reshape(1, -1)
            input_scaled = scaler_X.transform(input_variables)
            prediction_scaled = model.predict(input_scaled)
            ###### Fix #######
            # create empty table with 15 fields
            Predictions_dataset2 = np.zeros(shape=(len(prediction_scaled), 15))
            # put the predicted values in the right field
            Predictions_dataset2[:,0] = prediction_scaled
            # inverse transform and then select the right field
            prediction = scaler_Y.inverse_transform(Predictions_dataset2)[:,0]
            prediction = prediction[0]
            prediction = str(round(prediction, 4))

        elif Model == 'RFR':
            model = modelRFR
            scaler_X = scalerALL_X
            scaler_Y = scalerALL_Y

            input_variables = pd.DataFrame([[Reaction_time, Age, Height, Weight, Men_Backstroke, Men_Breaststroke, Men_Butterfly, Men_Freestyle, Women_Backstroke, Women_Breaststroke, Women_Butterfly, Women_Freestyle, F, M]],
                                       columns=['Reaction_time', 'Age', 'Height', 'Weight', 'Men_Backstroke', 'Men_Breaststroke', 'Men_Butterfly', 'Men_Freestyle', 'Women_Backstroke', 'Women_Breaststroke', 'Women_Butterfly', 'Women_Freestyle', 'F', 'M'],
                                       dtype=float)

            input_variables = input_variables.to_numpy()[0]
            input_variables = input_variables.reshape(1, -1)
            input_scaled = scaler_X.transform(input_variables)
            prediction_scaled = model.predict(input_scaled)
            ###### Fix #######
            # create empty table with 15 fields
            Predictions_dataset2 = np.zeros(shape=(len(prediction_scaled), 15))
            # put the predicted values in the right field
            Predictions_dataset2[:,0] = prediction_scaled
            # inverse transform and then select the right field
            prediction = scaler_Y.inverse_transform(Predictions_dataset2)[:,0]
            prediction = prediction[0]
            prediction = str(round(prediction, 4))

        elif Model == 'DTR':
            model = modelDTR
            scaler_X = scalerALL_X
            scaler_Y = scalerALL_Y

            input_variables = pd.DataFrame([[Reaction_time, Age, Height, Weight, Men_Backstroke, Men_Breaststroke, Men_Butterfly, Men_Freestyle, Women_Backstroke, Women_Breaststroke, Women_Butterfly, Women_Freestyle, F, M]],
                                       columns=['Reaction_time', 'Age', 'Height', 'Weight', 'Men_Backstroke', 'Men_Breaststroke', 'Men_Butterfly', 'Men_Freestyle', 'Women_Backstroke', 'Women_Breaststroke', 'Women_Butterfly', 'Women_Freestyle', 'F', 'M'],
                                       dtype=float)

            input_variables = input_variables.to_numpy()[0]
            input_variables = input_variables.reshape(1, -1)
            input_scaled = scaler_X.transform(input_variables)
            prediction_scaled = model.predict(input_scaled)
            ###### Fix #######
            # create empty table with 15 fields
            Predictions_dataset2 = np.zeros(shape=(len(prediction_scaled), 15))
            # put the predicted values in the right field
            Predictions_dataset2[:,0] = prediction_scaled
            # inverse transform and then select the right field
            prediction = scaler_Y.inverse_transform(Predictions_dataset2)[:,0]
            prediction = prediction[0]
            prediction = str(round(prediction, 4))

        else:
            model = modelANN
            scaler_X = scalerALL_X
            scaler_Y = scalerALL_Y

            input_variables = pd.DataFrame([[Reaction_time, Age, Height, Weight, Men_Backstroke, Men_Breaststroke, Men_Butterfly, Men_Freestyle, Women_Backstroke, Women_Breaststroke, Women_Butterfly, Women_Freestyle, F, M]],
                                       columns=['Reaction_time', 'Age', 'Height', 'Weight', 'Men_Backstroke', 'Men_Breaststroke', 'Men_Butterfly', 'Men_Freestyle', 'Women_Backstroke', 'Women_Breaststroke', 'Women_Butterfly', 'Women_Freestyle', 'F', 'M'],
                                       dtype=float)

            input_variables = input_variables.to_numpy()[0]
            input_variables = input_variables.reshape(1, -1)
            input_scaled = scaler_X.transform(input_variables)
            prediction_scaled = model.predict(input_scaled)
            ###### Fix #######
            # create empty table with 15 fields
            Predictions_dataset2 = np.zeros(shape=(len(prediction_scaled), 15))
            # put the predicted values in the right field
            Predictions_dataset2[:,0] = prediction_scaled
            # inverse transform and then select the right field
            prediction = scaler_Y.inverse_transform(Predictions_dataset2)[:,0]
            prediction = prediction[0]
            prediction = str(round(prediction, 4))

        return flask.render_template('main.html',
                                     original_input={'const':const,
                                                     'Reaction_time':Reaction_time,
                                                     'Age':Age,
                                                     'Height':Height,
                                                     'Weight':Weight,
                                                     'Men_Backstroke':Men_Backstroke,
                                                     'Men_Breaststroke':Men_Breaststroke,
                                                     'Men_Butterfly':Men_Butterfly,
                                                     'Men_Freestyle':Men_Freestyle,
                                                     'Women_Backstroke':Women_Backstroke,
                                                     'Women_Breaststroke':Women_Breaststroke,
                                                     'Women_Butterfly':Women_Butterfly,
                                                     'Women_Freestyle':Women_Freestyle,
                                                     'F':F,
                                                     'M':M},
                                     result=prediction, pred=input_scaled)

        return flask.render_template('main.html')

def main():
    return(flask.render_template('main.html'))
if __name__ == '__main__':
    app.run()