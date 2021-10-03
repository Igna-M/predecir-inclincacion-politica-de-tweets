from flask import Flask, jsonify, render_template, url_for, request, redirect, send_file, Response, make_response
import pickle



UPLOAD_FOLDER = 'static/uploads'
# pd.options.mode.chained_assignment = None  # default='warn'

application = Flask(__name__)
# app = application
# app.secret_key = b'{\xef~\x17\xe9\xc3\xd0\x1d\x806F\xb2\xc9\xed\xf9!\x91\xc5\xf0\x0f!\xde\x97V'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


with open('pickle/model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('pickle/vectorizer.pickle', 'rb') as v:
    vectorizer = pickle.load(v)



@app.route('/', methods=['POST', 'GET'])
def inicio():

    mensaje = 'Ingresa el texto de un tweet'




    return render_template('inicio.html', mensaje=mensaje)


@app.route('/predict', methods=['POST', 'GET'])
def predict():

    global vectorizer
    global model

    if request.method == 'POST':

        tweet = request.form['tweet']
        
        print(tweet)

    pred = model.predict(vectorizer.transform([tweet]))[0]
    
    if pred == 0:
        bloque = 'Juntos por el Cambio'
    else:
        bloque = 'Frente de Todos'

    print ("predicted class:", pred)


    return render_template('predict.html', bloque=bloque)







######## Para que corra en mi compu ########
if __name__ == '__main__':
    app.run(debug=True, port=7000)



### PARA ENCENDER: 
# source venv/bin/activate
# python3 app.py

## PARA DESCONECTAR:
# deactivate


### TAMBIEN, SE PUEDE:
# export FLASK_APP="application.py"
# flask run
