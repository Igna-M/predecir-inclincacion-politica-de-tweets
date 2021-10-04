from flask import Flask, jsonify, render_template, url_for, request, redirect, send_file, Response, make_response
import pickle
import unidecode
import re



# UPLOAD_FOLDER = 'static/uploads'
# pd.options.mode.chained_assignment = None  # default='warn'

app = Flask(__name__)
# app = application
# app.secret_key = b'{\xef~\x17\xe9\xc3\xd0\x1d\x806F\xb2\xc9\xed\xf9!\x91\xc5\xf0\x0f!\xde\x97V'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


with open('pickle/model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('pickle/vectorizer.pickle', 'rb') as v:
    vectorizer = pickle.load(v)








@app.route('/', methods=['POST', 'GET'])
def inicio():



    return render_template('inicio.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():

    def limpiar_tweets(tweet):
        t_lower_no_accents=unidecode.unidecode(tweet.lower()); # sacamos acentos y llevamos a minuscula
        t_lower_no_accents_no_punkt=re.sub(r'([^\s\w]|_)+','',t_lower_no_accents); # quitamos signos de puntuacion usando una regex que reemplaza todo lo q no sean espacios o palabras por un string vacio
        t_no_new_line = t_lower_no_accents_no_punkt.replace('\n', ' ')
        t_remove_http = re.sub(r'http\S+', '', t_no_new_line).strip()
        x = re.sub("(.)\\1{2,}", "\\1", t_remove_http)
        return x

    global vectorizer
    global model

    if request.method == 'POST':

        tweet = request.form['tweet']



    cleaned_tweet = limpiar_tweets(tweet)
    print(cleaned_tweet)

    pred = model.predict(vectorizer.transform([cleaned_tweet]))[0]
    
    if pred == 0:
        bloque = 'Juntos por el Cambio'
    else:
        bloque = 'Frente de Todos'

    return render_template('predict.html', bloque=bloque)







######## Para que corra en mi compu ########
if __name__ == '__main__':
    app.run(debug=True, port=7001)



### PARA ENCENDER: 
# source venv/bin/activate
# python3 app.py

## PARA DESCONECTAR:
# deactivate


### TAMBIEN, SE PUEDE:
# export FLASK_APP="application.py"
# flask run
