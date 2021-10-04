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
    pred = model.predict(vectorizer.transform([cleaned_tweet]))[0]
    
    if pred == 0:
        bloque = 'Juntos por el Cambio'
        imagen = '../static/images/2560px-Juntos-Por-El-Cambio-Logo.png'

    else:
        bloque = 'Frente de Todos'
        imagen = '../static/images/2560px-Frente_de_Todos_logo.png'

    return render_template('partido-politico-probable.html', bloque=bloque, imagen=imagen, tweet=tweet)




@app.route('/promedio-10-tweets', methods=['POST', 'GET'])
def mean_10_tweets():



    return render_template('promedio-10-tweets.html')



@app.route('/predecir-promedio-10-tweets', methods=['POST', 'GET'])
def mean_10_tweets_predict():

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

        print('')
        print('')

        todos_los_tweets = []
        for i in range(1, 11):
            tweet = request.form['tweet' + str(i)]
            if len(tweet) > 0:
                todos_los_tweets.append(tweet)

        print('')
        print(todos_los_tweets)
        print('')

        valoracion_de_tweets = []
        for j in range(len(todos_los_tweets)):
            print(todos_los_tweets[j])
            cleaned_tweet = limpiar_tweets(todos_los_tweets[j])
            pred = model.predict(vectorizer.transform([cleaned_tweet]))[0]
            valoracion_de_tweets.append(pred)

        promedio = sum(valoracion_de_tweets) / len(valoracion_de_tweets)
        porcentaje = promedio * 100

        porcentaje_jc = 50 + (50 - porcentaje)
        porcentaje_ft = 50 + (porcentaje - 50)
        
        len_tweets = len(todos_los_tweets)

        if promedio < 0.5:
            bloque = 'Juntos por el Cambio'
            imagen = '../static/images/2560px-Juntos-Por-El-Cambio-Logo.png'
            grado = porcentaje_jc
        else:
            bloque = 'Frente de Todos'
            imagen = '../static/images/2560px-Frente_de_Todos_logo.png'
            grado = porcentaje_ft

    return render_template('promedio-10-tweets-resultado.html', bloque=bloque, imagen=imagen, promedio=promedio, tweets=todos_los_tweets, len_tweets=len_tweets, grado=grado)







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
