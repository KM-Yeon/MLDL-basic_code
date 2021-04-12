from flask import Flask, render_template,request
import pickle
import json
import numpy as np

app = Flask(__name__)

@app.route('/')
def myfunc1():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def myfunc2():
    # //return 'Hello'

    # params = json.loads(request.get_data(), encoding='utf-8')
    # if len(params) == 0:
    #     return 'No parameter'
    #
    # params_str = ''
    # for key in params.keys():
    #     params_str += 'key: {}, value: {}<br>'.format(key, params[key])

    myfile = request.form['myfile']
    # mypram = myfile(1,28,28,3)

    sepal_length = request.form['sepal_length']
    sepal_width = request.form['sepal_width']
    petal_length = request.form['petal_length']
    petal_width = request.form['petal_width']

    list = [sepal_length,sepal_width,petal_length,petal_width]
    myparam = np.array(list).reshape(1,-1)

    print(myparam.shape)  #(1, 4)

    loaded_model = pickle.load(open('mymodel.pkl', 'rb'))
    result = loaded_model.predict(myparam)
    print(result)

    return render_template('index.html', MYRESULT=result)

if __name__ == '__main__':
    app.debug = True
    app.run(port=9999)
