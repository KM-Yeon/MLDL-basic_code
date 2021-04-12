import tensorflow as tf
from flask import Flask, request
from PIL import Image
import numpy as np
import pickle


# def convert_gray_image_to_numpy(file_path, width=8, height=8):
#     if type(file_path) == str:
#         image = Image.open(file_path)
#     else:
#         image = file_path
#     image = image.convert('L') #'L' (그레이 스케일), '1' (이진화), 'RGB' , 'RGBA', 'CMYK' #색상 모드 변경
#     if width != None and height != None:
#         image = image.resize((width, height))
#         image_numpy = np.array(image) #이미지 타입을 넘파이 타입으로 변환
#         image_numpy = image_numpy.reshape((width, height, 1))
#     else:
#         image_numpy = np.array(image)
#         width, height = image.size
#         image_numpy = image_numpy.reshape((width, height, 1))
#
#     return image_numpy

##########모델 로드

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
loaded_model = pickle.load(open('./mymodel_mnist.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    html = '''
<html>
<body>
    <center>
    MNIST 손글씨 숫자 예측<br>
    <form action="/predict" method="post" enctype="multipart/form-data">
        이미지 파일 <input type="file" name="file"><br>
        <input type="submit" value="예측하기">
    </form>
    </center>
</body>
</html>
'''
    return html

@app.route('/predict', methods=['post'])
def predict():
    image = Image.open(request.files['file'].stream)
    image = image.convert('L')  # 'L': greyscale, '1': 이진화, 'RGB' , 'RGBA', 'CMYK'
    image = image.resize((8, 8))

    image_numpy = np.array(image)
    x_test = image_numpy.reshape(-1,8*8)
    #x_test = x_test / 255
    print(x_test.shape)

    pred = loaded_model.predict(x_test)
    print("예측결과:",pred)
    return labels[pred[0]]

# app.run(host='127.0.0.1', port=5000, debug=False)
if __name__ == '__main__':
    app.debug = True
    app.run(port=8888)
