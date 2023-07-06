from flask import Flask, render_template, request
import cv2
import base64
import numpy as np
import torch
import torch.nn as nn

app = Flask(__name__)


# a cnn sequential model by pytorch
class SK_CNN(nn.Module):

    def __init__(self):
        super(SK_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 22)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(-1, 128 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

sf_model = SK_CNN()


@app.route('/')
def index():
    return render_template('index.html')

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def preprocess_img(img):
    # img = cv2.imread(os.path.join(image_path,filename), cv2.IMREAD_GRAYSCALE)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_image = binary_image.astype(np.uint8)  # 이미지 형식 변환
    distance_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    img = cv2.resize(distance_transform, (32, 32)) # 이미지 크기 조정
    img = img.transpose(1, 0)
    res = img.astype(np.float32) / 255.0

    return res

g_img = []
g_joints = []

@app.route('/input_img', methods=['POST'])
def request_sf():
    global g_img, g_joints

    b64 = request.get_data(as_text=True)
    g_img = data_uri_to_cv2_img(b64)
    cv2.imwrite('input_imgs/sf_input.png', g_img)

    g_joints = predict_joints(g_img).reshape(-1,2)

    res = {
        "joints": g_joints.tolist()
    }

    return res

def predict_joints(img):
    w = img.shape[1]
    print(img.shape)
    p_img = preprocess_img(img).reshape(1, 1, 32, 32)
    cv2.imwrite('p_sf_input.png', p_img[0][0]*255) 
    joints = sf_model(torch.tensor(p_img)).detach().numpy()[0]
    joints = np.round(joints*w)
    joints = np.clip(joints, 0, w-1)
    return joints


def move_to_local_min(df_img, x, y):
    min_x = x
    min_y = y
    min_val = df_img[y][x]
    for i in range(-1, 2):
        for j in range(-1, 2):
            xi = x + i
            yj = y + j
            if xi < 0 or xi > df_img.shape[1] - 1:
                continue
            if yj < 0 or yj > df_img.shape[0] - 1:
                continue
            if df_img[yj][xi] < min_val:
                min_val = df_img[yj][xi]
                min_x = xi
                min_y = yj

    return min_x, min_y

def move_to_local_max(df_img, x, y):
    max_x = x
    max_y = y
    max_val = df_img[y][x]
    for i in range(-1, 2):
        for j in range(-1, 2):
            xi = x + i
            yj = y + j
            if xi < 0 or xi > df_img.shape[1] - 1:
                continue
            if yj < 0 or yj > df_img.shape[0] - 1:
                continue
            if df_img[yj][xi] > max_val:
                max_val = df_img[yj][xi]
                max_x = xi
                max_y = yj

    return max_x, max_y




@app.route('/opt_joints', methods=['POST'])
def opt_joints():
    global g_joints, g_img

    if len(g_joints) != 11:
        return {"joints": []}

    img = cv2.imread('input_imgs/sf_input.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_image = binary_image.astype(np.uint8)  
    d_img = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    cv2.imwrite('d_img.png', d_img)

    print(request.get_data())
    k = request.get_json()["opt_times"]

    for _k in range(k):
        for i in range(11):
            x = int(g_joints[i][0])
            y = int(g_joints[i][1])

            if i == 2 :
                x, y = move_to_local_max(d_img, x, y)
            else:
                x, y = move_to_local_min(d_img, x, y)

            g_joints[i][0] = x
            g_joints[i][1] = y
    return {"joints": g_joints.tolist()}

if __name__ == '__main__':
    sf_model.load_state_dict(torch.load('model/trained_model.pt'))
    sf_model.eval()
    print(predict_joints(cv2.imread('input_imgs/sf_input.png')))
    app.run(debug=True, port=8888)
