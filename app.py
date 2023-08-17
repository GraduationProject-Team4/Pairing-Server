import os
import shutil

from flask import Flask, request, jsonify
import pickle
import torch
from torch import device
from audioutils import spec_to_image, get_melspectrogram_db

app = Flask(__name__)

# GPU 사용 가능 시
if torch.cuda.is_available():
    device = torch.device('cuda:0')

# GPU 사용 못하면 그냥 CPU 사용하여 연산
else:
    device = torch.device('cpu')

with open('saved/indtocat.pkl', 'rb') as f:
    indtocat = pickle.load(f)

resnet_model = torch.load('saved/esc50resnet.pth', map_location=device)

UPLOAD_FOLDER = 'uploads'  # 업로드한 WAV 파일을 저장할 폴더
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

SEPERATED_FOLDER = 'seperated'  # 분리된 WAV 파일을 저장할 폴더
if not os.path.exists(SEPERATED_FOLDER):
    os.makedirs(SEPERATED_FOLDER)


@app.route('/')
def main():
    return jsonify({'message': 'This is a model server'}), 200


# 앱에서 전송된 wav를 여러 wav로 분리해서 seperated 디렉토리에 저장.
@app.route('/upload', methods=['POST'])
def seperate_wav():
    # 분리되면 여러 파일이 생길텐데 이런 파일들은 어디에 저장하지..?
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']  # .wav 파일

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # 파일 업로드 성공시 수행할 코드 작성
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        return jsonify({'message': 'File uploaded successfully'}, {'filename': file.filename}), 200
    else:
        return jsonify({'error': 'Invalid file format. Only WAV files are allowed.'}), 400


# 앱에서 wav 파일을 받고 스펙토그램으로 전환
# --> 이 아니라 앱에서 받아온 복합적인 wav 파일이 단일 wav로 분리되면 각각의 wav가 어떤 음성인지 예측
# uploads 디렉토리를 돌면서 파일마다 예측값을 도출해서 리스트로 저장
@app.route('/predict', methods=['POST'])  # 아마 api가 아닌 그냥 함수로 사용할 듯
def predict_audio():
    audio_list = []
    path_dir = 'uploads'
    file_list = os.listdir(path_dir)
    for files in file_list:
        file_path = path_dir + '/' + files
        # print('file name is ', file.filename)
        spec = spec_to_image(get_melspectrogram_db(file_path))
        spec_t = torch.tensor(spec).to(device, dtype=torch.float32)
        pr = resnet_model.forward(spec_t.reshape(1, 1, *spec_t.shape))
        ind = pr.argmax(dim=1).cpu().detach().numpy().ravel()[0]
        audio_list.append(indtocat[ind])
    delete_directory_contents(path_dir)
    return jsonify({'message': 'File predicted successfully'}, {'audio prediction': audio_list}), 200


def allowed_file(filename):
    # 파일 확장자 validation
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'wav'


def delete_directory_contents(directory_path):
    try:
        shutil.rmtree(directory_path)  # 디렉토리와 하위 내용 삭제
        print(f"Directory '{directory_path}' All internal content has been deleted.")
    except OSError as e:
        print(f"Directory '{directory_path}' Error deleting internal content: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5050", debug=True)
