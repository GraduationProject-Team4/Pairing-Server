import os
import shutil
import time

from flask import Flask, request, jsonify
import pickle
import torch
from torch import device
from audioutils import spec_to_image, get_melspectrogram_db
from demucsUtils import separate_and_save

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

SEPARATED_FOLDER = 'separated'  # 분리된 WAV 파일을 저장할 폴더
if not os.path.exists(SEPARATED_FOLDER):
    os.makedirs(SEPARATED_FOLDER)


@app.route('/')
def main():
    return jsonify({'message': 'This is a model server'}), 200


# 앱에서 전송된 wav를 여러 wav로 분리해서 seperated 디렉토리에 저장.
@app.route('/pairing', methods=['POST'])
def seperate_audio():
    UPLOAD_FOLDER = 'uploads'  # 업로드한 WAV 파일을 저장할 폴더
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    SEPARATED_FOLDER = 'separated'  # 분리된 WAV 파일을 저장할 폴더
    if not os.path.exists(SEPARATED_FOLDER):
        os.makedirs(SEPARATED_FOLDER)

    # 분리되면 여러 파일이 생길텐데 이런 파일들은 어디에 저장하지..?
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']  # .wav 파일

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):

        origin_filename = file.filename
        filename = os.path.join(UPLOAD_FOLDER, file.filename)  # 저장할 필요 없을 수도 있음.
        file.save(filename)
        print("==================")
        print("file uploaded")
        file_size = os.path.getsize('uploads/' + file.filename)
        print('File Size:', convert_size(file_size), 'bytes')
        print("==================")
        # 음성 분리
        separate_and_save('uploads' + '/' + file.filename, SEPARATED_FOLDER)
        # separate_and_save(file, SEPARATED_FOLDER)
        print("audio separated")
        # 음성 분류
        predictionList = predict_audio(file.filename)
        print(predictionList)
        print("audio predicted")

        # delete_directory_contents(UPLOAD_FOLDER)
        # delete_directory_contents(SEPARATED_FOLDER)

        return jsonify({'filename': file.filename,
                        'predictMessage': 'File predicted successfully', 'audioPrediction': predictionList}), 200
        # return jsonify({'audioPrediction': predictionList}), 200
        # return jsonify({'message': 'File uploaded successfully'}, {'filename': file.filename}), 200
    else:
        return jsonify({'error': 'Invalid file format. Only .wav files are allowed.'}), 400


# 앱에서 wav 파일을 받고 스펙토그램으로 전환
# --> 이 아니라 앱에서 받아온 복합적인 wav 파일이 단일 wav로 분리되면 각각의 wav가 어떤 음성인지 예측
# uploads 디렉토리를 돌면서 파일마다 예측값을 도출해서 리스트로 저장
# @app.route('/predict', methods=['POST'])  # 아마 api가 아닌 그냥 함수로 사용할 듯
def predict_audio(filename):
    audio_list = []
    path_dir = 'separated/htdemucs/' + os.path.splitext(filename)[0]
    file_list = os.listdir(path_dir)
    for files in file_list:
        file_path = path_dir + '/' + files
        spec = spec_to_image(get_melspectrogram_db(file_path))
        spec_t = torch.tensor(spec).to(device, dtype=torch.float32)
        pr = resnet_model.forward(spec_t.reshape(1, 1, *spec_t.shape))
        ind = pr.argmax(dim=1).cpu().detach().numpy().ravel()[0]
        audio_list.append(indtocat[ind])
    return audio_list
    # delete_directory_contents(path_dir)
    # return jsonify({'message': 'File predicted successfully'}, {'audio prediction': audio_list}), 200


def allowed_file(filename):
    # 파일 확장자 validation
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'wav'


def delete_directory_contents(directory_path):
    try:
        shutil.rmtree(directory_path)  # 디렉토리와 하위 내용 삭제
        print(f"Directory '{directory_path}' All internal content has been deleted.")
    except OSError as e:
        print(f"Directory '{directory_path}' Error deleting internal content: {e}")


def convert_size(size_bytes):
    import math
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5050", debug=True)
