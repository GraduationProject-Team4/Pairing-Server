# Overview
Server Code for the Pairing App, a Daily Living Assistance Application for People with Hearing Impairments
- Receives a voice file with an extension ".wav" from the client as an input value.
- The file entered as the input value is ambient sound. This ambient sound is separated into a single voice file.
- It analyzes the separated voice files and outputs what type of voice it is.
---
# Installation Environment & Library
> better if you refer to the document "requirements.txt"
- python: 3.9.6
- **librosa: 0.8.0**
- numpy: 1.23.4
- **demucs: 4.0.1**
- torch: 2.0.1
- torchaudio: 2.0.2
- torchvision: 0.15.2
---
# API
## Uri: "/pairing"
- Method Type: post
- Request Body:
    form-data
    - key: "file"
    - value: file with an extension ".wav"
    - ![image](https://github.com/GraduationProject-Team4/Pairing-Server/assets/101577272/9f4d6b19-e4f3-46ef-a779-45a2120db0ff)
- Response:
  - <img width="1090" alt="pairing" src="https://github.com/GraduationProject-Team4/Pairing-Server/assets/101577272/63fdc840-1a46-48a2-9e00-afb422a2f5ed">

- Description:
  The ambient sound voice file received as input from the client is separated into up to four single voice files using demucs, and each file is analyzed through the librosa library and classified using an artificial intelligence model after preprocessing.
