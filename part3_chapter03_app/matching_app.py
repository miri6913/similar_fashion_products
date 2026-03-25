from flask import Flask, request, jsonify
import faiss
from PIL import Image
import io
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import json
import torch
import torch.nn as nn
from torchvision import models
import timm

from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import pickle

app = Flask(__name__)

class BranchClassifier(nn.Module):
    def __init__(self, num_classes_list, num_branches=4, model_name='efficientnet_b0', pretrained=True):
        super(BranchClassifier, self).__init__()

        # backbone 모델 로드
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1280, 256),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes_list[i])
                )
                for i in range(num_branches)
            ]
        )

    def forward(self, x):
        # backbone 모델을 통해 이미지 특징 추출
        features = self.backbone(x)
        # 각 분기별로 특징을 입력으로 받아 예측 수행
        outputs = [branch(features) for branch in self.branches]

        return outputs, features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 생성
model_save_dir = '/content/drive/MyDrive/Colab Notebooks/fast_campus_image_processing/similar_fashion_products/train_results'
num_branches = 4
with open(os.path.join(model_save_dir, 'detail_category_list.json'), 'r')as json_f:
    detail_category_list = json.load(json_f)
with open(os.path.join(model_save_dir, 'color_list.json'), 'r')as json_f:
    color_list = json.load(json_f)
with open(os.path.join(model_save_dir, 'fit_list.json'), 'r')as json_f:
    fit_list = json.load(json_f)
with open(os.path.join(model_save_dir, 'length_list.json'), 'r')as json_f:
    length_list = json.load(json_f)
num_classes_list = [len(detail_category_list), len(color_list), len(fit_list), len(length_list)]
model = BranchClassifier(num_classes_list=num_classes_list, num_branches=num_branches, pretrained=False).eval().to(device)
weight = torch.load(os.path.join(model_save_dir, 'best_model.pth'), map_location='cpu')
model.load_state_dict(weight)
model.eval()

transform = A.Compose([
    A.LongestMaxSize(max_size=224,
                     always_apply=True),
    A.PadIfNeeded(min_height=224,
                  min_width=224,
                  always_apply=True,
                  border_mode=0),
    A.Normalize(p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

## 인덱스 로드
index_save_dir = '/content/drive/MyDrive/Colab Notebooks/fast_campus_image_processing/similar_fashion_products/index'
index = faiss.read_index(os.path.join(index_save_dir, 'ivf_index.index'))
with open(os.path.join(index_save_dir, 'image_path_id_list.json') , 'r') as json_f:
    image_path_id_list = json.load(json_f)

## pca 로드
vector_save_dir = '/content/drive/MyDrive/Colab Notebooks/fast_campus_image_processing/similar_fashion_products/vectors'
normalizer = Normalizer(norm='l2')
pca_path = os.path.join(vector_save_dir, 'pca_model.pkl')
with open(pca_path, 'rb') as f:
    pca = pickle.load(f)


def predict_image(image_bytes):
    # 이미지를 PIL Image 객체로 변환
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # 이미지를 모델 입력에 맞게 전처리

    input_tensor = transform(image=np.array(image))['image']
    input_batch = input_tensor.unsqueeze(0)  # 모델은 배치를 기대하므로 차원을 추가합니다.

    # 이미지를 GPU 또는 CPU로 전송
    input_batch = input_batch.to(device)

    # 모델에 입력하여 예측
    with torch.no_grad():
        output, feature = model(input_batch)

    # 예측 결과를 확률로 변환
    probs = nn.functional.softmax(output[0], dim=0)
    preds = torch.argmax(probs).item()
    
    feature = feature.detach().cpu().numpy().tolist()

    # 예측 결과 반환
    return preds, feature

def matching(feature):
    # 1. L2 정규화
    norm_feature = normalizer.fit_transform(feature)

    # 2. PCA 적용 (128차원 으로 압축)
    comp_feature = pca.transform(norm_feature)

    # 3. 매칭
    distances, indices = index.search(comp_feature, 8)

    ## 매칭 결과 파일이름을 매핑한다.
    matched_files = []
    for idx in indices[0]:
        matched_files.append(image_path_id_list[idx])
    
    return distances, matched_files


# POST 요청을 처리하는 라우트 설정
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # POST 요청으로 받은 이미지 데이터를 처리
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            image_bytes = file.read()
            preds, feature = predict_image(image_bytes)
            distances, matched_files = matching(feature)

            return jsonify({'distances': distances[0].tolist(),
                            'matched_files': matched_files})

if __name__ == '__main__':
    app.run(debug=True)