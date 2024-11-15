import seaborn as sns
import pandas as pd

# 파일 경로와 라벨 리스트
filepaths = []
labels = []

####-----------------------------------------   주의   ----------------------------------------------------####
#### 만약 리사이즈된 폴더 안에 바로 test, val, train이 있으면 상관없지만 만약 RetinalOCT_Dataset폴더가     ####
#### 있으면 밑에 있는 resize_path 수정해야함.                                                              ####
#### Ex. 위에 resize_path가 r'C:\Users\Jonghun\Desktop\Resize' 이렇게 했다면 밑에 있는                     ####
#### resize_path = r'C:\Users\Jonghun\Desktop\Resize\RetinalOCT_Dataset' 으로 수정. 만약 수정 필요 없다면  ####
#### 밑에 resize_path 주석처리 부탁.                                                                       ####
####-------------------------------------------------------------------------------------------------------####

# resize_path를 RetinalOCT_Dataset 폴더로 재설정
#resize_path = r'C:\Users\Jonghun\Desktop\Resize\RetinalOCT_Dataset'

# 데이터셋 분할 목록과 질병별 폴더 순회
for split in ["test", "val", "train"]:
    split_path = os.path.join(resize_path, split)
    for condition in os.listdir(split_path):
        condition_path = os.path.join(split_path, condition)
        if os.path.isdir(condition_path):
            for file in os.listdir(condition_path):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(condition_path, file))
                    labels.append(f"{split}_{condition}")  # 분할명과 질병명을 결합하여 라벨 지정

# 데이터프레임 생성
df = pd.DataFrame({"filepath": filepaths, "labels": labels})

# 각 라벨별 이미지 수 계산
counts = df["labels"].value_counts()

# 데이터 정보와 이미지 수 출력
print(df.info())
print(counts)

# 바 그래프 시각화
plt.figure(figsize=(12, 8))
sns.barplot(x=counts.index, y=counts.values)
plt.title('The Count of Images in Each Folder')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=90)  # x축 라벨 회전
plt.show()

import numpy as np

def save_as_npy_by_category(resize_path, npy_save_path):
    for main_dir in ["test", "val", "train"]:
        main_dir_path = os.path.join(resize_path, main_dir)

        # npy 저장 경로 설정 (폴더가 없으면 생성)
        main_save_path = os.path.join(npy_save_path, main_dir)
        if not os.path.exists(main_save_path):
            os.makedirs(main_save_path)

        for condition in os.listdir(main_dir_path):
            condition_path = os.path.join(main_dir_path, condition)

            # 이미지 데이터와 레이블 초기화
            data = []
            labels = []

            for root, _, files in os.walk(condition_path):
                for file in files:
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)

                    # 이미지가 None이 아니면 진행
                    if img is not None:
                        # 이미지를 배열에 추가 (BGR에서 RGB로 변환, 필요 시)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        data.append(img_rgb)
                        labels.append(condition)  # 질병 이름을 레이블로 사용

            # numpy 배열로 변환 및 저장
            data = np.array(data)
            labels = np.array(labels)
            np.save(os.path.join(main_save_path, f"{condition}_data.npy"), data)
            np.save(os.path.join(main_save_path, f"{condition}_labels.npy"), labels)
            print(f"Saved {condition} data and labels in {main_dir}")

#-----------------------------------경로수정필요-----------------------------------#
npy_save_path = r'' # NPY 파일 저장 경로

# 각 폴더별 npy 파일 생성
save_as_npy_by_category(resize_path, npy_save_path)

def print_npy_shapes(npy_save_path):
    # test, validation, train 폴더 각각에 접근
    for main_dir in ["test", "val", "train"]:
        main_dir_path = os.path.join(npy_save_path, main_dir)

        # 해당 폴더에 있는 모든 npy 파일 순회
        for file in os.listdir(main_dir_path):
            file_path = os.path.join(main_dir_path, file)

            # data 파일인 경우만 로드하고 shape 출력
            if file.endswith("_data.npy"):
                data = np.load(file_path)
                print(f"{main_dir} - {file} shape: {data.shape}")

# 각 data npy 파일의 shape 출력
print_npy_shapes(npy_save_path)

def show_first_image_of_each_data(npy_save_path):
    # test, validation, train 폴더 각각에 접근
    for main_dir in ["test", "val", "train"]:
        main_dir_path = os.path.join(npy_save_path, main_dir)

        # 해당 폴더에 있는 모든 data npy 파일 순회
        for file in os.listdir(main_dir_path):
            if file.endswith("_data.npy"):
                file_path = os.path.join(main_dir_path, file)

                # npy 파일 로드 및 첫 번째 이미지 가져오기
                data = np.load(file_path)
                if data.size > 0:
                    first_image = data[0]

                    # 첫 번째 이미지 표시
                    plt.imshow(first_image)
                    plt.title(f"{main_dir} - {file}")
                    plt.axis("off")
                    plt.show()


# 각 data npy 파일의 첫 번째 이미지 표시
show_first_image_of_each_data(npy_save_path)

# 데이터셋 분할 목록
data_splits = ["test", "val", "train"]

# 8가지 질병 목록
conditions = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]

# 출력 배열을 자동으로 중간 생략하도록 설정
np.set_printoptions(threshold=100)  # threshold 값을 적절히 낮게 설정

for split in data_splits:
    print(f"\n===== {split.upper()} FOLDER =====\n")
    npy_base_path = os.path.join(npy_save_path, split)

    for condition in conditions:
        # 각 질병의 npy 파일 경로
        npy_file_path = os.path.join(npy_base_path, f"{condition}_data.npy")

        # npy 파일 로드
        data = np.load(npy_file_path)  # (샘플 수, 224, 224, 3)

        # 차원을 합쳐서 각 샘플의 평균값 계산
        flattened_data = data.reshape(data.shape[0], -1)  # (샘플 수, 224*224*3)
        sample_means = flattened_data.mean(axis=1)  # 각 샘플에 대해 평균 계산

        # 평균값 배열 출력
        print(f"{split} - {condition} sample means:")
        print(sample_means)
        print("\n" + "="*50 + "\n")  # 구분선

for split in data_splits:
    for condition in conditions:
        # 각 질병의 npy 파일 경로
        npy_file_path = os.path.join(npy_save_path, split, f"{condition}_data.npy")

        # npy 파일 로드 및 각 샘플의 평균값 계산
        data = np.load(npy_file_path)  # (샘플 수, 224, 224, 3)

        # 차원을 합쳐서 각 샘플의 평균값 계산
        flattened_data = data.reshape(data.shape[0], -1)  # (샘플 수, 224*224*3)
        sample_means = flattened_data.mean(axis=1)  # 각 샘플에 대해 평균 계산

        # 바 그래프 그리기
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(sample_means)), sample_means)
        plt.xlabel("Image Number")
        plt.ylabel("Average RGB Value")
        plt.title(f"{split.capitalize()} - Distribution of Average RGB Values Across {condition} Samples")
        plt.show()

  for split in data_splits:
    for condition in conditions:
        # 각 질병의 npy 파일 경로
        npy_file_path = os.path.join(npy_save_path, split, f"{condition}_data.npy")

        # npy 파일 로드
        data = np.load(npy_file_path)  # (샘플 수, 224, 224, 3)

        # 모든 이미지의 평균 이미지 생성
        average_image = np.mean(data, axis=0)  # 각 픽셀의 RGB 값을 평균

        # 평균 이미지 시각화
        plt.figure(figsize=(6, 6))
        plt.imshow(average_image.astype(np.uint8))  # 평균값을 uint8로 변환하여 시각화
        plt.title(f"Average Image of {condition} Data in {split.capitalize()} Set")
        plt.axis("off")
        plt.show()
