# 학습 결과를 `history` 객체에서 가져오기
history_dict = history.history

# 손실 (loss)과 정확도 (accuracy) 값 가져오기
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

# 그래프 크기 설정
plt.figure(figsize=(12, 5))

# 첫 번째 그래프: 손실 (Loss)
plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 서브 플롯
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 두 번째 그래프: 정확도 (Accuracy)
plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 서브 플롯
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 그래프 출력
plt.tight_layout()
plt.show()

# 테스트 데이터셋의 클래스 이름 불러오기
class_names = test_ds.class_names

# 테스트 이미지와 라벨의 한 배치 가져오기
for images, labels in test_ds.take(1):
    # 배치에서 몇 개의 랜덤한 인덱스 선택 (표시할 이미지 수)
    num_samples = 5  # 표시할 이미지의 개수
    random_indices = np.random.choice(range(images.shape[0]), num_samples, replace=False)

    plt.figure(figsize=(15, 15))

    for i, idx in enumerate(random_indices):
        ax = plt.subplot(1, num_samples, i + 1)

        # 선택된 이미지와 실제 라벨 가져오기
        img = images[idx]
        true_label = np.argmax(labels[idx])

        # 모델의 입력 형태에 맞게 이미지 차원 확장
        img_array = np.expand_dims(img, axis=0)

        # 모델의 예측 수행
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions)

        # 이미지와 실제 라벨, 예측 라벨 표시
        plt.imshow(img.numpy().astype("uint8"))
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}")
        plt.axis("off")

    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# 데이터셋 로드
batch_size = 256
test_ds = tf.keras.utils.image_dataset_from_directory(
    '../resize_path/test',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=batch_size,
    shuffle=True
)


# 데이터셋의 클래스 이름 추출
class_names = test_ds.class_names  # 레이블 이름 (예: ['1 레이블', '2 레이블', ..., '8 레이블'])
print("Class names:", class_names)

# 각 레이블 폴더에서 이미지 한 개씩 가져오기
images_to_display = []
labels_to_display = []
found_labels = set()

for images, labels in test_ds.unbatch():
    true_label_index = np.argmax(labels.numpy())
    true_label = class_names[true_label_index]

    # 레이블이 아직 리스트에 추가되지 않은 경우에만 추가
    if true_label not in found_labels:
        images_to_display.append(images.numpy())
        labels_to_display.append(true_label)
        found_labels.add(true_label)

    # 모든 레이블에 대해 한 개씩 이미지가 선택되면 종료
    if len(found_labels) == len(class_names):
        break

# 모델로 예측
predictions = model.predict(np.array(images_to_display))

# 이미지 출력 및 레이블 출력
plt.figure(figsize=(20, 10))

for i in range(len(images_to_display)):
    ax = plt.subplot(2, 4, i + 1)
    img = images_to_display[i]
    true_label = labels_to_display[i]
    predicted_label = class_names[np.argmax(predictions[i])]

    plt.imshow(img.astype("uint8"))
    plt.title(f"True: {true_label}, Pred: {predicted_label}")
    plt.axis("off")

plt.show()
