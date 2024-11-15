import os
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf

#-----------------------------------경로수정필요-----------------------------------
# 전처리 전 이미지의 절대 경로
path = ''
# crop 전처리 후 이미지의 절대 경로
cropped_path = ''
# padding 전처리 후 이미지의 절대 경로
padding_path = ''
# resize 전처리 후 이미지의 절대 경로
resize_path = ''

img_size = (224, 224)

  def crop_n_resize(path, resized_path):
    for root, dirs, files in os.walk(path):
        # 하위 디렉토리들에 있는 모든 파일들
        for file in files:
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            # '.DS_Store' 이런 파일은 넘어가기
            if img is None:
                print(f"Nonetype: {img_path}")
                continue

            # height, width
            height, width = img.shape[0], img.shape[1]

            # height, width 중 짧은 쪽에 맞추어 가운데에서 1:1로 crop
            if height > width:
                leftTopY = int(height / 2 - width / 2)
                rightBottomY = leftTopY + width
                # 왼쪽 위 좌표 (0, leftTopY) 오른쪽 아래 좌표 (width, rightBottomY) 로 crop
                cropped_img = img[leftTopY : rightBottomY, 0 : width]

            elif width > height:
                leftTopX = int(width / 2 - height / 2)
                rightBottomX = leftTopX + height
                # 왼쪽 위 좌표 (leftTopX, 0) 오른쪽 아래 좌표 (rightBottomX, height) 로 crop
                cropped_img = img[0 : height, leftTopX : rightBottomX]

            else:
                cropped_img = img

            # 224 * 224로 resizing
            resized_img = cv2.resize(cropped_img, img_size)

            # 전처리한 이미지의 base_path에 나머지 디렉토리를 붙여서 저장할 위치 생성
            resized_dir_path = os.path.join(resized_path, os.path.relpath(root, path))
            # 만약 해당 디렉토리가 존재하지 않으면 생성
            if not os.path.exists(resized_dir_path):
                os.makedirs(resized_dir_path)

            resized_img_path = os.path.join(resized_dir_path, file)
            cv2.imwrite(resized_img_path, resized_img)

                  def resize_n_padding(path, resized_path):
    for root, dirs, files in os.walk(path):
        # 하위 디렉토리들에 있는 모든 파일들
        for file in files:
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            # '.DS_Store' 파일은 넘어가기
            if img is None:
                print(f"Nonetype: {img_path}")
                continue

            # height, width
            height, width = img.shape[0], img.shape[1]

            # 원본 이미지의 비율
            img_ratio = float(img_size[0]) / max(height, width)

            # 원본 이미지의 height, width 중 큰 쪽이 224가 되도록 맞추어서 resize
            resized_img = cv2.resize(img, (int(width * img_ratio), int(height * img_ratio)))

            # resize한 이미지의 세로 여백
            vertical_margin = img_size[0] - resized_img.shape[0]
            padding_top =  vertical_margin // 2
            padding_bottom = vertical_margin - padding_top

            # resize한 이미지의 가로 여백
            horizontal_margin = img_size[1] - resized_img.shape[1]
            padding_left = horizontal_margin // 2
            padding_right = horizontal_margin - padding_left

            # resize한 이미지가 224 * 224가 될 수 있도록 패딩 추가
            black_padding = [0, 0, 0]
            with_padding_img = cv2.copyMakeBorder(
                resized_img,
                padding_top,
                padding_bottom,
                padding_left,
                padding_right,
                cv2.BORDER_CONSTANT,
                value = black_padding
            )

            # 전처리한 이미지의 base_path에 나머지 디렉토리를 붙여서 저장할 위치 생성
            resized_dir_path = os.path.join(resized_path, os.path.relpath(root, path))
            # 만약 해당 디렉토리가 존재하지 않으면 생성
            if not os.path.exists(resized_dir_path):
                os.makedirs(resized_dir_path)

            resized_img_path = os.path.join(resized_dir_path, file)
            cv2.imwrite(resized_img_path, with_padding_img)

              def just_resize(path, resized_path):
    # dimension이 다른 파일의 개수
    wrong_file_count = 0

    for root, dirs, files in os.walk(path):
        for file in files:
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            # '.DS_Store' 파일은 넘어가기
            if img is None:
                print(f"Nonetype: {img_path}")
                continue

            # 224 * 224로 resize
            resized_img = cv2.resize(img, img_size)

            # 전처리한 이미지의 base_path에 나머지 디렉토리를 붙여서 저장할 위치 생성
            resized_dir_path = os.path.join(resized_path, os.path.relpath(root, path))
            # 만약 해당 디렉토리가 존재하지 않으면 생성
            if not os.path.exists(resized_dir_path):
                os.makedirs(resized_dir_path)

            resized_img_path = os.path.join(resized_dir_path, file)
            cv2.imwrite(resized_img_path, resized_img)

      def check_dimensions(path):
    # dimension이 다른 파일의 개수
    wrong_file_count = 0

    for root, dirs, files in os.walk(path):
        for file in files:
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            # .DS_Store 파일 제외
            if img is not None:
                # 224 * 224 * 3이 아닌 파일 탐색
                if not img.shape == (224, 224, 3):
                    print(f"Dimensions Not Match: {img_path}")
                    wrong_file_count = wrong_file_count + 1
                    continue

    if wrong_file_count == 0:
        print("All Good")

      
# # crop 후 resize
# crop_n_resize(path, cropped_path)
#
# # dimension 확인
# check_dimensions(cropped_path)
#
#
# # resize 후 padding 추가
# resize_n_padding(path, padding_path)
#
# # dimension 확인
# check_dimensions(padding_path)


# resize
just_resize(path, resize_path)

# dimension 확인
check_dimensions(resize_path)
