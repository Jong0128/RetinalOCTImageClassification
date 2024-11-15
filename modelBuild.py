from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Activation


input_shape = (224, 224, 3)

resnet = ResNet50(
    include_top = True, # fully_connet_layer를 top_of_the_network에 사용여부
    weights = None, # 사전에 학습된 가중치를 사용할지 말지, None : random initialize, imagenet : pre_trained (on imagenet) 가중치를 사용함 If using `weights='imagenet'` with `include_top=True`, `classes` should be 1000. (capable of recognizing 1,000 different object categories, we encounter in our day-to-day lives )
    input_tensor = None,
    input_shape = (224,224,3),
    pooling = None, # include top이 false일 경우에만 pooling을 avg로 할 지 max로 할지 결정
    classes = 8, #최종 분류할 label의 개수
    classifier_activation = "softmax" # fully connected layer에서 사용할 activation 함수
    )

model = Model(inputs=resnet.input, outputs=resnet.output)
model.summary()

from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import  Model
from keras.layers import Dense,Input, Activation
### 위와는 다른 모델
### 최종 pooling 과정에서 max pooling을 사용,
### 초기 가중치로 imagenet data를 학습한 resnet 50의 가중치를 가져옴.

input_shape = (224, 224, 3)

resnet = ResNet50(
    include_top = None,
    weights = "imagenet",
    input_tensor = None,
    input_shape = (224,224,3),
    pooling = 'max',
    classes = None,
    classifier_activation = None
    )
x= resnet.output
x = Dense(8, activation='softmax')(x)



model = Model(inputs=resnet.input, outputs=x)
model.summary()

from keras.layers import Conv2D as Conv
from keras import layers
from keras.models import Model

### resnet의 conv layer중 conv 1,2 layer까지만 대략 구현해본 모델
###  skip_connection은 구현되어 있지 않음


def conv1 (x):
  x = Conv(filters = 64, kernel_size = (7,7), strides = 2, padding = 'same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPool2D(pool_size = (3,3), strides = 2, padding = 'same')(x)
  return x

def conv2 (x):
  for i in range(3):
    x = Conv(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = Conv(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters = 256, kernel_size = (1,1), strides = 1, padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
  return x

def conv (x):
  x = conv1(x)
  x = conv2(x)
  return x

inputs_layer = layers.Input(shape=(224,224,3))
conv_layer = conv(inputs_layer)
pooling_layer = layers.GlobalAveragePooling2D()(conv_layer)
outputs_layer = layers.Dense(8, activation='softmax')(pooling_layer)

model = Model(inputs=inputs_layer, outputs=outputs_layer, name = 'naive_resnet')
model.summary()

### Custom ResnNet

### Skip connection , Batch Normalization 모두 포함.

### 우선 직접 구현한 모델이 기존 import한 resnet과 유사한 정확도를 보이는지 확인해보기 위함.

from keras.layers import Conv2D as Conv
from keras.layers import BatchNormalization as BN
from keras.layers import Activation

from keras import layers
from keras.models import Model

def conv1 (x):
  x = Conv(filters = 64, kernel_size = (7,7), strides = 2, padding = 'same')(x)
  x = BN()(x)
  x = Activation('relu')(x)
  return x

def conv2 (x):

  x = layers.MaxPool2D(pool_size = (3,3), strides = 2, padding = 'same')(x)

  shortcut = x

  for i in range(3):

      x = Conv(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)

      x = Conv(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)

      x = layers.Conv2D(filters = 256, kernel_size = (1,1), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)
      # i=0 즉 첫번째 iteration인 경우에만 shortcut(이전 output)에 대하여 차원을 맞춰줄 필요가 있음.
      if i==0:
        shortcut = layers.Conv2D(filters = 256, kernel_size = (1,1), strides = 1, padding = 'same')(shortcut)
        shortcut = BN()(shortcut)
        shortcut = Activation('relu')(shortcut)

      x = layers.Add()([x,shortcut])
      shortcut = x


  return x

def conv3(x):

  shortcut = x

  for i in range(4):
    if i==0:
      x = Conv(filters = 128, kernel_size = (1,1), strides = 2, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)

      x = Conv(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)

      x = layers.Conv2D(filters = 512, kernel_size = (1,1), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)

      shortcut = layers.Conv2D(filters = 512, kernel_size = (1,1), strides = 2, padding = 'same')(shortcut)
      shortcut = BN()(shortcut)
      shortcut = Activation('relu')(shortcut)

      #x = layers.Add()([x,shortcut])
      #shortcut = x

    else :
      x = Conv(filters = 128, kernel_size = (1,1), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)

      x = Conv(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)

      x = layers.Conv2D(filters = 512, kernel_size = (1,1), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)

    x = layers.Add()([x,shortcut])
    shortcut = x


  return x

def conv4(x):

    shortcut = x

    for i in range(6):
      if i==0:
        x = Conv(filters = 256, kernel_size = (1,1), strides = 2, padding = 'same')(x)
        x = BN()(x)
        x = Activation('relu')(x)

        x = Conv(filters = 256, kernel_size = (3,3), strides = 1, padding = 'same')(x)
        x = BN()(x)
        x = Activation('relu')(x)

        x = layers.Conv2D(filters = 1024, kernel_size = (1,1), strides = 1, padding = 'same')(x)
        x = BN()(x)
        x = Activation('relu')(x)

        shortcut = layers.Conv2D(filters = 1024, kernel_size = (1,1), strides = 2, padding = 'same')(shortcut)
        shortcut = BN()(shortcut)
        shortcut = Activation('relu')(shortcut)

      else :
        x = Conv(filters = 256, kernel_size = (1,1), strides = 1, padding = 'same')(x)
        x = BN()(x)
        x = Activation('relu')(x)
        x = Conv(filters = 256, kernel_size = (3,3), strides = 1, padding = 'same')(x)
        x = BN()(x)
        x = Activation('relu')(x)
        x = layers.Conv2D(filters = 1024, kernel_size = (1,1), strides = 1, padding = 'same')(x)
        x = BN()(x)
        x = Activation('relu')(x)

      x = layers.Add()([x,shortcut])
      shortcut = x

    return x

def conv5(x):

  shortcut = x

  for i in range(3):
    if i==0 :
      x = Conv(filters = 512, kernel_size = (1,1), strides = 2, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)

      x = Conv(filters = 512, kernel_size = (3,3), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)

      x = Conv(filters = 2048, kernel_size = (1,1), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)

      shortcut = layers.Conv2D(filters = 2048, kernel_size = (1,1), strides = 2, padding = 'same')(shortcut)
      shortcut = BN()(shortcut)
      shortcut = Activation('relu')(shortcut)
    else :
      x = Conv(filters = 512, kernel_size = (1,1), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)
      x = Conv(filters = 512, kernel_size = (3,3), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)
      x = layers.Conv2D(filters = 2048, kernel_size = (1,1), strides = 1, padding = 'same')(x)
      x = BN()(x)
      x = Activation('relu')(x)

    x = layers.Add()([x,shortcut])
    shortcut = x

  return x

def conv_stage (x):
  x = conv1(x)
  x = conv2(x)
  x = conv3(x)
  x = conv4(x)
  x = conv5(x)
  return x

inputs_layer = layers.Input(shape=(224,224,3))
conv_layer = conv_stage(inputs_layer)
pooling_layer = layers.GlobalAveragePooling2D()(conv_layer)
outputs_layer = layers.Dense(8, activation='softmax')(pooling_layer)

model = Model(inputs=inputs_layer, outputs=outputs_layer, name = 'naive_resnet')
model.summary()

model.compile(
  loss='categorical_crossentropy', #target이 one hot vector 인 경우가 아닌 경우에는 sparse
  optimizer='adam',
  metrics=['accuracy']
)
