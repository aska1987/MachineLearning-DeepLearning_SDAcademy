# -*- coding: utf-8 -*-

# 딥러닝 관련 Keras 라이브러리
import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator

# File I/O
import subprocess
import shutil
import os
from glob import glob
from datetime import datetime
import argparse

# 데이터 처리
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
# 이미지 처리
import cv2
from scipy.ndimage import rotate
import scipy.misc

# 학습 파라미터를 설정한다
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, default='vgg16', help='Model Architecture')
parser.add_argument('--weights', required=False, default='None')
parser.add_argument('--learning-rate', required=False, type=float, default=1e-4)
parser.add_argument('--semi-train', required=False, default=None)
parser.add_argument('--batch-size', required=False, type=int, default=8)
parser.add_argument('--random-split', required=False, type=int, default=1)
parser.add_argument('--data-augment', required=False, type=int, default=0)
args = parser.parse_args()
import sklearn
print(sklearn.__version__)
fc_size = 2048
n_class = 10
seed = 10
nfolds = 3
test_nfolds = 3
img_row_size, img_col_size = 224, 224
train_path = 'C:\\Users\\SDEDU\\.kaggle\\competitions\\state-farm-distracted-driver-detection\\input\\train'
if args.semi_train is not None:
    train_path = args.semi_train
    args.semi_train = True
test_path ='C:\\Users\\SDEDU\\.kaggle\\competitions\\state-farm-distracted-driver-detection\\input\\test'
labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']



def _clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

text=int(input('1. 이미지 새로 생성 2. 기존 이미지 \n' ))
if text==1:
    suffix = 'm{}.w{}.lr{}.s{}.nf{}.semi{}.b{}.row{}col{}.rsplit{}.augment{}.d{}'.format(args.model, args.weights, args.learning_rate, seed, nfolds, args.semi_train, args.batch_size, img_row_size, img_col_size, args.random_split, args.data_augment, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    temp_train_fold = 'C:\\Users\\SDEDU\\.kaggle\\competitions\\state-farm-distracted-driver-detection\\input\\train_{}'.format(suffix)
    temp_valid_fold = 'C:\\Users\\SDEDU\\.kaggle\\competitions\\state-farm-distracted-driver-detection\\input\\valid_{}'.format(suffix)
    cache = 'C:\\Users\\SDEDU\\.kaggle\\competitions\\state-farm-distracted-driver-detection\\cache\\{}'.format(suffix)
    subm = 'C:\\Users\\SDEDU\\.kaggle\\competitions\\state-farm-distracted-driver-detection\\subm\\{}'.format(suffix)

    for path in [temp_train_fold, temp_valid_fold, cache, subm]:
        _clear_dir(path)

    
elif text==2:       
    date_text=input('원하는 날짜의 데이터 입력(ex 2018-12-18-12-22) : \n')
    
    #suffix = 'mvgg16.wNone.lr0.0001.s10.nf5.semiNone.b8.row224col224.rsplit1.augment0.d2018-12-18-12-21'
    suffix = 'mvgg16.wNone.lr0.0001.s10.nf5.semiNone.b8.row224col224.rsplit1.augment0.d'+date_text
    temp_train_fold = 'C:\\Users\\SDEDU\\.kaggle\\competitions\\state-farm-distracted-driver-detection\\input\\train_{}'.format(suffix)
    temp_valid_fold = 'C:\\Users\\SDEDU\\.kaggle\\competitions\\state-farm-distracted-driver-detection\\input\\valid_{}'.format(suffix)
    cache = 'C:\\Users\\SDEDU\\.kaggle\\competitions\\state-farm-distracted-driver-detection\\cache\\{}'.format(suffix)
    subm = 'C:\\Users\\SDEDU\\.kaggle\\competitions\\state-farm-distracted-driver-detection\\subm\\{}'.format(suffix)

def get_model():
    print('line 67! get model')
    # 최상위 전결층을 제외한 모델을 불러온다
    if args.weights == 'None':
        args.weights = None
    if args.model in ['vgg16']:
        base_model = keras.applications.vgg16.VGG16(include_top=False, weights=args.weights, input_shape=(img_row_size, img_col_size,3))
    elif args.model in ['vgg19']:
        base_model = keras.applications.vgg19.VGG19(include_top=False, weights=args.weights, input_shape=(img_row_size, img_col_size,3))
    elif args.model in ['resnet50']:
        base_model = keras.applications.resnet50.ResNet50(include_top=False, weights=args.weights, input_shape=(img_row_size, img_col_size,3))
    else:
        print('# {} is not a valid value for "--model"'.format(args.model))
        exit()

    # 최상위 전결층을 정의한다
    out = Flatten()(base_model.output)
    out = Dense(fc_size, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(fc_size, activation='relu')(out)
    out = Dropout(0.5)(out)
    output = Dense(n_class, activation='softmax')(out)
    model = Model(inputs=base_model.input, outputs=output)

    # SGD Optimizer를 사용하여, 모델을 compile한다
    sgd = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def read_image(path):
    print('line 96 read image')
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

print('# Data Load')
drivers = pd.read_csv('C:\\Users\SDEDU\.kaggle\competitions\state-farm-distracted-driver-detection\driver_imgs_list.csv\driver_imgs_list.csv')
img_to_driver = {}
uniq_drivers = []

for i, row in drivers.iterrows():
    label_n_driver = {}
    label_n_driver['label'] = row['classname']
    label_n_driver['driver'] = row['subject']
    img_to_driver[row['img']] = label_n_driver

    if row['subject'] not in uniq_drivers:
        uniq_drivers.append(row['subject'])

def generate_driver_based_split(img_to_driver, train_drivers):
    print('line 116 split')
    # 이미지 생성기를 위하여 임시 훈련/검증 폴더를 생성한다
    def _generate_temp_folder(root_path):
        _clear_dir(root_path)
        for i in range(n_class):
            os.mkdir('{}/c{}'.format(root_path, i))
    _generate_temp_folder(temp_train_fold)
    _generate_temp_folder(temp_valid_fold)

    # 임시 훈련/검증 폴더에 데이터를 랜덤하게 복사한다
    train_samples = 0
    valid_samples = 0
    if not args.random_split:
        for img_path in img_to_driver.keys():
            cmd = 'copy {}\\{}\{} {}\{}\{}'
            label = img_to_driver[img_path]['label']
            if not os.path.exists('{}\{}\{}'.format(train_path, label, img_path)):
                continue
            if img_to_driver[img_path]['driver'] in train_drivers:
                cmd = cmd.format(train_path, label, img_path, temp_train_fold, label, img_path)
                train_samples += 1
            else:
                cmd = cmd.format(train_path, label, img_path, temp_valid_fold, label, img_path)
                valid_samples += 1
            # copy image
            ##print('cmd:',cmd)
            subprocess.call(cmd, stderr=subprocess.STDOUT, shell=True)
    else:
        for label in labels:
            files = glob('{}\{}\*jpg'.format(train_path, label))
            for fl in files:
                cmd = 'copy {} {}\{}\{}'
                if np.random.randint(nfolds) != 1:
                    # 데이터의 4/5를 훈련 데이터에 추가한다
                    cmd = cmd.format(fl, temp_train_fold, label, os.path.basename(fl))
                    train_samples += 1
                else:
                    # 데이터의 1/5를 검증 데이터에 추가한다
                    cmd = cmd.format(fl, temp_valid_fold, label, os.path.basename(fl))
                    valid_samples += 1
                # 원본 훈련 데이터를 임시 훈련/검증 데이터에 복사한다
                ##print('cmd:',cmd)
                subprocess.call(cmd, stderr=subprocess.STDOUT, shell=True)

    # 훈련/검증 데이터 개수를 출력한다
    print('# {} train samples | {} valid samples'.format(train_samples, valid_samples))
    return train_samples, valid_samples

def crop_center(img, cropx, cropy):
    print('line 163 crop center')
    # 이미지 중간을 Crop하는 함수를 정의한다
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def preprocess(image):
    print('line 171 preprocess')
    # rescale
    image /= 255.

    # rotate
    rotate_angle = np.random.randint(40) - 20
    image = rotate(image, rotate_angle)

    # translate
    rows, cols, _ = image.shape
    width_translate = np.random.randint(60) - 30
    height_translate = np.random.randint(60) - 30
    M = np.float32([[1,0,width_translate],[0,1,height_translate]])
    image = cv2.warpAffine(image,M,(cols,rows))    
    print('line 180!')
    # zoom
    width_zoom = int(img_row_size * (0.8 + 0.2 * (1 - np.random.random())))
    height_zoom = int(img_col_size * (0.8 + 0.2 * (1 - np.random.random())))
    final_image = np.zeros((height_zoom, width_zoom, 3))
    final_image[:,:,0] = crop_center(image[:,:,0], width_zoom, height_zoom)
    final_image[:,:,1] = crop_center(image[:,:,1], width_zoom, height_zoom)
    final_image[:,:,2] = crop_center(image[:,:,2], width_zoom, height_zoom)

    # resize
    image = cv2.resize(final_image, (img_row_size, img_col_size))
    return image

print('# Train Model')
# 이미지 데이터 전처리를 수행하는 함수를 정의한다
# 실시간 전처리를 추가할 경우, 전처리 함수를 설정값에 넣어준다
if args.data_augment:
    datagen = ImageDataGenerator(preprocessing_function=preprocess)
else:
    datagen = ImageDataGenerator() # 데이터를 이리저리 변형시켜서 새로운 학습 데이터를 만들어줌

# 테스트 데이터를 불러오는 ImageGenerator를 생성한다
test_generator = datagen.flow_from_directory(
        test_path,
        target_size=(img_row_size, img_col_size),
        batch_size=1,
        class_mode=None,
        shuffle=False)
test_id = [os.path.basename(fl) for fl in glob('{}/imgs/*.jpg'.format(test_path))]

# 운전자별 5-Fold 교차 검증을 진행한다
kf = KFold(len(uniq_drivers), n_folds=nfolds, shuffle=True, random_state=20)
for fold, (train_drivers, valid_drivers) in enumerate(kf):
    # 새로운 모델을 정의한다
    model = get_model()

    # 훈련/검증 데이터를 생성한다
    print('line 217!')
    train_drivers = [uniq_drivers[j] for j in train_drivers]
    ####train_samples, valid_samples = generate_driver_based_split(img_to_driver, train_drivers)
    

    if text==1:
        train_samples, valid_samples = generate_driver_based_split(img_to_driver, train_drivers)
    
    elif text==2:
        train_samples=17909
        valid_samples=4515

    # 훈련/검증 데이터 생성기를 정의한다
    train_generator = datagen.flow_from_directory(
            directory=temp_train_fold,
            target_size=(img_row_size, img_col_size),
            batch_size=args.batch_size,
            class_mode='categorical',
            seed=seed)
    valid_generator = datagen.flow_from_directory(
            directory=temp_valid_fold,
            target_size=(img_row_size, img_col_size),
            batch_size=args.batch_size,
            class_mode='categorical',
            seed=seed)

    weight_path = 'C://Users/SDEDU/.kaggle/competitions/state-farm-distracted-driver-detection/cache/{}/weight.fold_{}.h5'.format(suffix, fold)
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),
            ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, verbose=0)]
    # 모델을 학습한다. val_loss 값이 3 epoch 연속 개악되면, 학습을 멈추고 최적 weight를 저장한다 
    #EarlyStopping: 조기학습 중단
    #patience: 개선이 없다고 바로 종료하지 않고 개선이 없는 epoch를 얼마나 기다려줄것인지 정함. ex) 10이라고 지정하면 개선이 없는 epoch가 10번째 지속될 경우 학습 종료
    # https://tykimos.github.io/2017/07/09/Early_Stopping/ 참조
    
    model.fit_generator(
            train_generator,
            steps_per_epoch=train_samples/(args.batch_size*50),
            ##steps_per_epoch=train_samples/args.batch_size,
            ##epochs=500,
            epochs=3,
            validation_data=valid_generator,
            validation_steps=valid_samples/args.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=1)

    # 테트스 테이터에 실시간 전처리를 수행하여 n번 예측한 결과값의 평균을 최종 예측값으로 사용한다
    for j in range(test_nfolds):
        preds = model.predict_generator(
                test_generator,
                steps=len(test_id),
                verbose=1)

        if j == 0:
            result = pd.DataFrame(preds, columns=labels)
        else:
            result += pd.DataFrame(preds, columns=labels)
    result /= test_nfolds
    result.loc[:, 'img'] = pd.Series(test_id, index=result.index)
    sub_file = '../subm/{}/f{}.csv'.format(suffix, fold)
    result.to_csv(sub_file, index=False)

    # 캐글에 제출한다
    #submit_cmd = 'kaggle competitions submit -c state-farm-distracted-driver-detection -f {} -m {}.fold{}'.format(sub_file, suffix, fold)
    #subprocess.call(submit_cmd, stderr=subprocess.STDOUT, shell=True)

    # 5-Fold 교차 검증 과정에서 생성한 훈련/검증 데이터를 삭제한다
    shutil.rmtree(temp_train_fold)
    shutil.rmtree(temp_valid_fold)

print('# Ensemble')
# 5-Fold 교차 검증의 결과물을 단순 앙상블한다
ensemble = 0
for fold in range(nfolds):
    ensemble += pd.read_csv('../subm/{}/f{}.csv'.format(suffix, fold), index_col=-1).values * 1. / nfolds
ensemble = pd.DataFrame(ensemble, columns=labels)
ensemble.loc[:, 'img'] = pd.Series(test_id, index=ensemble.index)
sub_file = '../subm/{}/ens.csv'.format(suffix)
ensemble.to_csv(sub_file, index=False)

# 캐글에 제출한다
#submit_cmd = 'kaggle competitions submit -c state-farm-distracted-driver-detection -f {} -m {}'.format(sub_file, suffix)
#subprocess.call(submit_cmd, stderr=subprocess.STDOUT, shell=True)
