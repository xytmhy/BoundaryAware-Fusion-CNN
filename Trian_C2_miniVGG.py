import matplotlib

matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from miniVGG import MiniVGGNet
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import numpy as np
import argparse
import dataset
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
ap.add_argument("-m", "--model", required=True, help="path to save train model")
args = vars(ap.parse_args())

print("[INFO] loading dataset")

validation_size = 0.2
img_size = 48
num_channels = 2
train_path = '/media/data2/mhy/data/0911'
labelNames = ['r1', 'r2' ]

data = dataset.read_train_sets(train_path, img_size, labelNames, validation_size)
print("Complete reading input data.Will Now print a snippet of it")


trainX=data.train.images
trainY=data.train.labels
testX=data.valid.images
testY=data.vaild.labels
# ((trainX, trainY), (testX, testY)) = cifar10.load_data()
# trainX = trainX.astype("float") / 255.0
# testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 70, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=48, height=48, depth=2, classes=2)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
print(model.summary())
print("[INFO] training network Lenet-5")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=70,
              callbacks=callbacks, verbose=1)

model.save(args["model"])

print("[INFO] evaluating Lenet-5..")
preds = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
                            target_names=labelNames))

# 保存可视化训练结果
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 70), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 70), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 70), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 70), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("# Epoch")
plt.ylabel("Loss/Accuracy without BatchNormalization")
plt.legend()
plt.savefig(args["output"])