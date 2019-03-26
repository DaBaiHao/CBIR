import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
base_model = load_model('my_model.h5')

num_classes = 10
data_augmentation = True


# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

pred = base_model.predict(x_test)
print('Predicted:', pred.shape)
print(y_train.shape)
sample = Image.open("sampledeer.jpg")
sample = sample.resize((32,32))
sample = np.array(sample)
print(sample.shape)
sample = sample.astype('float32')
sample /= 255

sample = sample[np.newaxis, :]
pred_sample = base_model.predict(sample)
print("pred_sample",pred_sample)
print("pred_sample shape",pred_sample.shape)

sample_list = np.squeeze(pred_sample)
sample_list = list(sample_list)
print(max(max(pred_sample)))
sample_max_index = sample_list.index(max(max(pred_sample)))
print(sample_max_index)

print(pred.shape)
retrieved_image = list([])
for i in range(10000):
    if pred[i][sample_max_index] - 0.05 <= pred_sample[0][sample_max_index] <= pred[i][sample_max_index] + 0.1:
        retrieved_image.append(x_test[i,:,:,:])
print(np.array(retrieved_image).shape)


import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(5, 5)
gs.update(wspace=0.05, hspace=0.05)

for i, sample in enumerate(retrieved_image[:25][:][:][:]):

    if i > 24:
        break
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample, cmap='Greys_r')

    # return fig
plt.show()
plt.savefig('g.jpg')
plt.close()


# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.utils.fixes import signature
# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
#
# step_kwargs = ({'step': 'post'}
#                if 'step' in signature(plt.fill_between).parameters
#                else {})
#
# Y_test = y_test
# n_classes = Y_test.shape[1]
# print(Y_test[0])
# y_score = pred
# print(y_score[0])
#
#
# precision = dict()
# recall = dict()
# average_precision = dict()
# for i in range(n_classes):
#     precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
#                                                         y_score[:, i])
#     average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
#
# # A "micro-average": quantifying score on all classes jointly
# precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
#     y_score.ravel())
# average_precision["micro"] = average_precision_score(Y_test, y_score,
#                                                      average="micro")
# print('Average precision score, micro-averaged over all classes: {0:0.2f}'
#       .format(average_precision["micro"]))
#
# # plot
# plt.figure()
# plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
#          where='post')
# plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
#                  **step_kwargs)
#
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title(
#     'OurNet, 90ep, micro-averaged over all classes: AP={0:0.2f}'
#     .format(average_precision["micro"]))
plt.show()

