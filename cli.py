
import numpy as np
# img = image.load_img('frog1.jpg', target_size=(120,120))
# img_to_arr = image.img_to_array(img)
# results = model.predict(np.array([img_to_arr]))
# labels = [np.where(result == np.amax(result)) for result in results]
# print(labels)
import sys
import os
import cv2
from keras.models import load_model
from keras import optimizers
from datetime import datetime

labels_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Command: python cli.py image_folder!')
    else:
        folder_img = sys.argv[1]
        if os.path.isdir(folder_img):
            list_imgs = []
            list_imgs_preprocessed = []
            for file in os.listdir(folder_img):
                img = cv2.imread(folder_img+'/'+file)
                if img is not None:
                    list_imgs.append(img)
                    list_imgs_preprocessed.append(cv2.resize(img,(120,120)) * 1.0 / 255.0)

            model = load_model('./model/cifar10_model.h5')
            sgd = optimizers.SGD(lr=0.02, momentum=0.9, nesterov=False)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            results = model.predict(np.array(list_imgs_preprocessed))
            labels = [np.where(result == np.amax(result))[0][0] for result in results]

            for i in range(0, len(list_imgs)):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(list_imgs[i], labels_list[labels[i]], (100, 100), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("img",list_imgs[i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # labels_text = [labels_list[label] for label in labels]
                if len(sys.argv) == 3:
                    if sys.argv[2] == 'save':
                        print('here')
                        if not os.path.isdir('./results'):
                            os.mkdir('./results')
                        print('./results/'+labels_list[labels[i]]+datetime.now().strftime('%d-%m-%Y_%H-%M-%S')+'.png')
                        cv2.imwrite('./results/'+labels_list[labels[i]]+datetime.now().strftime('%d-%m-%Y_%H-%M-%S')+'.png', list_imgs[i])
        else:
            print("Image folder doesn't exist!")