import os
import shutil

img_path = "img_annotations/images"
labels_path = "img_annotations/annotations"

img_dir_list = os.listdir(img_path)
label_dir_list = os.listdir(labels_path)

image_paths = []
label_path = []

for img_files_, label_files in zip(img_dir_list, label_dir_list):
    img_absolute_path = os.path.join(img_path, img_files_)
    label_absolute_path = os.path.join(labels_path, label_files)
    image_paths.append(img_absolute_path)
    label_path.append(label_absolute_path)

testing_length = len(image_paths) // 3
training_length = len(image_paths) - testing_length

train_img = image_paths[0:training_length]
test_img = image_paths[training_length:]

train_label = label_path[0:training_length]
test_label = label_path[training_length:]

train_dir_img = 'train/images'
test_dir_img = 'test/images'

train_dir_label = 'train/label'
test_dir_label = 'test/label'

new_path = 'img_annotations'
# Path
train_path_img = os.path.join(new_path, train_dir_img)
test_path_img = os.path.join(new_path, test_dir_img)

train_path_label = os.path.join(new_path, train_dir_label)
test_path_label = os.path.join(new_path, test_dir_label)
# Create the directory

istrain_imgExist = os.path.exists(train_path_img)
istest_imgExist = os.path.exists(test_path_img)

istrain_labelExist = os.path.exists(train_path_label)
istest_labelExist = os.path.exists(test_path_label)

if not istrain_imgExist:
    os.makedirs(train_path_img)

if not istest_imgExist:
    os.makedirs(test_path_img)

if not istrain_labelExist:
    os.makedirs(train_path_label)

if not istest_labelExist:
    os.makedirs(test_path_label)

for i, j in zip(train_img, train_label):
    img_dst_path = "img_annotations/train/images"
    label_dst_path = "img_annotations/train/label"
    shutil.move(i, img_dst_path)
    shutil.move(j, label_dst_path)

for m, n in zip(test_img, test_label):
    img_dst_path = "img_annotations/test/images"
    label_dst_path = "img_annotations/test/label"
    shutil.move(m, img_dst_path)
    shutil.move(n, label_dst_path)


os.rmdir(img_path)
os.rmdir(labels_path)

dataset_path = 'frames/'
list_name = os.listdir(dataset_path)

list_ = []



list_.append(str('train:'+ ' '+str(train_path_img)))
list_.append(str('test:'+ ' '+str(test_path_img)))
list_.append(str('nc:'+' '+str(len(list_name))))
list_.append(str('names:'+' '+str(list_name)))

complete_voc_objects = '\n'.join(list_)
with open('img_annotations/data.yml','w+') as f:
    f.write(complete_voc_objects)

