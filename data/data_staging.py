from random import shuffle
import os

data_directory = '/Users/nicholasjoodi/Documents/ucdavis/computerScience/VisualRecognition/diagrams_with_google_label'
triplets = []
split =0.75
train_file = 'train.txt'
test_file = 'test.txt'
data_dir_array =[os.path.join(data_directory, o) for o in os.listdir(data_directory)  if os.path.isdir(os.path.join(data_directory,o))]
size_of_dir = len(data_dir_array)
print(size_of_dir)
for j in range(size_of_dir):
    relative_path = data_dir_array[j]
    # inner_dir_array = [name[0] for name in os.walk(data_directory+'/'+label)]
    inner_dir_array = []
    for (dirpath, dirnames, filenames) in os.walk(relative_path):
        inner_dir_array.extend(filenames)
        break
    satelite_image = None
    report_image = None
    if len(inner_dir_array)!=2:
        print(len(inner_dir_array))
        print('too many images in folder: '+ relative_path)
        continue
    for i in range(len(inner_dir_array)):
        if 'jpeg' in inner_dir_array[i]:
            report_image = inner_dir_array[i]
        elif 'png' in inner_dir_array[i]:
            satelite_image = inner_dir_array[i]
    if satelite_image==None or report_image==None:
        print('Could not find satelite and report image for'+relative_path)
        continue
    _,class_name = os.path.split(relative_path)
    # print(class_name)
    satelite_image= class_name+'/'+satelite_image
    report_image = class_name+'/'+report_image
    triplets.append(report_image + '\t'+ satelite_image)
shuffle(triplets)
size = len(triplets)
print(size)
train_size = split*size
with open(train_file, 'w') as _file,open(test_file, 'w') as _file_test :
    for i in range(size):
        row = triplets.pop(0)
        if i < train_size:
            _file.write("%s\n" % row)
        else:
            _file_test.write("%s\n" % row)


    # satelite_string = sess.run(encoded_png, feed_dict={image: images[j]})
    # report_string = sess.run(encoded_jpeg, feed_dict={image: images[j]})
