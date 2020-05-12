import os
import shutil

dataset_path = os.path.expanduser('~/Documents/research/VISAGE_a/validation/curated/')

# iterate each image then copy the file to the respective folder

dataset_path_output = os.path.expanduser('~/Documents/research/VISAGE_a/visage_dataset/')
validation_path = os.path.expanduser('~/Documents/research/VISAGE_a/visage_dataset/validation/')

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

if not os.path.exists(validation_path):
    os.mkdir(validation_path)


def get_age(f):

    fileHandle = open('metadata.txt', 'r')

    for line in fileHandle:
        fields = line.split('|')

        if fields[0] == f:
            fileHandle.close()
            return int(fields[1].strip()) - 1


counter = 0

for fi in os.listdir(dataset_path):
    if not fi.startswith('.'):
        #print(fi)

        age = get_age(fi)

        if age is not None:

            # copy file to path

            copy_directory_destination = os.path.join(validation_path, str(age))

            if not os.path.exists(copy_directory_destination):
                os.mkdir(copy_directory_destination)

            copy_path_source = os.path.join(dataset_path, fi)
            copy_path_destination = os.path.join(copy_directory_destination, fi)

            dest = shutil.copy(copy_path_source, copy_directory_destination)
            print(dest)

        else:
            counter += 1
            print('not found', counter)









