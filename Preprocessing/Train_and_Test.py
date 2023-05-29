import os

def TrainTest_files(new_file_path, train_test_dir, type):
    class_label = 0

    with open(new_file_path, 'w') as new_file:
        for i in range(0, 28):
            file_to_change = os.path.join(train_test_dir, str(i) + f'_{type}.txt')

            with open(file_to_change, 'r') as file: #Open for reading
                for line in file:
                    line = line.split()
                    if line[1] == '1':
                        new_line = line[0] + '\t' + str(class_label) + '\n'
                        new_file.write(new_line)
            class_label += 1

train_test_dir = '/home/xnmaster/data/ImageSets/0/'     #Dire

new_path_train = '/home/xnmaster/data/train.txt'
new_path_test  = '/home/xnmaster/data/test.txt'

TrainTest_files(new_path_train, train_test_dir, 'train')
TrainTest_files(new_path_test,  train_test_dir, 'test')

