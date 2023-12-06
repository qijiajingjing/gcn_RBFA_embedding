import numpy as np
import os

def dict_processing(model_name, result_dir, dict_data):
    save_path = result_dir + model_name + '/'
    if not os.path.exists(save_path): os.mkdir(save_path)  # 应该是存在的

    valid_dir = save_path + 'valid.txt'
    train_dir = save_path + 'train.txt'
    test_dir = save_path + 'test.txt'

    l1 = []
    l1.extend(dict_data['train_index'].data.cpu().numpy())
    l1.extend(dict_data['train_logits'].data.cpu().numpy())
    l1.extend(dict_data['train_label'].data.cpu().numpy())
    l1 = np.array(l1).reshape(3, -1)
    f1 = open(train_dir, 'w')
    for i in range(len(l1[0])):
        f1.write(str(l1[0][i]) + ' ' + str(l1[1][i]) + ' ' + str(l1[2][i]))
        f1.write('\n')
    f1.close()

    l2 = []
    l2.extend(dict_data['valid_index'].data.cpu().numpy())
    l2.extend(dict_data['valid_logits'].data.cpu().numpy())
    l2.extend(dict_data['valid_label'].data.cpu().numpy())
    l2 = np.array(l2).reshape(3, -1)
    f2 = open(valid_dir, 'w')
    for i in range(len(l2[0])):
        f2.write(str(l2[0][i]) + ' ' + str(l2[1][i]) + ' ' + str(l2[2][i]))
        f2.write('\n')
    f2.close()

    l3 = []
    l3.extend(dict_data['test_index'].data.cpu().numpy())
    l3.extend(dict_data['test_logits'].data.cpu().numpy())
    l3.extend(dict_data['test_label'].data.cpu().numpy())
    l3 = np.array(l3).reshape(3, -1)
    f3 = open(test_dir, 'w')
    for i in range(len(l3[0])):
        f3.write(str(l3[0][i]) + ' ' + str(l3[1][i]) + ' ' + str(l3[2][i]))
        f3.write('\n')
    f3.close()