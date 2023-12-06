import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from data import DDDataset
from model import normalization, tensor_from_numpy
from model import RBFAgcn
from dict_processing import dict_processing


def parser():
    parse = argparse.ArgumentParser(description='GCN-RBFA Model Training')
    parse.add_argument('--RANDOM_STATE', '-random_state', type = int ,default=1, help='The random_state of dataset')
    parse.add_argument('--EPOCHS','-epochs', type = int ,default=10, help='Training epoch')
    parse.add_argument('--INPUT_DIM','-input_dim', type = int ,default=6, help='The input_dim of GCN')
    parse.add_argument('--HIDDEN_DIM','-hidden_dim',type = int ,default=64, help='The hidden_dim of GCN')
    parse.add_argument('--LAYER','-layer',type = int ,default=5, help='The layer of GCN')
    parse.add_argument('--LEARNING_RATE','-lr',type = float ,default=0.001, help='The learning_rate of GCN')
    parse.add_argument('--WEIGHT_DECAY','-wd',type = float ,default=0.0001, help='The weight_decay of GCN')
    parse.add_argument('--NUM_CLASSES', '-num_classes', type=int, default=1, help='Number of classes')

    parse.add_argument('--CUDA', '-cuda', type=str, default='cuda:0', help='Select the cuda')
    parse.add_argument('--BEST_LOSS', '-best_loss', type=float, default=0.04, help='The initial best loss')
    parse.add_argument('--MODEL_DIR', '-model_dir', type=str, default='./result/model/', help='The path of model')
    parse.add_argument('--RESULT_DIR', '-result_dir', type=str, default='./result/data/', help='The path of result')
    parse.add_argument('--MODEL_NAME', '-model_name', type=str, default='RBFA', help='The path of result')
    args = parse.parse_args()
    return args

def main():

    args = parser()

    # load data
    dataset = DDDataset(random_state=args.RANDOM_STATE)
    DEVICE = args.CUDA if torch.cuda.is_available() else "cpu"
    # adjacency matrix corresponding to all graphs
    adjacency = dataset.sparse_adjacency
    # laplacian matrix
    normalize_adjacency = normalization(adjacency).to(DEVICE)
    # eigenvectors of all nodes
    node_features = tensor_from_numpy(dataset.node_features, DEVICE)
    # graph index of each node
    graph_indicator = tensor_from_numpy(dataset.graph_indicator, DEVICE)
    # label for each diagram
    graph_labels = tensor_from_numpy(dataset.graph_labels, DEVICE)
    # nodes of focused attention
    atten_score = tensor_from_numpy(dataset.atten_score, DEVICE)
    # graph index of training set and test set
    train_index = tensor_from_numpy(dataset.train_index, DEVICE)
    valid_index = tensor_from_numpy(dataset.valid_index, DEVICE)
    test_index = tensor_from_numpy(dataset.test_index, DEVICE)
    # labels of training set and test set
    train_label = tensor_from_numpy(dataset.train_label, DEVICE)
    valid_label = tensor_from_numpy(dataset.valid_label, DEVICE)
    test_label = tensor_from_numpy(dataset.test_label, DEVICE)

    args.INPUT_DIM = node_features.size(1)
    args.best_loss = 1
    args.MODEL_NAME = str(args.LAYER) + '_' + str(args.RANDOM_STATE)

    # load model
    model = RBFAgcn(args.INPUT_DIM, args.HIDDEN_DIM, args.NUM_CLASSES).to(DEVICE)
    mse = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), args.LEARNING_RATE, weight_decay = args.WEIGHT_DECAY)
    model_path = args.MODEL_DIR + args.MODEL_NAME +'.pt'

    for epoch in range(args.EPOCHS):
        # train for one epoch
        # switch to train mode
        model.train()
        logits = model(normalize_adjacency, node_features, graph_indicator, atten_score)  # 对所有数据(图)前向传播 得到输出
        train_logits = logits[train_index]
        train_logits = train_logits.squeeze(-1)
        loss = mse(train_logits, train_label)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate on validation set
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            logits = model(normalize_adjacency, node_features, graph_indicator, atten_score)
            valid_logits = logits[valid_index]
            valid_logits = valid_logits.squeeze(-1)
            val_loss = mse(valid_logits, valid_label)

        # remember the best mae_eror and save checkpoint
        if val_loss<args.best_loss:
            torch.save({
                'epoch':epoch+1,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'train_loss':loss.item(),
                'val_loss':val_loss.item(),
            },model_path)
            args.best_loss = val_loss

        print("Epoch {:03d}: Loss {:.4f} , valid_loss {:.4f}".format(
            epoch + 1, loss.item(), val_loss.item()))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # test best model
    # model.load_state_dict(torch.load(model_path))
    logits = model(normalize_adjacency, node_features, graph_indicator,atten_score)
    test_logits = logits[test_index]
    test_logits = test_logits.squeeze(-1)
    test_loss = mse(test_logits, test_label)
    print('val_loss', args.best_loss)
    print('test_loss', test_loss)

    dict_data = {'train_index':train_index, 'train_logits':train_logits, 'train_label':train_label,
                 'valid_index': valid_index, 'valid_logits': valid_logits, 'valid_label': valid_label,
                 'test_index': test_index, 'test_logits': test_logits, 'test_label': test_label,}

    # save training data
    dict_processing(args.MODEL_NAME, args.RESULT_DIR, dict_data)

    print('finished')

if __name__ == '__main__':
    main()