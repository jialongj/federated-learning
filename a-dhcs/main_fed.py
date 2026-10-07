#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import csv
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=206, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    new_glob=copy.deepcopy(w_glob)

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    all_pre=[]
    bits = []
    all_bit = 0
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for i in range(1):
        # net_glob.load_state_dict(new_glob)
        # evr_pre=[]
        # frac= 0.3
        # print("当前参与率：",frac)

        for iter in range(args.epochs):
            ever_bit = 0
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            frac = 0.3
            m = max(int(frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))


            ever_bit = 163780 * 8*0.3*100
            all_bit+=ever_bit
            bits.append(copy.deepcopy(all_bit))
            # update global weights
            w_glob = FedAvg(w_locals)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
        #     evr_pre.append(acc_test)
        # all_pre.append(evr_pre)
            all_pre.append(copy.deepcopy(acc_test))





    # 保存all_test
    # 指定要保存的 CSV 文件名
    filename = ('noiid_all_pre.csv')

    # 将列表保存到 CSV 文件
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(all_pre)

    # 保存bits
    # 指定要保存的 CSV 文件名
    filename = ('noiid_all_bits.csv')

    # 将列表保存到 CSV 文件
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(bits)
    # # 绘制图表
    # # 设置matplotlib的字体为中文
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # # 准备数据用于绘图
    # x_values = list(range(1, len(all_pre) + 1))  # x轴的值，从1到all_test长度
    # # 检查 all_test 是否包含四个子列表
    # assert len(all_pre) == len(all_pre), "all_test 需要包含子列表"
    # # 创建图表
    # plt.figure(figsize=(10, 6))  # 可以调整图表大小
    #
    # # 对 all_test 中的每个子列表绘制一条线
    # for i, acc_values in enumerate(all_pre, start=1):
    #     plt.plot(x_values, acc_values, label=f'参与率 {i * 0.1}')
    #
    # # 添加图例
    # plt.legend()
    #
    # # 添加标题和轴标签
    # plt.title('Acc测试结果')
    # plt.xlabel('测试次数')
    # plt.ylabel('准确率')
    #
    # # 旋转x轴的标签，避免重叠
    # plt.xticks(rotation=45)
    #
    # # 展示图表
    # plt.tight_layout()  # 调整布局以适应所有的标签
    # plt.savefig('0.001_iid_adeptive.png')
    # plt.show()
    #
    # # 保存数据到CSV文件
    # csv_file = 'iid_all_pre.csv'
    #
    # # 使用 'w' 模式打开文件进行写入
    # with open(csv_file, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     # 写入标题（可选）
    #     writer.writerow(['Round'] + [f'Line {i + 1}' for i in range(len(all_pre))])
    #     # 写入数据
    #     for i in range(len(x_values)):
    #         row = [i + 1] + [all_pre[j][i] for j in range(len(all_pre))]
    #         writer.writerow(row)
    #
    # print(f'Data saved to {csv_file}')
    #
    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

