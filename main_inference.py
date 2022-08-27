#! /usr/bin/env python

import os
import time
import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision import datasets, transforms
from torchvision.models import resnet


from torch2trt import TRTModule


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", type=str, default="")
    parser.add_argument("-n", "--num_workers", type=int, default=8)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    return args


def inference(model_path,batch_size):

    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # テンソルに変換・正規化
    tf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ]
    transform = transforms.Compose(tf)

    # データセット読み込み
    test_set = CIFAR10(root="~/.datasets",
                       train=False,
                       download=True,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=args.num_workers)
    
    print(test_set.data.shape)

    # モデルの定義
    low_dim = 128
    net = ResNet18(low_dim=low_dim)
    norm = Normalize(2)

    net, norm = net.to(device), norm.to(device)

    # 重みのロード
    #net = TRTModule()
    net.load_state_dict(torch.load(model_path))

    print("バッチサイズ:{}".format(batch_size))

    # 推論
    start = time.time()
    print("推論を開始します")

    net.eval()
    features_buffer = []
    for inputs, _, _ in test_loader:
        with torch.no_grad():
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            features = norm(net(inputs)).cpu()

        features_buffer.append(features)
        del features
        torch.cuda.empty_cache()

    inference_time = time.time()
    print("推論の実行時間:{}".format((inference_time - start)) + "[秒]")
    
    features_buffer = torch.cat(features_buffer,dim=0)
    targets = test_loader.dataset.targets
    targets = targets[0:8192]
    print(len(targets))
    acc, nmi, ari =  calc_clustering_metrics(features_buffer, targets)
    
    clustering_time = time.time() - inference_time
    print("クラスタリングの実行時間:{}".format(clustering_time) + "[秒]")
    
    total_time = time.time() - start
    print("トータルの実行時間:{}".format(total_time) + "[秒]\n")
    print("ACC:{} NMI:{} ARI:{}".format(acc,nmi,ari))
    
    return (inference_time - start), clustering_time, total_time


class CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index
    
    def __len__(self):
        return 8192


class metrics:
    ari = adjusted_rand_score
    nmi = normalized_mutual_info_score

    @staticmethod
    def acc(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row, col = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(row, col)]) * 1.0 / y_pred.size


def calc_clustering_metrics(features, targets):
    z = features.detach().numpy()
    y = np.array(targets)
    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z)
    return metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def ResNet18(low_dim=128):
    net = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], low_dim)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                          stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


if __name__=="__main__":
    model_path = "idfd_epoch_1999.pth"
    model_trt = "IDFD_trt.pth"
    batch_list = [64,128,256,512]
    # batch_list = [1024,2048]
    
    inference(model_path,batch_size=512)

#     # 推論
#     for batch_size in batch_list:
#         inf = []
#         clu = []
#         tot = []
#         for _ in range(3):
#             inference_time, clustering_time, total_time = inference(model_path,batch_size=batch_size)
#             inf.append(inference_time)
#             clu.append(clustering_time)
#             tot.append(total_time)
        
#         print("バッチサイズ{}の時の推論時間の平均:{}".format(batch_size,np.mean(inf)))
#         print("バッチサイズ{}の時のクラスタリング時間の平均:{}".format(batch_size,np.mean(clu)))
#         print("バッチサイズ{}の時の全体実行時間の平均:{}".format(batch_size,np.mean(tot)))