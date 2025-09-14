import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gzip, struct, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader


X_train = torch.from_numpy(X_train).float().div(255.0).unsqueeze(1)
X_test  = torch.from_numpy(X_test ).float().div(255.0).unsqueeze(1)
y_train = torch.from_numpy(y_train).long()
y_test  = torch.from_numpy(y_test ).long()

perm = torch.randperm(X_train.shape[0])
X_train = X_train[perm]
y_train = y_train[perm]

X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
X_test  = torch.as_tensor(X_test,  dtype=torch.float32, device=device)
y_train = torch.as_tensor(y_train, dtype=torch.long, device=device)
y_test  = torch.as_tensor(y_test,  dtype=torch.long, device=device)

# flatten -> [m, d], d = H*W
X_train = X_train.view(X_train.size(0), -1)  
X_test  = X_test.view(X_test.size(0), -1)    

num_classes = int(y_train.max().item() + 1)

# move to GPU
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)



class Architecture:
    def __init__(self, blocks, model_dir, num_classes, batch_size=10):
        self.blocks = blocks
        self.model_dir = model_dir
        self.num_classes = num_classes
        self.batch_size = batch_size

    def __call__(self, Z, y=None):
        for b, block in enumerate(self.blocks):
            block.load_arch(self, b)
            self.init_loss()
            Z = block.preprocess(Z)
            Z = block(Z, y)
            Z = block.postprocess(Z)
        return Z

    def __getitem__(self, i):
        return self.blocks[i]

    def init_loss(self):
        self.loss_dict = {"loss_total": [], "loss_expd": [], "loss_comp": []}

    def update_loss(self, layer, loss_total, loss_expd, loss_comp):
        self.loss_dict["loss_total"].append(loss_total)
        self.loss_dict["loss_expd"].append(loss_expd)
        self.loss_dict["loss_comp"].append(loss_comp)
        print(f"layer: {layer} | loss_total: {loss_total:5f} | loss_expd: {loss_expd:5f} | loss_comp: {loss_comp:5f}")


## utils
import os
import logging
import json
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize

def sort_dataset(data, labels, classes, stack=False):
    if isinstance(classes, int):
        classes = torch.arange(classes, device=data.device)
    sorted_data = []
    sorted_labels = []
    for c in classes:
        idx = (labels == c)
        data_c = data[idx]
        labels_c = labels[idx]
        sorted_data.append(data_c)
        sorted_labels.append(labels_c)
    if stack:
        sorted_data = torch.vstack(sorted_data)
        sorted_labels = torch.hstack(sorted_labels)
    return sorted_data, sorted_labels

def save_params(model_dir, params, name='params.json'):
    path = os.path.join(model_dir, name)
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)

def load_params(model_dir):
    path = os.path.join(model_dir, "params.json")
    with open(path, 'r') as f:
        return json.load(f)

def create_csv(model_dir, filename, headers):
    csv_path = os.path.join(model_dir, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w+') as f:
        f.write(','.join(map(str, headers)))
    return csv_path

def save_loss(loss_dict, model_dir, name):
    save_dir = os.path.join(model_dir, "loss")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{name}.csv")
    pd.DataFrame(loss_dict).to_csv(file_path)

def save_features(model_dir, name, features, labels, layer=None):
    save_dir = os.path.join(model_dir, "features")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(features.detach().cpu(), os.path.join(save_dir, f"{name}_features.pt"))
    torch.save(labels.detach().cpu(), os.path.join(save_dir, f"{name}_labels.pt"))








#functionals

import os
from tqdm import tqdm
import numpy as np
import scipy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_n_each(X, y, n, b=0):
    classes = torch.unique(y)
    _X, _y = [], []
    for c in classes:
        idx = y == c
        X_class = X[idx][b*n:(b+1)*n]
        y_class = y[idx][b*n:(b+1)*n]
        _X.append(X_class)
        _y.append(y_class)
    return torch.vstack(_X), torch.hstack(_y)

def translate1d(data, labels, n=None, stride=1):
    n_samples, _, n_dim = data.shape
    data_new = []
    if n is None:
        shifts = torch.arange(0, n_dim, stride)
    else:
        shifts = torch.arange(-n*stride, (n+1)*stride, stride)
    for r in shifts:
        data_new.append(torch.roll(data, shifts=r.item(), dims=2))
    return torch.vstack(data_new), labels.repeat(len(shifts))

def translate2d(data, labels, n=None, stride=1):
    n_samples, _, H, W = data.shape
    data_new = []
    if n is None:
        vshifts = torch.arange(0, H, stride)
        hshifts = torch.arange(0, W, stride)
    else:
        hshifts = torch.arange(-n*stride, (n+1)*stride, stride)
        vshifts = torch.arange(-n*stride, (n+1)*stride, stride)
    for h in vshifts:
        for w in hshifts:
            data_new.append(torch.roll(data, shifts=(h.item(), w.item()), dims=(2, 3)))
    return torch.vstack(data_new), labels.repeat(len(hshifts) * len(vshifts))

def shuffle(data, labels, seed=10):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(data.shape[0], generator=g)
    return data[idx], labels[idx]

def filter_class(data, labels, classes, n=None, b=0):
    if isinstance(classes, int):
        classes = torch.arange(classes, device=labels.device)
    data_filter = []
    labels_filter = []
    for cls in classes:
        idx = labels == cls
        data_filter.append(data[idx][b*n:(b+1)*n])
        labels_filter.append(labels[idx][b*n:(b+1)*n])
    data_new = torch.vstack(data_filter)
    labels_cat = torch.hstack(labels_filter)
    unique_labels, labels_new = torch.unique(labels_cat, return_inverse=True)
    return data_new, labels_new

def normalize(X, p=2):
    X_flat = X.view(X.size(0), -1)
    norm = torch.norm(X_flat, p=p, dim=1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8)
    return X_flat / norm

def batch_cov(V, bs):
    m = V.shape[0]
    cov_sum = torch.zeros(V.shape[1], V.shape[1], device=V.device)
    for i in range(0, m, bs):
        chunk = V[i:i+bs]
        cov_sum += chunk.T @ chunk
    return cov_sum

def generate_kernel(mode, size, seed=10):
    torch.manual_seed(seed)
    if mode == 'gaussian':
        return torch.randn(size, device='cuda')
    elif mode == 'ones':
        return torch.ones(size, device='cuda')

def convert2polar(images, channels, timesteps):
    mid_pt = images.shape[1] // 2
    r = np.linspace(0, mid_pt, channels).astype(np.int32)
    angles = np.linspace(0, 360, timesteps)
    polar_imgs = []
    for angle in angles:
        X_rot = scipy.ndimage.rotate(images.cpu().numpy(), angle, axes=(1, 2), reshape=False)
        polar_imgs.append(torch.tensor(X_rot[:, mid_pt, r], dtype=torch.float32))
    polar_imgs = torch.stack(polar_imgs).permute(1, 2, 0)
    return polar_imgs.to(images.device)

# === Evaluation ===
import argparse
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def asnumpy(x):
    if isinstance(x,torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC(verbose=0, random_state=10)
    clf.fit(train_features, train_labels)
    acc_train = clf.score(train_features, train_labels)
    acc_test = clf.score(test_features, test_labels)
    print("SVM: {}".format(acc_test))
    return acc_train, acc_test

def knn(train_features, train_labels, test_features, test_labels, k=5):
    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(k=k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values
    acc = compute_accuracy(test_pred.cpu().numpy(), test_labels.cpu().numpy())
    print("kNN: {}".format(acc))
    return acc

def nearsub(train_features, train_labels, test_features, test_labels, n_comp=10):
    scores = []
    classes = np.unique(test_labels)
    features_sort, _ = sort_dataset(train_features, train_labels, classes=classes, stack=False)
    fd = features_sort[0].shape[1]
    if n_comp >= fd:
        n_comp = fd - 1
    for j in range(len(classes)):
        svd = TruncatedSVD(n_components=n_comp).fit(asnumpy(features_sort[j]).astype(np.float32))
        U = svd.components_.T
        proj = (np.eye(fd) - U @ U.T) @ asnumpy(test_features).astype(np.float32).T
        scores.append(np.linalg.norm(proj, axis=0))
    pred = np.argmin(scores, axis=0)
    acc = compute_accuracy(classes[pred], asnumpy(test_labels).astype(np.float32))
    print("SVD: {}".format(acc))
    return acc

def nearsub_pca(train_features, train_labels, test_features, test_labels, n_comp=10):
    scores = []
    classes = np.unique(test_labels)
    features_sort, _ = sort_dataset(train_features, train_labels, classes=classes, stack=False)
    fd = features_sort[0].shape[1]
    if n_comp >= fd:
        n_comp = fd - 1
    for j in range(len(classes)):
        pca = PCA(n_components=n_comp).fit(features_sort[j].cpu().numpy())
        U = pca.components_.T
        mean = np.mean(features_sort[j].cpu().numpy(), axis=0)
        proj = (np.eye(fd) - U @ U.T) @ (test_features.cpu().numpy() - mean).T
        scores.append(np.linalg.norm(proj, axis=0))
    pred = np.argmin(scores, axis=0)
    acc = compute_accuracy(classes[pred], test_labels.cpu().numpy())
    print("PCA: {}".format(acc))
    return acc

def compute_accuracy(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size

def baseline(train_features, train_labels, test_features, test_labels):
    models = {
        'log_l2': SGDClassifier(loss='log_loss', max_iter=10000, random_state=42),
        'SVM_linear': LinearSVC(max_iter=10000, random_state=42),
        'SVM_RBF': SVC(kernel='rbf', random_state=42),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier()
    }
    for name, model in models.items():
        model.fit(train_features, train_labels)
        score = model.score(test_features, test_labels)
        print(f"{name}: {score}")



#Vector
import os
import numpy as np
import torch
from torch.linalg import inv, slogdet, norm
from scipy.special import softmax

class Vector:
    def __init__(self, layers, eta, eps, lmbda=500):
        self.layers = layers
        self.eta = eta
        self.eps = eps
        self.lmbda = lmbda

    def __call__(self, Z, y=None):
        for layer in range(self.layers):
            Z, y_approx = self.forward(layer, Z, y)
            self.arch.update_loss(layer, *self.compute_loss(Z, y_approx))
        return Z

    def forward(self, layer, Z, y=None):
        if y is not None:
            self.feature = Z
            self.label = y
            self.init(Z, y)
            self.save_weights(layer)
            self.save_gam(layer)

            m, d = Z.shape
            c = d / (m * self.eps)
            for j in range(self.num_classes):
                Z_j = Z[y == j]
                m_j = Z_j.shape[0]
                c_j = d / (m_j * self.eps)
                I_mj = torch.eye(m_j, device=Z.device)
                I_d = torch.eye(d, device=Z.device)
                pre_Ej = inv(I_mj + c * Z_j @ Z_j.T)
                pre_Cj = inv(I_mj + c_j * Z_j @ Z_j.T)
                E_j = c * (I_d - c * Z_j.T @ pre_Ej @ Z_j)
                C_j = c_j * (I_d - c_j * Z_j.T @ pre_Cj @ Z_j)
                expd = Z_j @ E_j.T
                clus = self.gam[j] * Z_j @ C_j.T
                Z_j = Z_j + self.eta * (expd - clus)
                if j == 0:
                    out = Z_j
                else:
                    out = torch.cat((out, Z_j), dim=0)
            out = self.postprocess(out)
            return out, y
        else:
            self.load_weights(layer)
            self.load_gam(layer)
            m, d = self.feature.shape
            c = d / (m * self.eps)
            comp = []
            for j in range(self.num_classes):
                Z_j = self.feature[self.label == j]
                m_j = Z_j.shape[0]
                I_mj = torch.eye(m_j, device=Z.device)
                I_d = torch.eye(d, device=Z.device)
                c_j = d / (m_j * self.eps)
                pre_Cj = inv(I_mj + c_j * Z_j @ Z_j.T)
                C_j = c_j * (I_d - c_j * Z_j.T @ pre_Cj @ Z_j)
                comp.append(Z @ C_j.T)
            comp = torch.stack(comp)
            comp = comp.view(self.num_classes, Z.shape[0], d)
            pred_pi, y_approx = self.nonlinear(comp)
            out = torch.zeros(Z.shape[0],Z.shape[1],device=Z.device,dtype=Z.dtype)
            for j in range(self.num_classes):
                Z_j = self.feature[self.label == j]
                m_j = Z_j.shape[0]
                I_mj = torch.eye(m_j, device=Z.device)
                I_d = torch.eye(d, device=Z.device)
                pre_Ej = inv(I_mj + c * Z_j @ Z_j.T)
                E_j = c * (I_d - c * Z_j.T @ pre_Ej @ Z_j)
                update = Z + self.eta * (Z @ E_j.T - self.gam[j] * comp[j])

                w=pred_pi[j]
                w=w.view(-1,1)
                #print(f"Z_shape{Z.shape}")
                #print(f"pred_pi shape{pred_pi.shape}")
                out.addcmul_(w,update)
            out = self.postprocess(out)
            return out, y_approx

    def first_ortho(self, Z, y):
        y_1 = []
        Z_0 = Z[y == 0].T
        y_1.extend([0] * Z_0.shape[1])
        Q, _ = torch.linalg.qr(Z_0)
        A = Q
        output = Z_0
        for j in range(1, self.num_classes):
            U = A @ A.T
            Z_j = Z[y == j].T
            y_1.extend([j] * Z_j.shape[1])
            Z_j = (torch.eye(U.shape[0], device=Z.device) - U) @ Z_j
            B, _ = torch.linalg.qr(Z_j)
            A = torch.cat((A, B), dim=1)
            output = torch.cat((output, Z_j), dim=1)
        return output.T, np.array(y_1)

    def load_arch(self, arch, block_id):
        self.arch = arch
        self.block_id = block_id
        self.num_classes = self.arch.num_classes

    def init(self, Z, y):
        self.compute_gam(y)

    def compute_gam(self, y):
        m_j = torch.tensor([(y == j).sum().item() for j in range(self.num_classes)], dtype=torch.float32, device=y.device)
        self.gam = m_j / y.size(0)

    def compute_loss(self, Z, y):
        m, d = Z.shape
        c = d / (m * self.eps)
        I = torch.eye(d, device=Z.device)
        logdet_expd = slogdet(I + c * Z.T @ Z)[1] / 2
        loss_comp = 0.
        for j in range(self.num_classes):
            Z_j = Z[y == j]
            m_j = Z_j.shape[0]
            if m_j == 0:
                continue
            c_j = d / (m_j * self.eps)
            logdet_j = slogdet(I + c_j * Z_j.T @ Z_j)[1]
            loss_comp += self.gam[j] * logdet_j / 2
        return (logdet_expd - loss_comp).item(), logdet_expd.item(), loss_comp.item()

    def preprocess(self, X):
        m = X.shape[0]
        return torch.nn.functional.normalize(X.view(m, -1), p=2, dim=1)

    def postprocess(self, X):
        return torch.nn.functional.normalize(X, p=2, dim=1)

    def nonlinear(self, Bz):
        flat = Bz.view(Bz.size(0), Bz.size(1), -1)
        norms = norm(flat, dim=2)
        norms = torch.clamp(norms, min=1e-8).cpu().numpy()
        prob = softmax(-self.lmbda * norms, axis=0)
        pred = torch.tensor(prob, dtype=torch.float32, device=Bz.device)
        y = torch.argmax(pred, dim=0)
        return pred.unsqueeze(-1), y

    def save_weights(self, layer):
        weight_dir = os.path.join(self.arch.model_dir, "weights")
        os.makedirs(weight_dir, exist_ok=True)
        torch.save({"feature": self.feature, "label": self.label}, os.path.join(weight_dir, f"{self.block_id}_{layer}.pt"))

    def load_weights(self, layer):
        path = os.path.join(self.arch.model_dir, "weights", f"{self.block_id}_{layer}.pt")
        state = torch.load(path)
        self.feature = state["feature"]
        self.label = state["label"]

    def save_gam(self, layer):
        weight_dir = os.path.join(self.arch.model_dir, "weights")
        os.makedirs(weight_dir, exist_ok=True)
        np.save(os.path.join(weight_dir, f"{self.block_id}_{layer}_gam.npy"), self.gam.detach().cpu().numpy())

    def load_gam(self, layer):
        path = os.path.join(self.arch.model_dir, "weights", f"{self.block_id}_{layer}_gam.npy")
        self.gam = torch.tensor(np.load(path), dtype=torch.float32).to(self.feature.device)




#######
import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default=5, help="number of layers")
parser.add_argument('--eta', type=float, default=0.1, help='learning rate')
parser.add_argument('--eps', type=float, default=0.01, help='eps squared')
parser.add_argument('--tail', type=str, default='', help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='saved_models/', help='base directory for saving PyTorch model.')
parser.add_argument('--data_dir', type=str, default='saved_models/data/', help='base directory for saving PyTorch model.')
args = parser.parse_args(args=[])

# pipeline setup
model_dir = os.path.join("saved_models", "devana_multi_redunet", f"layers{args.layers}_eps{args.eps}_eta{args.eta}{args.tail}")
os.makedirs(model_dir, exist_ok=True)
#save_params(model_dir, vars(args))
print(model_dir)

# data setup
# X_train, y_train, X_test, y_test, num_classes assumed to be defined

# model setup
layers = [Vector(args.layers, eta=args.eta, eps=args.eps)]
model = Architecture(layers, model_dir, num_classes)

# train/test pass
print("Forward pass - train features")
start_time = time.time()
Z_train = model(X_train, y_train)
end_time = time.time()
save_loss(model.loss_dict, model_dir, "train")

print("Forward pass - test features")
Z_test = model(X_test)
save_loss(model.loss_dict, model_dir, "test")

# evaluation
Z_train_np = Z_train.detach().cpu().numpy()
y_train_np = y_train.detach().cpu().numpy()
Z_test_np = Z_test.detach().cpu().numpy()
y_test_np = y_test.detach().cpu().numpy()



_, acc_svm = svm(Z_train_np, y_train_np, Z_test_np, y_test_np)
acc_knn = knn(torch.tensor(Z_train_np), torch.tensor(y_train_np), torch.tensor(Z_test_np), torch.tensor(y_test_np), k=5)
acc_svd = nearsub(Z_train_np, y_train_np, Z_test_np, y_test_np, n_comp=5)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd}
save_params(model_dir, acc, name="acc_test.json")

elapsed_time = end_time - start_time
print(f"Model execution time: {elapsed_time:.4f} seconds")
