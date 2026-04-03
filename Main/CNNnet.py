from tools import *
from Agents import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MNISTTrainer:
    def __init__(self, batch_size=64, learning_rate=0.001, num_epochs=20, num_classes=7, in_channels=3):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device("cpu")

        self.model = SimpleLeNet(num_classes=num_classes, in_channels=in_channels).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_loader = None

    def prepare_data(self, train_images, train_labels):
        """
        准备数据加载器
        输入形状:
        - images: (n, 28, 28, 3) or (n, 28, 28)
        - labels: (n, 1)
        """
        train_images = torch.FloatTensor(train_images)
        if train_images.dim() == 3:
            # 在第3个位置（最后）增加一个维度，变成 (N, 28, 28, 1)
            train_images = train_images.unsqueeze(-1)
        train_images = train_images.permute(0, 3, 1, 2)
        # train_images = torch.FloatTensor(train_images).permute(0, 3, 1, 2)
        train_labels = torch.LongTensor(train_labels).squeeze()
        train_dataset = TensorDataset(train_images, train_labels)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def predict_img(self, images):
        images = torch.FloatTensor(images)
        if images.dim() == 3:
            images = images.unsqueeze(-1)
        images = images.permute(0, 3, 1, 2)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(images.to(self.device))
            probs = torch.softmax(outputs, dim=1).numpy()
        return probs

    def predict_feat(self, features):
        features = torch.FloatTensor(features)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model.classifier2(features.to(self.device))
            probs = torch.softmax(outputs, dim=1).numpy()
        return probs

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc

    def evaluate(self, loader):
        """评估模型性能"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        eval_loss = running_loss / len(loader)
        eval_acc = 100. * correct / total
        return eval_loss, eval_acc

    def run_training(self, isLog=False, path='', log=None, mute=True):
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch()
            if not mute:
                if epoch % (self.num_epochs // 10) == max((self.num_epochs // 10) - 1, 1):
                    log(f'Epoch [{epoch + 1}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%', path, isLog)

    def run_training_classifier2(self, isLog=False, path='', log=None, mute=True):
        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.classifier1.parameters():
            param.requires_grad = False
        for param in self.model.classifier2.parameters():
            param.requires_grad = True
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch()
            if not mute:
                if epoch % (self.num_epochs // 10) == max((self.num_epochs // 10) - 1, 1):
                    log(f'Epoch [{epoch + 1}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%', path, isLog)
        for param in self.model.features.parameters():
            param.requires_grad = True
        for param in self.model.classifier1.parameters():
            param.requires_grad = True

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

    def save(self, filepath):
        torch.save(
            self.model.state_dict(),
            filepath
        )

    def save_(self, filepath):
        noise_ratio = 0.01
        state_dict = self.model.state_dict()
        noisy_state_dict = {}
        for key, tensor in state_dict.items():
            median_abs = torch.median(tensor.abs())
            noise_scale = noise_ratio * median_abs
            noise = torch.randn_like(tensor) * noise_scale
            noise = torch.clamp(noise, -3 * noise_scale, 3 * noise_scale)
            noisy_state_dict[key] = tensor + noise
        torch.save(noisy_state_dict, filepath)

    def covx(self, images):
        images = torch.FloatTensor(images)
        if images.dim() == 3:
            images = images.unsqueeze(-1)
        images = images.permute(0, 3, 1, 2).to(self.device)
        return self.model.covx(images).detach().numpy()

class SimpleLeNet(nn.Module):
    def __init__(self, num_classes=7, in_channels=3):
        super(SimpleLeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.LayerNorm(10)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier2(self.classifier1(x))
        return x

    def covx(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier1(x)
        return x

class MNISTTrainer_FirstK:
    def __init__(self, mnist_trainer: MNISTTrainer, k=10):
        self.mnist_trainer = mnist_trainer
        self.k = k

    def predict_feat(self, features):
        return self.mnist_trainer.predict_feat(features[:, :self.k])

def calScoreCl_img(pred:MNISTTrainer, X:np.ndarray, Y:np.ndarray):
    # X 是 28*28*3
    prob = pred.predict_img(X)
    S = 1 - prob[np.arange(X.shape[0]), np.int_(Y.squeeze())]
    return S.reshape((-1,1))

def calScoreCl_feat(pred:MNISTTrainer, X:np.ndarray, Y:np.ndarray):
    # X 下一步由 classifier2 处理
    prob = pred.predict_feat(X)
    S = 1 - prob[np.arange(X.shape[0]), Y.squeeze()]
    return S.reshape((-1,1))

if __name__ == '__main__':
    train_images = np.random.rand(1000, 28, 28, 3).astype(np.float32)
    train_labels = np.random.randint(0, 7, (1000, 1))

    trainer = MNISTTrainer(
        batch_size=64,
        learning_rate=0.001,
        num_epochs=10
    )

    trainer.prepare_data(
        train_images=train_images,
        train_labels=train_labels,
    )

    # prob = trainer.predict_prob(train_images)
    # S = prob[np.arange(prob.shape[0]), train_labels.squeeze()]
    # S0 = prob.max(-1)
    # trainer.run_training(mute=False)
    # trainer.predict_prob(train_images).sum(-1).mean()