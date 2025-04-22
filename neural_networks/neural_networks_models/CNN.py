import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from datetime import datetime

from neural_networks.neural_networks_helpers.helpers_CNN.get_metrics import get_metrics_of_one_CNN
from paths import SAVED_NETS_PATH
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, conv1_out=32, conv2_out=64, kernel1_size=3, kernel2_size=2, dropout_rate=0.2, input_size=400):
        super(CNN, self).__init__()
        self.LEAD_NAME = None
        self.POINT_TYPE = None
        self.F1 = 0.0
        self.mean_err = 0.0
        self.input_size = input_size

        # Свёрточные слои (как в первой версии)
        self.conv1 = nn.Conv1d(1, conv1_out, kernel_size=kernel1_size, padding=kernel1_size // 2)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=kernel2_size)
        self.dropout_layer = nn.Dropout(dropout_rate)

        # Динамический расчёт размера
        with torch.no_grad():
            dummy = torch.randn(1, 1, self.input_size)
            dummy = F.relu(self.conv1(dummy))
            dummy = F.max_pool1d(dummy, 2)
            dummy = F.relu(self.conv2(dummy))
            dummy = F.max_pool1d(dummy, 2)
            self.flatten_size = dummy.view(-1).shape[0]

        # Полносвязные слои (как в первой версии)
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Классификатор (как в первой версии)
        self.classifier = nn.Sequential(
            nn.Linear(64, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_reconstructed=False):
        if x.dim() not in [2, 3]:
            raise ValueError(f"Ожидаемый вход: (batch_size, {self.input_size}) или (batch_size, 1, {self.input_size}). Получено: {x.shape}")

        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_size]

        # Forward conv (аналогично первой версии)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = self.dropout_layer(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.dropout_layer(x)

        # Полносвязные слои
        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        output = self.classifier(x)

        if return_reconstructed:
            return output, None  # Для совместимости, но декодер не реализован
        return output

    def get_info(self):
        return self.F1, self.mean_err, self.input_size, self.POINT_TYPE, self.LEAD_NAME

    def get_metrics(self):
        metrics = {
            'F1_seg': 0.0,
            'mean_err_seg': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        return metrics

    # def forward(self, x):
    #     # Проверяем, что входные данные имеют правильную форму
    #     if x.dim() not in [2, 3]:
    #         raise ValueError(f"Ожидаемый вход: (batch_size, 500) или (batch_size, 1, 500). Получено: {x.shape}")
    #
    #     # Добавляем размерность канала, если нужно
    #     if x.dim() == 2:
    #         x = x.unsqueeze(1)  # (batch_size, 1, 500)
    #     print(f"Input shape: {x.shape}")
    #
    #     # Энкодер
    #     x = self.encoder[0](x)  # Conv1d
    #     print(f"After encoder[0] (Conv1d): {x.shape}")
    #     x = self.encoder[1](x)  # ReLU
    #     print(f"After encoder[1] (ReLU): {x.shape}")
    #     x = self.encoder[2](x)  # MaxPool1d
    #     print(f"After encoder[2] (MaxPool1d): {x.shape}")
    #     x = self.encoder[3](x)  # Conv1d
    #     print(f"After encoder[3] (Conv1d): {x.shape}")
    #     x = self.encoder[4](x)  # ReLU
    #     print(f"After encoder[4] (ReLU): {x.shape}")
    #     x = self.encoder[5](x)  # MaxPool1d
    #     print(f"After encoder[5] (MaxPool1d): {x.shape}")
    #     x = self.encoder[6](x)  # Flatten
    #     print(f"After encoder[6] (Flatten): {x.shape}")
    #     x = self.encoder[7](x)  # Linear
    #     print(f"After encoder[7] (Linear): {x.shape}")
    #     x = self.encoder[8](x)  # ReLU
    #     print(f"After encoder[8] (ReLU): {x.shape}")
    #     x = self.encoder[9](x)  # Linear
    #     print(f"After encoder[9] (Linear): {x.shape}")
    #     encoded = x
    #
    #     # Сигмоидный выход
    #     x = self.sigm[0](x)  # Linear
    #     print(f"After sigm[0] (Linear): {x.shape}")
    #     x = self.sigm[1](x)  # Tanh
    #     print(f"After sigm[1] (Tanh): {x.shape}")
    #     x = self.sigm[2](x)  # Linear
    #     print(f"After sigm[2] (Linear): {x.shape}")
    #     x = self.sigm[3](x)  # Sigmoid
    #     print(f"After sigm[3] (Sigmoid): {x.shape}")
    #     sigm = x
    #
    #     return sigm

    def get_win_len(self):
        return self.input_size

    def apply(self, signal):
        # Преобразуем сигнал в тензор, если он еще не является тензором
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32)

        # Проверяем, что входные данные имеют правильную форму
        if signal.dim() not in [1, 2]:
            raise ValueError(f"Ожидаемый вход: (500,) или (batch_size, 500). Получено: {signal.shape}")

        # Проверяем длину сигнала
        expected_length = self.input_size
        if signal.size(-1) != expected_length:
            raise ValueError(f"Ожидаемая длина сигнала: {expected_length}. Получено: {signal.size(-1)}")

        # Добавляем измерение батча, если его нет
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        with torch.no_grad():  # Отключаем вычисление градиентов
            binvote = self.forward(signal)

        return binvote

    def add_info(self, F1, mean_err, POINT_TYPE, LEAD_NAME):
        self.F1 = F1
        self.mean_err = mean_err
        self.POINT_TYPE = POINT_TYPE
        self.LEAD_NAME = LEAD_NAME


def save_model(binary_dataset, POINT_TYPE, LEAD_NAME, epochs):
    signals_train, labels_train = binary_dataset.get_train()
    class1 = torch.from_numpy(signals_train[labels_train == 1]).float()
    class2 = torch.from_numpy(signals_train[labels_train == 0]).float()

    train_loader = DataLoader(class1, batch_size=50, shuffle=False)
    train_loader_2 = DataLoader(class2, batch_size=50, shuffle=False)

    model = CNN(32, 128, 7, 3, 0.339, 1000)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    losses1 = []
    losses2 = []

    num_epoch = epochs
    # Обучение
    model.train()
    for epoch in range(num_epoch):
        for i, (data_batch_1, data_batch_2) in enumerate(zip(train_loader, train_loader_2)):
            optimizer.zero_grad()
            # Обучаем модель на батче из train_loader
            sigm_1 = model(data_batch_1)

            target1 = torch.ones(data_batch_1.shape[0], 1)
            loss_sigm_1 = criterion(sigm_1, target1)
            loss_sigm_1.backward()

            # Обучаем модель на батче из train_loader_2
            sigm_2 = model(data_batch_2)
            target2 = torch.zeros(data_batch_2.shape[0], 1)
            loss_sigm_2 = criterion(sigm_2, target2)
            loss_sigm_2.backward()

            optimizer.step()

        print(f"Epoch [{epoch}/{num_epoch}] Loss binary vote: {loss_sigm_1.item():.3f}, {loss_sigm_2.item():.3f}")
        losses1.append(loss_sigm_1.item())
        losses2.append(loss_sigm_2.item())
    model.eval()

    F1, mean_err, precision, recall = get_metrics_of_one_CNN(model, binary_dataset.get_test()[0], binary_dataset.get_test()[1], threshold=0.8)
    model.add_info(F1, mean_err, POINT_TYPE, LEAD_NAME)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    os.makedirs(f"{SAVED_NETS_PATH}", exist_ok=True)
    torch.save(model, f"{SAVED_NETS_PATH}/{binary_dataset.get_name()}_{timestamp}.pth")

