import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os
from datetime import datetime

from neural_networks.neural_networks_helpers.helpers_CNN import get_F1_of_one_CNN
import torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler


class CNN(nn.Module):
    def __init__(self, conv1_out=32, conv2_out=64, kernel1_size=3, kernel2_size=2, dropout_rate=0.2, input_size=400):
        super(CNN, self).__init__()
        self.LEAD_NAME = None
        self.POINT_TYPE = None
        self.F1 = None
        self.mean_err = None
        self.input_size = input_size

        self.conv1 = nn.Conv1d(1, conv1_out, kernel_size=kernel1_size, padding=kernel1_size // 2)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=kernel2_size)
        self.dropout_layer = nn.Dropout(dropout_rate)

        with torch.no_grad():
            dummy = torch.randn(1, 1, self.input_size)
            dummy = F.relu(self.conv1(dummy))
            dummy = F.max_pool1d(dummy, 2)
            dummy = F.relu(self.conv2(dummy))
            dummy = F.max_pool1d(dummy, 2)
            self.flatten_size = dummy.view(-1).shape[0]

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise ValueError(
                f"Ожидаемый вход: (batch_size, {self.input_size}) или (batch_size, 1, {self.input_size}). Получено: {x.shape}")

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = self.dropout_layer(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.dropout_layer(x)

        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        output = self.classifier(x)

        return output

    def get_info(self):
        return self.F1, self.mean_err, self.input_size, self.POINT_TYPE, self.LEAD_NAME

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
        if signal.size(-1) != self.input_size:
            raise ValueError(f"Ожидаемая длина сигнала: {self.input_size}. Получено: {signal.size(-1)}")

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

    # Применяем RandomOverSampler для балансировки
    oversampler = RandomOverSampler()
    signals_reshaped = signals_train.reshape(len(signals_train), -1)
    signals_resampled, labels_resampled = oversampler.fit_resample(signals_reshaped, labels_train)

    # Возвращаем обратно в исходную форму
    signals_resampled = signals_resampled.reshape(-1, *signals_train.shape[1:])

    # Преобразуем в тензоры PyTorch
    signals_tensor = torch.from_numpy(signals_resampled).float()
    labels_tensor = torch.from_numpy(labels_resampled).float().unsqueeze(1)

    # Создаем единый DataLoader
    dataset = TensorDataset(signals_tensor, labels_tensor)
    train_loader = DataLoader(dataset, batch_size=50, shuffle=True)

    model = CNN(16, 64, 7, 3, 0.3, 300)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    losses = []

    num_epoch = epochs
    # Обучение
    model.train()
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        for signals, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.3f}")
    model.eval()

    F1, mean_err = get_F1_of_one_CNN(model, binary_dataset.get_test()[0], binary_dataset.get_test()[1], threshold=0.8)
    model.add_info(F1, mean_err, POINT_TYPE, LEAD_NAME)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    os.makedirs("SAVED_NETS", exist_ok=True)
    torch.save(model, f"SAVED_NETS/{binary_dataset.get_name()}_{epochs}_16-64_{timestamp}.pth")
