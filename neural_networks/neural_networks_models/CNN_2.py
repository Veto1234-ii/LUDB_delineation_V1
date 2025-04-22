import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os
from datetime import datetime

from neural_networks.neural_networks_helpers.helpers_CNN import get_F1_of_one_CNN
import torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler

from neural_networks.neural_networks_helpers.helpers_CNN.F1_of_CNN import get_F1_segmentation_of_one_CNN
from neural_networks.neural_networks_helpers.helpers_CNN.get_metrics import get_metrics_of_one_CNN
from paths import SAVED_NETS_PATH


class CNN_2(nn.Module):
    def __init__(self, conv1_out=32, conv2_out=64, kernel1_size=3, kernel2_size=2, dropout_rate=0.2, input_size=400, kernel3_size=3, conv3_out=128):
        super(CNN_2, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.LEAD_NAME = None
        self.POINT_TYPE = None
        self.F1 = 0.0
        self.mean_err = 0.0
        self.input_size = input_size
        self.metrics = {
            'F1_seg': 0.0,
            'mean_err_seg': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

        self.conv1 = nn.Conv1d(1, conv1_out, kernel_size=kernel1_size, padding=kernel1_size // 2)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=kernel2_size)
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.conv3 = nn.Conv1d(conv2_out, conv3_out, kernel_size=kernel3_size)
        self.bn3 = nn.BatchNorm1d(conv3_out)
        self.dropout_layer = nn.Dropout(dropout_rate)

        with torch.no_grad():
            dummy = torch.randn(1, 1, self.input_size)
            dummy = F.relu(self.conv1(dummy))
            dummy = F.avg_pool1d(dummy, 2)
            dummy = F.relu(self.conv2(dummy))
            dummy = F.avg_pool1d(dummy, 2)
            dummy = F.relu(self.conv3(dummy))  # Без пулинга
            self.flatten_size = dummy.view(-1).shape[0]

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise ValueError(
                f"Ожидаемый вход: (batch_size, {self.input_size}) или (batch_size, 1, {self.input_size}). Получено: {x.shape}")

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.avg_pool1d(x, 2)  # AvgPool вместо MaxPool
        x = self.dropout_layer(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool1d(x, 2)  # AvgPool вместо MaxPool
        x = self.dropout_layer(x)

        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.avg_pool1d(x, 2)  # AvgPool вместо MaxPool
        x = self.dropout_layer(x)

        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        output = self.classifier(x)

        return output

    def get_info(self):
        return self.F1, self.mean_err, self.input_size, self.POINT_TYPE, self.LEAD_NAME

    def get_metrics(self):
        return self.metrics

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

    def add_metrics(self, F1_seg, mean_err_seg, precision, recall):
        self.metrics['F1_seg'] = F1_seg
        self.metrics['mean_err_seg'] = mean_err_seg
        self.metrics['precision'] = precision
        self.metrics['recall'] = recall


def save_model_2(binary_dataset, POINT_TYPE, LEAD_NAME, epochs, binary_dataset_seg, input_size, binary_dataset_qrs):
    print("save_model_2")
    signals_train, labels_train = binary_dataset.get_train()
    # signals_train_qrs, labels_train_qrs = binary_dataset_qrs.get_train()

    # signals_train_qrs = signals_train_qrs[labels_train_qrs == 1]
    # labels_train_qrs = np.zeros(len(signals_train_qrs))

    # signals_train = np.concatenate([signals_train, signals_train_qrs])
    # labels_train = np.concatenate([labels_train, labels_train_qrs])

    # # Применяем RandomOverSampler для балансировки
    # oversampler = RandomOverSampler()
    # signals_reshaped = signals_train.reshape(len(signals_train), -1)
    # signals_resampled, labels_resampled = oversampler.fit_resample(signals_reshaped, labels_train)
    #
    # # Возвращаем обратно в исходную форму
    # signals_resampled = signals_resampled.reshape(-1, *signals_train.shape[1:])

    # Преобразуем в тензоры PyTorch
    signals_tensor = torch.from_numpy(signals_train).float()
    labels_tensor = torch.from_numpy(labels_train).float().unsqueeze(1)

    # Создаем единый DataLoader
    dataset = TensorDataset(signals_tensor, labels_tensor)
    train_loader = DataLoader(dataset, batch_size=25, shuffle=True)

    model = CNN_2(32, 64, 5, 3, 0.3, input_size, 3, 128)
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

    F1, mean_err, precision, recall = get_metrics_of_one_CNN(model, binary_dataset.get_test()[0], binary_dataset.get_test()[1], threshold=0.8)
    model.add_info(F1, mean_err, POINT_TYPE, LEAD_NAME)

    F1_seg, mean_err_seg = get_F1_segmentation_of_one_CNN(model, binary_dataset_seg.get_train()[0],
                                                          binary_dataset_seg.get_train()[1], threshold=0.8,
                                                          seg_size=len(binary_dataset_seg.get_train()[0][0])//2)

    model.add_metrics(F1_seg, mean_err_seg, precision, recall)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    os.makedirs(f"{SAVED_NETS_PATH}", exist_ok=True)
    torch.save(model, f"{SAVED_NETS_PATH}/{binary_dataset.get_name()}_{timestamp}.pth")

