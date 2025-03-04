import torch
import torch.nn as nn
import torch.nn.functional as F
class UNet1D(nn.Module):
    def __init__(self):
        super(UNet1D, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 32,
                      kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 500 -> 250
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64,
                      kernel_size=3, padding=1, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 250 -> 125
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 128,
                      kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64,
                               kernel_size=3, stride=2,
                               output_padding=1),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64 + 64, 32,
                               kernel_size=3, stride=2,
                               output_padding=1),
            nn.ReLU(),
        )

        self.out_conv = nn.Conv1d(32 + 32, 1, kernel_size=3)


    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # (50, 500)

        x = x.unsqueeze(1)  # Добавляем канал (batch, 1, 500)
        # print(f"After unsqueeze: {x.shape}")  # (50, 1, 500)

        # Encoder path
        e1 = self.enc1(x)
        # print(f"After enc1: {e1.shape}")  # (50, 32, 250)

        e2 = self.enc2(e1)
        # print(f"After enc2: {e2.shape}")  # (50, 64, 125)

        # Bottleneck
        b = self.bottleneck(e2)
        # print(f"After bottleneck: {b.shape}")  # (50, 128, 125)

        # Decoder path (с объединением скипов)
        d1 = self.dec1(b)
        # print(f"After dec1: {d1.shape}")  # (50, 64, 250)

        e2_upsampled = F.interpolate(e2, size=d1.shape[-1], mode="nearest")  # (50, 64, 250)
        d1 = torch.cat((d1, e2_upsampled), dim=1)
        # print(f"After skip connection 1: {d1.shape}")  # (50, 128, 250)

        d2 = self.dec2(d1)
        # print(f"After dec2: {d2.shape}")  # (50, 32, 500)

        e1_upsampled = F.interpolate(e1, size=d2.shape[-1], mode="nearest")  # (50, 32, 500)
        d2 = torch.cat((d2, e1_upsampled), dim=1)
        # print(f"After skip connection 2: {d2.shape}")  # (50, 64, 500)

        # Output layer
        out = self.out_conv(d2)
        # print(f"After out_conv: {out.shape}")  # (50, 1, 500)

        # out = torch.sigmoid(out)

        out = out.squeeze(1)  # Убираем канал
        # print(f"Final output shape: {out.shape}")  # (50, 500)


        return out

    def get_win_len(self):
        return 500

    def apply(self, signal):
        # Преобразуем сигнал в тензор, если он еще не является тензором
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32)

            # Проверяем, что входные данные имеют правильную форму
            if signal.dim() not in [1, 2]:
                raise ValueError(f"Ожидаемый вход: (500,) или (batch_size, 500). Получено: {signal.shape}")

            # Проверяем длину сигнала
            expected_length = 500
            if signal.size(-1) != expected_length:
                raise ValueError(f"Ожидаемая длина сигнала: {expected_length}. Получено: {signal.size(-1)}")

            # Добавляем измерение батча, если его нет
            if signal.dim() == 1:
                signal = signal.unsqueeze(0)

        # Передаем сигнал через модель и получаем маску
        with torch.no_grad():  # Отключаем вычисление градиентов
            mask = self.forward(signal)

        return mask