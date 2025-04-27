import os
from paths import SAVED_NETS_PATH
import torch

def load_best_net(point_type, lead_name):
    folder_path = SAVED_NETS_PATH
    F1_max = 0
    best_net = ''

    # Проверяем, существует ли папка
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не существует.")
        return

    for model_file in os.listdir(folder_path):
        if model_file.endswith(".pth"):
            model_path = os.path.join(folder_path, model_file)

            try:
                model = torch.load(model_path, weights_only=False)
                F1, _, _, point_type_, lead_name_ = model.get_info()
                metrics = model.get_metrics()

                # устанавливаем метрику, по которой будем выбирать лучшую сеть
                # если на всем сигнале - F1
                # если на некотором сегменте - metrics['F1_seg']
                metric = metrics['F1_seq']
                if point_type_ == point_type and lead_name_ == lead_name:
                    if metric > F1_max:
                        F1_max = metric
                        best_net = model_path
            except Exception as e:
                print(f"Ошибка при загрузке или применении модели {model_file}: {e}")

    if not os.path.exists(best_net):
        raise FileNotFoundError(f"Файл не существует: {best_net}")

    appliable = torch.load(best_net, weights_only=False)
    return appliable
