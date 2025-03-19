import os
import torch
import pandas as pd
from colorama import Fore, Style, init

# Инициализация colorama
init()

# Словарь, содержащий минимальные значения длин столбцов
SIZES_COLS = {
    "Файл модели": len("Файл модели"),
    "F1": len("F1"),
    "Ошибка": len("Ошибка"),
    "Длина окна": len("Длина окна"),
    "Тип точки": len("Тип точки"),
    "Имя отведения": len("Имя отведения")
}

def run(folder_path):
    # Проверяем, существует ли папка
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не существует.")
        return

    # Создаем список для хранения данных
    data = []

    for model_file in os.listdir(folder_path):
        # Проверяем, что файл имеет расширение .pth
        if model_file.endswith(".pth"):
            model_path = os.path.join(folder_path, model_file)

            try:
                # Загружаем модель
                model = torch.load(model_path, weights_only=False)

                # Получаем информацию о модели
                F1, err, win_len, point_type, lead_name = model.get_info()
                point_type = str(point_type).split(".")[1]

                lengths = [model_file, f"{F1:.2f}", f"{err:.2f}", str(win_len), point_type, str(lead_name)]
                for i, key in enumerate(SIZES_COLS.keys()):
                    SIZES_COLS[key] = max(SIZES_COLS[key], len(lengths[i]))

                data.append({
                    "Файл модели": model_file,
                    "F1": F1,
                    "Ошибка": err,
                    "Длина окна": win_len,
                    "Тип точки": point_type,
                    "Имя отведения": lead_name
                })

            except Exception as e:
                print(f"Ошибка при загрузке или применении модели {model_file}: {e}")

    # Создаем DataFrame из списка данных
    df = pd.DataFrame(data)

    # Возвращаем таблицу
    return df

def highlight_f1(value):
    if value > 0.8:
        return f"{Fore.GREEN}{value:.2f}{Style.RESET_ALL}"  # Зеленый цвет для F1 > threshold
    else:
        return f"{Fore.RED}{value:.2f}{Style.RESET_ALL}"  # Красный цвет для F1 <= threshold

def print_table(df):
    # Определяем ширину столбцов
    col_widths = {
        "Файл модели": SIZES_COLS["Файл модели"],
        "F1": SIZES_COLS["F1"],
        "Ошибка": SIZES_COLS["Ошибка"],
        "Длина окна": SIZES_COLS["Длина окна"],
        "Тип точки": SIZES_COLS["Тип точки"],
        "Имя отведения": SIZES_COLS["Имя отведения"]
    }

    # Выводим заголовки
    headers = [f"{col:<{width}}" for col, width in col_widths.items()] # Выравнивание по левому краю
    print("|" + "|".join(headers) + "|")
    print("|" + "+".join(["-" * width for width in col_widths.values()]) + "|")

    # Выводим данные
    for _, row in df.iterrows():
        row_data = [
            str(row["Файл модели"]).ljust(col_widths["Файл модели"]),
            highlight_f1(row["F1"]).ljust(col_widths["F1"]),
            f"{row['Ошибка']:.2f}".ljust(col_widths["Ошибка"]),
            str(row["Длина окна"]).ljust(col_widths["Длина окна"]),
            str(row["Тип точки"]).ljust(col_widths["Тип точки"]),
            str(row["Имя отведения"]).ljust(col_widths["Имя отведения"])
        ]
        print("|" + "|".join(row_data) + "|")

if __name__ == "__main__":
    folder_path = "SAVED_NETS"
    table = run(folder_path)

    # Сортируем таблицу по F1
    table = table.sort_values(by="F1", ascending=False)

    # Выводим таблицу
    print_table(table)