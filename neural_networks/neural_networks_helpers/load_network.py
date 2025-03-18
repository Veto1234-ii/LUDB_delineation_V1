import torch
import os

def get_appliable(name):
    
    # Получаем текущий путь к файлу
    current_file_path = os.path.abspath(__file__)
    
    # Поднимаемся на два уровня выше
    one_level_up = os.path.dirname(current_file_path)
    two_level_up = os.path.dirname(one_level_up)
    three_Level_up = os.path.dirname(two_level_up)
    
    # Путь к целевому файлу
    target_file_path = os.path.join(three_Level_up, "SAVED_NETS", f"{name}.pth")
    

    
    appliable = torch.load(target_file_path, weights_only=False)
    
    return appliable