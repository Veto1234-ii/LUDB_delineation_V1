import math

def get_test_and_train_ids(LUDB_data):
    patients_ids=list(LUDB_data.keys())
    # 70 % трейн, 30% тест
    num_train = math.ceil(len(patients_ids) * 0.7)
    train_ids = patients_ids[0:num_train]
    test_ids = patients_ids[num_train:]
    return train_ids, test_ids