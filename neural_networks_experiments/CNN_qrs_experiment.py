# создаем микродатасет (datasets.binary_datasets.binary_dataset_creator)
# обучаем на нем сеть (neural_networks_models.CNN) на трейновой части получившегося датасета (.get_train) + сохраняем график падения ошибки
# вытскиваем истинные разметки и сигнал соотв. отведений из всех тестовых пациентах
# ( id-шники тестовых пациентов взять  из datasets.LUDB_utils.train_test_patients_ids_split)
# (вытаскивание сигнала из пациента datasets.LUDB_utils.get_signal_by_patient_id )
# меряем F1 и среднее отклонение вызовом ф-ции neural_networks_helpers.helpers_CNN.get_F1_of_CNN


# сохраняем чекпоинт
# сохраняем картинки вида : нарисован сигнал, на нем правильная разметка данной точки и разметка от сети
# распечатываем f1  консоль

