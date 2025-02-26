from settings import LEADS_NAMES, POINTS

def get_one_lead_delineation_by_patient_id(patient_id,  LUDB_data, lead_name=LEADS_NAMES.i, point_type=POINTS.QRS_PEAK):
    """ Для данного пациента получить координаты точек такого-то типа в таком-то отведении. """
    result_coords = []

    wave = None
    if point_type in [POINTS.QRS_PEAK, POINTS.QRS_END, POINTS.QRS_START]:
        wave = 'qrs'
    else:
        if point_type in [POINTS.T_PEAK, POINTS.T_END, POINTS.T_START]:
            wave = 't'
        else:
            if point_type in [POINTS.P_PEAK, POINTS.P_END, POINTS.P_START]:
                wave = 'p'

    id_in_triplet = None
    if point_type in [POINTS.QRS_PEAK, POINTS.T_PEAK, POINTS.P_PEAK]:
        id_in_triplet=1
    else:
        if point_type in [POINTS.QRS_START, POINTS.T_START, POINTS.P_START]:
            id_in_triplet = 0
        else:
            if point_type in [POINTS.QRS_END, POINTS.T_END, POINTS.P_END]:
                id_in_triplet = 2

    points_triplets = LUDB_data[patient_id]['Leads'][lead_name]['Delineation'][wave]
    for triplet in points_triplets:
        result_coords.append(triplet[id_in_triplet])

    return result_coords


if __name__ == "__main__":
    from get_LUDB_data import get_LUDB_data
    from get_some_test_patient_id import get_some_test_patient_id

    LUDB_data = get_LUDB_data()
    patient_id =  get_some_test_patient_id()

    delineation = get_one_lead_delineation_by_patient_id(patient_id,  LUDB_data, lead_name=LEADS_NAMES.i, point_type=POINTS.QRS_PEAK)
    print(delineation)