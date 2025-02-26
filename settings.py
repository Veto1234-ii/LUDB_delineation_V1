from enum import Enum

PATH_TO_LUDB = 'C:\\Users\\User\\PycharmProjects\\LUDB_delineation_V1\\LUDB\\ecg_data_200.json'

class LeadsNames:
    def __init__(self):
        self.i='i'
        self.ii='ii'
        self.iii='iii'
        self.avr='avr'
        self.avl='avl'
        self.avf='avf'
        self.v1 ='v1'
        self.v2 ='v2'
        self.v3 ='v3'
        self.v4='v4'
        self.v5='v5'
        self.v6='v6'
LEADS_NAMES = LeadsNames()

LEADS_NAMES_ORDERED = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

class WAVES_NAMES:
    def __init__(self):
        self.P = 'p'
        self.QRS = 'qrs'
        self.T = 't'

class POINTS(Enum):
    T_START = 1
    T_PEAK = 2
    T_END = 3

    P_START = 4
    P_PEAK = 5
    P_END = 6

    QRS_START = 7
    QRS_PEAK = 8
    QRS_END = 9

FREQUENCY = 500 # измерений в секунду