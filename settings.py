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

POINTS_TYPES_COLORS = {
    POINTS.T_START:'#71234255',
    POINTS.T_PEAK:'#37126138',
    POINTS.T_END:'#60196214',

    POINTS.QRS_PEAK : '#239731',
    POINTS.QRS_END : '#5621472',
    POINTS.QRS_START : '#6225081',

    POINTS.P_END : '#71234255',
    POINTS.P_START : '#71114255',
    POINTS.P_PEAK : '#71444255'
    }


FREQUENCY = 500 # измерений в секунду

class WavesTypes:
    def __init__(self):
        self.P = 'p'
        self.QRS = 'qrs'
        self.T = 't'

WAVES_TYPES = WavesTypes()

#      НАСТРОЙКИ РИСОВАНИЯ

# вертикальные линии разметки
DELINEATION_LINEWIDTH = 0.9

# рисование сигнала на миллиметровке:
SIGNAL_LINEWIDTH = 1
MINOR_GRID_LINEWIDTH = 0.2
MAJOR_GRID_LINEWITH = 0.6
SIGNAL_COLOR =  (0.1,0.3,0.5)
GRID_COLOR = 'gray'
