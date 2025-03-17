from enum import Enum

# Настройки LUDB:
PATH_TO_LUDB = "D:\\6 семестр\\Курсовая\\ECG\\ecg_data_200.json"

FREQUENCY = 500 # измерений в секунду

# Переменные, связанные с отведениями:
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


# Переменные, связанные с точками разметки
class POINTS_TYPES(Enum):
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
    POINTS_TYPES.T_START: '#712342',
    POINTS_TYPES.T_PEAK: '#371261',
    POINTS_TYPES.T_END: '#601962',

    POINTS_TYPES.QRS_PEAK : '#239731',
    POINTS_TYPES.QRS_END : '#562147',
    POINTS_TYPES.QRS_START : '#622508',

    POINTS_TYPES.P_END : '#712342',
    POINTS_TYPES.P_START : '#711142',
    POINTS_TYPES.P_PEAK : '#714442'
    }

# Переменные, связанные с целыми волнами/сегментами
class WavesTypes:
    def __init__(self):
        self.P = 'p'
        self.QRS = 'qrs'
        self.T = 't',
        self.NO_WAVE = 'nowave'

WAVES_TYPES = WavesTypes()

WAVES_TYPES_COLORS = {
            WAVES_TYPES.NO_WAVE: 'gray',
            WAVES_TYPES.QRS: POINTS_TYPES_COLORS[POINTS_TYPES.QRS_PEAK],
            WAVES_TYPES.P: POINTS_TYPES_COLORS[POINTS_TYPES.P_PEAK],
            WAVES_TYPES.T: POINTS_TYPES_COLORS[POINTS_TYPES.T_PEAK]
        }

#  Общие настройки рисования:

# вертикальные линии разметки
DELINEATION_LINEWIDTH = 1.1

# рисование сигнала на миллиметровке:
SIGNAL_LINEWIDTH = 1
MINOR_GRID_LINEWIDTH = 0.2
MAJOR_GRID_LINEWITH = 0.6
SIGNAL_COLOR =  (0.1,0.3,0.5)
GRID_COLOR = 'gray'
