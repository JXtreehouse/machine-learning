DATA_PATH = 'D:\workspace\zimu'
DATA_FILE = 'subtitle.corpus'
CPT_PATH = 'checkpoints'


THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3


BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]

NUM_LAYERS = 3
HIDDEN_SIZE = 256  # 神经元个数
BATCH_SIZE = 64

LR = 0.05

MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512

VOCAB_SIZE = 1000
SENTENCE_MAX_LEN = 30
