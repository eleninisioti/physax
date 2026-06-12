import jax.numpy as jnp

NOP = 0
IN = 1
OUT = 2
LOAD = 3
STORE = 4
MOVE = 5
ALLOCATE = 6
COMPARE = 7
IFZERO = 8
JUMP = 9
DEC = 10
INC = 11
DIVIDE = 12
SDIR = 13
GDIR = 14
SEND = 15
RECEIVE = 16
ADD = 17
SUB = 18
MUL = 19
DIV_OP = 20
MOD = 21
AND = 22
OR = 23
XOR = 24
NEG = 25
NOT = 26
SHIFT_L = 27
SHIFT_R = 28
FORK_TH = 29
KILL_TH = 30
R = 31
S = 32
Q = 33
I = 34
B = 35
SEP = 36
CLEAR = 37
CINC = 38
CDEC = 39
IS_SEP = 40
REL_LOAD = 41
REL_STORE = 42
IFNOTZERO = 43

UP_IS_SIZE = 44
BLANK = -1

# Number of operands for each opcode (index = opcode, 44 entries)
# From the instruction set definition file
N_OPERANDS = jnp.array([
    0,  # 0: NOP
    1,  # 1: IN
    1,  # 2: OUT
    2,  # 3: LOAD
    2,  # 4: STORE
    2,  # 5: MOVE
    1,  # 6: ALLOCATE
    3,  # 7: COMPARE
    1,  # 8: IFZERO
    1,  # 9: JUMP
    1,  # 10: DEC
    1,  # 11: INC
    0,  # 12: DIVIDE
    1,  # 13: SDIR
    1,  # 14: GDIR
    1,  # 15: SEND
    1,  # 16: RECEIVE
    3,  # 17: ADD
    3,  # 18: SUB
    3,  # 19: MUL
    3,  # 20: DIV
    3,  # 21: MOD
    3,  # 22: AND
    3,  # 23: OR
    3,  # 24: XOR
    2,  # 25: NEG
    2,  # 26: NOT
    2,  # 27: SHIFT_L
    2,  # 28: SHIFT_R
    1,  # 29: FORK_TH
    0,  # 30: KILL_TH
    0,  # 31: R
    0,  # 32: S
    0,  # 33: Q
    0,  # 34: I
    0,  # 35: B
    0,  # 36: SEPARATOR
    1,  # 37: CLEAR
    1,  # 38: CINC
    1,  # 39: CDEC
    2,  # 40: IS_SEP
    3,  # 41: REL_LOAD
    3,  # 42: REL_STORE
    1,  # 43: IFNOTZERO
], dtype=jnp.int32)

# SS: plotting values for percentile plots
LS = ['dotted','dashdot','dashed','solid','dashed','dashdot','dotted']
PERCENTILES = jnp.array([5,10,25,50,75,90,95])

