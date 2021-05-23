"""
ECE586 - Final Project
Team: Khoi, Melinda, Nguyen, Thong

Provide all tools/API for working with Lite MIPS
"""

from enum import Enum, auto
import os.path
import copy
import re

# -------------------- DEFINES/CONSTANTS --------------------

# offset for each portions in an instruction
LEN_OP = 6
LEN_REG = 5
LEN_IMM = 16
OFFSET_OP = 0
OFFSET_RS = OFFSET_OP + LEN_OP
OFFSET_RT = OFFSET_RS + LEN_REG
OFFSET_RD = OFFSET_RT + LEN_REG
OFFSET_IMM = OFFSET_RT + LEN_REG

# max number of register
MAX_REGS = 32

# mask for 32-bit integer
INT_MASK_32 = 0xFFFFFFFF

# max length for a hex string in the memory image
MAX_HEX_LENGTH = 8

# max length in bit of an instruction
LEN_INS = MAX_HEX_LENGTH * 4

# min imm value
MIN_IMM = -int(2 ** LEN_IMM / 2)

# max imm value
MAX_IMM = int(2 ** LEN_IMM / 2) - 1

# number of stages in the pipeline
N_STAGES = 5

# -------------------- GLOBAL VARIABLES --------------------

# debug level to determine whether a message should be displayed using the debugPrint
__debug_level = False


class MessageType(Enum):
    """
    Type assigned to each debug message
    """

    INFO = auto(),
    DEBUG = auto()


class DebugLevel(Enum):
    """
    All levels supported in the debugPrint function
    """

    # will not show anything
    SILENCE = auto(),

    # will shown INFO msg only
    INFO = auto(),

    # will shown INFO/DEBUG msg
    DEBUG = auto()


# -------------------- FUNCS --------------------


def genBitMask(bit_length: int):
    """
    Generate a bit mask based on the number of required bit-length

    :param bit_length:
    :return:
    """

    bin_str = ''
    for i in range(bit_length):
        bin_str += '1'

    return int(bin_str, 2)


def convertHex2Bin(hex_str: str):
    """
    Convert a given string of hex values to binary values
    :return: a string of corresponding binary values
    """

    bin_str = ''

    # convert each hex value to binary
    for a_hex in hex_str:
        a_dec = int(a_hex, 16)
        a_bin = bin(a_dec)[2:].zfill(4)
        bin_str += a_bin

    return bin_str


def convertBin2SignedInt(bin_str: str, bit_length) -> int:
    """
    Convert a given binary-string into its corresponding signed integer (2-complement)

    :param bin_str: the binary string
    :param bit_length: max number of bit used for the number
    :return:
    """

    # trim the string to bit-length digits (counting from the right) & convert to number
    bin_str = bin_str[-bit_length:]
    num = int(bin_str, 2)

    # handle 2-complement
    if bin_str[0] == '1':
        num = num - 2 ** bit_length
    return num


def convertSignedInt2Hex(signed_int: int, bit_length=LEN_INS) -> str:
    """
    Convert the given signed integer into its corresponding hex string

    :param bit_length:
    :param signed_int: the signed integer
    :return:
    """

    # mask to indicate how many bits we want to take into account of
    mask = genBitMask(bit_length)
    hex_str = hex(signed_int & mask)

    # fill with leading 0
    hex_len = int(bit_length / 4)
    hex_str = hex_str[2:].zfill(hex_len)
    return hex_str


def convertSignedInt2Bin(signed_int: int, bit_length: int) -> str:
    """
    Convert the given signed integer into its corresponding binary string

    :param bit_length:
    :param signed_int: the signed integer
    :return:
    """

    # mask to indicate how many bits we want to take into account of
    mask = genBitMask(bit_length)
    bin_str = bin(signed_int & mask)

    # fill with leading 0
    bin_str = bin_str[2:].zfill(bit_length)
    return bin_str


def convertBin2Hex(bin_str: str, hex_length=MAX_HEX_LENGTH) -> str:
    """
    Convert the given binary string to corresponding hex string

    :param bin_str:
    :param hex_length: the number of digits in the resulting hex string
    :return:
    """

    # convert to hex
    hex_str = hex(int(bin_str, 2))

    # fill leading 0 to hex
    return hex_str[2:].zfill(hex_length)


def getNameList(list_enum: []) -> []:
    """
    Return a list of name string corresponding to the given list of enum

    :param list_enum: list of input enums
    :return: list of enum names
    """

    list_names = []
    for enum in list_enum:
        list_names.append(enum.name)
    return list_names


def debugPrint(*args, sep=' ', end='\n', file=None, msg_type=MessageType.DEBUG):
    """
    Print if the debug level allows for the message type. All the arguments have the same meaning as in the print() function
    """

    if __debug_level == DebugLevel.SILENCE:
        return

    if __debug_level == DebugLevel.DEBUG \
            or (__debug_level == DebugLevel.INFO and msg_type == MessageType.INFO):
        print(*args, sep=sep, end=end, file=file)


def setDebugLevel(debug_level: DebugLevel):
    """
    Set debug level

    :param debug_level:
    :return:
    """

    global __debug_level
    __debug_level = debug_level


def getDebugLevel():
    """
    Get debug level

    :return:
    """

    return __debug_level


# -------------------- CLASS --------------------

class Opcode(Enum):
    """
    All possible opcode in this Lite MIPS
    """

    ADD = 0
    ADDI = 1
    SUB = 2
    SUBI = 3
    MUL = 4
    MULI = 5
    OR = 6
    ORI = 7
    AND = 8
    ANDI = 9
    XOR = 10
    XORI = 11
    LDW = 12
    STW = 13
    BZ = 14
    BEQ = 15
    JR = 16
    HALT = 17
    NOOP = 0b111110
    UNKNOWN = 0b111111

    def __int__(self):
        """
        Convert this enum to its corresponding integer value

        :return:
        """
        return self.value

    @staticmethod
    def getList() -> []:
        """
        Get list of all enum in this class

        :return:
        """

        return list(map(lambda opcode: opcode, Opcode))


class WriteDst(Enum):
    """
    All write destinations, metadata used by the stages
    """
    REG = auto()
    MEM = auto()
    PC = auto()
    NOT_SET = auto()


class Instruction:
    """
    Represent an instruction in Lite MIPS architecture
    """

    # set of instruction types
    list_R_types = [Opcode.AND, Opcode.OR, Opcode.XOR, Opcode.ADD, Opcode.SUB, Opcode.MUL]
    list_arithmetic_types = [Opcode.ADD, Opcode.ADDI, Opcode.SUB, Opcode.SUBI, Opcode.MUL, Opcode.MULI]
    list_logic_types = [Opcode.AND, Opcode.ANDI, Opcode.OR, Opcode.ORI, Opcode.XOR, Opcode.XORI]
    list_control_types = [Opcode.BZ, Opcode.BEQ, Opcode.JR, Opcode.HALT]
    list_mem_types = [Opcode.LDW, Opcode.STW]

    # register symbol
    R_str = 'R'

    def __init__(self):
        """
        Constructor.
        NOT recommended to create an obj directly using this instructor.
        Should use the parse function of this class.
        """

        # opcode & operands
        self.opcode = Opcode.UNKNOWN
        self.rs = 0
        self.rt = 0
        self.rd = 0
        self.imm = 0

        # Original string of hex values
        self.hex_str = ''

    def __str__(self):
        """
        String representation of this instruction

        :return: a description of this instruction
        """

        return self.toString()

    def isRType(self):
        """
        Check if this instruction is of R-type

        :return: True if R-type, False otherwise
        """

        return self.opcode in self.list_R_types

    def isArithmetic(self):
        """
        Check if this instruction is Arithmetic

        :return:
        """

        return self.opcode in self.list_arithmetic_types

    def isLogical(self):
        """
        Check if this instruction is Logical

        :return:
        """

        return self.opcode in self.list_logic_types

    def isMemoryAccess(self):
        """
        Check if this instruction is Memory Access

        :return:
        """

        return self.opcode in self.list_mem_types

    def isControlTransfer(self):
        """
        Check if this instruction is Control Transfer

        :return:
        """

        return self.opcode in self.list_control_types

    def isIType_RegWrite(self):
        """
        Return True if this instruction is I-type with a register write as the final impact

        :return:
        """

        return (self.opcode in self.list_logic_types) or (
                self.opcode in self.list_arithmetic_types) or self.opcode == Opcode.LDW

    def isNoop(self):
        """
        Return TRUE if this is a no-operation instruction

        :return:
        """

        return self.opcode == Opcode.NOOP or self.opcode == Opcode.HALT

    def getType(self) -> str:
        """
        Return type of this instruction as string

        :return:
        """

        if self.isArithmetic():
            return 'Ari'
        if self.isLogical():
            return 'Log'
        if self.isMemoryAccess():
            return 'Mem'
        if self.isControlTransfer():
            return 'Con'

        return ''

    def toString(self, add_type=False):
        """
        Return a string representation of this instruction

        :param add_type: flag to add type info to the string
        :return:
        """

        # opcode
        result = str(self.opcode.name)

        # unknown, halt, noop
        if self.opcode in [Opcode.UNKNOWN, Opcode.HALT, Opcode.NOOP]:
            return result

        # add values based on the opcode
        result += ' '
        if self.isRType():
            result += f'{self.R_str}{self.rd}, {self.R_str}{self.rs}, {self.R_str}{self.rt}'
        elif self.opcode == Opcode.BZ:
            result += f'{self.R_str}{self.rs}, {self.imm}'
        elif self.opcode == Opcode.JR:
            result += f'{self.R_str}{self.rs}'
        else:
            result += f'{self.R_str}{self.rt}, {self.R_str}{self.rs}, {self.imm}'

        # add type
        if add_type:
            result += f' ({self.getType()})'

        return result

    @staticmethod
    def parse(hex_str: str):
        """
        Parse a given string of hex values to a lite-MIPS instruction

        :param hex_str: the string containing hex values
        :return: UNKNOWN instruction if failed, the instruction with correct opcode if success
        """

        # prepare result
        ins = Instruction()

        try:
            # convert to binary string
            bin_str = convertHex2Bin(hex_str)

            # opcode
            op = bin_str[OFFSET_OP:OFFSET_RS]
            op = int(op, 2)
            ins.opcode = Opcode(op)

            # regs
            ins.rs = int(bin_str[OFFSET_RS:OFFSET_RT], 2)
            ins.rt = int(bin_str[OFFSET_RT:OFFSET_RD], 2)
            ins.rd = int(bin_str[OFFSET_RD:OFFSET_RD + LEN_REG], 2)

            # imm
            ins.imm = convertBin2SignedInt(bin_str[OFFSET_IMM:], LEN_IMM)
            # ins.imm = int(bin_str[OFFSET_IMM:], 2)
            # if bin_str[OFFSET_IMM] == '1':
            #     ins.imm = ins.imm - 2 ** LEN_IMM

            # record the hex string
            ins.hex_str = hex_str
        except Exception as e:
            # debugPrint('Error while parsing instruction: [', hex_str, '], error=', e, sep='')
            ins.opcode = Opcode.UNKNOWN

        return ins

    @staticmethod
    def checkValidReg(reg_str: str) -> int:
        """
        Check & parse if the given reg-string is valid, that is "Rx", with x is from 0 to 31

        :param reg_str:
        :return: register-index if success (>=0), -1 if failed
        """

        # check valid
        if len(reg_str) <= 1:
            return -1
        if reg_str[0] != 'R':
            return -1
        if not reg_str[1:].isnumeric():
            return -1

        # check range
        reg = int(reg_str[1:])
        if reg >= MAX_REGS:
            return -1

        return reg

    @staticmethod
    def checkValidImm(imm_str: str) -> (bool, int):
        """
        Check & parse if the given imm-string is valid, that is numeric and in range

        :param imm_str:
        :return: (True, imm) if success, (False, ...) if failed
        """

        # check valid
        if len(imm_str) <= 0:
            return False, 0
        if not imm_str.isnumeric():
            # invalid positive
            if imm_str[0] != '-':
                return False, 0
            # invalid negative
            if not imm_str[1:].isnumeric():
                return False, 0

        # check range
        imm = int(imm_str)
        if imm < MIN_IMM or imm > MAX_IMM:
            return False, 0

        return True, imm

    @staticmethod
    def parseInsStr(ins_str: str) -> str:
        """
        Parse a given string of instruction to a string of corresponding hex values, if valid

        :param ins_str: the instruction string, e.g.: "ADD R3, R1, R2", to parse
        :return: empty string if the given string is invalid format
        """

        # format & break the string into tokens
        ins_str = ins_str.upper().strip()
        tokens = re.findall(r"[-]*[\w']+", ins_str)
        if len(tokens) <= 0:
            return ''

        # prepare a binary-string & list of integers
        bin_str = ''
        list_vals = []
        list_bit_len = []

        # parse the opcode to enum if valid
        opcode = tokens[0]
        if opcode not in getNameList(Opcode.getList()):
            return ''
        opcode = Opcode[opcode]
        bin_str += convertSignedInt2Bin(int(opcode), LEN_OP)
        debugPrint(f'opcode={opcode}, bin_str={bin_str}')

        if len(tokens) == 4:

            # parse R-type
            if opcode in Instruction.list_R_types:
                # parse registers
                for i in range(1, 4):
                    reg = Instruction.checkValidReg(tokens[i])
                    if reg < 0:
                        return ''
                    list_vals.append(reg)
                    list_bit_len.append(LEN_REG)
                # rearrange register order
                tmp = list_vals.pop(0)
                list_vals.append(tmp)

            # parse I-type (no BZ/JR/HALT)
            elif opcode not in [Opcode.BZ, Opcode.JR, Opcode.HALT]:
                # parse registers
                for i in range(1, 3):
                    reg = Instruction.checkValidReg(tokens[3 - i])
                    if reg < 0:
                        return ''
                    list_vals.append(reg)
                    list_bit_len.append(LEN_REG)
                # parse imm
                (is_ok, imm) = Instruction.checkValidImm(tokens[3])
                if not is_ok:
                    return ''
                list_vals.append(imm)
                list_bit_len.append(LEN_IMM)
            else:
                return ''

        # parse BZ/JR
        elif (opcode == Opcode.BZ and len(tokens) == 3) or (opcode == Opcode.JR and len(tokens) == 2):
            # parse rs
            reg = Instruction.checkValidReg(tokens[1])
            if reg < 0:
                return ''
            list_vals.append(reg)
            list_bit_len.append(LEN_REG)
            # insert empty rt
            list_vals.append(0)
            list_bit_len.append(LEN_REG)
            # parse imm if BZ
            if opcode == Opcode.BZ:
                (is_ok, imm) = Instruction.checkValidImm(tokens[2])
                if not is_ok:
                    return ''
                list_vals.append(imm)
                list_bit_len.append(LEN_IMM)
        elif opcode != Opcode.HALT and opcode != Opcode.NOOP:
            return ''

        # parse integers & concatenate to binary-string
        for (val, bit_len) in list(zip(list_vals, list_bit_len)):
            bin_str += convertSignedInt2Bin(val, bit_len)
        if len(bin_str) < LEN_INS:
            bin_str += '0' * (LEN_INS - len(bin_str))

        hex_str = convertBin2Hex(bin_str).upper()
        return hex_str

    @staticmethod
    def parseInsFile(file_in: str, file_out: str):
        """
        Parse all instruction in the given input file & write to the given output file
        :param file_in:
        :param file_out:
        :return:
        """

        # check & open files
        if not os.path.isfile(file_in):
            debugPrint('File not exist:', file_in)
            return
        fin = open(file_in, "r")
        fout = open(file_out, "w")

        # add all lines from the file to the memory
        for ins_str in fin:
            ins_str = ins_str.strip()
            hex_str = Instruction.parseInsStr(ins_str) + '\n'
            fout.write(hex_str)

        # close file
        fin.close()
        fout.close()


class StageData:
    """
    Store intermediate results for a stage.
    Note that the data of a stage always store the final result at the end of a cycle.
    """

    def __init__(self):
        """
        Constructor
        """

        # the instruction processed in this stage
        self.ins = Instruction()
        self.ins.opcode = Opcode.NOOP

        # flag: if HALT is fetched
        self.is_halt_done_IF = False

        # flag: if HALT is processed in stage WB
        self.is_halt_done_WB = False

        # flag: determine if we should take the branch
        self.is_branch = False

        # flag: whether the output of this stage is ready to propagate or not
        self.is_output_ready = False

        # intermediate computation values
        self.alu_result = 0  # store computed ALU result used by stages MEM and WB
        self.alu_op_a = 0
        self.alu_op_b = 0

        # store:
        # - mem val to write to reg (LDW)
        # - reg val to write to mem (STW)
        # - ALU val to write to reg (R/I-type)
        self.data_to_write = 0

        # register destination
        self.reg_to_write = 0

        # destination: rd, mem or PC
        self.dst_type = WriteDst.NOT_SET

        # branch address in case we should take the branch
        self.branch_addr = 0

        # memory address used for LDW/STW
        self.mem_addr = 0

        # record indices of input registers to check for hazards
        self.input_indices = []


class PipelineStage:
    """
    Represent a stage in the pipeline, should not be used to create any specific stage,
    but use the other classes instead.
    """

    def __init__(self):
        """
        Constructor
        """

        # flag to enable/disable this stage (stall)
        self.is_stall = False

        # intermediate output results
        self.data = StageData()

    def checkStallAndPrepare(self):
        """
        Check if the stage is stalled and prepare necessary stuffs

        :return: True if stalled, False otherwise
        """

        if self.is_stall:
            return True

        return False

    def setStall(self, is_stall: bool):
        """
        Enable/Disable this stage

        :param is_stall:
        :return:
        """

        self.is_stall = is_stall

    def getStatus(self) -> str:
        """
        Return a string summarizing current status of the stage. Should be called after execute method to get up-to-date
        status.

        :return:
        """

        status = f'{self.data.ins.toString()}'
        if (self.data.ins.opcode != Opcode.NOOP and self.data.ins.opcode != Opcode.HALT) and (
                self.is_stall or not self.data.is_output_ready):
            status += ' [stall]'

        # if (not self.is_stall) and self.data.is_output_ready:
        #     return f'{self.data.ins.toString()}'
        # else:
        #     return 'stall'

        return status

    def isDoingNothing(self):
        """
        :return: True if NOOP or HALT or stall or output-is-not-ready
        """

        return self.data.ins.opcode == Opcode.HALT or self.data.ins.opcode == Opcode.NOOP or self.is_stall or not self.data.is_output_ready

    def flush(self):
        """
        Flush the current output out from the stage, replaced by a NOOP

        :return:
        """

        self.data.ins.opcode = Opcode.NOOP


class StageIF(PipelineStage):
    """
    Stage 01: Instruction Fetch
    """

    def execute(self, emu_data: 'EmuData'):
        """
        Execute this stage to emulate one step/cycle in the pipeline

        :return:
        """

        # check stall
        if self.checkStallAndPrepare():
            self.data.is_output_ready = False
            return

        # mark output ready by default
        self.data.is_output_ready = True

        # do nothing if HALT is encountered or stalled
        if self.data.is_halt_done_IF:
            self.data.ins.opcode = Opcode.NOOP
            return

        # fetch instruction from memory
        self.data.ins = emu_data.getInsAtPC()

        # determine to stop fetching
        self.data.is_halt_done_IF = (self.data.ins.opcode == Opcode.HALT)

        # ----------
        # Sneak peek at input/output registers
        # ----------
        # Note: even though we suppose to decode the instruction in stage ID, we take a "sneak peek" in the
        # registers in this stage in order to detect hazards easier later
        #
        self.data.input_indices.clear()
        self.data.input_indices.append(self.data.ins.rs)
        if self.data.ins.isRType() or self.data.ins.opcode == Opcode.BEQ or self.data.ins.opcode == Opcode.STW:
            self.data.input_indices.append(self.data.ins.rt)
        if self.data.ins.isRType():
            self.data.reg_to_write = self.data.ins.rd
        elif self.data.ins.isIType_RegWrite():
            self.data.reg_to_write = self.data.ins.rt
        else:
            self.data.reg_to_write = -1

    def getStatus(self) -> str:
        status = super().getStatus()
        if self.isDoingNothing():
            return status

        status += f' (' \
                  f'input={self.data.input_indices}' \
                  f', reg_to_write={self.data.reg_to_write}' \
                  f')'

        return status


class StageID(PipelineStage):
    """
    Stage 02: Instruction Decode
    """

    def execute(self, prev_stage: 'PipelineStage', emu_data: 'EmuData'):
        """
        Execute this stage to emulate one step/cycle in the pipeline

        :return:
        """

        # check stall
        if self.checkStallAndPrepare():
            self.data.is_output_ready = False
            return

        # propagate data from previous stage if output is ready, skip otherwise
        if not prev_stage.data.is_output_ready:
            self.data.is_output_ready = False
            return
        self.data = copy.deepcopy(prev_stage.data)
        self.data.is_output_ready = True

        # ignore if HALT/NOOP
        if self.isDoingNothing():
            return

        # ----------
        # Decode
        # ----------
        # - input registers, ALU operands
        # - destination registers
        # - branch address
        #

        # ALU operand a: is always reg[rs]
        self.data.alu_op_a = emu_data.getRegister(self.data.ins.rs)

        # ALU operand b: depend on R-type or I-type
        if self.data.ins.isRType():
            self.data.alu_op_b = emu_data.getRegister(self.data.ins.rt)
        else:
            # special cases for BEQ/BZ/STW
            if self.data.ins.opcode == Opcode.STW:
                # TODO: remember to check STW and forward to data_to_write
                self.data.data_to_write = emu_data.getRegister(self.data.ins.rt)
                self.data.alu_op_b = self.data.ins.imm
            elif self.data.ins.opcode == Opcode.BZ:
                self.data.alu_op_b = 0
            elif self.data.ins.opcode == Opcode.BEQ:
                self.data.alu_op_b = emu_data.getRegister(self.data.ins.rt)
            else:
                self.data.alu_op_b = self.data.ins.imm

        # determine dst type
        if self.data.ins.opcode == Opcode.STW:
            """
            STW:
                dst-to-write: a memory address, retrieved from the ALU result computed in stage EX
            """

            self.data.dst_type = WriteDst.MEM
        elif self.data.ins.isControlTransfer():
            """
            BEQ/BZ/JR:
                dst-to-write: is the PC
            """

            self.data.dst_type = WriteDst.PC
        else:
            """
            others:
                dst-to-write: a register index
            """

            self.data.dst_type = WriteDst.REG

    def getStatus(self):
        """
        Return a string summarizing current status of the stage

        :return:
        """

        status = super().getStatus()
        if self.isDoingNothing():
            return status

        status += f' (' \
                  f'alu_a={self.data.alu_op_a}, ' \
                  f'alu_b={self.data.alu_op_b}, ' \
                  f'dst_type={self.data.dst_type.name}, ' \
                  f'data_to_write={self.data.data_to_write}' \
                  f')'

        return status


class StageEX(PipelineStage):
    """
    Stage 03: Execution
    """

    def compute(self, opcode: Opcode, op_a: int, op_b: int) -> int:
        """
        Compute & return result based on the given opcode

        :param opcode: opcode to determine operation to perform on the two given operands
        :param op_a: the 1st operand
        :param op_b: the 2nd operand
        :return: the computed value
        """

        if opcode == Opcode.ADD or opcode == Opcode.ADDI or opcode == Opcode.LDW or opcode == Opcode.STW:
            return op_a + op_b

        if opcode == Opcode.SUB or opcode == Opcode.SUBI:
            return op_a - op_b

        if opcode == Opcode.MUL or opcode == Opcode.MULI:
            return op_a * op_b

        if opcode == Opcode.OR or opcode == Opcode.ORI:
            return op_a | op_b

        if opcode == Opcode.AND or opcode == Opcode.ANDI:
            return op_a & op_b

        if opcode == Opcode.XOR or opcode == Opcode.XORI:
            return op_a ^ op_b

        raise Exception(f'StageEX: unsupported ALU operation for opcode {opcode.name}')

    def execute(self, prev_stage: 'PipelineStage', emu_data: 'EmuData'):
        """
        Execute this stage to emulate one step/cycle in the pipeline

        :return:
        """

        # check stall
        if self.checkStallAndPrepare():
            self.data.is_output_ready = False
            return

        # propagate data from previous stage if output is ready, skip otherwise
        if not prev_stage.data.is_output_ready:
            self.data.is_output_ready = False
            return
        self.data = copy.deepcopy(prev_stage.data)
        self.data.is_output_ready = True

        # ignore if HALT/NOOP
        if self.isDoingNothing():
            return

        # handle BEQ/BZ/JR: should we branch? branch-addr?
        self.data.is_branch = False
        if self.data.ins.isControlTransfer():
            if self.data.ins.opcode == Opcode.JR:
                self.data.is_branch = True
                # TODO: double check the branch-addr: use as is OR divide by 4 ???.
                #  Use /4 for now. Since "as is" does not make sense for the example "sample_memory_image"
                self.data.branch_addr = int(self.data.alu_op_a / 4)
            else:
                self.data.alu_result = self.data.alu_op_a - self.data.alu_op_b
                self.data.is_branch = (self.data.alu_result == 0)
                self.data.branch_addr = self.data.ins.imm + emu_data.pc - 2
            return

        # perform ALU operation based on the opcode
        self.data.alu_result = self.compute(self.data.ins.opcode, self.data.alu_op_a, self.data.alu_op_b)

        # prepare data-to-write
        if self.data.ins.opcode != Opcode.STW:
            self.data.data_to_write = self.data.alu_result

        # handle LDW/STW: prepare memory address
        if self.data.ins.isMemoryAccess():
            self.data.mem_addr = int(self.data.alu_result / 4)

    def getStatus(self) -> str:
        status = super().getStatus()
        if self.isDoingNothing():
            return status

        status += f' (' \
                  f'alu_result={self.data.alu_result}, ' \
                  f'data_to_write={self.data.data_to_write}, ' \
                  f'is_branch={self.data.is_branch}, ' \
                  f'branch_addr={self.data.branch_addr}, ' \
                  f'mem_addr={self.data.mem_addr}' \
                  f')'

        return status


class StageMEM(PipelineStage):
    """
    Stage 04: Memory Access
    """

    def execute(self, prev_stage: 'PipelineStage', emu_data: 'EmuData'):
        """
        Execute this stage to emulate one step/cycle in the pipeline

        :return:
        """

        # check stall
        if self.checkStallAndPrepare():
            self.data.is_output_ready = False
            return

        # propagate data from previous stage if output is ready, skip otherwise
        if not prev_stage.data.is_output_ready:
            self.data.is_output_ready = False
            return
        self.data = copy.deepcopy(prev_stage.data)
        self.data.is_output_ready = True

        # ignore if HALT/NOOP
        if self.isDoingNothing():
            return

        # handle LDW
        if self.data.ins.opcode == Opcode.LDW:
            self.data.data_to_write = emu_data.getMemInt(self.data.mem_addr)
            return

        # handle STW
        if self.data.ins.opcode == Opcode.STW:
            emu_data.setMemInt(self.data.mem_addr, self.data.data_to_write)
            return

    def getStatus(self) -> str:

        status = super().getStatus()
        if self.isDoingNothing():
            return status

        if self.data.ins.opcode == Opcode.LDW:
            status += f' (data_to_write = mem[{self.data.mem_addr}] = {self.data.data_to_write})'
        elif self.data.ins.opcode == Opcode.STW:
            status += f' (mem[{self.data.mem_addr}] = {self.data.data_to_write})'

        return status


class StageWB(PipelineStage):
    """
    Stage 05: Write Back
    """

    def execute(self, prev_stage: 'PipelineStage', emu_data: 'EmuData'):
        """
        Execute this stage to emulate one step/cycle in the pipeline

        :return:
        """

        # check stall
        if self.checkStallAndPrepare():
            self.data.is_output_ready = False
            return

        # propagate data from previous stage if output is ready, skip otherwise
        if not prev_stage.data.is_output_ready:
            self.data.is_output_ready = False
            return
        self.data = copy.deepcopy(prev_stage.data)
        self.data.is_output_ready = True

        # determine if HALT is processed
        self.data.is_halt_done_WB = self.data.ins.opcode == Opcode.HALT

        # ignore if HALT/NOOP
        if self.isDoingNothing():
            return

        # write back to registers
        if self.data.dst_type == WriteDst.REG:
            emu_data.setRegister(self.data.reg_to_write, self.data.data_to_write)

    def getStatus(self) -> str:

        status = super().getStatus()
        if self.isDoingNothing():
            return status

        if self.data.dst_type == WriteDst.REG:
            status += f' (R{self.data.reg_to_write} = {self.data.data_to_write})'

        return status


class EmuData:
    """
    Store data necessary for emulation: registers and memory
    """

    def __init__(self):
        """
        Constructor
        """

        # memory image: contain 1024 lines
        self.mem = []

        # registers
        self.regs = []

        # program counter
        self.pc = 0

        # reset all
        self.reset()

    def clone(self) -> 'EmuData':
        """
        Clone this instance and return
        """

        cloned = EmuData()
        cloned.mem = self.mem.copy()
        cloned.regs = self.regs.copy()
        cloned.pc = self.pc
        return cloned

    def resetRegisters(self):
        """
        Reset all registers/PC to zero

        :return:
        """

        # set all registers to 0
        self.regs.clear()
        for i in range(MAX_REGS):
            self.regs.append(0)

        # set PC to 0
        self.pc = 0

    def reset(self):
        """
        Reset and clear all data

        :return:
        """

        self.resetRegisters()

        # clear memory
        self.mem.clear()

    def loadFromFile(self, file_path: str):
        """
        Load memory image from the given file path

        :param file_path: the file to read the memory image
        :return:
        """

        # check & open file
        if not os.path.isfile(file_path):
            debugPrint('File not exist:', file_path)
            return
        f = open(file_path, "r")

        # add all lines from the file to the memory
        for a_line in f:
            a_line = a_line.strip()
            if len(a_line) > 0:
                self.mem.append(a_line)

        # close file
        f.close()

    def getInsStr(self):
        """
        Return a string of all instructions in the memory.

        :return: a multi-line-string of instructions
        """

        # concatenate all instruction strings
        result = ''
        for ins in self.getIns():
            result += str(ins) + '\n'
            if ins.opcode == Opcode.HALT:
                break

        return result

    def getIns(self) -> []:
        """
        Return a list of all parse-able instructions in the memory.
        Assuming the CODE segment start at line 0 in the memory
        and stop at a HALT instruction.

        :return:
        """

        # parse instruction until HALT
        result = []
        for line in self.mem:
            ins = Instruction.parse(line)
            result.append(ins)
            if ins.opcode == Opcode.HALT:
                break

        return result

    def getInsAtPC(self) -> Instruction:
        """
        Return the Instruction at the PC

        :return:
        """

        # check bound
        if self.pc >= len(self.mem) or self.pc < 0:
            return Instruction()

        ins_str = self.mem[self.pc]
        return Instruction.parse(ins_str)

    def getRegister(self, reg_id):
        """
        Return register value at the desired index
        :return:
        """

        # check
        if reg_id < 0 or reg_id >= MAX_REGS:
            return 0

        return self.regs[reg_id]

    def setRegister(self, reg_id, reg_val):
        """
        Set value to the desired register
        :return:
        """

        # check
        if reg_id < 0 or reg_id >= MAX_REGS:
            return

        self.regs[reg_id] = reg_val

    def getMemInt(self, mem_addr) -> int:
        """
        Get memory content at the desired address as a signed integer

        :param mem_addr:
        :return:
        """

        # get memory string
        mem_str = self.getMemStr(mem_addr)

        # convert to signed integer
        bin_str = convertHex2Bin(mem_str)
        signed_int = convertBin2SignedInt(bin_str, LEN_INS)

        return signed_int

    def setMemInt(self, mem_addr, signed_int: int):
        """
        Set the given signed integer to the desired address

        :param signed_int: the content to overwrite
        :param mem_addr: the desired memory address
        :return:
        """

        # check
        if mem_addr < 0 or mem_addr >= len(self.mem):
            raise Exception('Memory out-of-bound: ' + str(mem_addr))

        # convert to hex string & store
        hex_str = convertSignedInt2Hex(signed_int)
        self.mem[mem_addr] = hex_str

    def getMemStr(self, mem_addr) -> str:
        """
        Get memory content at the desired address as a string

        :param mem_addr:
        :return:
        """

        # check
        if mem_addr < 0 or mem_addr >= len(self.mem):
            raise Exception('Memory out-of-bound: ' + str(mem_addr))

        return self.mem[mem_addr]

    def getMemLen(self):
        """
        Get length of memory, should not exceed 1024

        :return:
        """

        return len(self.mem)


class Emulator:
    """
    Emulate a Lite MIPS processor. Support: pipeline & data forwarding
    """

    def __init__(self):
        """
        Constructor
        """

        # store input memory image
        self.mem_in = EmuData()

        # store output memory image, after emulation
        self.mem_out = EmuData()

        # flag for data forwarding
        self.is_forwarding = False

        # flag to set if end of instructions (HALT is encountered and processed)
        self.is_halted = False

        # count the number of cycles spent in the emulation
        self.count_cycles = 0

        # count executed instructions
        self.count_ins = 0

        # count stalls
        self.count_stalls = 0

        # count instructions: arithmetic, logical, mem-access, control
        self.count_ins_ari = 0
        self.count_ins_log = 0
        self.count_ins_mem = 0
        self.count_ins_con = 0

        # pipeline intermediate data
        self.stage_IF = StageIF()
        self.stage_ID = StageID()
        self.stage_EX = StageEX()
        self.stage_MEM = StageMEM()
        self.stage_WB = StageWB()

    def loadFromFile(self, file_path: str):
        """
        Read memory image into this emulator

        :param file_path: the file to read the memory image
        """
        self.mem_in.loadFromFile(file_path)

    def getInsStr(self):
        """
        Return a string of all instructions in the memory files.

        :return: a multi-line-string of instructions
        """

        return self.mem_in.getInsStr()

    def setForwardingEnabled(self, forwarding_enable: bool):
        """
        Set whether data-forwarding will be enabled when dealing with hazard

        :param forwarding_enable:
        :return:
        """

        self.is_forwarding = forwarding_enable

    def forward(self, src_data: StageData, dst_data: StageData):
        """
        Forward data from source stage to destination stage, based on the involved instructions

        :param src_data:
        :param dst_data:
        :return:
        """

        debug_str = ''

        # determine value to forward
        debug_str += '  forward_val='
        if src_data.ins.opcode != Opcode.LDW:
            forward_val = src_data.alu_result
            debug_str += 'alu_result'
        else:
            forward_val = src_data.data_to_write
            debug_str += 'data_to_write'
        debug_str += f'={forward_val}'

        # forward to alu-op-a
        if len(dst_data.input_indices) > 0 and dst_data.input_indices[0] == src_data.reg_to_write:
            dst_data.alu_op_a = forward_val
            debug_str += ' --> alu_op_a'

        # forward to alu-op-b/data-to-write
        if len(dst_data.input_indices) > 1 and dst_data.input_indices[1] == src_data.reg_to_write:
            if dst_data.ins.isRType() or dst_data.ins.opcode == Opcode.BEQ:
                dst_data.alu_op_b = forward_val
                debug_str += ' --> alu_op_b'
            elif dst_data.ins.opcode == Opcode.STW:
                dst_data.data_to_write = forward_val
                debug_str += ' --> data_to_write'

        debugPrint(debug_str)

    def execute_step(self):
        """
        Run the emulator for one step/cycle

        :return:
        """

        # do nothing if end of code is reached
        if self.is_halted:
            return

        """
        1/ BEGIN OF CYCLE:
            - emulate the pipeline with 5 stages
        """

        # Stage 5: WB
        self.stage_WB.execute(self.stage_MEM, self.mem_out)
        debugPrint('WB  --> ', self.stage_WB.getStatus(), sep='')

        # Stage 4: MEM
        self.stage_MEM.execute(self.stage_EX, self.mem_out)
        debugPrint('MEM --> ', self.stage_MEM.getStatus(), sep='')

        # Stage 3: EX
        self.stage_EX.execute(self.stage_ID, self.mem_out)
        debugPrint('EX  --> ', self.stage_EX.getStatus(), sep='')

        # Stage 2: ID
        self.stage_ID.execute(self.stage_IF, self.mem_out)
        debugPrint('ID  --> ', self.stage_ID.getStatus(), sep='')

        # Stage 1: IF
        self.stage_IF.execute(self.mem_out)
        debugPrint('IF  --> ', self.stage_IF.getStatus(), sep='')

        """
        2/ END OF CYCLE:
            - recover/flush incorrect fetched instruction
            - PC
            - check for hazards & stall (by disabling appropriate stages)
            - check to forward (if stall)
        """

        # determine if the emulation should be halted
        self.is_halted = self.stage_WB.data.is_halt_done_WB

        # handle PC (sequential or branch) and hazards
        debugPrint(f'curr PC = {self.mem_out.pc}')
        suffix = ''
        if self.stage_EX.data.is_branch:
            """
            [Branch]
                Ignore hazards if we branch, since IF and ID are flushed
            """

            self.mem_out.pc = self.stage_EX.data.branch_addr
            suffix = '(branch, flush IF & ID)'

            # flush 2 incorrect instructions in IF and ID stages
            self.stage_IF.flush()
            self.stage_IF.setStall(False)
            self.stage_ID.flush()
            self.stage_ID.setStall(False)

            # undone IF.is_halt_done_IF
            if self.stage_IF.data.is_halt_done_IF:
                self.stage_IF.data.is_halt_done_IF = False
        else:
            """
            [Sequential]
                Handle hazards by forwarding/stalling
            """

            # detect & handle hazards
            self.handleHazards()

            # do not increase PC if IF is stalled
            if self.stage_IF.is_stall or self.stage_IF.data.is_halt_done_IF:
                suffix = '(stall)'
            else:
                self.mem_out.pc += 1
        debugPrint(f'next PC = {self.mem_out.pc} {suffix}')

        # count cycles
        self.count_cycles += 1

        # count executed instructions & ins frequency: must be output-ready & including HALT
        if (
                (self.stage_WB.data.ins.opcode == Opcode.HALT or not self.stage_WB.data.ins.isNoop())
                and self.stage_WB.data.is_output_ready
        ):
            self.count_ins += 1
            if self.stage_WB.data.ins.isArithmetic():
                self.count_ins_ari += 1
            if self.stage_WB.data.ins.isLogical():
                self.count_ins_log += 1
            if self.stage_WB.data.ins.isMemoryAccess():
                self.count_ins_mem += 1
            if self.stage_WB.data.ins.isControlTransfer():
                self.count_ins_con += 1

        # print a new line to separate between cycles
        debugPrint()

    def handleHazards(self):
        """
        Detect & handle hazards by forwarding or stall, depending on current settings
        """

        # check for hazards to enable/disable stages for next cycle
        if self.is_forwarding:
            """
            [Cases if forwarding]

                    Producer        Consumer        Notes
            1/      LDW             any             stall 1 cycle, forward from MEM, forward  data_to_write instead of alu_result
            2/      Logic/Arith     any             no stall, forward from EX/MEM
            3/      any             STW             forward to: data_to_write, alu_op_a
            """

            # enable IF/ID/EX in case they are stalled
            self.stage_IF.setStall(False)
            self.stage_IF.data.is_output_ready = True
            self.stage_ID.setStall(False)
            self.stage_ID.data.is_output_ready = True
            self.stage_EX.setStall(False)

            if not self.stage_ID.data.ins.isNoop():

                # stall: case 1
                if (
                        self.stage_EX.data.is_output_ready
                        and (self.stage_EX.data.reg_to_write in self.stage_ID.data.input_indices)
                        and self.stage_EX.data.ins.opcode == Opcode.LDW
                ):
                    self.stage_IF.setStall(True)
                    self.stage_ID.setStall(True)
                    self.stage_EX.setStall(True)
                    self.count_stalls += 1
                # else:

                # forward: case 1, 2, 3
                if (
                        self.stage_EX.data.is_output_ready
                        and (self.stage_EX.data.reg_to_write in self.stage_ID.data.input_indices)
                        and (self.stage_EX.data.ins.isLogical() or self.stage_EX.data.ins.isArithmetic())
                ):
                    debugPrint(f'  forward EX to to next EX')
                    self.forward(self.stage_EX.data, self.stage_ID.data)
                if (
                        self.stage_MEM.data.is_output_ready
                        and (self.stage_MEM.data.reg_to_write in self.stage_ID.data.input_indices)
                        and
                        (
                                self.stage_MEM.data.ins.isLogical()
                                or self.stage_MEM.data.ins.isArithmetic()
                                or self.stage_MEM.data.ins.opcode == Opcode.LDW
                        )
                ):
                    debugPrint(f'  forward MEM to to next EX')
                    self.forward(self.stage_MEM.data, self.stage_ID.data)

        else:

            """
            [Cases if NOT forwarding]
            """

            if (
                    (not self.stage_IF.data.ins.isNoop())
                    and (
                    (
                            (self.stage_ID.data.reg_to_write in self.stage_IF.data.input_indices)
                            and self.stage_ID.data.is_output_ready
                            and not self.stage_ID.data.ins.isNoop()
                    )
                    or
                    (
                            (self.stage_EX.data.reg_to_write in self.stage_IF.data.input_indices)
                            and self.stage_EX.data.is_output_ready
                            and not self.stage_EX.data.ins.isNoop()
                    )
            )
            ):
                self.stage_IF.setStall(True)
                self.stage_ID.setStall(True)
                self.count_stalls += 1
            else:
                self.stage_IF.setStall(False)
                self.stage_IF.data.is_output_ready = True
                self.stage_ID.setStall(False)

    def reset(self):
        """
        Reset this emulator before re-run the emulation

        :return:
        """

        self.is_halted = False
        self.count_cycles = 0
        self.count_ins = 0
        self.count_stalls = 0
        self.stage_IF = StageIF()
        self.stage_ID = StageID()
        self.stage_EX = StageEX()
        self.stage_MEM = StageMEM()
        self.stage_WB = StageWB()

    def getReportString(self):
        """
        Return a report string for the emulation
        :return:
        """

        report = ''

        # cycles
        report += f'Total cycles = {self.count_cycles}\n'

        # stalls
        report += f'Total stalls = {self.count_stalls}\n'

        # count executed instructions & frequency
        report += f'Total executed instructions = {self.count_ins}\n'
        report += f'  Arithmetic      : {self.count_ins_ari}\n'
        report += f'  Logical         : {self.count_ins_log}\n'
        report += f'  Memory access   : {self.count_ins_mem}\n'
        report += f'  Control transfer: {self.count_ins_con}\n'

        # register
        report += '\nRegisters:\n'
        for i in range(MAX_REGS):
            reg_in = self.mem_in.getRegister(i)
            reg_out = self.mem_out.getRegister(i)
            if reg_in != reg_out:
                report += f'  R[{i}] = {reg_out}\n'

        # memory
        report += '\nMemory:\n'
        for i in range(self.mem_in.getMemLen()):
            val_in = self.mem_in.getMemInt(i)
            val_out = self.mem_out.getMemInt(i)
            if val_in != val_out:
                report += f'  [{i * 4}] = {val_out}\n'

        return report

    def execute(self):
        """
        Run the emulator by performing operations corresponding to the instructions in the loaded memory

        :return:
        """

        # skip if empty memory
        if self.mem_in.getMemLen() == 0:
            debugPrint('Memory is empty. Skipped.')
            return

        # do nothing if end of code is reached
        if self.is_halted:
            return

        # prepare output memory
        self.mem_out = self.mem_in.clone()
        self.mem_out.resetRegisters()

        # run emulation until 'HALT'
        while not self.is_halted:
            self.execute_step()

