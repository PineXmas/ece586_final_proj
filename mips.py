"""
ECE586 - Final Project
Team: Khoi, Melinda, Nguyen, Thong

Provide all tools/API for working with Lite MIPS
"""

from enum import Enum, auto
import os.path

# -------------------- DEFINES --------------------

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


# -------------------- FUNCS --------------------


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
    UNKNOWN = 0b111111


class Instruction:
    """
    Represent an instruction in Lite MIPS architecture
    """

    # set of instruction types
    list_R_types = [Opcode.AND, Opcode.OR, Opcode.XOR, Opcode.ADD, Opcode.SUB, Opcode.MUL]
    list_arithmetic_types = [Opcode.ADD, Opcode.ADDI, Opcode.SUB, Opcode.SUBI, Opcode.MUL, Opcode.MULI]
    list_logic_types = [Opcode.AND, Opcode.ANDI, Opcode.OR, Opcode.ORI, Opcode.XOR, Opcode.XORI]
    list_control_types = [Opcode.BZ, Opcode.BEQ, Opcode.JR]
    list_mem_types = [Opcode.LDW, Opcode.STW]

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

        # unknown, halt
        if self.opcode in [Opcode.UNKNOWN, Opcode.HALT]:
            return result

        # add values based on the opcode
        result += ' '
        if self.isRType():
            result += f'R{self.rd}, R{self.rs}, R{self.rt}'
        elif self.opcode == Opcode.BZ:
            result += f'R{self.rs}, {self.imm}'
        elif self.opcode == Opcode.JR:
            result += f'R{self.rs}'
        else:
            result += f'R{self.rt}, R{self.rs}, {self.imm}'

        # add type
        if add_type:
            result += f' ({self.getType()})'

        return result

    @staticmethod
    def parse(hex_str: str):
        """
        Parse a given string of hex values to an lite-MIPS instruction

        :param hex_str: the string containing hex values
        :return: NULL if failed, the Instruction obj if success
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

            # imm & handle 2-complement
            ins.imm = int(bin_str[OFFSET_IMM:], 2)
            if bin_str[OFFSET_IMM] == '1':
                ins.imm = ins.imm - 2 ** LEN_IMM

            # record the hex string
            ins.hex_str = hex_str
        except Exception as e:
            # print('Error while parsing instruction: [', hex_str, '], error=', e, sep='')
            ins.opcode = Opcode.UNKNOWN

        return ins


class PipelineStage:
    """
    Represent one stage in the pipeline, capable of emulating any of the 5 stages
    """

    def __init__(self):
        """
        Constructor
        """

        # the instruction processed in this stage
        self.ins = Instruction()

        # flag to enable/disable this stage (stall)
        self.is_stall = False

    def executeIF(self, emu_data: 'EmuData'):
        """
        Emulate stage IF

        :return:
        """

        # fetch instruction from memory
        self.ins = emu_data.getInsAtPC()

    def executeID(self, prev_stage: 'PipelineStage'):
        """
        Emulate stage ID

        :return:
        """

    def executeEX(self, prev_stage: 'PipelineStage'):
        """
        Emulate stage EX

        :return:
        """

    def executeMEM(self, prev_stage: 'PipelineStage'):
        """
        Emulate stage MEM

        :return:
        """

    def executeWB(self, prev_stage: 'PipelineStage'):
        """
        Emulate stage WB

        :return:
        """


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
            print('File not exist:', file_path)
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

        ins_str = self.mem[self.pc]
        return Instruction.parse(ins_str)


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
        self.forwarding_enabled = False

        # pipeline intermediate data
        self.stage_IF = PipelineStage()
        self.stage_ID = PipelineStage()
        self.stage_EX = PipelineStage()
        self.stage_MEM = PipelineStage()
        self.stage_WB = PipelineStage()

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

        self.forwarding_enabled = forwarding_enable

    def execute(self):
        """
        Run the emulator by performing operations corresponding to the instructions in the loaded memory

        :return:
        """

        # prepare output memory
        self.mem_out = self.mem_in.clone()
        self.mem_out.resetRegisters()

        # run emulation until 'HALT'
        while True:

            # Stage 5: WB

            # Stage 4: MEM

            # Stage 3: EX

            # Stage 2: ID

            # Stage 1: IF
            self.stage_IF.executeIF(self.mem_out)
            print('IF --> ', self.stage_IF.ins.toString(), sep='')
