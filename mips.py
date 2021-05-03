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

    # ----------
    # Elements of an instruction
    # ----------

    opcode = Opcode.UNKNOWN
    rs = 0
    rt = 0
    rd = 0
    imm = 0

    # Original string of hex values
    hex_str = ''

    # ----------
    # Methods
    # ----------

    def __init__(self):
        """
        Constructor.
        NOT recommended to create an obj directly using this instructor.
        Should use the parse function of this class.
        """

    def __str__(self):
        """
        String representation of this instruction
        :return: a description of this instruction
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

        return result

    def isRType(self):
        """
        Check if this instruction is of R-type
        :return: True if R-type, False otherwise
        """

        return self.opcode in [Opcode.AND, Opcode.OR, Opcode.XOR, Opcode.ADD, Opcode.SUB, Opcode.MUL]

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


class Emulator:
    """
    Emulate a Lite MIPS processor. Support: pipeline & data forwarding
    """

    # memory image: contain 1024 lines
    mem = []

    def __init__(self):
        """
        Constructor
        """

    def loadFromFile(self, file_path: str):
        """
        Load memory image in the given file path
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
            self.mem.append(a_line)

        # close file
        f.close()

    def getInsStr(self):
        """
        Return a string of all instructions in the memory files.
        Assuming the CODE segment start at line 0 in the memory
        and stop at a HALT instruction.

        :return: a multi-line-string of instructions
        """

        # parse instruction until HALT
        result = ''
        for line in self.mem:
            ins = Instruction.parse(line)
            result += str(ins) + '\n'
            if ins.opcode == Opcode.HALT:
                break

        return result
