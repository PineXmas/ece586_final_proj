"""
ECE586 - Final Project
Team: Khoi, Melinda, Nguyen, Thong

Provide all tools/API for working with Lite MIPS
"""

from enum import Enum, auto


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
    Represent an instruction in MIPS
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

    @staticmethod
    def parse(hex_str: str):
        """
        Parse a given string of hex values to an lite-MIPS instruction

        :param hex_str: the string containing hex values
        :return: NULL if failed, the Instruction obj if success
        """
        ins = Instruction()

        return ins


class Parser:
    """
    This class provides methods to parse MIPS codes in hex to instructions
    """

    def __init__(self):
        print('A Parser obj is created')

    def parse(self, s: str):
        print('Input string = ', s)
