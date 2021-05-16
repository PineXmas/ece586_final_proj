"""
ECE586 - Final Project
Team: Khoi, Melinda, Nguyen, Thong

Provide all tools/API for working with Lite MIPS
"""

from enum import Enum, auto
import os.path
import copy

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

# mask for 32-bit integer
INT_MASK_32 = 0xFFFFFFFF

# max length for a hex string in the memory image
MAX_HEX_LENGTH = 8

# max length in bit of an instruction
LEN_INS = MAX_HEX_LENGTH * 4

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


def convertSignedInt2Hex(signed_int: int) -> str:
    """
    Convert the given signed integer into its corresponding hex string

    :param signed_int: the signed integer
    :return:
    """

    hex_str = hex(signed_int & INT_MASK_32)
    hex_str = hex_str[2:].zfill(MAX_HEX_LENGTH)
    return hex_str

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
    list_control_types = [Opcode.BZ, Opcode.BEQ, Opcode.JR]
    list_mem_types = [Opcode.LDW, Opcode.STW]

    # register symbol
    R_str = '$'

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

    # def clone(self) -> 'Instruction':
    #     """
    #     Clone this object & return (deep copy)
    #     :return:
    #     """
    #
    #     cloned = Instruction()
    #     cloned.opcode = self.

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

        return (self.opcode in self.list_logic_types) or (self.opcode in self.list_arithmetic_types) or self.opcode == Opcode.LDW

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

            # imm
            ins.imm = convertBin2SignedInt(bin_str[OFFSET_IMM:], LEN_IMM)
            # ins.imm = int(bin_str[OFFSET_IMM:], 2)
            # if bin_str[OFFSET_IMM] == '1':
            #     ins.imm = ins.imm - 2 ** LEN_IMM

            # record the hex string
            ins.hex_str = hex_str
        except Exception as e:
            # print('Error while parsing instruction: [', hex_str, '], error=', e, sep='')
            ins.opcode = Opcode.UNKNOWN

        return ins


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

        # intermediate computation values
        self.alu_result = 0             # store computed ALU result used by stages MEM and WB
        self.alu_op_a = 0
        self.alu_op_b = 0

        # store:
        # - mem val to write to reg (LDW)
        # - reg val to write to mem (STW)
        # - ALU val to write to reg (R/I-type)
        self.data_to_write = 0

        # store destination in terms of: mem addr OR reg index
        self.dst_to_write = 0

        # destination: rd, mem or PC
        self.dst_type = WriteDst.NOT_SET

        # branch address in case we should take the branch
        self.branch_addr = 0

        # memory address used for LDW/STW
        self.mem_addr = 0

        # record indices of input registers to check for hazards
        self.input_indices = set()


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

        # intermediate results
        self.data = StageData()

    def checkStallAndSetNOOP(self):
        """
        Check if the stage is stalled and set instruction to NOOP for doing nothing
        :return: True if stalled, False otherwise
        """

        if self.is_stall:
            self.data.ins = Instruction()
            self.data.ins.opcode = Opcode.NOOP
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
        Return a string summarizing current status of the stage

        :return:
        """

        return f'{self.data.ins.toString()}'

    def isDoingNothing(self):
        """
        :return: True if NOOP or HALT
        """

        return self.data.ins.opcode == Opcode.HALT or self.data.ins.opcode == Opcode.NOOP


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
        if self.checkStallAndSetNOOP():
            return

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
        self.data.input_indices.add(self.data.ins.rs)
        if self.data.ins.isRType() or self.data.ins.opcode == Opcode.BEQ or self.data.ins.opcode == Opcode.STW:
            self.data.input_indices.add(self.data.ins.rt)
        if self.data.ins.isRType():
            self.data.dst_to_write = self.data.ins.rd
        elif self.data.ins.isIType_RegWrite():
            self.data.dst_to_write = self.data.ins.rt
        else:
            self.data.dst_to_write = -1

        # TODO: PC + 1 should be placed in stage IF

    def getStatus(self) -> str:
        status = super().getStatus()
        if self.isDoingNothing():
            return status

        status += f' ('\
                  f'input={self.data.input_indices}' \
                  f', dst_to_write={self.data.dst_to_write}' \
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
        if self.checkStallAndSetNOOP():
            return

        # take data from previous stage
        self.data = copy.deepcopy(prev_stage.data)

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
                self.data.data_to_write = emu_data.getRegister(self.data.ins.rt)
            elif self.data.ins.opcode == Opcode.BZ:
                self.data.alu_op_b = 0
                self.data.branch_addr = self.data.ins.imm
            elif self.data.ins.opcode == Opcode.BEQ:
                self.data.alu_op_b = emu_data.getRegister(self.data.ins.rt)
                self.data.branch_addr = self.data.ins.imm
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

        status += f' ('\
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

    def execute(self, prev_stage: 'PipelineStage'):
        """
        Execute this stage to emulate one step/cycle in the pipeline

        :return:
        """

        # check stall
        if self.checkStallAndSetNOOP():
            return

        # take data from previous stage
        self.data = copy.deepcopy(prev_stage.data)

        # ignore if HALT/NOOP
        if self.isDoingNothing():
            return

        # handle BEQ/BZ/JR: should we branch? branch-addr?
        self.data.is_branch = False
        if self.data.ins.isControlTransfer():
            if self.data.ins.opcode == Opcode.JR:
                self.data.is_branch = True
                self.data.branch_addr = self.data.alu_op_a
            else:
                self.data.alu_result = self.data.alu_op_a - self.data.alu_op_b
                self.data.is_branch = (self.data.alu_result == 0)
                self.data.branch_addr = self.data.ins.imm
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
                  f'is_branch={self.data.is_branch}, ' \
                  f'branch_addr={self.data.branch_addr}' \
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
        if self.checkStallAndSetNOOP():
            return

        # take data from previous stage
        self.data = copy.deepcopy(prev_stage.data)

        # ignore if HALT/NOOP
        if self.isDoingNothing():
            return

        # # handle LDW
        # if self.data.ins.opcode == Opcode.LDW:
        #     self.data.data_to_write = emu_data.getMemInt(self.data.mem_addr)
        #     return


class StageWB(PipelineStage):
    """
    Stage 05: Write Back
    """

    def execute(self, prev_stage: 'PipelineStage'):
        """
        Execute this stage to emulate one step/cycle in the pipeline

        :return:
        """

        # check stall
        if self.checkStallAndSetNOOP():
            return

        # take data from previous stage
        self.data = copy.deepcopy(prev_stage.data)

        # determine if HALT is processed
        self.data.is_halt_done_WB = self.data.ins.opcode == Opcode.HALT

        # ignore if HALT/NOOP
        if self.isDoingNothing():
            return


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

        # flag to set if end of instructions (HALT is encountered and processed)
        self.is_halted = False

        # count the number of cycles spent in the emulation
        self.count_cycles = 0

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

        self.forwarding_enabled = forwarding_enable

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
            - check for hazards & stall (by disabling appropriate stages)
            - check to forward (if stall)
        """

        # check for hazards to enable/disable stages

        """
        2/ DURING CYCLE:
            - emulate the pipeline with 5 stages
        """

        # Stage 5: WB
        self.stage_WB.execute(self.stage_MEM)
        print('WB  --> ', self.stage_WB.getStatus(), sep='')

        # Stage 4: MEM
        self.stage_MEM.execute(self.stage_EX, self.mem_out)
        print('MEM --> ', self.stage_MEM.getStatus(), sep='')

        # Stage 3: EX
        self.stage_EX.execute(self.stage_ID)
        print('EX  --> ', self.stage_EX.getStatus(), sep='')

        # Stage 2: ID
        self.stage_ID.execute(self.stage_IF, self.mem_out)
        print('ID  --> ', self.stage_ID.getStatus(), sep='')

        # Stage 1: IF
        self.stage_IF.execute(self.mem_out)
        print('IF  --> ', self.stage_IF.getStatus(), sep='')

        # determine next PC
        self.mem_out.pc += 1

        # determine if the emulation should be halted
        self.is_halted = self.stage_WB.data.is_halt_done_WB

        # count cycles
        self.count_cycles += 1

        """
        3/ END OF CYCLE:
            - recover/flush incorrect fetched instruction
        """

        print()

    def reset(self):
        """
        Reset this emulator before re-run the emulation

        :return:
        """

        self.is_halted = False
        self.count_cycles = 0
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
        report += 'Total cycles = ' + str(self.count_cycles)
        return report

    def execute(self):
        """
        Run the emulator by performing operations corresponding to the instructions in the loaded memory

        :return:
        """

        # do nothing if end of code is reached
        if self.is_halted:
            return

        # prepare output memory
        self.mem_out = self.mem_in.clone()
        self.mem_out.resetRegisters()

        # run emulation until 'HALT'
        while not self.is_halted:
            self.execute_step()
