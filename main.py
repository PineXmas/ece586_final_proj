# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import mips


def testInstruction():
    print('32510010=', mips.Instruction.parse('32510010'))
    print('3251FFFE=', mips.Instruction.parse('3251FFFE'))


def testEmuData():
    # load memory image
    file_path = 'data/sample_memory_image.txt'
    d1 = mips.EmuData()
    d1.loadFromFile(file_path)
    print('EmuData memory loaded')

    # print all instructions
    list_ins = d1.getIns()
    print('Instructions:')
    for ins in list_ins:
        print(ins.toString(add_type=True))

    # set/get mem
    print('Test set/get memory:')
    mem_val = -32767
    mem_addr = 0
    mem_val_read = d1.getMemInt(mem_addr)
    mem_str_read = d1.getMemStr(mem_addr)
    print('[Before] Get memory at address', mem_addr, ':', mem_val_read, '(str=', mem_str_read, ')')
    d1.setMemInt(mem_addr, -32767)
    print('Set', mem_val, 'address', mem_addr)
    mem_val_read = d1.getMemInt(mem_addr)
    print('[After ] Get memory at address', mem_addr, ':', mem_val_read)


def testConvertSignedInt2Bin():
    num = -32767
    max_num = 32766
    bit_length = 16
    while num <= max_num:
        num_from_hex = mips.convertBin2SignedInt(mips.convertHex2Bin(mips.convertSignedInt2Hex(num)), bit_length)
        num_from_bin = mips.convertBin2SignedInt(mips.convertSignedInt2Bin(num, bit_length), bit_length)
        if num_from_bin != num_from_hex:
            print(f'Error: num={num}, num_from_hex={num_from_hex}, num_from_bin={num_from_bin}')
            return
        num += 1

    print('PASS OK')


def testEmulator():
    # load memory image
    emu = mips.Emulator()
    file_path = "data/sample_memory_image.txt"
    # file_path = "data/test_03.txt"
    # file_path = "data/extra_ex_07.txt"
    # file_path = "data/test_branch_02.txt"
    emu.loadFromFile(file_path)
    print('Number of mem lines =', len(emu.mem_in.mem))

    # display instructions
    print(emu.getInsStr())

    # execute
    emu.execute()
    print(emu.getReportString())


def testParseInsStr():

    list_tests = [
        ('add r1, r2, r9', '00490800'),
        ('LDW R3, r30, 10', '33C3000A'),
        ('SUB R5, R1, R3', '08232800'),
        ('ADDI R17, R18, -32767', '06518001'),
        ('BEQ R17, R18, 16', '3E510010'),
        ('BEQ R17, R18, -2', '3E51FFFE'),
        ('BZ R6, 2', '38C00002'),
        ('JR R12', '41800000'),
        ('halt', '44000000')
    ]

    for test in list_tests:
        ins_str = test[0]
        expected = test[1]
        hex_str = mips.Instruction.parseInsStr(ins_str)
        if hex_str != expected:
            print('ERROR:')
            print(f'parsed  ={hex_str}')
            print(f'expected={expected}')
            print()


def testParseInsFile():

    file_in = 'data/test_branch_02_output.txt'
    file_out = file_in.replace('_output', '')
    mips.Instruction.parseInsFile(file_in, file_out)


def main():
    print('Hello from main')

    # testParseInsFile()
    testEmulator()
    # testEmuData()
    # testConvertSignedInt2Bin()
    # testParseInsStr()


if __name__ == '__main__':
    main()
