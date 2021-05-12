# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import mips


def testInstruction():
    print('32510010=', mips.Instruction.parse('32510010'))
    print('3251FFFE=', mips.Instruction.parse('3251FFFE'))


def testEmuData():
    # load memory image
    file_path = 'data/test_02.txt'
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


def testEmulator():
    # load memory image
    emu = mips.Emulator()
    file_path = "data/sample_memory_image.txt"
    # file_path = "data/test_02.txt"
    # file_path = "data/extra_ex_07.txt"
    emu.loadFromFile(file_path)
    print('Number of mem lines =', len(emu.mem_in.mem))

    # display instructions
    print(emu.getInsStr())

    # execute
    emu.execute()
    print(emu.getReportString())


def main():
    print('Hello from main')

    testEmulator()
    # testEmuData()


if __name__ == '__main__':
    main()
