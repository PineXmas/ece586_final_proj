# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import mips


def testInstruction():
    print('32510010=', mips.Instruction.parse('32510010'))
    print('3251FFFE=', mips.Instruction.parse('3251FFFE'))


def testEmuData():
    # load memory image
    d1 = mips.EmuData()
    d1.loadFromFile('data/sample_memory_image.txt')

    # print all instructions
    list_ins = d1.getIns()
    for ins in list_ins:
        print(ins.toString(add_type=True))


def testEmulator():
    # load memory image
    emu = mips.Emulator()
    file_path = "data/test_02.txt"
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
