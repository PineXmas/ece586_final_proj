# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import mips


def testInstruction():
    print('32510010=', mips.Instruction.parse('32510010'))
    print('3251FFFE=', mips.Instruction.parse('3251FFFE'))


def testEmulator():
    # load memory image
    emu = mips.Emulator()
    file_path = "data/test_02.txt"
    emu.loadFromFile(file_path)
    print('Number of mem lines =', len(emu.mem))

    # display instructions
    print(emu.getInsStr())


def main():
    print('Hello from main')

    testEmulator()


if __name__ == '__main__':
    main()

