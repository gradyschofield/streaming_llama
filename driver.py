#!/usr/bin/env python
from codellama.tokenizer import Tokenizer
from localsock import Socket
import subprocess
from codellama.generation import get_token
import os
import signal
import sys
import curses

B_INST, E_INST = "[INST]", "[/INST]"

def cursesPrint(responseTokens, tokenizer, stdscr):
    decodedStr = tokenizer.decode(responseTokens)
    if len(decodedStr) > 10:
        shortStr = decodedStr[:-10]
        strs = shortStr.split('\n')
        stdscr.clear()
        i = 0
        numRows, numCols = stdscr.getmaxyx()
        strs2 = []
        for s in strs[-numRows:]:
            while True:
                strs2.append(s[:numCols-1])
                s = s[numCols-1:]
                if len(s) == 0:
                    break
        try:
            for s in strs2[-numRows:]:
                stdscr.addstr(i, 0, f"{s}")
                i += 1
            stdscr.refresh()
        except Exception as e:
            with open("errfile", "w") as errfile:
                errfile.write(f"failed on {shortStr}")
            raise e

def cursesInput(stdscr):
    curses.noecho()
    stdscr.nodelay(False)
    stdscr.clear()
    stdscr.addstr(0, 0, "# ")
    ret = ""
    ch = stdscr.getch()
    i = 2
    while chr(ch) != '\n':
        if ch == curses.KEY_BACKSPACE or ch == 8 or ch == 127:
            if len(ret) > 0:
                i -= 1
                stdscr.addstr(0, i, ' ')
                stdscr.move(0, i)
                ret = ret[:-1]
        else:
            stdscr.addstr(0, i, chr(ch))
            ret += chr(ch)
            i += 1
        ch = stdscr.getch()
    stdscr.addstr(1, 0, ret)
    return ret

#if __name__ == "__main__":
def main(stdscr):
    socketPath = '/tmp/codellama_evaluator.sock'
    if os.path.exists(socketPath):
        os.remove(socketPath)
    evaluator = subprocess.Popen("release/evaluator")

    tokenizer_path='codellama/tokenizer_7b_13b.model'
    tokenizer = Tokenizer(model_path=tokenizer_path)

    socket = Socket()

    def cleanup(signum, frame):
        socket.close()
        sys.exit(1)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    topP = 0.95
    temperature = 0.2

    disconnected = False
    initialInput = None
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            initialInput = file.read()
    while not disconnected:
        if initialInput is None:
            user_input = cursesInput(stdscr).strip()
            if user_input == "exit":
                break
            if user_input.startswith("from file "):
                filename = user_input[len("from file "):]
                if os.path.exists(filename):
                    with open(filename, 'r') as file:
                        user_input = file.read()
                        print(f'Submitting the following prompt from file:\n-------\n{user_input}\n-------\n')
                else:
                    print("File not found: {filename}.")
                    continue
        else:
            stdscr.clear()
            stdscr.addstr(0, 0, f'Submitting the following prompt:\n-------\n{initialInput}\n-------\n')
            stdscr.refresh()
            user_input = initialInput
            initialInput = None

        toks = tokenizer.encode(f"{B_INST}{user_input}{E_INST}",
                                bos=True,
                                eos=False)
        socket.sendTokens(toks)

        responseTokens = []
        (logits, disconnected) = socket.recvLogits()
        responseTokens.append(get_token(logits, temperature, topP))
        stdscr.nodelay(True)
        while not disconnected and responseTokens[-1] != tokenizer.eos_id:
            socket.sendTokens([responseTokens[-1]]);
            (logits, disconnected) = socket.recvLogits()
            responseTokens.append(get_token(logits, temperature, topP))
            cursesPrint(responseTokens, tokenizer, stdscr)
            ch = stdscr.getch()
            if ch == 27:
                break

        if not disconnected:
            cursesPrint(responseTokens, tokenizer, stdscr)

    stdscr.addstr(0, 0, 'exited main loop')
    socket.close()
    evaluator.wait(2)

curses.wrapper(main)