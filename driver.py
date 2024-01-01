#!/usr/bin/env python
from codellama.tokenizer import Tokenizer
from localsock import Socket
import subprocess
from codellama.generation import get_token
import os
import signal
import sys

B_INST, E_INST = "[INST]", "[/INST]"

if __name__ == "__main__":
    socketPath = '/tmp/codellama_evaluator.sock'
    if os.path.exists(socketPath):
        os.remove(socketPath)
    evaluator = subprocess.Popen("release/evaluator")

    tokenizer_path='/Users/grady/src/codellama/CodeLlama-7b-Instruct/tokenizer.model'
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
            user_input = input("# ").strip()
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
            print(f'Submitting the following prompt:\n-------\n{initialInput}\n-------\n')
            user_input = initialInput
            initialInput = None

        toks = tokenizer.encode(f"{B_INST}{user_input}{E_INST}",
                                bos=True,
                                eos=False)
        socket.sendTokens(toks)

        responseTokens = []
        (logits, disconnected) = socket.recvLogits()
        responseTokens.append(get_token(logits, temperature, topP))
        while not disconnected and responseTokens[-1] != tokenizer.eos_id:
            socket.sendTokens([responseTokens[-1]]);
            (logits, disconnected) = socket.recvLogits()
            responseTokens.append(get_token(logits, temperature, topP))
            if len(responseTokens) % 10 == 0:
                print(f"decoded response so far: {tokenizer.decode(responseTokens)}")

        if not disconnected:
            print(f"decoded response: {tokenizer.decode(responseTokens)}")

    socket.close()
    evaluator.wait(2)
