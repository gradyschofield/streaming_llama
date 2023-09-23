from codellama.tokenizer import Tokenizer
from localsock import Socket
import subprocess
from codellama.generation import get_token
import sys

B_INST, E_INST = "[INST]", "[/INST]"

if __name__ == "__main__":
    evaluator = subprocess.Popen("release/evaluator")

    tokenizer_path='/Users/grady/src/codellama/CodeLlama-7b-Instruct/tokenizer.model'
    tokenizer = Tokenizer(model_path=tokenizer_path)

    socket = Socket()

    topP = 0.95
    temperature = 0.2

    disconnected = False
    while not disconnected:
        user_input = input("# ")
        if user_input == "exit":
            break

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