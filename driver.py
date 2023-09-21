from codellama.tokenizer import Tokenizer
import os
import socket
import struct
import subprocess
import sys
import time

B_INST, E_INST = "[INST]", "[/INST]"

if __name__ == "__main__":
    subprocess.Popen("release/evaluator")
    socketPath = '/tmp/codellama_evaluator.sock'
    while not os.path.exists(socketPath):
        time.sleep(0.1)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(socketPath)
    while True:
        user_input = input("Enter a comma-separated list of integers (or 'exit' to quit): ")
        if user_input == "exit":
            break

        try:
            numbers = [int(i.strip()) for i in user_input.split(",")]
        except ValueError:
            print("Invalid input. Please enter a comma-separated list of integers.")
            continue

        data = struct.pack(f"{len(numbers)}i", *numbers)
        client.sendall(data)

        response_data = client.recv(1024)
        if not response_data:
            print("Server disconnected.")
            break

        incremented_numbers = struct.unpack(f"{len(numbers)}i", response_data)
        print("Received from server:", ", ".join(map(str, incremented_numbers)))

    client.close()
    sys.exit(0)
    tokenizer_path='/Users/grady/src/codellama/CodeLlama-7b-Instruct/tokenizer.model'
    tokenizer = Tokenizer(model_path=tokenizer_path)
    toks = tokenizer.encode(f"{B_INST} int main(int argc, char ** argv){E_INST}",
                            bos=True,
                            eos=False)
    for t in toks:
        print(t)
    print(tokenizer.decode(toks))