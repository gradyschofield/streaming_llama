#!/usr/bin/env python
from dataclasses import dataclass
import glob
import numpy as np
import pickle
import struct
import sys
from typing import Any, Callable, IO
import zipfile

# I read llama.cpp code to figure out how the pytorch/fairscale files work.
# See https://github.com/ggerganov/llama.cpp/blob/7e50d34be68aae2cc766203703dd188e910e033a/convert.py#L649

@dataclass
class TensorInfo:
    fileNumber: str
    offset: int
    outFileOffset: int

@dataclass
class TensorStorageLoader:
    load: Callable[[], np.ndarray]
    basePath: str
    fileNumber: str
    outFile: IO[bytes]
    offsetDirectory: {}

class ModelUnpickler(pickle.Unpickler):

    def __init__(self, fp: IO[bytes], path: str, zipFile: zipfile.ZipFile,
                 outFile: IO[bytes], fileIdx: int, numFiles: int, offsetDirectory: {}):
        super().__init__(fp)
        self.path = path
        self.outFile = outFile
        self.fileIdx = fileIdx
        self.numFiles = numFiles
        self.offsetDirectory = offsetDirectory
        self.zipFile = zipFile

    # pid: (storage, dtype, filestem, 'cpu', num elements)
    def persistent_load(self, pid: Any) -> Any:
        assert pid[0] == 'storage'
        dataType = pid[1]
        filenameStem = pid[2]
        filename = f'{self.path}/{filenameStem}'
        print(f'persistent load called pid:{pid} filename:{filename}')
        zipFileInfo = self.zipFile.getinfo(filename)

        def load(offset: int, count: int) -> np.ndarray:
            with self.zipFile.open(zipFileInfo) as file:
                elementSize = dataType.itemsize
                file.seek(offset * elementSize)
                size = count * elementSize
                print(f'load called dtype:{dataType} offset: {offset} size: {size}')
                data = file.read(size)
                assert len(data) == size
                if dataType != np.dtype(np.uint16) and dataType != np.dtype(np.float32):
                    raise "We encountered an array that is neither bfloat16 or float32.  You need to modify the code to handle this."
                if np.dtype(np.float32) == dataType:
                    int32_array = np.frombuffer(data, np.dtype(np.int32))
                    uint16_array = (int32_array >> 16).astype(np.dtype(np.uint16))
                    return uint16_array
                return np.frombuffer(data, np.dtype(np.uint16))
        return TensorStorageLoader(load, filename, filenameStem, outFile, offsetDirectory)

    @staticmethod
    def rebuild_tensor_v2(storage: Any, offset: Any, size: Any, stride: Any,
                          requires_grad: Any, backward_hooks: Any, metadata: Any = None) -> TensorInfo:
        assert isinstance(storage, TensorStorageLoader)
        print(f'rebuild_tensor_v2 called file no: {storage.fileNumber} offset: {offset} size: {size} stride: {stride}')
        count = stride[0] * size[0]

        outFileOffset = storage.outFile.tell()
        outFile.write(struct.pack('i', size[0]))  # rows
        if len(size) == 1:
            outFile.write(struct.pack('i', 1))  # columns
            storage.load(offset, count).tofile(outFile)
        else:
            outFile.write(struct.pack('i', size[1]))  # columns
            storage.load(offset, count).reshape(size).transpose().tofile(outFile)  # transpose to get column major layout
        return TensorInfo(storage.fileNumber, offset, outFileOffset)

    @staticmethod
    def rebuild_from_type_v2(func, new_type, state, args):
        return func(*args)

    CLASSES = {
        ('torch._tensor', '_rebuild_from_type_v2'): getattr(rebuild_from_type_v2, '__func__'),
        ('torch._utils', '_rebuild_tensor_v2'): getattr(rebuild_tensor_v2, '__func__'),
        ('torch', 'BFloat16Storage'): np.dtype(np.uint16),
        ('torch', 'HalfStorage'): np.dtype(np.float16),
        ('torch', 'FloatStorage'): np.dtype(np.float32),
        ('torch', 'IntStorage'): np.dtype(np.int16),
        ('torch', 'Tensor'): TensorInfo
    }

    def find_class(self, module: str, name: str):
        if not module.startswith('torch'):
            return super().find_class(module, name)
        return self.CLASSES[(module, name)]

if __name__ == "__main__":
    if(len(sys.argv) == 1):
        print('USAGE: model_unpickler.py [model path]')
        sys.exit(1)
    basePath = sys.argv[1]
    pthFileList = glob.glob(f'{basePath}/*.pth')
    pthFileList.sort()
    print('Reading files:')
    for f in pthFileList:
        print(f)
    print(f'llama_model.bin will be written to {basePath}')
    numFiles = len(pthFileList)
    assert numFiles > 0
    offsetDirectory = {}  #TODO: remove
    outFile = open(f'{basePath}/llama_model.pbin', 'wb')
    outFile.write(struct.pack('Q', 0))  # save space for the location of the tensor offset table
    tensorNameToOffset = []  #TODO: rename
    for fileIdx, zipFilePath in enumerate(pthFileList):
        with open(zipFilePath, 'rb') as zipFileHandle:
            zipFile = zipfile.ZipFile(zipFileHandle)

            picklePaths = [s for s in zipFile.namelist() if s.endswith('.pkl')]
            assert len(picklePaths) == 1, picklePaths
            with zipFile.open(picklePaths[0], 'r') as inFile:
                #TODO: remove numFiles
                unpickler = ModelUnpickler(inFile, f'{picklePaths[0][:-4]}', zipFile,
                                           outFile, fileIdx, numFiles, offsetDirectory)
                layers = unpickler.load()
                print(f'type: {type(layers)} size: {len(layers)}')
                for key,val in layers.items():
                    tensorNameToOffset.append((f'{fileIdx:02}.'+key, val.outFileOffset))
    tensorOffsetTablePos = outFile.tell()
    print(f'Tensor table pos: {tensorOffsetTablePos}')
    outFile.write(struct.pack('i', len(tensorNameToOffset)))
    for tensorName, offset in tensorNameToOffset:
        print(f'{tensorName}: {offset}')
        outFile.write(struct.pack('i', len(tensorName)))
        outFile.write(tensorName.encode('ascii'))
        outFile.write(struct.pack('Q', offset))
    outFile.seek(0)
    outFile.write(struct.pack('Q', tensorOffsetTablePos))
    outFile.close()
