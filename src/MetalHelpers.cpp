//
// Created by Grady Schofield on 12/10/23.
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include<MetalHelpers.h>
#include<Exception.h>
#include<filesystem>

namespace fs = std::filesystem;

static MTL::Device * device = nullptr;
static MTL::CaptureManager * captureManager = nullptr;
static MTL::CaptureDescriptor * captureDescriptor = nullptr;

static unordered_map<void const *, MTL::Buffer*> pointerToMetalBuffer;
static unordered_map<MTL::Buffer *, void *> metalBufferToPointer;

namespace Metal {
    MTL::Device * getDevice() {
        if (!device) {
            device = MTLCreateSystemDefaultDevice();
        }
        return device;
    }

    MTL::Buffer * newBuffer(void * p, long len) {
        MTL::Buffer * ret = device->newBuffer(p, len, MTL::StorageModeShared);
        pointerToMetalBuffer.emplace(p, ret);
        metalBufferToPointer.emplace(ret, p);
        return ret;
    }

    void releaseBuffer(MTL::Buffer * buffer) {
        if (buffer) {
            void *p = mapAt(metalBufferToPointer, buffer);
            pointerToMetalBuffer.erase(p);
            metalBufferToPointer.erase(buffer);
            buffer->release();
        }
    }

    MTL::Buffer * getBuffer(void const * p) {
        return mapAt(pointerToMetalBuffer, p);
    }

    void startCapture(string path, bool eraseExisting) {
        const char * env = getenv("MTL_CAPTURE_ENABLED");
        if (!env || strncmp("1", env, 1) != 0) {
            cout << "The MTL_CAPTURE_ENABLED env variable is not set to 1\n";
        }
        if (!path.ends_with(".gputrace")) {
            cout << "The gputrace path does not end with .gputrace\n";
            return;
        }
        if (!captureManager) {
            captureManager = MTL::CaptureManager::sharedCaptureManager();
        }
        if (captureDescriptor) {
            stringstream sstr;
            sstr << "Tried to start a capture for " << path << " but one is already running.";
            throw Exception(sstr.str());
        }
        captureDescriptor = MTL::CaptureDescriptor::alloc();
        captureDescriptor->setCaptureObject(getDevice());
        captureDescriptor->setDestination(MTL::CaptureDestination::CaptureDestinationGPUTraceDocument);
        if (eraseExisting && fs::exists(path)) {
            if (!fs::is_directory(path)) {
                cout << "The gputrace path already exists, but it is not a directory.  I won't delete it.\n";
                return;
            }
            try {
                for (auto const & e : fs::directory_iterator(path)) {
                    fs::remove_all(e.path());
                }
                fs::remove(path);
            } catch (exception const & e) {
                cout << "error removing existing gputrace directory " << e.what() << "\n";
                return;
            }
        }
        captureDescriptor->setOutputURL(
                NS::URL::fileURLWithPath(
                        NS::String::string(path.c_str(), NS::ASCIIStringEncoding)));
        NS::Error * err = nullptr;
        captureManager->startCapture(captureDescriptor, &err);
        if (err) {
            cout << "capture error: " << err->localizedDescription()->utf8String() << endl;
        }
    }

    void finishCapture() {
        if (captureManager && captureDescriptor) {
            captureManager->stopCapture();
            captureDescriptor->release();
        }
    }


    MTL::Library * newLibrary(string const & src) {
        NS::String * shaderStr = NS::String::string(src.c_str(), NS::ASCIIStringEncoding);
        NS::Error * error = nullptr;
        MTL::Library * library = device->newLibrary(shaderStr, nullptr, & error);
        if (error) {
            cout << "newLibrary error: " << error->localizedDescription()->utf8String() << endl;
        }
        return library;
    }

    Function getFunction(MTL::Library * library, string name) {
        NS::String * matvecName = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
        MTL::Function * function = library->newFunction(matvecName);
        MTL::Device * device = getDevice();
        NS::Error * error;
        MTL::ComputePipelineState * computePipelineState = device->newComputePipelineState(function, & error);
        if (error) {
            cout << "newComputePipelineState error: " << error->localizedDescription()->utf8String() << endl;
            function->release();
            throw 1;
        }
        return Function(function, computePipelineState);
    }
}