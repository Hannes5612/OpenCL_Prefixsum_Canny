#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

#define FAILURE 1
#define SUCCESS 0

#include "ImageFX.h"
#include "OpenCLMgr.h"

/* convert the kernel file into a string */
int convertToString(const char *filename, std::string& s) {
    size_t size;
    char*  str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if(f.is_open()) {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size+1];
        if(!str) {
            f.close();
            return 0;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    cout<<"Error: failed to open file\n:"<<filename<<endl;
    return FAILURE;
}

// ==========================
// ==== filenames for UI ====
// ==========================

std::string inputFilename = "Banane.png";
std::string outputFilename = "ImageFX.bmp";

struct ImgFXWindow : Window {
    cl_mem inmem, outmem;
    cl_kernel procKernel;

    int32_t *outbuf;
    SDL_Surface *image;

	OpenCLMgr *mgr;

    ImgFXWindow() {
		inmem = outmem = 0;
		outbuf = 0;

		mgr = new OpenCLMgr();

		procKernel = mgr->procKernel;

        onReset();
    }

	~ImgFXWindow()
	{
		delete mgr;
	}

    virtual void onReset() {
		// =================================================
		// ==== called when clicking the "reset" button ====
		// =================================================
        if (inmem) clReleaseMemObject(inmem);
        if (outmem) clReleaseMemObject(outmem);
        delete[] outbuf; outbuf = 0;

		image = loadImageRGB32(inputFilename);
        if (!image) 
			throw "loading failed";
		cl_int width = image->w;
		cl_int height = image->h;

		// create buffer for input image
        inmem = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                               sizeof(int32_t)*width*height, image->pixels, NULL);

        outmem = clCreateBuffer(mgr->context, CL_MEM_WRITE_ONLY, sizeof(int32_t)*width*height, NULL, NULL);


		// copy input to output for first display
        outbuf = new int32_t[width*height];
        memcpy(outbuf, image->pixels, sizeof(int32_t)*width*height);
        useBuffer(outbuf, width, height);
    }

	virtual void onSave() {
		// ================================================
		// ==== called when clicking the "save" button ====
		// ================================================
		saveBMP(outputFilename);
	}

    virtual void onApply() {
		// ==================================================
		// ==== called when clicking the "apply" button ====
		// ==================================================
        if (!outbuf || !image) return;

        size_t gdims[] = { image->w, image->h };

		cl_int dim = 3;
		//cl_float FilterMat[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
		cl_float FilterMat[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };

        cl_int status=0;
		status = clSetKernelArg(procKernel, 0, sizeof(inmem), &inmem);
		status |= clSetKernelArg(procKernel, 1, sizeof(outmem), &outmem);
		status |= clSetKernelArg(procKernel, 2, sizeof(cl_int), &image->w);
		status |= clSetKernelArg(procKernel, 3, sizeof(cl_int), &image->h);

		// TODO 



		if (status)
			throw "set kernel arg";
		status |= clEnqueueNDRangeKernel(mgr->commandQueue, procKernel, 2, NULL, gdims, NULL, 0, NULL, NULL);
        status |= clEnqueueReadBuffer(mgr->commandQueue, outmem, CL_TRUE, 0, sizeof(*outbuf)*image->w*image->h, outbuf, 0, NULL, NULL);
        if (status) throw "enqueue commands";
    }
};

int main(int argc, char *argv[]) {


	// ====== show the UI
    ImgFXWindow w;
    w.run();


    return SUCCESS;
}


#include <stdafx.h>

int _tmain(int argc, _TCHAR* argv[])
{
	main(argc, NULL);
	return 0;
}

