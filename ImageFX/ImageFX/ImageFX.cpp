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

std::string inputFilename = "tore.png";
std::string outputFilename = "ImageFX.bmp";

struct ImgFXWindow : Window {
    cl_mem inmem, outmem;
    cl_kernel procKernel;
    cl_kernel greyKernel;

    int32_t *outbuf;
    SDL_Surface *image;

	OpenCLMgr *mgr;

    ImgFXWindow() {
		inmem = outmem = 0;
		outbuf = 0;

		mgr = new OpenCLMgr();

		procKernel = mgr->procKernel;
        greyKernel = mgr->greyKernel;

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


    virtual cl_mem applyFilter(cl_mem in_buffer,  cl_float* FilterMat, cl_int dim, cl_int flag){
        size_t gdims[] = { image->w, image->h };
        
        cl_mem out_buffer = clCreateBuffer(mgr->context, CL_MEM_WRITE_ONLY, sizeof(int32_t)*image->w*image->h, NULL, NULL);
        cl_mem filtermem = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * dim * dim, FilterMat, NULL);

        cl_int status = 0;
        status = clSetKernelArg(procKernel, 0, sizeof(inmem), &in_buffer);
		status |= clSetKernelArg(procKernel, 1, sizeof(out_buffer), &out_buffer);
		status |= clSetKernelArg(procKernel, 2, sizeof(cl_int), &image->w);
		status |= clSetKernelArg(procKernel, 3, sizeof(cl_int), &image->h);
        status |= clSetKernelArg(procKernel, 4, sizeof(cl_int), &dim);
        status |= clSetKernelArg(procKernel, 5, sizeof(filtermem), &filtermem);
        status |= clSetKernelArg(procKernel, 6, sizeof(cl_int), &flag);

        if (status)
			throw "set kernel arg";
        
        status |= clEnqueueNDRangeKernel(mgr->commandQueue, procKernel, 2, NULL, gdims, NULL, 0, NULL, NULL);
        std::cout << status;

        return out_buffer;
    }


    virtual cl_mem applyGrey(cl_mem in_buffer) {
        size_t gdims[] = { image->w, image->h };
        cl_int status = 0;

        // creating and setting parameters for grey_buffer kernel call
        cl_mem greyMem = clCreateBuffer(mgr->context, CL_MEM_WRITE_ONLY, sizeof(int32_t) * image->w * image->h, NULL, NULL);
        status = clSetKernelArg(greyKernel, 0, sizeof(inmem), &inmem);
        status |= clSetKernelArg(greyKernel, 1, sizeof(greyMem), &greyMem);

        // starting the greyscale kernel
        status |= clEnqueueNDRangeKernel(mgr->commandQueue, greyKernel, 2, NULL, gdims, NULL, 0, NULL, NULL);

        return greyMem;
    }


    virtual void onApply() {
		// ==================================================
		// ==== called when clicking the "apply" button ====
		// ==================================================
        if (!outbuf || !image) return;

        size_t gdims[] = { image->w, image->h };

		cl_int dim = 3;
        cl_int flag = 1; // 1 if absolute values wanted, 0 if scaled values wanted

        cl_float FilterMat[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
        //cl_float FilterMat[9] = { 0, 0, 0, 0, 1, 0, 0, 0, 0 };
        //cl_float FilterMat[9] = { 0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625 };    // Blurr
        // cl_float FilterMat[9] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };                                       // Sharpen
        // cl_mem filtermem = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * dim * dim, FilterMat, NULL);

        cl_int status = 0;
		//status = clSetKernelArg(procKernel, 0, sizeof(inmem), &inmem);
		//status |= clSetKernelArg(procKernel, 1, sizeof(outmem), &outmem);
		//status |= clSetKernelArg(procKernel, 2, sizeof(cl_int), &image->w);
		//status |= clSetKernelArg(procKernel, 3, sizeof(cl_int), &image->h);
        //status |= clSetKernelArg(procKernel, 4, sizeof(cl_int), &dim);
        //status |= clSetKernelArg(procKernel, 5, sizeof(filtermem), &filtermem);
        //status |= clSetKernelArg(procKernel, 6, sizeof(cl_int), &flag);
		

		//if (status)
		//	throw "set kernel arg";
		//status |= clEnqueueNDRangeKernel(mgr->commandQueue, procKernel, 2, NULL, gdims, NULL, 0, NULL, NULL);

        //cl_mem gaussMem = applyFilter(inmem, FilterMat, dim, flag);

        //status = clEnqueueReadBuffer(mgr->commandQueue, gaussMem, CL_TRUE, 0, sizeof(*outbuf)*image->w*image->h, outbuf, 0, NULL, NULL);
        //if (status) throw "enqueue commands";

        // using the applyGrey function to greyscale the input image
        cl_mem greyMem = applyGrey(inmem);

        // reading the greyscale result
        //status = clEnqueueReadBuffer(mgr->commandQueue, greyMem, CL_TRUE, 0, sizeof(*outbuf) * image->w * image->h, outbuf, 0, NULL, NULL);


        // create smoothin filter
        cl_float glattFilterMat[9] = { 0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625 };

        // applying the smoothing filter on the greyscaled picture
        cl_mem glattMem = applyFilter(greyMem, glattFilterMat, dim, flag);

        // reading the smoothing result
        status = clEnqueueReadBuffer(mgr->commandQueue, glattMem, CL_TRUE, 0, sizeof(*outbuf) * image->w * image->h, outbuf, 0, NULL, NULL);


        // creating bufer for greyscale kernel
        // cl_mem filtermem = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*dim*dim, FilterMat, NULL);

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

