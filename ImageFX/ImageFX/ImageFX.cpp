#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

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
    cl_kernel greyKernel;
    cl_kernel absEdgeKernel;
    cl_kernel nmsKernel;
    cl_kernel ucharKernel;

    int32_t *outbuf;
    SDL_Surface *image;

	OpenCLMgr *mgr;

    ImgFXWindow() {
		inmem = outmem = 0;
		outbuf = 0;

		mgr = new OpenCLMgr();

		procKernel = mgr->procKernel;
        greyKernel = mgr->greyKernel;
        absEdgeKernel = mgr->absEdgeKernel;
        nmsKernel = mgr->nmsKernel;
        ucharKernel = mgr->ucharKernel;

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
        size_t local_size[] = { 16, 16 };


        cl_mem out_buffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(cl_float)*image->w*image->h, NULL, NULL);
        cl_mem filtermem = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * dim * dim, FilterMat, NULL);

        cl_int status = 0;
        status = clSetKernelArg(procKernel, 0, sizeof(in_buffer), &in_buffer);
		status |= clSetKernelArg(procKernel, 1, sizeof(out_buffer), &out_buffer);
		status |= clSetKernelArg(procKernel, 2, sizeof(cl_int), &image->w);
		status |= clSetKernelArg(procKernel, 3, sizeof(cl_int), &image->h);
        status |= clSetKernelArg(procKernel, 4, sizeof(cl_int), &dim);
        status |= clSetKernelArg(procKernel, 5, sizeof(filtermem), &filtermem);
        status |= clSetKernelArg(procKernel, 6, sizeof(cl_int), &flag);

        if (status)
			throw "set kernel arg";
        
        status |= clEnqueueNDRangeKernel(mgr->commandQueue, procKernel, 2, NULL, gdims, local_size, 0, NULL, NULL);
        std::cout << status;

        return out_buffer;
    }


    virtual cl_mem applyGrey(cl_mem in_buffer) {
        size_t gdims[] = { image->w, image->h };
        cl_int status = 0;

        // creating and setting parameters for grey_buffer kernel call
        cl_mem greyMem = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(cl_float) * image->w * image->h, NULL, NULL);
        status = clSetKernelArg(greyKernel, 0, sizeof(in_buffer), &in_buffer);
        status |= clSetKernelArg(greyKernel, 1, sizeof(greyMem), &greyMem);

        // starting the greyscale kernel
        status |= clEnqueueNDRangeKernel(mgr->commandQueue, greyKernel, 2, NULL, gdims, NULL, 0, NULL, NULL);

        return greyMem;
    }

    virtual cl_mem* applyAbsEdge(cl_mem xSobelMem, cl_mem ySobelMem) {
        // Create Buffer for abs strengths and gradients
        cl_mem* absEdgeMem = new cl_mem[2];
        absEdgeMem[0] = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(cl_float) * image->w * image->h, NULL, NULL);
        absEdgeMem[1] = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(cl_float) * image->w * image->h, NULL, NULL);

        cl_int status = 0;
        size_t gdims[] = { image->w, image->h };

        status = clSetKernelArg(absEdgeKernel, 0, sizeof(xSobelMem), &xSobelMem);
        status |= clSetKernelArg(absEdgeKernel, 1, sizeof(ySobelMem), &ySobelMem);
        status |= clSetKernelArg(absEdgeKernel, 2, sizeof(absEdgeMem[0]), &absEdgeMem[0]);
        status |= clSetKernelArg(absEdgeKernel, 3, sizeof(absEdgeMem[1]), &absEdgeMem[1]);

        status = clEnqueueNDRangeKernel(mgr->commandQueue, absEdgeKernel, 2, NULL, gdims, NULL, 0, NULL, NULL);

        return absEdgeMem;
    }

    virtual cl_mem applyNms(cl_mem* absEdgeMem) {
        cl_mem nmsMem = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(cl_float) * image->w * image->h, NULL, NULL);

        cl_int status = 0;
        size_t gdims[] = { image->w, image->h };


        status = clSetKernelArg(nmsKernel, 0, sizeof(absEdgeMem[0]), &absEdgeMem[0]);
        status |= clSetKernelArg(nmsKernel, 1, sizeof(absEdgeMem[1]), &absEdgeMem[1]);
        status |= clSetKernelArg(nmsKernel, 2, sizeof(nmsMem), &nmsMem);

        status = clEnqueueNDRangeKernel(mgr->commandQueue, nmsKernel, 2, NULL, gdims, NULL, 0, NULL, NULL);
        
        return nmsMem;
    }

    virtual cl_mem toUchar(cl_mem floatMem) {
        cl_mem ucharMem = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(int32_t) * image->w * image->h, NULL, NULL);
        
        cl_int status = 0;
        size_t gdims[] = { image->w, image->h };


        status = clSetKernelArg(ucharKernel, 0, sizeof(floatMem), &floatMem);
        status |= clSetKernelArg(ucharKernel, 1, sizeof(ucharMem), &ucharMem);

        status = clEnqueueNDRangeKernel(mgr->commandQueue, ucharKernel, 2, NULL, gdims, NULL, 0, NULL, NULL);

        return ucharMem;
    }

    virtual void onApply() {
        // ==================================================
        // ==== called when clicking the "apply" button ====
        // ==================================================
        if (!outbuf || !image) return;

        size_t gdims[] = { image->w, image->h };
        cl_int status = 0;
        cl_int dim = 3;
        cl_int flag = 1; // 1 if absolute values wanted, 0 if scaled values wanted


        // ==== 1. Greyscale image ====
        cl_mem greyMem = applyGrey(inmem);


        // ==== 2. Gauss smoothing ====
        // create smoothin filter
        cl_float gaussFilterMat[9] = { 0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625 };

        // applying the smoothing filter on the greyscaled picture
        cl_mem gaussMem = applyFilter(greyMem, gaussFilterMat, dim, flag);
        for (int i = 0; i < 2; i++) {
        gaussMem = applyFilter(gaussMem, gaussFilterMat, dim, flag);
        }

        // release not anymore needed memory buffer of greyscaled image
        //clReleaseMemObject(greyMem);


        // ==== 3. x- and y-Sobel edge detection ====
        // x and y sobel filters
        cl_float sobelXFilterMat[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        cl_float sobelYFilterMat[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
        //cl_float sobelXFilterMat[9] = { -1, -2, -1, 0, 0, 0, -1, -2, -1 };
        //cl_float sobelYFilterMat[9] = { 1, 0, -1, 2, 0, -2, 1, 0, 1 };

        //applying both filters on the smoothed picture
        cl_mem xSobelMem = applyFilter(gaussMem, sobelXFilterMat, dim, flag);
        cl_mem ySobelMem = applyFilter(gaussMem, sobelYFilterMat, dim, flag);

        // release not anymore needed memory buffer of smoothed image memory buffer
        // clReleaseMemObject(gaussMem);


        // ==== 4. Calculating absolute edgestrength ====
        cl_mem* absEdgeMem = applyAbsEdge(xSobelMem, ySobelMem);


        // ==== 5. Applying non-maximum supression
        cl_mem nmsMem = applyNms(absEdgeMem);


        // ==== Lastly convert back to uchar
        cl_mem ucharMem = toUchar(nmsMem);



        // reading the result

        status = clEnqueueReadBuffer(mgr->commandQueue, ucharMem, CL_TRUE, 0, sizeof(*outbuf) * image->w * image->h, outbuf, 0, NULL, NULL);
        std::cout << status;
        float* outbuf_2;
        outbuf_2 = new float[image->w * image->h];

        // reading the result
        status = clEnqueueReadBuffer(mgr->commandQueue, absEdgeMem[1], CL_TRUE, 0, sizeof(*outbuf_2) * image->w * image->h, outbuf_2, 0, NULL, NULL);

        std::cout << *std::max_element(outbuf_2, outbuf_2 + (image->w * image->h)) << "\n";
        std::cout << *std::min_element(outbuf_2, outbuf_2 + (image->w * image->h));

        //int j = image->w * image->h;
        //for (int i = 0; i < j; i++) {
        //std::cout << outbuf_2[i] << '\n';
        //}

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
