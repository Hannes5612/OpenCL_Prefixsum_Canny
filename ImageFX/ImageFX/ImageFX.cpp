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

std::string inputFilename = "images/hate.png";
std::string outputFilename = "ImageFX.bmp";

struct ImgFXWindow : Window {
    cl_mem inmem, outmem;
    cl_kernel procKernel;
    cl_kernel greyKernel;
    cl_kernel absEdgeKernel;
    cl_kernel nmsKernel;
    cl_kernel ucharKernel;
    cl_kernel hystereseKernel;

    int32_t *outbuf;
    SDL_Surface *image;

    cl_int status = 0;

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
        hystereseKernel = mgr->hystereseKernel;

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
		// ==== Called when clicking the "save" button ====
		// ================================================
		saveBMP(outputFilename);
	}
    

    virtual cl_mem applyFilter(cl_mem in_buffer,  cl_float* FilterMat, cl_int dim, cl_int flag, size_t* global_size){

        cl_mem outBuffer = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(cl_float)*image->w*image->h, NULL, NULL);
        cl_mem filterMem = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * dim * dim, FilterMat, NULL);

        status = clSetKernelArg(procKernel, 0, sizeof(in_buffer), &in_buffer);
		status |= clSetKernelArg(procKernel, 1, sizeof(outBuffer), &outBuffer);
        status |= clSetKernelArg(procKernel, 2, sizeof(cl_int), &dim);
        status |= clSetKernelArg(procKernel, 3, sizeof(filterMem), &filterMem);
        status |= clSetKernelArg(procKernel, 4, sizeof(cl_int), &flag);

        status |= clEnqueueNDRangeKernel(mgr->commandQueue, procKernel, 2, NULL, global_size, NULL, 0, NULL, NULL);

        return outBuffer;
    }


    virtual cl_mem applyGrey(cl_mem in_buffer, size_t* global_size) {
        // creating and setting parameters for grey_buffer kernel call
        cl_mem greyMem = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(cl_float) * image->w * image->h, NULL, NULL);
        status = clSetKernelArg(greyKernel, 0, sizeof(in_buffer), &in_buffer);
        status |= clSetKernelArg(greyKernel, 1, sizeof(greyMem), &greyMem);

        // starting the greyscale kernel
        status |= clEnqueueNDRangeKernel(mgr->commandQueue, greyKernel, 2, NULL, global_size, NULL, 0, NULL, NULL);

        return greyMem;
    }

    virtual cl_mem applyAbsEdge(cl_mem xSobelMem, cl_mem ySobelMem, size_t* global_size) {
        size_t gdims[] = { image->w, image->h };
        // Create Buffer for abs strengths and gradients
        cl_mem absEdgeMem = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(cl_float) * image->w * image->h, NULL, NULL);

        status = clSetKernelArg(absEdgeKernel, 0, sizeof(xSobelMem), &xSobelMem);
        status |= clSetKernelArg(absEdgeKernel, 1, sizeof(ySobelMem), &ySobelMem);
        status |= clSetKernelArg(absEdgeKernel, 2, sizeof(absEdgeMem), &absEdgeMem);

        status = clEnqueueNDRangeKernel(mgr->commandQueue, absEdgeKernel, 2, NULL, global_size, NULL, 0, NULL, NULL);

        return absEdgeMem;
    }

    virtual cl_mem applyNms(cl_mem absEdgeMem, size_t* global_size) {
        size_t gdims[] = { image->w, image->h };
        cl_mem nmsMem = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(cl_float) * image->w * image->h, NULL, NULL);

        status = clSetKernelArg(nmsKernel, 0, sizeof(absEdgeMem), &absEdgeMem);
        status |= clSetKernelArg(nmsKernel, 1, sizeof(nmsMem), &nmsMem);

        status = clEnqueueNDRangeKernel(mgr->commandQueue, nmsKernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
        
        return nmsMem;
    }

    virtual cl_mem toUchar(cl_mem floatMem, size_t* global_size) {
        size_t gdims[] = { image->w, image->h };
        cl_mem ucharMem = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(int32_t) * image->w * image->h, NULL, NULL);

        status = clSetKernelArg(ucharKernel, 0, sizeof(floatMem), &floatMem);
        status |= clSetKernelArg(ucharKernel, 1, sizeof(ucharMem), &ucharMem);

        status = clEnqueueNDRangeKernel(mgr->commandQueue, ucharKernel, 2, NULL, global_size, NULL, 0, NULL, NULL);

        return ucharMem;
    }

    virtual cl_mem applyHysterese(cl_mem floatMem, size_t* global_size, int minThreas, int maxThreas) {
        size_t gdims[] = { image->w, image->h };
        cl_mem hystMem = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE, sizeof(cl_float) * image->w * image->h, NULL, NULL);

        status = clSetKernelArg(hystereseKernel, 0, sizeof(floatMem), &floatMem);
        status |= clSetKernelArg(hystereseKernel, 1, sizeof(hystMem), &hystMem);
        status |= clSetKernelArg(hystereseKernel, 2, sizeof(minThreas), &minThreas);
        status |= clSetKernelArg(hystereseKernel, 3, sizeof(maxThreas), &maxThreas);

        status = clEnqueueNDRangeKernel(mgr->commandQueue, hystereseKernel, 2, NULL, global_size, NULL, 0, NULL, NULL);

        return hystMem;
    }

    virtual void onApply() {
        // ==================================================
        // ==== called when clicking the "apply" button =====
        // ==================================================

        std::string minStr;
        std::string maxStr;
        std::string gaussStr;
        std::cout << "Enter gauss blurr strength (pref. between 1 and 10): ";
        std::getline(std::cin, gaussStr);
        std::cout << "Enter hysterisis min threashold (pref. between 10 and 30): ";
        std::getline(std::cin, minStr);
        std::cout << "Enter hysterisis max threashold (pref. between 50 and 150): ";
        std::getline(std::cin, maxStr);
        int minThres = std::stoi(minStr);
        int maxThres = std::stoi(maxStr);
        int gauss = std::stoi(gaussStr);
        std::cout << "==== Calculating ====\n";

        if (!outbuf || !image) return;
        // Dimension of our filters
        cl_int dim = 3;
        // Flag to determine whether absolute (1) values are wanted, or if scaled (0) values wanted
        cl_int flag = 1;
        // Calculate global work size to pass into methods
        size_t gdims[] = { image->w, image->h };
        
        // ==== 1. Greyscale image ====
        cl_mem greyMem = applyGrey(inmem, gdims);

        // ==== 2. Gauss smoothing ====
        // create smoothin filter
        cl_float gaussFilterMat[9] = { 0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625 };

        // applying the smoothing filter on the greyscaled picture
        cl_mem gaussMem = applyFilter(greyMem, gaussFilterMat, dim, flag, gdims);
        for (int i = 1; i < gauss; i++) {
            gaussMem = applyFilter(gaussMem, gaussFilterMat, dim, flag, gdims);
        }

        // ==== 3. x- and y-Sobel edge detection ====
        // x and y sobel filters
        cl_float sobelXFilterMat[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        cl_float sobelYFilterMat[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

        //applying both filters on the smoothed picture
        cl_mem xSobelMem = applyFilter(gaussMem, sobelXFilterMat, dim, flag, gdims);
        cl_mem ySobelMem = applyFilter(gaussMem, sobelYFilterMat, dim, flag, gdims);

        // ==== 4. Calculating absolute edgestrength, release previous buffer ====
        cl_mem absEdgeMem = applyAbsEdge(xSobelMem, ySobelMem, gdims);


        // ==== 5. Applying non-maximum supression, release previous buffer ======
        cl_mem nmsMem = applyNms(absEdgeMem, gdims);


        // ==== 6. Apply hyterese claculations, release previous buffer ==========
        cl_mem hystMem = applyHysterese(nmsMem, gdims, minThres, maxThres);


        // ==== Lastly convert back to uchar, release previous buffer ============
        cl_mem ucharMem = toUchar(hystMem, gdims);


        // reading the result
        status = clEnqueueReadBuffer(mgr->commandQueue, ucharMem, CL_TRUE, 0, sizeof(*outbuf) * image->w * image->h, outbuf, 0, NULL, NULL);
        
        // Release allocated memory
        clReleaseMemObject(greyMem);
        clReleaseMemObject(gaussMem);
        clReleaseMemObject(xSobelMem);
        clReleaseMemObject(ySobelMem);
        clReleaseMemObject(absEdgeMem);
        clReleaseMemObject(nmsMem);
        clReleaseMemObject(hystMem);       
        clReleaseMemObject(ucharMem);
        
        // Debug code
        {
            // std::cout << status;
            //float* outbuf_2;
            //outbuf_2 = new float[image->w * image->h];
        
            // reading data for debugging information
            //status = clEnqueueReadBuffer(mgr->commandQueue, absEdgeMem[1], CL_TRUE, 0, sizeof(*outbuf_2) * image->w * image->h, outbuf_2, 0, NULL, NULL);

            //std::cout << *std::max_element(outbuf_2, outbuf_2 + (image->w * image->h)) << "\n";
            //std::cout << *std::min_element(outbuf_2, outbuf_2 + (image->w * image->h));

            //int j = image->w * image->h;
            //for (int i = 0; i < j; i++) {
            //std::cout << outbuf_2[i] << '\n';
            //}
        }

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

