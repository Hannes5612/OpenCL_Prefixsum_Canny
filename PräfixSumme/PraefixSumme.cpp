/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

// For clarity,error checking has been omitted.

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

#include "OpenCLMgr.h"



// size of arrays must be exactly 256
int praefixsumme(cl_int *input, cl_int *output, int size, OpenCLMgr& mgr)
{
	cl_int status;

	int clsize = 256;

	// create OpenClinput buffer
	cl_mem inputBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, clsize * sizeof(cl_int),NULL, NULL);
	status = clEnqueueWriteBuffer(mgr.commandQueue, inputBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), input, 0, NULL, NULL);
	CHECK_SUCCESS("Error: writing buffer!")

	// create OpenCl buffer for output
	cl_mem outputBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, clsize * sizeof(cl_int), NULL, NULL);

	// Set kernel arguments.
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 1!")
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 1, sizeof(cl_mem), (void *)&outputBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")

	// Run the kernel.
	size_t global_work_size[1] = { clsize };
	size_t local_work_size[1] = { clsize };
	status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.praefixsumme256_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_SUCCESS("Error: enqueuing kernel!")

	// get resulting array
	status = clEnqueueReadBuffer(mgr.commandQueue, outputBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), output, 0, NULL, NULL);
	CHECK_SUCCESS("Error: reading buffer!")

	// release buffers
	status = clReleaseMemObject(inputBuffer);		
	CHECK_SUCCESS("Error: releasing buffer!")
	status = clReleaseMemObject(outputBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")

	return SUCCESS;
}

int main(int argc, char* argv[])
{
	OpenCLMgr mgr;

	// Initial input,output for the host and create memory objects for the kernel
	int size = 256;
	cl_int *input = new cl_int[size];
	cl_int *output = new cl_int[size];

	for (int i=0 ; i<size ; i++)
		input[i] = 1;

	// call function
	praefixsumme(input, output, size, mgr);
	
	delete [] input;
	delete[] output;

	std::cout<<"Passed!\n";
	return SUCCESS;
}



// eigener code
int praefixsumme_own(cl_int* input, cl_int* output, int size, OpenCLMgr& mgr)
{
	cl_int status;

	int clsize = 256;


	// größe für gesamten Inputarray berechnen, muss Vielfaches von 256 sein
	size_t new_size = (size + 255) / 256 * 256;

	// inputBuffer erstellen
	cl_mem inputBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, new_size * sizeof(cl_int), NULL, NULL);

	// mit nullen auffüllen
	status = clEnqueueWriteBuffer(mgr.commandQueue, inputBuffer, CL_TRUE, 0, new_size * sizeof(cl_int), input, 0, NULL, NULL);

	// outputBuffer erstellen
	cl_mem outputBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, new_size * sizeof(cl_int), NULL, NULL);

}