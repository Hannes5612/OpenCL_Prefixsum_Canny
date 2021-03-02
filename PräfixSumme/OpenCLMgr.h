#pragma once


#include <CL/cl.h>
#include <string>
#include <iostream>
#include <string>


#define SUCCESS 0
#define FAILURE 1


#define CHECK_SUCCESS(msg) \
		if (status!=SUCCESS) { \
			cout << msg << endl; \
			return FAILURE; \
		}


class OpenCLMgr
{
public:
	OpenCLMgr();
	~OpenCLMgr();

	int isValid() {return valid;}

	
	cl_context context;
	cl_command_queue commandQueue;
	cl_program program;
	
	cl_kernel praefixsumme256_kernel;	// array size exactly 256. Only one workgroup uses

private:
	static int convertToString(const char *filename, std::string& s);

	int init();
	int valid;
};