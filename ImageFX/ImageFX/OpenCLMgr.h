#include <CL/cl.h>


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

	int isValid() { return valid; }


	cl_context context;
	cl_command_queue commandQueue;
	cl_program program;

	cl_kernel procKernel;
	cl_kernel greyKernel;
	cl_kernel absEdgeKernel;
	cl_kernel nmsKernel;
	cl_kernel ucharKernel;
	cl_kernel hystereseKernel;

private:
	static int convertToString(const char* filename, std::string& s);

	int init();
	int valid;
};