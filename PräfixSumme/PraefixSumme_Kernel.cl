__kernel praefixsumme256_kernel(__global int* input_buffer_a, __global int* output_buffer_b, __global int* blocksum_buffer_c)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int gpid = get_group_id(0);

    // Copy to local memory
    __local int localArray[256];
    localArray[lid] = input_buffer[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Save last element to later calculate 
    int lastElementValue = 0;
    if (lid == 255) {
        lastElementValue = localArray[lid];
    }

    // ### Up-Sweep
    int index1,                 // Used in addition
        index2;                 // Used in addition, result being written into

    int noItemsThatWork = 128;  // Limit to select indices
    int spacing = 1;            // Space between the two used indices

    for (int d = 0; d < 8; d++) // 8, since log_2(256)
    { 

        // Check, that workitem is in allowed range and that the resulting index 
        // In the following caluclation will not exceed 255
        if (lid < noItemsThatWork) { 
            index1 = lid * (spacing << 1) + spacing - 1;                    // Calculate first index
            index2 = index1 + spacing;                                      // Calc second index with spacing
            localArray[index2] = localArray[index1] + localArray[index2];   // Write resulting sum in localArray[index2]
        }

        // Wait for every work item to finish
        barrier(CLK_LOCAL_MEM_FENCE);

        // Modify for next loop
        noItemsThatWork >>= 1;   // Halve
        spacing <<= 1;           // Double
    }

    // ### Down-Sweep
    if (lid == 255) localArray[255] = 0;    // Since being used again, reset
    barrier(CLK_LOCAL_MEM_FENCE);           // Wait for every work item to finish

    noItemsThatWork = 1;    // Limit to select indices
    spacing = 128;          // Space between the two used indices

    for (d = 0; d < 8; d++) // 8, since log_2(256)
    {
        if (lid < noItemsThatWork) {
            index1 = lid * (spacing << 1) + spacing - 1;    // Calc index 1
            index2 = index1 + spacing;                      // Calc index 2
            int tmp = localArray[index1];                   // Temp save localArray[index1] value for later use, since we write over it
            localArray[index1] = localArray[index2];        // Swap the two values
            localArray[index2] = tmp + localArray[index2];  // Write the right value as an addition of both
        }

        // Wait
        barrier(CLK_LOCAL_MEM_FENCE);

        // Modify for next loop
        noItemsThatWork <<= 1;  // Double
        spacing >>= 1;          // Halve
    }

    // Write result to global memory
    output_buffer[gid] = localArray[lid];

    // Write 
    if (lid == 255) {
        blocksum_output_buffer[gpid] = localArray[lid] + lastElementValue;
    }

}

__kernel void final_prefixsum(global int* buffer_b, global int* helper_buffer_d)
{
    int gid = get_global_id(0);
    int gpid = get_group_id(0);

    // jedes Element im B Array mit dem zugehörigen Element aus D addieren und in den Outputbuffer schreiben
    buffer_b[gid] += helper_buffer_d[gpid];

}