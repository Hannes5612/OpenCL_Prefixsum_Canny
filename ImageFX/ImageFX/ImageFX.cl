
#define GX get_global_id(0)
#define GY get_global_id(1)

#define GW get_global_size(0)
#define GH get_global_size(1)

#define LX get_local_id(0)
#define LY get_local_id(1)

// Work item position
#define pos get_global_size(0) * get_global_id(1) + get_global_id(0)

// ====== Convolution Kernel ======
__kernel void imgfx(__global float* in, __global float* out, int dim, __global float* filterMat, int flag) {

	// Pixel we save our convoluted values in
	float calculated_pixel = 0.0;
	// Filter radius
	int offset = (dim - 1) / 2;


	// Filter index incremented during loop
	int filter_index = 0;

	// Go through filter matrix indices and calculate endpixel
	for (int y_axis_index = GY - offset; y_axis_index <= GY + offset; y_axis_index++) {		// Go through horizontally
		for (int x_axis_index = GX - offset; x_axis_index <= GX + offset; x_axis_index++) {	// Go through vertically
			if (x_axis_index >= 0 && y_axis_index >= 0										// Make sure to not multiply by negative values -> would skip whole x=0 and y=0
				&& x_axis_index <= GY * GW + GX && y_axis_index <= GY * GW + GX) 
			{									
				int one_dim_ind = y_axis_index * GW + x_axis_index;							// Calculate used index
				float pixel = in[one_dim_ind];												// Get pixel values of index
				calculated_pixel += filterMat[filter_index] * pixel;						// Add weighted values to end pixel
			}
			filter_index++;																	// Increase filter matrix index
		}
	}
	
	// Check if we are at border pixels -> set black if
	if (GX < offset || GY < offset || GX >= (GW - offset) || GY >= (GH - offset)) {
		calculated_pixel = 0.0;
	} 

	// Clamp value between max and min value
	calculated_pixel = (float)clamp((float)calculated_pixel, (float)-255, (float)255);

	// Apply scaling or absolution
	if ( flag == 1) {	// -> We want absolute values
		calculated_pixel = fabs(calculated_pixel);
	}
	else {				// -> We want scaled values
		calculated_pixel = (calculated_pixel + 255) /2;
	}

	// Write in output buffer
	out[pos] = calculated_pixel;
}

// ====== Greyscaling kernel ======
__kernel void to_grey(__global uchar4* in, __global float* out) {
	// Convert uchar4 to float4 for calculations
	float4 pixel = convert_float4(in[GX + GY * GW]);
	// Calculate greyscaled value
	float average_rgb = (pixel.x + pixel.y + pixel.z) / 3;

	// Write only float values to ease up succeeding steps
	out[GX + GY * GW] = average_rgb;
}

// ====== Convert float to uchar4 kernel ======
__kernel void to_uchar(__global float* in, __global uchar4* out) {
	// Get input value 
	float pixel = in[pos];
	// Turn single color value to turn into BGRA pixel
	out[pos] = (uchar4)(pixel, pixel, pixel, 1);
}

// ====== Calculate absolute edgestrengths kernel ====== 
kernel void abs_edge(global float* x_sobel, global float* y_sobel, global float* out) {
	
	// Get input values
	float x = x_sobel[pos];
	float y = y_sobel[pos];

	// Write to output buffer
	out[pos] = min(255.0f, hypot(x, y));
}

// ====== Non maximum suppression kernel ======
__kernel void nms(__global float* in, __global float* out) {

	// get input pixel
	float pixel = in[pos];

	// save edges
	float newPixel = 0.0;
	
	// Check for left and right values
	if (GX > 0 && GX < GW - 1) {
		float leftPixel = in[pos - 1];
		float rightPixel = in[pos + 1];
		// check if our current position has a higher value than the neighbour pixels
		if (leftPixel <= pixel && rightPixel < pixel) {
			newPixel = pixel;
		}
	}


	// Check for top and bottom values
	if (GY > 0 && GY < GH - 1) {
		float topPixel = in[pos - GW];
		float bottomPixel = in[pos + GW];
		// check if our current position has a higher value than the neighbour pixels
		if (topPixel <= pixel && bottomPixel < pixel) {
			newPixel += pixel;
		}
	}

	out[pos] = newPixel;


}

__kernel void hysterese(__global float* in, __global float* out) {
	float low = 30;
	float high = 80;

	float edge = 255;

	float pixel = in[pos];

	if (pixel >= high) {
		out[pos] = edge;
	}
	else if (pixel <= low) {
		out[pos] = 0;
	}
	else {
		// Check for neighbour pixels being over threshold
		// avoid being at border
		if (GX > 0 && GX < GW - 1 && GY > 0 && GY < GH - 1) {
			float leftPixel = in[pos - 1];
			float rightPixel = in[pos + 1];
			float topPixel = in[pos - GW];
			float bottomPixel = in[pos + GW];
			if (leftPixel > high || rightPixel > high || topPixel > high || bottomPixel > high) {
				out[pos] = edge;
			}
		}
		else {
			out[pos] = 0;
		}
	}
}