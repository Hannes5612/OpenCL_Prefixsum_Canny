
#define GX get_global_id(0)
#define GY get_global_id(1)

#define GW get_global_size(0)
#define GH get_global_size(1)

#define LX get_local_id(0)
#define LY get_local_id(1)


#define L_SIZE 16


__kernel void imgfx(__global float* in, __global float* out, int width, int height, int dim, __global float* filterMat, int flag) {


	//int kernel_rad = (dim - 1) / 2;
	//if (GX > kernel_rad && GX < width - kernel_rad && GY>kernel_rad && GY < height - kernel_rad) {
	//	float rSum = 0, kSum = 0;
	//	for (uint i = 0; i < dim; i++)
	//	{
	//		for (uint j = 0; j < dim; j++)
	//		{
	//			int pixelPosX = GX + (i - (dim / 2));
	//			int pixelPosY = GY + (j - (dim / 2));

	//			if ((pixelPosX < 0) ||
	//				(pixelPosX >= width) ||
	//				(pixelPosY < 0) ||
	//				(pixelPosY >= height)) continue;

	//			float r = in[pixelPosX + pixelPosY * width];
	//			float kernelVal = filterMat[i + j * dim];
	//			rSum += r * kernelVal;

	//			kSum += kernelVal;
	//		}
	//	}
	//	if (kSum == 0) kSum = 1;
	//	rSum /= kSum;
	//	out[GX + GY * width] = rSum;
	//}
	//else {
	//	out[GX + GY * width] = 255;
	//}
	//barrier(CLK_LOCAL_MEM_FENCE);

	{
	//// Die Pixel sind vom Datentyp uchar4. Ein solcher Vektor bestehend aus 4 uchar-Werten (rot, grün, blau, alpha)
	//// kann mit einem einzelnen uchar multipliziert werden, aber nicht mit anderen Datentypen, wie z.B. einem float
	//// Um mit einem float multiplizieren zu können, muss zuerst konvertiert werden. Zur Konvertierung von Vektordatentypen
	//// gibt es Fukntionen.
	//// Beispiel
	////
	//// float a,b;
	//// uchar4 pix1 = in[0];
	//// uchar4 pix2 = in[1];
	//// uchar4 value = a*convert_float4(pix1) + b*convert_float4(pix2);
	//// out[0] = convert_uchar4(value);

	float calculated_pixel = 0.0;
	int offset = (dim - 1) / 2;
	size_t pos = width * GY + GX;

	int filter_index = 0;

	// Go through filter matrix indices and calculate endpixel
	for (int y_axis_index = GY - offset; y_axis_index <= GY + offset; y_axis_index++) {		// Go through horizontally
		for (int x_axis_index = GX - offset; x_axis_index <= GX + offset; x_axis_index++) {	// Go through vertically
			if (x_axis_index >= 0 && y_axis_index >= 0										// Make sure to not multiply by negative values -> would skip whole x=0 and y=0
				&& x_axis_index <= GY * GW + GX && y_axis_index <= GY * GW + GX) 
			{									
				int one_dim_ind = y_axis_index * width + x_axis_index;						// Calculate used index
				float pixel = in[one_dim_ind];												// Get pixel values of index
				calculated_pixel += filterMat[filter_index] * pixel;						// Add weighted values to end pixel
			}
			filter_index++;																	// Increase filter matrix index
		}
	}
	
	// Check if we are at border pixels
	if (GX < offset || GY < offset || GX >= (GW - offset) || GY >= (GH - offset)) {
		calculated_pixel = 0.0;
	} 

	barrier(CLK_LOCAL_MEM_FENCE);

	// clamp value in allowed range
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
}}


__kernel void to_grey(__global uchar4* in, __global float* out) {

	float4 pixel = convert_float4(in[GX + GY * GW]);
	float average_rgb = (pixel.x + pixel.y + pixel.z) / 3;

	out[GX + GY * GW] = average_rgb;
}

__kernel void to_uchar(__global float* in, __global uchar4* out) {
	float pixel = in[GY * GW + GX];
	out[GY * GW + GX] = (uchar4)(pixel, pixel, pixel, 1);
}


kernel void abs_edge(global float* x_sobel, global float* y_sobel, global float* out) {

	size_t pos = GW * GY + GX;

	float x = x_sobel[pos];
	float y = y_sobel[pos];

	out[pos] = min(255.0f, hypot(x, y));
}

__kernel void nms(__global float* in, __global float* out) {
	size_t pos = GY * GW + GX;
	float pixel = in[pos];
	float newPixel = 0.0;
	
	if (GX > 0 && GX < GW - 1) {
		float leftPixel = in[pos - 1];
		float rightPixel = in[pos + 1];
		if (leftPixel <= pixel && rightPixel < pixel) {
			newPixel = pixel;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (GY > 0 && GY < GH - 1) {
		float topPixel = in[pos - GW];
		float bottomPixel = in[pos + GW];
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
	int pos = GY * GW + GX;

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