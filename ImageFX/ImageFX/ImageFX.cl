
#define GX get_global_id(0)
#define GY get_global_id(1)

#define GW get_global_size(0)
#define GH get_global_size(1)

#define LX get_local_id(0)
#define LY get_local_id(1)


#define L_SIZE 16


__kernel void imgfx(__global float* in, __global float* out, int width, int height, int dim, __global float* filterMat, int flag) {


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
	float average_rgb = dot(pixel, (float4)(1)) / 3;
	pixel = (float4)(average_rgb, average_rgb, average_rgb, pixel.w);

	out[GX + GY * GW] = pixel.z * 0.299 + pixel.y * 0.587 + pixel.x * 0.114;
}

__kernel void to_uchar(__global float* in, __global uchar4* out) {
	int pixel = (int)in[GY * GW + GX];
	out[GY * GW + GX] = (uchar4)(pixel, pixel, pixel, 255);
}


kernel void abs_edge(global float* x_sobel, global float* y_sobel, global float* outAbs, __global float* outGrad) {

	size_t pos = GW * GY + GX;

	float x = x_sobel[pos];
	float y = y_sobel[pos];

	outAbs[pos] = min(hypot(x, y), 255.0f);
}

__kernel void nms(__global float* in, __global float* helper, __global float* out) {
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
	helper[pos] = newPixel;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (GY > 0 && GY < GH - 1) {
		float topPixel = in[pos - GH];
		float bottomPixel = in[pos + GH];
		if (topPixel <= pixel && bottomPixel < pixel) {
			newPixel = pixel;
		}
	}

	out[pos] = newPixel;

	barrier(CLK_LOCAL_MEM_FENCE);

	out[pos] = out[pos] + helper[pos];

}