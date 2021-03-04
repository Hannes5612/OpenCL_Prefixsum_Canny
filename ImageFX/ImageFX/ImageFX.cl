
#define GX get_global_id(0)
#define GY get_global_id(1)

#define GW get_global_size(0)
#define GH get_global_size(1)

#define LX get_local_id(0)
#define LY get_local_id(1)


__kernel void imgfx(__global uchar4* in, __global uchar4* out, int width, int height, int dim, __global float* filterMat, int flag) {

	// Die Pixel sind vom Datentyp uchar4. Ein solcher Vektor bestehend aus 4 uchar-Werten (rot, grün, blau, alpha)
	// kann mit einem einzelnen uchar multipliziert werden, aber nicht mit anderen Datentypen, wie z.B. einem float
	// Um mit einem float multiplizieren zu können, muss zuerst konvertiert werden. Zur Konvertierung von Vektordatentypen
	// gibt es Fukntionen.
	// Beispiel
	//
	// float a,b;
	// uchar4 pix1 = in[0];
	// uchar4 pix2 = in[1];
	// uchar4 value = a*convert_float4(pix1) + b*convert_float4(pix2);
	// out[0] = convert_uchar4(value);

	// Radius of the filter matrix
	int offset = (dim - 1) / 2;

	// Pixel to calculate
	float4 calculated_pixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

	// 
	int filter_index = 0;

	// Go through filter matrix indices and calculate endpixel
	for (int y_axis_index = GY - offset; y_axis_index <= GY + offset; y_axis_index++) {		// Go through horizontally
		for (int x_axis_index = GX - offset; x_axis_index <= GX + offset; x_axis_index++) {	// Go through vertically
			if (x_axis_index >= 0 && y_axis_index >= 0) {									// Make sure to not multiply by negative values
				int one_dim_ind = y_axis_index * width + x_axis_index;						// Calculate used index
				float4 pixel = convert_float4(in[one_dim_ind]);								// Get pixel values of index
				calculated_pixel += filterMat[filter_index] * pixel;						// Add weighted values to end pixel
			}
			filter_index++;																	// Increase filter matrix index
		}
	}
	

	if (GX < offset || GY < offset || GX >= (GW - offset) || GY >= (GH - offset)) {
		calculated_pixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	}

	if ( flag == 1) {	// -> We want absolute values
		calculated_pixel = fabs(calculated_pixel);
	}
	else {				// -> We want scaled values
		calculated_pixel = (calculated_pixel + (float4)(256.0f, 256.0f, 256.0f, 0.0f)) / 2;
	}

	out[GY * GW + GX] = convert_uchar4(calculated_pixel);
}


__kernel void to_grey(__global uchar4* in, __global uchar4* out) {

	float4 pixel = convert_float4(in[GX + GY * GW]);
	float average_rgb = dot(pixel, (float4)(1)) / 3;
	pixel = (float4)(average_rgb, average_rgb, average_rgb, pixel.w);

	out[GX + GY * GW] = convert_uchar4(pixel);
}
