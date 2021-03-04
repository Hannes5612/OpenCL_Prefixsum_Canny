
#define GX get_global_id(0)
#define GY get_global_id(1)

#define GW get_global_size(0)
#define GH get_global_size(1)

#define LX get_local_id(0)
#define LY get_local_id(1)


__kernel void imgfx(__global uchar4 *in, __global uchar4 *out, int width, int height, int dim, __global float* filterMat, int flag) { 

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

	// TODO

	// Radius of the filter matrix
	int offset = (dim - 1) / 2;

	// Pixel to calculate
	float4 calculated_pixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

	//
	int filter_index = 0;
	
	// Go through filter matrix indices
	for (int x_axis = GX - offset; x_axis <= GX + offset; x_axis++) {				// go through horizontally
		for (int y_axis = GY - offset; y_axis <= GY + offset; y_axis++) {			// go through vertically
			if (x_axis < 0 || y_axis < 0 || x_axis >= width || y_axis >= height) {	// out of image borders
				calculated_pixel += (float4)(1.0f, 1.0f, 1.0f, 1.0f);
			}
			else {
				int one_dim_ind = y_axis * width + x_axis;
				float4 pixel = convert_float4(in[one_dim_ind]);
				calculated_pixel += (filterMat[filter_index] * pixel);
			}

			filter_index++;
		}
	}

	out[GY*GW + GX] = convert_uchar4(calculated_pixel);
}