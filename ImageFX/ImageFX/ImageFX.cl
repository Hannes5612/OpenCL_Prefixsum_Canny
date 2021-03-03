
#define GX get_global_id(0)
#define GY get_global_id(1)

#define GW get_global_size(0)
#define GH get_global_size(1)

#define LX get_local_id(0)
#define LY get_local_id(1)


__kernel void imgfx(__global uchar4 *in, __global uchar4 *out, int width, int height) { 

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
	
	out[GY*GW + GX] = (uchar4)(0,0,0,0);
}
