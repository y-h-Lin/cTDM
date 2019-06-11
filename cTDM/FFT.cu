#define _USE_MATH_DEFINES
#include "FFT.cuh"
using namespace std;

/*-------------------------------------------------------------------------*/
/* This computes an in-place complex-to-complex FFT                        */
/* x and y are the real and imaginary arrays of 2^m points.                */
/* dir =  1 gives forward transform                                        */
/* dir = -1 gives reverse transform                                        */
/*                                                                         */  
/*  Formula: forward                                                       */
/*                N-1                                                      */
/*                ---                                                      */
/*            1   \          - j k 2 pi n / N                              */
/*    X(n) = ---   >   x(k) e                    = forward transform       */
/*            N   /                                n=0..N-1                */
/*                ---                                                      */
/*                k=0                                                      */
/*                                                                         */
/*    Formula: reverse                                                     */
/*                N-1                                                      */
/*                ---                                                      */
/*                \          j k 2 pi n / N                                */
/*    X(n) =       >   x(k) e                    = forward transform       */
/*                /                                n=0..N-1                */
/*                ---                                                      */
/*                k=0                                                      */
/*-------------------------------------------------------------------------*/
int FFT(int dir,int m,double *x,double *y)
{
	long nn,i,i1,j,k,i2,l,l1,l2;
	double c1,c2,tx,ty,t1,t2,u1,u2,z;

	/* Calculate the number of points */
	nn = 1;
	for (i=0;i<m;i++)
		nn *= 2;
	/* Do the bit reversal */
	i2 = nn >> 1;
	j = 0;
	for (i=0;i<nn-1;i++)
	{
		if (i < j)
		{
			tx = x[i];
			ty = y[i];
			x[i] = x[j];
			y[i] = y[j];
			x[j] = tx;
			y[j] = ty;
		}
		k = i2;
		while (k <= j)
		{
			j -= k;
			k >>= 1;
		}
		j += k;
	}

	/* Compute the FFT */
	c1 = -1.0;
	c2 = 0.0;
	l2 = 1;
	for (l=0;l<m;l++)
	{
		l1 = l2;
		l2 <<= 1;
		u1 = 1.0;
		u2 = 0.0;
		for (j=0;j<l1;j++)
		{
			for (i=j;i<nn;i+=l2)
			{
				i1 = i + l1;
				t1 = u1 * x[i1] - u2 * y[i1];
				t2 = u1 * y[i1] + u2 * x[i1];
				x[i1] = x[i] - t1;
				y[i1] = y[i] - t2;
				x[i] += t1;
				y[i] += t2;
			}
			z =  u1 * c1 - u2 * c2;
			u2 = u1 * c2 + u2 * c1;
			u1 = z;
		}
		c2 = sqrt((1.0 - c1) / 2.0);
		if (dir == 1)
			c2 = -c2;
		c1 = sqrt((1.0 + c1) / 2.0);
	}

	/* Scaling for forward transform */
	if (dir == 1)
	{
		for (i=0;i<nn;i++)
		{
			x[i] /= (double)nn;
			y[i] /= (double)nn;
		}
	}
   return(true);
}
/*-------------------------------------------------------------------------*/
int DFT(int dir,int m,double *x1,double *y1)
{
   long i,k;
   double arg;
   double cosarg,sinarg;

   double *x2 = (double *)malloc(m*sizeof(double));
   double *y2 = (double *)malloc(m*sizeof(double));

   if (x2 == NULL || y2 == NULL)
      return(false);

   for (i=0;i<m;i++) {
      x2[i] = 0;
      y2[i] = 0;
      arg = - dir * 2.0 * M_PI * (double)i / (double)m;
      for (k=0;k<m;k++) {
         cosarg = cos(k * arg);
         sinarg = sin(k * arg);
         x2[i] += (x1[k] * cosarg - y1[k] * sinarg);
         y2[i] += (x1[k] * sinarg + y1[k] * cosarg);
      }
   }

   /* Copy the data back */
   if (dir == 1) {
      for (i=0;i<m;i++) {
         x1[i] = x2[i] / (double)m;
         y1[i] = y2[i] / (double)m;
      }
   } else {
      for (i=0;i<m;i++) {
         x1[i] = x2[i];
         y1[i] = y2[i];
      }
   }

   free(x2);
   free(y2);
   return(true);
}
/*-------------------------------------------------------------------------*/
/* Butterworth filter                                                      */
/*-------------------------------------------------------------------------*/
void bfilter(complex<float> *filter, int width, int height)
{
	for (int v=0; v<height; v++)
		for (int u=0; u<width; u++)
		{
			int pos = u + v*width;
			float temp_v = (u-width/2)*(u-width/2) + (v-height/2)*(v-height/2);
			double distance = sqrt(temp_v);
			double H = 1 / (1 + pow(distance / 2.0, 0.1));
			filter[u + v*width] = complex<float>(real(filter[u + v*width])*H,imag(filter[u + v*width])*H);
		}
}
/*-------------------------------------------------------------------------*/
void FFT1Dshift(complex<float> *input, int length)
{
	complex<float> tmp;
	
	for (int i = 0; i < length/2; i++)
	{
		tmp = input[i];
		input[i] = input[i+length/2];
		input[i+length/2] = tmp;
	}
}
/*-------------------------------------------------------------------------*/
/*void cuFFT1Dshift(cufftComplex *input, int length)
{
	cufftComplex tmp;
	
	for (int i = 0; i < length/2; i++)
	{
		tmp = input[i];
		input[i] = input[i+length/2];
		input[i+length/2] = tmp;
	}
}*/
/*-------------------------------------------------------------------------*/
__global__ void cuFFT1Dshift(cufftComplex *input, int width)
{
	cufftComplex tmp;

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i<width / 2)
	{
		// interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
		tmp = input[i];
		input[i] = input[i + width / 2];
		input[i + width / 2] = tmp;
	}
}
/*-------------------------------------------------------------------------*/
/*Ref.: goo.gl/DR9Pqs*/
void FFT2Dshift(complex<float> *input, int width, int height)
{
	complex<float> tmp13, tmp24;
	
	// interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
	for (int k = 0; k < height/2; k++)
		for (int i = 0; i < width/2; i++)
		{
			tmp13         = input[i+k*width];
			input[i+k*width]       = input[(i+width/2)+(k+height/2)*width];
			input[(i+width/2)+(k+height/2)*width] = tmp13;
			tmp24         = input[(i+width/2)+k*width];
			input[(i+width/2)+k*width]    = input[i+(k+height/2)*width];
			input[i+(k+height/2)*width]    = tmp24;
		}
}
/*-------------------------------------------------------------------------*/
#define IDX2R(i,j,N) (((j)*(N))+(i))
__global__ void cuFFT2Dshift(cufftComplex *input, int width, int height)
{
	cufftComplex tmp13, tmp24;
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(i<width/2 && j<height/2)
	{
		// interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
		tmp13								  = input[i+j*width];
		input[i+j*width]					  = input[(i+width/2)+(j+height/2)*width];
		input[(i+width/2)+(j+height/2)*width] = tmp13;
		tmp24								  = input[(i+width/2)+j*width];
		input[(i+width/2)+j*width]			  = input[i+(j+height/2)*width];
		input[i+(j+height/2)*width]			  = tmp24;
	}

	/*int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int ind = idy*width + idx;
	int x, y, indshift;
	cufftComplex v;


	if (idx < width && idy < height / 2)
	{
		// A OPTIMISER
		if (idx<width / 2 && idy<height / 2)
		{
			x = idx + width / 2;
			y = idy + height / 2;
		}
		else if (idx >= width / 2 && idy<height / 2)
		{
			x = idx - width / 2;
			y = idy + height / 2;
		}

		indshift = y*width + x;
		v.x = input[ind].x;
		v.y = input[ind].y;

		input[ind].x = input[indshift].x;
		input[ind].y = input[indshift].y;

		input[indshift].x = v.x;
		input[indshift].y = v.y;
	}*/
}
/*-------------------------------------------------------------------------*/
/*Ref.: goo.gl/3ZEKgN*/
void FFT3Dshift(complex<float> *input, int width, int height, int slice)
{
	complex<float> tmp1, tmp2, tmp3, tmp4;
	
	for (int k = 0; k < slice/2; k++)
		for (int j = 0; j < height/2; j++)
			for (int i = 0; i < width/2; i++)
			{
				tmp1 = input[i+j*width+k*width*height];
				input[i+j*width+k*width*height] = input[(width/2+i)+(height/2+j)*width+(slice/2+k)*width*height];
				input[(width/2+i)+(height/2+j)*width+(slice/2+k)*width*height] = tmp1;

				tmp2 = input[i+(height/2+j)*width+k*width*height];
				input[i+(height/2+j)*width+k*width*height] = input[(width/2+i)+j*width+(slice/2+k)*width*height];
				input[(width/2+i)+j*width+(slice/2+k)*width*height] = tmp2;

				tmp3 = input[(width/2+i)+j*width+k*width*height];
				input[(width/2+i)+j*width+k*width*height] = input[i+(height/2+j)*width+(slice/2+k)*width*height];
				input[i+(height/2+j)*width+(slice/2+k)*width*height] = tmp3;

				tmp4 = input[(width/2+i)+(height/2+j)*width+k*width*height];
				input[(width/2+i)+(height/2+j)*width+k*width*height] = input[i+j*width+(slice/2+k)*width*height];
				input[i+j*width+(slice/2+k)*width*height] = tmp4;
			}
}
/*-------------------------------------------------------------------------*/
void FFT3Dshift_cufftComplex(cufftComplex *input, int width, int height, int slice)
{
	cufftComplex tmp1, tmp2, tmp3, tmp4;
	
	for (int k = 0; k < slice/2; k++)
		for (int j = 0; j < height/2; j++)
			for (int i = 0; i < width/2; i++)
			{
				tmp1 = input[i+j*width+k*width*height];
				input[i+j*width+k*width*height] = input[(width/2+i)+(height/2+j)*width+(slice/2+k)*width*height];
				input[(width/2+i)+(height/2+j)*width+(slice/2+k)*width*height] = tmp1;

				tmp2 = input[i+(height/2+j)*width+k*width*height];
				input[i+(height/2+j)*width+k*width*height] = input[(width/2+i)+j*width+(slice/2+k)*width*height];
				input[(width/2+i)+j*width+(slice/2+k)*width*height] = tmp2;

				tmp3 = input[(width/2+i)+j*width+k*width*height];
				input[(width/2+i)+j*width+k*width*height] = input[i+(height/2+j)*width+(slice/2+k)*width*height];
				input[i+(height/2+j)*width+(slice/2+k)*width*height] = tmp3;

				tmp4 = input[(width/2+i)+(height/2+j)*width+k*width*height];
				input[(width/2+i)+(height/2+j)*width+k*width*height] = input[i+j*width+(slice/2+k)*width*height];
				input[i+j*width+(slice/2+k)*width*height] = tmp4;
			}
}
/*-------------------------------------------------------------------------*/
__global__ void cuFFT3Dshift(cufftComplex *input, int width, int height, int slice)
{
	cufftComplex tmp1, tmp2, tmp3, tmp4;

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

	if ((i<width / 2) && (j<height / 2) && (k<slice / 2))
	{
		tmp1 = input[i + j*width + k*width*height];
		input[i + j*width + k*width*height] = input[(width / 2 + i) + (height / 2 + j)*width + (slice / 2 + k)*width*height];
		input[(width / 2 + i) + (height / 2 + j)*width + (slice / 2 + k)*width*height] = tmp1;

		tmp2 = input[i + (height / 2 + j)*width + k*width*height];
		input[i + (height / 2 + j)*width + k*width*height] = input[(width / 2 + i) + j*width + (slice / 2 + k)*width*height];
		input[(width / 2 + i) + j*width + (slice / 2 + k)*width*height] = tmp2;

		tmp3 = input[(width / 2 + i) + j*width + k*width*height];
		input[(width / 2 + i) + j*width + k*width*height] = input[i + (height / 2 + j)*width + (slice / 2 + k)*width*height];
		input[i + (height / 2 + j)*width + (slice / 2 + k)*width*height] = tmp3;

		tmp4 = input[(width / 2 + i) + (height / 2 + j)*width + k*width*height];
		input[(width / 2 + i) + (height / 2 + j)*width + k*width*height] = input[i + j*width + (slice / 2 + k)*width*height];
		input[i + j*width + (slice / 2 + k)*width*height] = tmp4;
	}
}
/*-------------------------------------------------------------------------*/
int Powerof2(int n, int *m, int *twopm)
{
	if (n <= 1)
	{
		*m = 0;
		*twopm = 1;
		return(false);
	}
	
	*m = 1;
	*twopm = 2;
	do{
		(*m)++;
		(*twopm) *= 2;
	}while (2*(*twopm) <= n);
	
	if (*twopm != n)
		return(false);
	else
		return(true);
}
/*-------------------------------------------------------------------------*/
/* Perform a 2D FFT inplace given a complex 2D array                       */
/* The direction dir, 1 for forward, -1 for reverse                        */
/* The size of the array (nx,ny)                                           */
/* Return false if there are memory problems or                            */
/*    the dimensions are not powers of 2                                   */
/*-------------------------------------------------------------------------*/
int FFT2D(complex<float> *c, int nx, int ny, int dir)
{
	int m,twopm;	
	double *realC, *imagC;

	/* Transform the rows */
	realC = (double *)malloc(nx * sizeof(double));
	imagC = (double *)malloc(nx * sizeof(double));
	if (realC == NULL || imagC == NULL)
		return(false);
	if (!Powerof2(nx,&m,&twopm) || twopm != nx)
		return(false);
	for (int j=0;j<ny;j++)
	{
		for (int i=0;i<nx;i++)
		{
			realC[i] = (double)real(c[i*ny + j]);
			imagC[i] = (double)imag(c[i*ny + j]);
		}

		FFT(dir,m,realC,imagC);

		for (int i=0;i<nx;i++)
		{
			c[i*ny + j] = complex<float>((float)realC[i],(float)imagC[i]);
		}
	}
	
	/* Transform the columns */
	realC = (double *)realloc(realC, nx * sizeof(double));
	imagC = (double *)realloc(imagC, nx * sizeof(double));
	if (realC == NULL || imagC == NULL)
		return(false);
	if (!Powerof2(ny,&m,&twopm) || twopm != ny)
		return(false);
	for (int i=0;i<nx;i++)
	{
		for (int j=0;j<ny;j++)
		{
			realC[j] = (double)real(c[i*ny + j]);
			imagC[j] = (double)imag(c[i*ny + j]);
		}

		FFT(dir,m,realC,imagC);

		for (int j=0;j<ny;j++)
		{
			c[i*ny + j] = complex<float>((float)realC[j],(float)imagC[j]);
		}
	}
	free(realC);
	free(imagC);
	
	return(true);
}
/*-------------------------------------------------------------------------*/
void FFT3D(complex<float> *c, int nx, int ny, int nz, int dir)
{
	#pragma omp parallel for
	for(int z = 0; z<nz; z++)
	{
		complex <float> *temp_f = (complex<float> *)malloc(nx*ny*sizeof(complex<float>));
		
		for(int i =0; i < nx*ny; i++)
		{
			temp_f[i] = c[i+z*nx*ny];
		}
		
		FFT2D(temp_f,nx,ny,dir);
		for(int i =0; i < nx*ny; i++)
		{
			c[i+z*nx*ny] = temp_f[i];
		}
		free(temp_f);
	}
	#pragma omp barrier

	//int m,twopm;	
	//double *realC, *imagC;

	// Transform the rows
	/*realC = (double *)malloc(nx * sizeof(double));
	imagC = (double *)malloc(nx * sizeof(double));
	if (realC == NULL || imagC == NULL)
		return(false);
	if (!Powerof2(nx,&m,&twopm) || twopm != nx)
		return(false);
	for(int k=0;k<nz;k++)
	{
		for (int j=0;j<ny;j++)
		{
			for (int i=0;i<nx;i++)
			{
				realC[i] = (double)real(c[k*nx*ny + j*nx + i]);
				imagC[i] = (double)imag(c[k*nx*ny + j*nx + i]);
			}

			FFT(dir,m,realC,imagC);

			for (int i=0;i<nx;i++)
			{
				c[k*nx*ny + j*nx + i] = complex<float>((float)realC[i],(float)imagC[i]);
			}
		}
	}
	free(realC);
	free(imagC);
	
	// Transform the columns 
	realC = (double *)malloc(nx * sizeof(double));
	imagC = (double *)malloc(nx * sizeof(double));
	if (realC == NULL || imagC == NULL)
		return(false);
	if (!Powerof2(ny,&m,&twopm) || twopm != ny)
		return(false);
	for(int k=0;k<nz;k++)
	{
		for (int i=0;i<nx;i++)
		{
			for (int j=0;j<ny;j++)
			{
				realC[j] = (double)real(c[k*nx*ny + j*nx + i]);
				imagC[j] = (double)imag(c[k*nx*ny + j*nx + i]);
			}

			FFT(dir,m,realC,imagC);

			for (int j=0;j<ny;j++)
			{
				c[k*nx*ny + j*nx + i] = complex<float>((float)realC[j],(float)imagC[j]);
			}
		}
	}
	free(realC);
	free(imagC);*/

	//Transform the slices
	#pragma omp parallel for
	for(int i=0;i<nx*ny;i++)
	{
		double *realC = (double *)malloc(nz * sizeof(double));
		double *imagC = (double *)malloc(nz * sizeof(double));

		int m,twopm;		
		Powerof2(nz,&m,&twopm);			

		for(int k=0;k<nz;k++)
		{
			realC[k] = (double)real(c[k*nx*ny + i]);
			imagC[k] = (double)imag(c[k*nx*ny + i]);
		}

		FFT(dir,m,realC,imagC);

		for(int k=0;k<nz;k++)
		{
			c[k*nx*ny + i] = complex<float>((float)realC[k],(float)imagC[k]);
		}
		free(realC);
		free(imagC);
	}
	#pragma omp barrier

}
/*-------------------------------------------------------------------------*/
/*void FFTW2D(complex<float> *input, int m, int n, int type)
{
	int Nx = ceil(log((double)m)/log(2.0));
	int Ny = ceil(log((double)n)/log(2.0));
	Nx = pow(2.0,Nx);
	Ny = pow(2.0,Ny);

	if(Nx>=Ny)
		Ny=Nx;
	else
		Nx=Ny;

	//fftwf_init_threads();
	//int nthreads = 1;
	//fftwf_plan_with_nthreads (nthreads) ; 
	
	fftwf_plan pFFT;
	fftwf_complex *in = (fftwf_complex*)fftwf_malloc(Nx*Ny*sizeof(fftwf_complex));
	fftwf_complex *out = (fftwf_complex*)fftwf_malloc(Nx*Ny*sizeof(fftwf_complex));

	for(int i=0;i<m*n;i++)
	{
		in[i][0] = (real(input[i]));
		in[i][1] = (imag(input[i]));
		out[i][0] = 0.0;
		out[i][1] = 0.0;
	}
	
	if(type == -1)	//forward FFT
	{
		pFFT=fftwf_plan_dft_2d(m,n,in,out,FFTW_FORWARD,FFTW_ESTIMATE);
		fftwf_execute(pFFT);
	}
	else if(type == 1)	//backword FFT
	{
		pFFT=fftwf_plan_dft_2d(m,n,in,out,FFTW_BACKWARD,FFTW_ESTIMATE);
		fftwf_execute(pFFT);
	}

	for(int i=0;i<m*n;i++)
	{
		input[i] = complex<float>((float)out[i][0],(float)out[i][1]);
	}

	fftwf_destroy_plan(pFFT);
	fftwf_cleanup();
	fftwf_cleanup_threads() ; 
	fftwf_free(in);
	fftwf_free(out);
}

void FFTW3D(complex<float> *input, int m, int n, int z, int type)
{
	//fftwf_init_threads();
	//int nthreads = 1;
	//fftwf_plan_with_nthreads (nthreads) ; 
	
	fftwf_plan pFFT;
	fftwf_complex *in = (fftwf_complex*)fftwf_malloc(m*n*z*sizeof(fftwf_complex));
	fftwf_complex *out = (fftwf_complex*)fftwf_malloc(m*n*z*sizeof(fftwf_complex));

	for(int i=0;i<m*n*z;i++)
	{
		in[i][0] = (real(input[i]));
		in[i][1] = (imag(input[i]));
		out[i][0] = 0.0;
		out[i][1] = 0.0;
	}
	
	if(type == -1)	//forward FFT
	{
		pFFT=fftwf_plan_dft_3d(m,n,z,in,out,FFTW_FORWARD,FFTW_ESTIMATE);
		fftwf_execute(pFFT);
	}
	else if(type == 1)	//backword FFT
	{
		pFFT=fftwf_plan_dft_3d(m,n,z,in,out,FFTW_BACKWARD,FFTW_ESTIMATE);
		fftwf_execute(pFFT);
	}

	for(int i=0;i<m*n*z;i++)
	{
		input[i] = complex<float>((float)out[i][0],(float)out[i][1]);
	}

	fftwf_destroy_plan(pFFT);
	fftwf_cleanup();
	fftwf_cleanup_threads() ; 
	fftwf_free(in);
	fftwf_free(out);
}*/
void cuFFT1D(cufftComplex *ImgArray, int size, int batch, int dir)
{
	//Create a 1D FFT plan. 
	cufftHandle plan;
	cufftPlan1d(&plan, size, CUFFT_C2C, batch);

	if (dir == -1)
	{
		// Use the CUFFT plan to transform the signal out of place. 
		cufftExecC2C(plan, (cufftComplex *)ImgArray, (cufftComplex *)ImgArray, CUFFT_FORWARD);

//		cudaThreadSynchronize();
	}
	else if (dir == 1)
	{
		// Note: idata != odata indicates an out-of-place transformation to CUFFT at execution time. 
		//Inverse transform the signal in place 
		cufftExecC2C(plan, (cufftComplex *)ImgArray, (cufftComplex *)ImgArray, CUFFT_INVERSE);

		cudaThreadSynchronize();

		int grid = (size + 1024 - 1) / 1024;
		int block = 32*32;
		scaleFFT1D << <grid, block >> >(ImgArray, size, 1.f / size);
	}
	else if (dir == 2)
	{
		// Note: idata != odata indicates an out-of-place transformation to CUFFT at execution time. 
		//Inverse transform the signal in place 
		cufftExecC2C(plan, (cufftComplex *)ImgArray, (cufftComplex *)ImgArray, CUFFT_INVERSE);

		cudaThreadSynchronize();
	}

	// Destroy the CUFFT plan.
	cufftDestroy(plan);
}

void cuFFT1D_test(cufftComplex *d_odata, cufftReal *d_idata, int size, int batch, int dir)
{
	cufftHandle plan;
	if (dir == -1)
	{
		cufftPlan1d(&plan, size, CUFFT_R2C, batch);
		cufftExecR2C(plan, (cufftReal*)d_idata, (cufftComplex*)d_odata);

	}
	else if (dir == 1)
	{		
		cufftPlan1d(&plan, size, CUFFT_C2R, batch);
		cufftExecC2R(plan, (cufftComplex*)d_odata, (cufftReal*)d_idata);

		/*int grid = (size + 1024 - 1) / 1024;
		int block = 32 * 32;
		scaleFFT1D << <grid, block >> >(ImgArray, size, 1.f / size);*/
	}

	// Destroy the CUFFT plan.
	cufftDestroy(plan);
}

void cuFFT2D(cufftComplex *ImgArray, int sizeX, int sizeY, int dir)
{
	//Create a 2D FFT plan. 
	cufftHandle plan;
	cufftPlan2d(&plan, sizeX, sizeY, CUFFT_C2C);
	/*const int NRANK = 2;
	const int BATCH = 10;

	int n [NRANK] = {sizeX, sizeY} ;
	cufftPlanMany(&plan , 2 , n ,
					NULL , 1 , sizeX*sizeY ,
					NULL , 1 , sizeX*sizeY ,
					CUFFT_C2C , BATCH );*/

	
	if(dir == -1)
	{
		// Use the CUFFT plan to transform the signal out of place. 
		cufftExecC2C(plan, (cufftComplex *)ImgArray, (cufftComplex *)ImgArray, CUFFT_FORWARD);

		cudaThreadSynchronize();
	}
	else if(dir == 1)
	{
		// Note: idata != odata indicates an out-of-place transformation to CUFFT at execution time. 
		//Inverse transform the signal in place 
		cufftExecC2C(plan, (cufftComplex *)ImgArray, (cufftComplex *)ImgArray, CUFFT_INVERSE);
		
		cudaThreadSynchronize();

		int blocksInX = (sizeX+32-1)/32;
		int blocksInY = (sizeY+32-1)/32;
		dim3 grid(blocksInX, blocksInY);
		dim3 block(32, 32);
		scaleFFT2D<<<grid,block>>>(ImgArray,sizeX,sizeY, 1.f/(sizeX*sizeY));
	}

	// Destroy the CUFFT plan.
	cufftDestroy(plan);
}

void cuFFT2D_Batch(cufftComplex *ImgArray, int sizeX, int sizeY, int sizeZ, int dir)
{
	//Create a 2D FFT plan. 
	cufftHandle plan;
	const int NRANK = 2;
	const int BATCH = sizeZ;

	int n [NRANK] = {sizeX, sizeY} ;
	cufftPlanMany(&plan , 2 , n ,
	NULL , 1 , sizeX*sizeY ,
	NULL , 1 , sizeX*sizeY ,
	CUFFT_C2C , BATCH );


	if (dir == -1)
	{
		// Use the CUFFT plan to transform the signal out of place. 
		cufftExecC2C(plan, (cufftComplex *)ImgArray, (cufftComplex *)ImgArray, CUFFT_FORWARD);

		cudaThreadSynchronize();
	}
	else if (dir == 1)
	{
		// Note: idata != odata indicates an out-of-place transformation to CUFFT at execution time. 
		//Inverse transform the signal in place 
		cufftExecC2C(plan, (cufftComplex *)ImgArray, (cufftComplex *)ImgArray, CUFFT_INVERSE);

		cudaThreadSynchronize();

		int blocksInX = (sizeX + 32 - 1) / 32;
		int blocksInY = (sizeY + 32 - 1) / 32;
		dim3 grid(blocksInX, blocksInY);
		dim3 block(32, 32);
		//scaleFFT2D << <grid, block >> >(ImgArray, sizeX, sizeY, 1.f / (sizeX*sizeY));
	}

	// Destroy the CUFFT plan.
	cufftDestroy(plan);
}

void cuFFT2Dz(cufftDoubleComplex *ImgArray, int sizeX, int sizeY, int dir)
{
	//Create a 2D FFT plan. 
	cufftHandle plan;
	cufftPlan2d(&plan, sizeX, sizeY, CUFFT_Z2Z);	//cufftSafeCall(cufftPlan2d(&plan, sizeX, sizeY, CUFFT_C2C));

	
	if(dir == -1)
	{
		// Use the CUFFT plan to transform the signal out of place. 
		cufftExecZ2Z(plan, (cufftDoubleComplex *)ImgArray, (cufftDoubleComplex *)ImgArray, CUFFT_FORWARD);
	}
	else if(dir == 1)
	{
		// Note: idata != odata indicates an out-of-place transformation to CUFFT at execution time. 
		//Inverse transform the signal in place 
		cufftExecZ2Z(plan, (cufftDoubleComplex *)ImgArray, (cufftDoubleComplex *)ImgArray, CUFFT_INVERSE);
		
		int blocksInX = (sizeX+32-1)/32;
		int blocksInY = (sizeY+32-1)/32;
		dim3 grid(blocksInX, blocksInY);
		dim3 block(32, 32);
		scaleFFT2Dz<<<grid,block>>>(ImgArray,sizeX,sizeY, 1.f/(sizeX*sizeY));
	}

	// Destroy the CUFFT plan.
	cufftDestroy(plan);
}

void cuFFT3D(cufftComplex *ImgArray, int sizeX, int sizeY, int sizeZ, int dir)
{
	//Create a 3D FFT plan. 
	cufftHandle plan;
	cufftPlan3d(&plan, sizeX, sizeY, sizeZ, CUFFT_C2C);
	//int batch = 10;
	//int dims[] = {sizeZ, sizeY, sizeX}; // reversed order
	//cufftPlanMany(&plan, 3, dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batch);

	if(dir == -1)
	{
		// Use the CUFFT plan to transform the signal out of place. 
		cufftExecC2C(plan, (cufftComplex *)ImgArray, (cufftComplex *)ImgArray, CUFFT_FORWARD);
	}
	else if(dir == 1)
	{
		// Note: idata != odata indicates an out-of-place transformation to CUFFT at execution time. 
		//Inverse transform the signal in place 
		cufftExecC2C(plan, (cufftComplex *)ImgArray, (cufftComplex *)ImgArray, CUFFT_INVERSE);

		int blocksInX = (sizeX+8-1)/8;
		int blocksInY = (sizeY+8-1)/8;
		int blocksInZ = (sizeZ+8-1)/8;
		dim3 grid(blocksInX, blocksInY, blocksInZ);
		dim3 block(8, 8, 8);

		scaleFFT3D<<<grid,block>>>(ImgArray,sizeX,sizeY,sizeZ, 1.f/(sizeX*sizeY*sizeZ));
	}
	
	// Destroy the CUFFT plan.
	cufftDestroy(plan);
}

__global__ void scaleFFT1D(cufftComplex *cu_F, int nx, float scale)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	float tempX, tempY;
	if (xIndex<nx)
	{
		tempX = cu_F[xIndex].x * scale;
		tempY = cu_F[xIndex].y * scale;
		cu_F[xIndex].x = tempX;
		cu_F[xIndex].y = tempY*(-1);
	}
}

__global__ void scaleFFT2D(cufftComplex *cu_F, int nx, int ny, float scale)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int index_out;
	float tempX, tempY;

	if((xIndex<nx) && (yIndex<ny))
	{
		index_out = xIndex + nx*yIndex;
		tempX = cu_F[index_out].x * scale;
		tempY = cu_F[index_out].y * scale;
		cu_F[index_out].x = tempX;
		cu_F[index_out].y = tempY*(-1);
	}
}

__global__ void scaleFFT2Dz(cufftDoubleComplex *cu_F, int nx, int ny, double scale)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	
	if((xIndex<nx) && (yIndex<ny))
	{
		unsigned int index_out = xIndex + nx*yIndex;
		double tempX = cu_F[index_out].x * scale;
		double tempY = cu_F[index_out].y * scale;
		cu_F[index_out].x = (double)tempX;
		cu_F[index_out].y = (double)tempY*(-1);
	}
}


__global__ void scaleFFT2DReal(float *cu_F, int nx, int ny, float scale)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int index_out;

	if((xIndex<nx) && (yIndex<ny))
	{
		index_out = xIndex + nx*yIndex;
		cu_F[index_out] = cu_F[index_out] * scale;
	}
}


__global__ void scaleFFT3D(cufftComplex *cu_F, int nx, int ny, int nz, float scale)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;
	unsigned int index_out;
	float tempX, tempY;

	if((xIndex<nx) && (yIndex<ny) && (zIndex<nz))
	{
		index_out = xIndex + nx*yIndex + nx*ny*zIndex;
		tempX = cu_F[index_out].x * scale;
		tempY = cu_F[index_out].y * scale;
		cu_F[index_out].x = tempX;
		cu_F[index_out].y = tempY;
	}
}



void cuFFT_Real(cufftComplex *freq, float *img, const unsigned int Nx, const unsigned int Ny, int dir)
{
	size_t   Ny_pad = ((Ny >> 1) + 1);
	//size_t   Ny_pad = Ny;
	size_t   N_pad  = Nx * Ny_pad;
	size_t   stride = 2*Ny_pad; // stride on real data	

// step 1: transfer data to device, sequence by sequence
	cufftReal *img_plane;
	cudaMalloc((void**)&img_plane, sizeof(cufftReal)*Nx*Ny);
	cudaMemcpy(img_plane, img, sizeof(cufftReal)*Nx*Ny, cudaMemcpyDeviceToDevice);

	cufftComplex *FFT_plane;
	cudaMalloc((void**)&FFT_plane, sizeof(cufftComplex)*Nx*Ny_pad);
	cudaMemcpy(FFT_plane, freq, sizeof(cufftComplex)*Nx*Ny_pad, cudaMemcpyDeviceToDevice);

// step 2: Create a 2D FFT plan. 
// step 3: Use the CUFFT plan to transform the signal in-place.
	cufftHandle plan;
	cufftResult flag;
	if(dir == -1)
	{
		cufftPlan2d(&plan, Nx, Ny   , CUFFT_R2C );
		flag = cufftExecR2C( plan, (cufftReal*)img_plane, (cufftComplex*)FFT_plane );
		
		cudaMemcpy(freq, FFT_plane, sizeof(cufftComplex)*Nx*Ny_pad, cudaMemcpyDeviceToDevice);
	}
	else if(dir == 1)
	{
		cufftPlan2d(&plan, Nx, Ny, CUFFT_C2R );
		flag = cufftExecC2R( plan, (cufftComplex*)FFT_plane, (cufftReal*)img_plane );

		int blocksInX = (Nx+32-1)/32;
		int blocksInY = (Ny+32-1)/32;
		dim3 grid(blocksInX, blocksInY);
		dim3 block(32, 32);
		scaleFFT2DReal<<<grid,block>>>(img_plane, Nx, Ny, (float)1.f/(Nx*Ny));

		cudaMemcpy(img, img_plane, sizeof(cufftReal)*Nx*Ny, cudaMemcpyDeviceToDevice);
	}

	if (flag != CUFFT_SUCCESS)	printf("2D: cufftExec fails\n");

// make sure that all threads are done
	cudaThreadSynchronize();

// step 4: copy data to host
	//cudaMemcpy(h_idata, d_idata, sizeof(cufftComplex)*N_pad, cudaMemcpyDeviceToHost);

// Destroy the CUFFT plan.
	cufftDestroy(plan);
	cudaFree(FFT_plane);
	cudaFree(img_plane);
}