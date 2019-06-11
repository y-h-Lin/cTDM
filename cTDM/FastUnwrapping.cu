#include <helper_cuda.h>
#include "FastUnwrapping.cuh"
using namespace std;

void FastUnwrapping(float *ImgSrc, float *ImgDst, int sizeX, int sizeY)
{
	PhaseNormalize(ImgSrc, sizeX, sizeY);

	int blocksInX = (sizeX+32-1)/32;
	int blocksInY = (sizeY+32-1)/32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);

	int newNx = sizeX<<1, newNy = sizeY<<1;
	int blocksInX2 = (newNx+32-1)/32;
	int blocksInY2 = (newNy+32-1)/32;
	dim3 grid2(blocksInX2, blocksInY2);
	dim3 block2(32, 32);

	int blocksInX3 = (newNx/2 + 32 - 1) / 32;
	int blocksInY3 = (newNy/2 + 32 - 1) / 32;
	dim3 grid3(blocksInX3, blocksInY3);
	dim3 block3(32, 32);

	float        *CosPhi, *SinPhi, *ImgDif, *ImgCont;
	cufftComplex *term1 , *term2, *PhiP ;
	cudaMalloc((void **)&SinPhi   ,sizeof(float       )*newNx*newNy);
	cudaMalloc((void **)&CosPhi   ,sizeof(float       )*newNx*newNy);
	cudaMalloc((void **)&ImgDif   ,sizeof(float       )*sizeX*sizeY);
	cudaMalloc((void **)&ImgCont  ,sizeof(float       )*sizeX*sizeY);
	cudaMalloc((void **)&term1    ,sizeof(cufftComplex)*newNx*newNy);
	cudaMalloc((void **)&term2    ,sizeof(cufftComplex)*newNx*newNy);
	cudaMalloc((void **)&PhiP     ,sizeof(cufftComplex)*newNx*newNy);

	s_unwrap = clock();
	//generate the symmertic matrix, SIN map and COS map from normalized phase map
	makeSimmetric<<<grid,block>>>(SinPhi, CosPhi, ImgSrc, newNx, newNy, sizeX, sizeY);
	
	//forward FFT
	Real2Complex<<<grid2,block2>>>(term1, SinPhi, newNx, newNy);
	Real2Complex<<<grid2,block2>>>(term2, CosPhi, newNx, newNy);
	//cuFFT2D(term1, newNx, newNy,-1);
	//cuFFT2D(term2, newNx, newNy,-1);
	cufftExecC2C(plan_2D_C2C_FORWARD_FTUP, term1, term1, CUFFT_FORWARD);
	cufftExecC2C(plan_2D_C2C_FORWARD_FTUP, term2, term2, CUFFT_FORWARD);
	//shift FFT 2D
	cuFFT2Dshift<<<grid3,block3>>>(term1, newNx, newNy);
	cuFFT2Dshift<<<grid3,block3>>>(term2, newNx, newNy);
	

	//time the parabolic matrix based on the distance map
	MultiplyPosition<<<grid2,block2>>>(term1, newNx, newNy);
	MultiplyPosition<<<grid2,block2>>>(term2, newNx, newNy);
	transferQuarter<<<grid,block>>>(ImgCont, term1, sizeX, sizeY, newNx, newNy);
	
	//inverse FFT
	cuFFT2Dshift<<<grid3,block3>>>(term1, newNx, newNy);
	cuFFT2Dshift<<<grid3,block3>>>(term2, newNx, newNy);
	//cuFFT2D(term1, newNx, newNy, 1);
	//cuFFT2D(term2, newNx, newNy, 1);
	cufftExecC2C(plan_2D_C2C_INVERSE_FTUP, term1, term1, CUFFT_INVERSE);
	scaleFFT2D << <grid2, block2 >> >(term1, newNx, newNy, 1.f / (newNx*newNy));
	cufftExecC2C(plan_2D_C2C_INVERSE_FTUP, term2, term2, CUFFT_INVERSE);
	scaleFFT2D << <grid2, block2 >> >(term2, newNx, newNy, 1.f / (newNx*newNy));
	
	
	//times the SIN & COS map, respectively
	MultiplyComplex<<<grid2,block2>>>(term1, CosPhi, newNx, newNy);
	MultiplyComplex<<<grid2,block2>>>(term2, SinPhi, newNx, newNy);
	
	//forward FFT
	//cuFFT2D(term1, newNx, newNy,-1);
	//cuFFT2D(term2, newNx, newNy,-1);
	cufftExecC2C(plan_2D_C2C_FORWARD_FTUP, term1, term1, CUFFT_FORWARD);
	cufftExecC2C(plan_2D_C2C_FORWARD_FTUP, term2, term2, CUFFT_FORWARD);
	cuFFT2Dshift<<<grid3,block3>>>(term1, newNx, newNy);
	cuFFT2Dshift<<<grid3,block3>>>(term2, newNx, newNy);

	//divide the parabolic matrix based on the distance map
	MultiplyInvPosition<<<grid2,block2>>>(term1, newNx, newNy);
	MultiplyInvPosition<<<grid2,block2>>>(term2, newNx, newNy);	

	//obtain the difference between Term1 and Term2
	SubtractComplex<<<grid2,block2>>>(PhiP, term1, term2, newNx, newNy);

	//inverse FFT
	cuFFT2Dshift<<<grid2,block2>>>(PhiP, newNx, newNy);
	//cuFFT2D(PhiP, newNx, newNy, 1);	
	cufftExecC2C(plan_2D_C2C_INVERSE_FTUP, PhiP, PhiP, CUFFT_INVERSE);
	scaleFFT2D << <grid2, block2 >> >(PhiP, newNx, newNy, 1.f / (newNx*newNy));
	
	//copy the quarter matrix as a float-type matrix
	transferQuarter<<<grid,block>>>(ImgCont, PhiP, sizeX, sizeY, newNx, newNy);

	//unwrapping the phase image 
	ObtainUnwrap<<<grid,block>>>(ImgDst, ImgDif, ImgCont, ImgSrc, sizeX, sizeY);
	e_unwrap = clock(); unwrap_time += e_unwrap - s_unwrap;

	//DeviceMemOut2("C:\\HilbertImg\\Data\\ImgCont.1024.1024.raw",ImgCont, sizeX, sizeY);
	//DeviceMemOut2("C:\\HilbertImg\\Data\\ImgDif.1024.1024.raw",ImgDif, sizeX, sizeY);
	int maxRec=0;
	for(int i=0; i<maxRec; i++)
	{
		printf("recurrence:%2d\n",i);
		ObtainUnwrap<<<grid,block>>>(ImgDst, ImgDif, ImgCont, ImgDst, sizeX, sizeY);
	}

	subtractMin(ImgDst, sizeX, sizeY);

	checkCudaErrors(cudaFree(SinPhi   ));
	checkCudaErrors(cudaFree(CosPhi   ));
	checkCudaErrors(cudaFree(ImgDif   ));
	checkCudaErrors(cudaFree(ImgCont  ));
	checkCudaErrors(cudaFree(term1    ));
	checkCudaErrors(cudaFree(term2    ));
	checkCudaErrors(cudaFree(PhiP     ));
}
//--------------------------------------------------------------------------------------
void PhaseNormalize(float *src, int width, int height)
{
	int blocksInX = (width+32-1)/32;
	int blocksInY = (height+32-1)/32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);

	float h_max_out, h_min_out;
	findMinMax(h_min_out, h_max_out, src, width, height);

	RescalePhaseMap<<<grid,block>>>(src, h_min_out, h_max_out, width, height);
}
//--------------------------------------------------------------------------------------
__global__ void RescalePhaseMap(float *src, float min, float max, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+j*width;

	if(i<width &&  j<height)
	{
		src[idx] = 2*M_PI * (src[idx]-min)/(max-min);
	}
}
//--------------------------------------------------------------------------------------
__global__ void makeSimmetric(float *sinMap, float *cosMap, float *src, 
							  int Nx2, int Ny2, int Nx1, int Ny1)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx, idx1, idx2, idx3, idx4;
	float sinP, cosP;
	float srcValue;

	if(i<Nx1 && j<Ny1)
	{
		idx1 = (Nx2-i-1) + (j)       * Nx2;	//right-top
        idx2 =         i + (Ny2-j-1) * Nx2;	//left-bottom
        idx3 = (Nx2-i-1) + (Ny2-j-1) * Nx2;	//right-bottom
        idx4 =         i +         j * Nx2;	//left-top
        idx  =         i +         j * Nx1;

		srcValue  = src[idx];

		sincos(srcValue, &sinP, &cosP);
		sinMap[idx4] = sinP;	cosMap[idx4] = cosP;
        sinMap[idx3] = sinP;	cosMap[idx3] = cosP;
        sinMap[idx2] = sinP;	cosMap[idx2] = cosP;
        sinMap[idx1] = sinP;	cosMap[idx1] = cosP;
	}
}
//--------------------------------------------------------------------------------------
__global__ void estimateCosArray(float *dst, float *src, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+width*j;

	if(i<width && j<height)
	{
		dst[idx] = cosf(src[idx]);
	}
}
//--------------------------------------------------------------------------------------
__global__ void estimateSinArray(float *dst, float *src, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+width*j;

	if(i<width && j<height)
	{
		dst[idx] = sinf(src[idx]);
	}
}
//--------------------------------------------------------------------------------------
__global__ void Real2Complex(cufftComplex *dst, float *src, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+width*j;

	if(i<width && j<height)
	{
		dst[idx].x = src[idx];
		dst[idx].y = 0;
	}
}
//--------------------------------------------------------------------------------------
__global__ void Complex2Real(float *dst, cufftComplex *src, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+width*j;

	if(i<width && j<height)
	{
		dst[idx] = src[idx].x;		
	}
}
//--------------------------------------------------------------------------------------
__global__ void transferQuarter  (float *dst, cufftComplex *src, int Nx2, int Ny2, int Nx1, int Ny1)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx1 = i+Nx1*j;
	unsigned int idx2 = i+Nx2*j;

	if(i<Nx2 && j<Ny2)
	{
		dst[idx2] = src[idx1].x;		
	}
}
//--------------------------------------------------------------------------------------
__global__ void RecoveryComplex(cufftComplex *src, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+width*j;
	int xc = width>>1;
	int yc = height>>1;
	int halfSize = width * yc + 1;

	if(idx > halfSize)
	{		
		src[idx].x =  src[idx%halfSize].x;
		src[idx].y = -src[idx%halfSize].y;
	}
}
//--------------------------------------------------------------------------------------
__global__ void MultiplyComplex(cufftComplex *src, float *mask, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+width*j;

	if(i<width && j<height)
	{
		src[idx].x *= mask[idx];
		src[idx].y *= mask[idx];
		//src[idx].y = 0.0;
	}
}
//--------------------------------------------------------------------------------------
__global__ void MultiplyPosition(cufftComplex *src, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+width*j;
	float con=1;
	int xc = width>>1;
	int yc = height>>1;

	if(i<width && j<height)
	{
		con = ((i-xc)*(i-xc))+((j-yc)*(j-yc));
		src[idx].x *= con;
		src[idx].y *= con;
		//src[idx].y = 0.0;
	}
}
//--------------------------------------------------------------------------------------
__global__ void MultiplyInvPosition(cufftComplex *src, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+width*j;
	float con;
	int xc = width>>1;
	int yc = height>>1;

	if(i<width && j<height)
	{
		con = ((i-xc)*(i-xc))+((j-yc)*(j-yc));

		if(con == 0.0) 
		{
			src[idx].x = 0;
			src[idx].y = 0;
		}
		else
		{
			src[idx].x /= con;
			src[idx].y /= con;
			//src[idx].y = 0.0;
		}
	}
}
//--------------------------------------------------------------------------------------
__global__ void MultiplyArray(float *src, float *mask, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+width*j;

	if(i<width && j<height)
	{
		src[idx] *= mask[idx];
	}
}
//--------------------------------------------------------------------------------------
__global__ void SubtractComplex(cufftComplex *dst, cufftComplex *src1, cufftComplex *src2, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+width*j;

	if(i<width && j<height)
	{
		dst[idx].x = (src1[idx].x - src2[idx].x);
		dst[idx].y = (src1[idx].y - src2[idx].y);		
	}
}
//--------------------------------------------------------------------------------------
__global__ void ObtainUnwrap(float *dst, float *diff, float *cont, float *src, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+width*j;

	if(i<width && j<height)
	{
		diff[idx] = (2*M_PI)*roundf( (cont[idx]-src[idx])*(0.5 * M_1_PI));
		dst [idx] = src[idx] + diff[idx];
	}
}
//--------------------------------------------------------------------------------------
void subtractMin(float *src, int width, int height)
{
	float h_max_out, h_min_out;
	findMinMax(h_min_out, h_max_out, src, width, height);

	int blocksInX = (width+32-1)/32;
	int blocksInY = (height+32-1)/32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);

	PhaseMapSubMin<<<grid,block>>>(src, h_min_out, width, height);
}
//--------------------------------------------------------------------------------------
__global__ void PhaseMapSubMin(float *src, float min, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+j*width;

	if(i<width &&  j<height)
	{
		src[idx] -= min;
	}
}
//--------------------------------------------------------------------------------------
void DeviceMemOut2(char *path, float *arr, int sizeX, int sizeY)
{
	int size = sizeX*sizeY;
	float *temp = (float *)malloc(size*sizeof(float));
	cudaMemcpy(temp, arr, size*sizeof(float),cudaMemcpyDeviceToHost);

	FILE *fp;
	fp = fopen(path,"wb");
	fwrite(temp,size,sizeof(float),fp);
	fclose(fp);
	free(temp);
}
//--------------------------------------------------------------------------------------
void findMinMax(float &h_min_out, float &h_max_out, float *arr, int Nx, int Ny)
{
	int points = Nx * Ny;
    int logPoints = ceil(log((float)points)/log((float)2));
    int sizePow = logPoints;
    int size = pow((float)2, sizePow);
    int numThreads = 1024;
    int numBlocks = size / numThreads;

    float *d_out;
    float *d_max_out, *d_min_out;

    checkCudaErrors(cudaMalloc((void **) &d_out, numBlocks * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_max_out, sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_min_out, sizeof(float)));
	
	//find max
    cudaDeviceSynchronize();
    reduce_max_kernel<<<numBlocks, numThreads, sizeof(float)*numThreads>>>(d_out, arr, points);

    cudaDeviceSynchronize();
    reduce_max_kernel2<<<1, numBlocks>>>(d_max_out, d_out);

	//find min
	cudaDeviceSynchronize();
    reduce_min_kernel<<<numBlocks, numThreads, sizeof(float)*numThreads>>>(d_out, arr, points);

    cudaDeviceSynchronize();
    reduce_min_kernel2<<<1, numBlocks>>>(d_min_out, d_out);

    //float h_out_max, h_out_min;
    checkCudaErrors(cudaMemcpy(&h_max_out, d_max_out, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&h_min_out, d_min_out, sizeof(float), cudaMemcpyDeviceToHost));

    //printf("max:%3.3f\tmin:%3.3f\n", h_out_max, h_out_min);

    checkCudaErrors(cudaFree(d_max_out));
	checkCudaErrors(cudaFree(d_min_out));
    checkCudaErrors(cudaFree(d_out));
}
//--------------------------------------------------------------------------------------
__global__ void reduce_max_kernel(float *d_out, const float *d_logLum, int size)
{
    int tid         = threadIdx.x;                              // Local thread index
    int myId        = blockIdx.x * blockDim.x + threadIdx.x;    // Global thread index

    extern __shared__ float temp[];

    // --- Loading data to shared memory. All the threads contribute to loading the data to shared memory.
    temp[tid] = (myId < size) ? d_logLum[myId] : -FLT_MAX;

    // --- Before going further, we have to make sure that all the shared memory loads have been completed
    __syncthreads();

    // --- Reduction in shared memory. Only half of the threads contribute to reduction.
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s) { temp[tid] = fmaxf(temp[tid], temp[tid + s]); }
        // --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = temp[0];
    }
}
//--------------------------------------------------------------------------------------
__global__ void reduce_max_kernel2(float *d_out, float *d_in)
{
    // Reduce log Lum with Max Operator
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
	{
        if (tid < s) {	d_in[myId] = fmaxf(d_in[myId + s], d_in[myId]);	}
        __syncthreads();   
    }

    if (tid == 0) {	d_out[0] = d_in[0];	}
}
//--------------------------------------------------------------------------------------
__global__ void reduce_min_kernel(float *d_out, const float *d_logLum, int size)
{
    int tid         = threadIdx.x;                              // Local thread index
    int myId        = blockIdx.x * blockDim.x + threadIdx.x;    // Global thread index

    extern __shared__ float temp[];

    // --- Loading data to shared memory. All the threads contribute to loading the data to shared memory.
    temp[tid] = (myId < size) ? d_logLum[myId] : -FLT_MIN;

    // --- Before going further, we have to make sure that all the shared memory loads have been completed
    __syncthreads();

    // --- Reduction in shared memory. Only half of the threads contribute to reduction.
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s) { temp[tid] = fminf(temp[tid], temp[tid + s]); }
        // --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = temp[0];
    }
}
//--------------------------------------------------------------------------------------
__global__ void reduce_min_kernel2(float *d_out, float *d_in)
{
    // Reduce log Lum with Max Operator
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
	{
        if (tid < s) {	d_in[myId] = fminf(d_in[myId + s], d_in[myId]);	}
        __syncthreads();   
    }

    if (tid == 0) {	d_out[0] = d_in[0];	}
}