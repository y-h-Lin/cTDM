// Ref1: http ://fourier.eng.hmc.edu/e161/lectures/dct/node2.html
// Ref2: 

#include "DCT_Unwrapping.cuh"
#include "dct8x8_kernel2.cuh"
using namespace std;
const int TILE_DIM = 32;
const int BLOCK_ROWS = 32;
//--------------------------------------------------------------------------------------
void DCT_UWLS_Unwrapped(float *ImgDst, float *ImgSrc, int sizeX, int sizeY)
{
	ROI Size;
	Size.width  = sizeX;
	Size.height = sizeY;

	int blocksInX = (sizeX+32-1)/32;
	int blocksInY = (sizeY+32-1)/32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);
	
	float *LaplaceArray, *outX, *outY, *inX, *inY;
	cudaMalloc((void **)&LaplaceArray ,sizeof(float)*Size.width*Size.height);	
	cudaMalloc((void **)&inX, sizeof(float)*Size.width*Size.height);
	cudaMalloc((void **)&inY, sizeof(float)*Size.width*Size.height);
	//cudaMalloc((void **)&outX, sizeof(float)*Size.width*Size.height);
	//cudaMalloc((void **)&outY, sizeof(float)*Size.width*Size.height);
	cudaMemset(inX, 0.0, sizeof(float)*Size.width*Size.height);
	cudaMemset(inY, 0.0, sizeof(float)*Size.width*Size.height);
	//cudaMemset(outX, 0.0, sizeof(float)*Size.width*Size.height);
	//cudaMemset(outY, 0.0, sizeof(float)*Size.width*Size.height);
	
	s_unwrap = clock();
	NumericDerivative1 << <grid, block >> >(inX, inY, ImgSrc, Size.width, Size.height);
	//TrigonometricF << <grid, block >> >(inX, inY, Size.width, Size.height);
	//NumericDerivative2 << <grid, block >> >(outX, outY, inX, inY, Size.width, Size.height);
	//TrigonometricF << <grid, block >> >(outX, outY, Size.width, Size.height);
	//SumDerivative << <grid, block >> >(LaplaceArray, outX, outY, ImgSrc, Size.width, Size.height);
	SumDerivative << <grid, block >> >(LaplaceArray, inX, inY, ImgSrc, Size.width, Size.height);
	//DeviceMemOutDCT("src.256.256.raw", ImgSrc, sizeX, sizeY);
	//LaplaceWithPU << <grid, block >> >(LaplaceArray, ImgSrc, Size.width, Size.height);
	e_unwrap = clock(); unwrap_time += e_unwrap - s_unwrap;

    //allocate device memory
    float *dst;
    size_t DeviceStride;
    checkCudaErrors(cudaMallocPitch((void **)&dst, &DeviceStride, Size.width * sizeof(float), Size.height));	

	
	
    //perform block-wise DCT processing and benchmarking
	myDCT(dst, LaplaceArray, sizeX, sizeY);
	//DeviceMemOutDCT("DCT.256.256.raw", dst, sizeX, sizeY);
	//solve the Poission's equation
	s_unwrap = clock();
	devConstant<<<grid,block>>>(dst, sizeX, sizeY);
	e_unwrap = clock(); unwrap_time += e_unwrap - s_unwrap;
	//DeviceMemOutDCT("DCT2.256.256.raw", dst, sizeX, sizeY);
    //perform block-wise IDCT processing
	myIDCT(ImgDst, dst, sizeX, sizeY);
	//DeviceMemOutDCT("IDCT.256.256.raw", ImgDst, sizeX, sizeY);
	

    //clean up memory
    checkCudaErrors(cudaFree(LaplaceArray));
	checkCudaErrors(cudaFree(inX));
	checkCudaErrors(cudaFree(inY));
	//checkCudaErrors(cudaFree(outX));
	//checkCudaErrors(cudaFree(outY));
	checkCudaErrors(cudaFree(dst));
}
//--------------------------------------------------------------------------------------
void myDCT(float *ImgDst, float *ImgSrc, int sizeX, int sizeY)
{
	cufftComplex *dSrc;
	cudaMalloc((void **)&dSrc, sizeof(cufftComplex)*(sizeX*2)*sizeY);

	//copy the floating array to cufftComplex
	dim3 dimGrid(sizeX / TILE_DIM, sizeY / TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);	

	int blocksX = (sizeX + 32 - 1) / 32;
	int blocksY = (sizeY + 32 - 1) / 32;
	dim3 grid(blocksX, blocksY);
	dim3 block(32, 32);
	
	//cufftHandle plan;
	//cufftPlan1d(&plan, sizeX * 2, CUFFT_C2C, sizeY);

	s_unwrap = clock();
	//DCT on x-axis
	float2cufft << <grid, block >> >(dSrc, ImgSrc, sizeX, sizeY);
	//cuFFT1D(dSrc, sizeX*2, sizeY, -1);	
	cufftExecC2C(plan_1D_C2C_FORWARD, dSrc, dSrc, CUFFT_FORWARD);
	countFDCT << <grid, block >> >(ImgDst, dSrc, sizeX, sizeY, 1 / sqrtf(2 *sizeY));
	transposeShared << <dimGrid, dimBlock >> >(ImgSrc, ImgDst);

	//DCT on y-axis	
	float2cufft << <grid, block >> >(dSrc, ImgSrc, sizeX, sizeY);
	//cuFFT1D(dSrc, sizeY*2, sizeX, -1);
	cufftExecC2C(plan_1D_C2C_FORWARD, dSrc, dSrc, CUFFT_FORWARD);
	countFDCT << <grid, block >> >(ImgSrc, dSrc, sizeY, sizeX, 1 / sqrtf(2 *sizeY));
	transposeShared << <dimGrid, dimBlock >> >(ImgDst, ImgSrc);
	e_unwrap = clock(); unwrap_time += e_unwrap - s_unwrap;
	cudaFree(dSrc);	
}
//--------------------------------------------------------------------------------------
void myIDCT(float *ImgDst, float *ImgSrc, int sizeX, int sizeY)
{
	cufftComplex *dSrc;
	cudaMalloc((void **)&dSrc, sizeof(cufftComplex)*(sizeX * 2)*sizeY);

	//copy the floating array to cufftComplex
	dim3 dimGrid(sizeX / TILE_DIM, sizeY / TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

	int blocksInX = (sizeX + 32 - 1) / 32;
	int blocksInY = (sizeY + 32 - 1) / 32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);

	s_unwrap = clock();
	//IDCT on x-axis
	transposeShared << <dimGrid, dimBlock >> >(ImgDst, ImgSrc);	
	countIDCT << <grid, block >> >(dSrc, ImgDst, sizeX, sizeY, 1/sqrtf(2 *sizeY));
	//cuFFT1D(dSrc, sizeX * 2, sizeY, 2); 
	cufftExecC2C(plan_1D_C2C_INVERSE, dSrc, dSrc, CUFFT_INVERSE);
	cufft2float << <grid, block >> >(ImgSrc, dSrc, sizeX, sizeY);
	
	//IDCT on y-axis
	transposeShared << <dimGrid, dimBlock >> >(ImgDst, ImgSrc);
	countIDCT << <grid, block >> >(dSrc, ImgDst, sizeY, sizeX, 1/sqrtf(2 *sizeY));
	//cuFFT1D(dSrc, sizeY * 2, sizeX, 2);	
	cufftExecC2C(plan_1D_C2C_INVERSE, dSrc, dSrc, CUFFT_INVERSE);
	cufft2float << <grid, block >> >(ImgDst, dSrc, sizeX, sizeY);
	//cudaMemcpy2D(ImgDst, 1 * sizeof(float), dSrc, sizeof(cufftComplex), sizeof(float), sizeX * sizeY, cudaMemcpyDeviceToDevice);

	e_unwrap = clock(); unwrap_time += e_unwrap - s_unwrap;

	cudaFree(dSrc);
}
//--------------------------------------------------------------------------------------
void DeviceMemOutDCTFFT(char *path, cufftComplex *arr, int sizeX, int sizeY)
{
	int size = sizeX*sizeY;
	cufftComplex *temp = (cufftComplex *)malloc(size*sizeof(cufftComplex));
	float *temp2 = (float *)malloc(size*sizeof(float));
	cudaMemcpy(temp, arr, size*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++)
	{
		temp2[i] = log10f(sqrt(temp[i].x*temp[i].x + temp[i].y*temp[i].y));
	}

	FILE *fp;
	fp = fopen(path, "wb");
	fwrite(temp2, size, sizeof(float), fp);
	fclose(fp);
	free(temp);
	free(temp2);
}
//--------------------------------------------------------------------------------------
__global__ void shift1DFFT(cufftComplex *device_FFT, int Nx, int Ny)
{
	cufftComplex tmp;

	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	if (xIndex<Nx / 2 && yIndex<Ny)
	{
		tmp = device_FFT[xIndex + Nx*yIndex];
		device_FFT[xIndex + Nx*yIndex] = device_FFT[(xIndex + Nx / 2) + Nx*yIndex];
		device_FFT[(xIndex + Nx / 2) + Nx*yIndex] = tmp;
	}
}
//--------------------------------------------------------------------------------------
__global__ void float2cufft(cufftComplex *odata, const float *idata, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx, idx1, idx2;
	float srcValue;

	if (i<sizeX && j<sizeY)
	{
		idx1 = (sizeX * 2 - i - 1) + (j)* sizeX * 2;	//right-top
		idx2 = i + j *sizeX * 2;	//left-top
		idx = i + j * sizeX;

		srcValue = idata[idx];

		odata[idx2].x = srcValue;	odata[idx2].y = 0;
		odata[idx1].x = srcValue;	odata[idx1].y = 0;
	}
}
//--------------------------------------------------------------------------------------
__global__ void cufft2float(float *odata, const cufftComplex *idata, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx, idx1;

	if (i<sizeX && j<sizeY)
	{
		idx1 = i + j *sizeX * 2;	//left-top
		idx = i + j * sizeX;

		odata[idx] = idata[idx1].x;		
	}
}
//--------------------------------------------------------------------------------------
__global__ void transposeShared(float *odata, float *idata)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y + j][threadIdx.x] = idata[(y + j)*width + x];

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
	{
		odata[(y + j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
	}

}
//--------------------------------------------------------------------------------------
__global__ void countFDCT(float *ImgDst, cufftComplex *ImgSrc, int sizeX, int sizeY, float dTemp)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i + sizeX*j;
	unsigned int idx2 = i + sizeX * 2 * j;
	float C, S;
	
	if (i < sizeX && j < sizeY)
	{
		C = ImgSrc[idx2].x * cos((float(i) *M_PI / (sizeX * 2)));
		S = ImgSrc[idx2].y * sin((float(i) *M_PI / (sizeX * 2)));
		ImgDst[idx] = (i == 0) ? ImgSrc[idx2].x *dTemp / sqrt(2.0): (C + S) *dTemp;
	}
}
//--------------------------------------------------------------------------------------
__global__ void countIDCT(cufftComplex *ImgDst, float *ImgSrc, int sizeX, int sizeY, float dTemp)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i + sizeX * j;							//orginal index
	unsigned int idx2 = i + sizeX * 2 * j;						//left side
	unsigned int idx3 = idx2 + sizeX;// (sizeX * 2 - i - 1) + sizeX * 2 * j;	//right side

	cufftComplex temp, temp2;
	float C = cos(float(i *M_PI / (sizeX * 2)));
	float S = sin(float(i *M_PI / (sizeX * 2)));
	float V = ImgSrc[idx] * dTemp;
	float S2 = sin(float((i + 1.0)*M_PI / (sizeX * 2)));
	float C2 = -cos(float((i + 1.0)*M_PI / (sizeX * 2)));
	float V2;
	
	if (i < sizeX && j < sizeY)
	{		
		temp.x = (i == 0) ? (C * V) * sqrt(2.0) : C * V;
		temp.y = (i == 0) ? (S * V) * sqrt(2.0) : S * V;

		V2 = (i == 0) ? 0 : ImgSrc[(sizeX - i) + sizeX * j] * dTemp;
		temp2.x = (i == 0) ? 0 : S2 * V2;
		temp2.y = (i == 0) ? 0 : C2 * V2;
		
		ImgDst[idx2] = temp;
		ImgDst[idx3] = temp2;
	}
}
//--------------------------------------------------------------------------------------
void DCT2D(float *ImgDst, float *ImgSrc, int sizeX, int sizeY, int dir)
{
	ROI Size;
	Size.width  = sizeX;
	Size.height = sizeY;

	float *src, *dst;
    size_t DeviceStride;
    checkCudaErrors(cudaMallocPitch((void **)&src, &DeviceStride, Size.width * sizeof(float), Size.height));
    checkCudaErrors(cudaMallocPitch((void **)&dst, &DeviceStride, Size.width * sizeof(float), Size.height));
    DeviceStride /= sizeof(float);

	int StrideF;	

	//setup execution parameters
    dim3 GridFullWarps(Size.width / KER2_BLOCK_WIDTH, Size.height / KER2_BLOCK_HEIGHT, 1);
    dim3 ThreadsFullWarps(8, KER2_BLOCK_WIDTH/8, KER2_BLOCK_HEIGHT/8);

	if(dir == -1)
	{
		//copy from host memory to device
		/*checkCudaErrors(cudaMemcpy2D(src, DeviceStride * sizeof(float),
                                 ImgSrc, StrideF * sizeof(float),
                                 Size.width * sizeof(float), Size.height,
                                 cudaMemcpyDeviceToDevice));*/

		//perform block-wise DCT processing and benchmarking
		const int numIterations = 100;

		for (int i = -1; i < numIterations; i++)
		{
			if (i == 0)
			{
				checkCudaErrors(cudaDeviceSynchronize());
			}

			CUDAkernel2DCT<<<GridFullWarps, ThreadsFullWarps>>>(ImgDst, ImgSrc, (int)DeviceStride);
			getLastCudaError("Kernel execution failed");
		}
	}
	else if(dir == 1)
	{
		CUDAkernel2IDCT<<<GridFullWarps, ThreadsFullWarps >>>(ImgSrc, ImgDst, (int)DeviceStride);
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("Kernel execution failed");
	}

	checkCudaErrors(cudaFree(dst));
    checkCudaErrors(cudaFree(src));

}
//--------------------------------------------------------------------------------------
__global__ void devConstant(float* buf, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i + sizeX*j;

	if (i == 0 && j == 0)
		buf[idx] = 0; 
	else if(i<sizeX && j<sizeY)
	{		
		buf[idx] = buf[idx] / (2*cosf(4*M_PI*i / (4*sizeX - 1)) + 2*cosf(4*M_PI*j / (4*sizeY - 1)) - 4);
	}
}
//--------------------------------------------------------------------------------------
__device__ int sign(float x)
{
	return x > 0 ? 1 : (x < 0 ? -1 : 0);
}

__global__ void LaplaceWithPU(float *dst, float *src, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i + sizeX*j;
	int signDst;
	if (i<sizeX && j<sizeY) {
		dst[idx] = src[idx];
		if (i>1 && j >1 && i < sizeX - 1 && j < sizeY - 1) {
			dst[idx] = src[idx - 1] + src[idx + 1] + src[idx - sizeX] + src[idx + sizeX] - (4 * src[idx]);
//			signDst = signbit(dst[idx]) ? -1 : 1;
			signDst = sign(dst[idx]);
			dst[idx] = dst[idx] - 2 * M_PI*signDst * floor(fabs(M_PI * signDst + dst[idx]) / (M_PI * 2));
		}
	}
}
//--------------------------------------------------------------------------------------
__global__ void NumericDerivative1(float *outX, float *outY, float *input, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i + sizeX*j;
	int signOutX, signOutY;
	float tmpX, tmpY;
	if (i<sizeX && j<sizeY)
	{
		//outX[idx] = (i == sizeX - 1) ? -input[idx] : input[idx + 1] - input[idx];//- input[idx - 1    ]
		//outY[idx] = (j == sizeY - 1) ? -input[idx] : input[idx + sizeX] - input[idx];//- input[idx - sizeX] 
		tmpX = (i == sizeX - 1) ? -input[idx] : input[idx + 1] - input[idx];//- input[idx - 1    ]
		tmpY = (j == sizeY - 1) ? -input[idx] : input[idx + sizeX] - input[idx];//- input[idx - sizeX] 

		idx = i + sizeX*j;
		//signOutX = signbit(outX[idx]) ? -1 : 1;
		//signOutY = signbit(outY[idx]) ? -1 : 1;
		signOutX = sign(tmpX);
		signOutY = sign(tmpY);
		outX[idx] = tmpX - 2 * M_PI* sign(tmpX) * floor(fabs(M_PI * signOutX + tmpX) / (M_PI * 2));
		outY[idx] = tmpY - 2 * M_PI* sign(tmpY) * floor(fabs(tmpY + M_PI * signOutY) / (M_PI * 2));
	}
}
//--------------------------------------------------------------------------------------
__global__ void NumericDerivative2(float *outX, float *outY, float *inX, float *inY, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i + sizeX*j;
	if (i<sizeX && j<sizeY)
	{
		outX[idx] = (i == sizeX - 1) ? -inX[idx] : inX[idx + 1] - inX[idx];//- inX[idx-1    ]
		outY[idx] = (j == sizeY - 1) ? -inY[idx] : inY[idx + sizeX] - inY[idx];//- inY[idx-sizeX]
		//outX[idx] = (i == 0) ? inX[idx+1] - inX[idx] : inX[idx] - inX[idx-1];// - inX[idx-1    ]
		//outY[idx] = (j == 0) ? inY[idx+sizeX] - inY[idx] : inY[idx] - inY[idx-sizeX];//- inY[idx-sizeX]
	}
}
//--------------------------------------------------------------------------------------
__global__ void TrigonometricF(float *outX, float *outY, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i + sizeX*j;
	int signOutX, signOutY;
	float tmpX, tmpY;

	if (i < sizeX && j < sizeY)	{
		//signOutX = signbit(outX[idx]) ? -1 : 1;
		//signOutY = signbit(outY[idx]) ? -1 : 1;
		signOutX = sign(outX[idx]);
		signOutY = sign(outY[idx]);
		tmpX = outX[idx];
		tmpY = outY[idx];
		outX[idx] = tmpX - 2 * M_PI*signOutX * floor(fabs(M_PI * signOutX + tmpX) / (M_PI * 2));
		outY[idx] = tmpY - 2 * M_PI*signOutY * floor(fabs(tmpY + M_PI * signOutY) / (M_PI * 2));
	}
}
//--------------------------------------------------------------------------------------
__global__ void SumDerivative(float *output, float *outX, float *outY, float *src, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;
	float c1, c2, c3, c4;
	if (i<sizeX && j<sizeY )
	{
		idx = i + sizeX*j;

		c1 = (i == sizeX - 1) ? 0 : outX[idx];
		c2 = (i == 0 ) ? 0 : outX[idx - 1];
		c3 = (j == sizeY - 1) ? 0 : outY[idx];
		c4 = (j == 0 ) ? 0 : outY[idx - sizeX];
		
		//if (i<sizeX - 1 && j<sizeY - 1 && i>1 && j>1)
			output[idx] = (c1-c2) + (c3-c4);
	}
	/*if (i<sizeX && j<sizeY)
	{
		idx = i + sizeX*j;
		output[idx] = (outX[idx] + outY[idx]);
	}*/
}
//--------------------------------------------------------------------------------------
__global__ void MultiplyOperator(float *input, float div, int width, int height)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i+width*j;

	if(i<width && j<height)
	{
		input[idx] *= div;
	}

}
//--------------------------------------------------------------------------------------
__global__ void shift2DArray(float *input, int width, int height)
{
	float tmp13, tmp24;
	
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
}
//--------------------------------------------------------------------------------------
void DeviceMemOutDCT(char *path, float *arr, int sizeX, int sizeY)
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
void exportRAW(char * fpath, float* buf, int size)
{
	FILE *fp;
	fp = fopen(fpath, "wb");
	if (!fp)
	{
		printf("\nCannot save the image: %s", fpath);
		exit(1);
	}
	else
	{
		fwrite(buf, size, sizeof(float), fp);
	}
	fclose(fp);
}
