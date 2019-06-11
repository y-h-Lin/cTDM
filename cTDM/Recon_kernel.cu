#include "Recon_kernel.cuh"
//#include "BatchRecon.cuh"

int ReconMode;
size_t availMem,totalMem;
//--------------------------------------------------------------------------------------
void MedianFilter(float *PhaseData, float *AmpData, float *Phase_img, float *Amp_img, int nx, int ny, int nz)
{
	int blocksInX = (nx+32-1)/32;
	int blocksInY = (ny+32-1)/32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);	

	//phase
	cudaMemcpy(PhaseData, Phase_img, sizeof(float)*nx*ny*nz, cudaMemcpyHostToDevice);
	cudaMemcpy(AmpData  , Amp_img  , sizeof(float)*nx*ny*nz, cudaMemcpyHostToDevice);

	for (int numZ = 0; numZ < nz; numZ++)
	{
		MedianFilter_gpu<<<grid, block>>>(PhaseData, nx, ny, numZ);
		MedianFilter_gpu<<<grid, block>>>(AmpData  , nx, ny, numZ);
		checkMaps<<<grid, block >>>(PhaseData, AmpData, nx, ny, numZ);
	}

	//cudaMemcpy(Phase_img, PhaseData, sizeof(float)*nx*ny*nz, cudaMemcpyDeviceToHost);
	
}
//--------------------------------------------------------------------------------------
__global__ void Median_Filter_GPU(float *Input_Image, float *Output_Image, int Image_Width, int Image_Height, int Image_Slice)
{	
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;

	if((xIndex<Image_Width-1) && (xIndex > 1) && (yIndex<Image_Height-1) && (yIndex > 1)  && (zIndex<Image_Slice))
	{		
		float temp[]={
						Input_Image[zIndex*Image_Width*Image_Height +(yIndex-1)*Image_Width+(xIndex-1)],
						Input_Image[zIndex*Image_Width*Image_Height +(yIndex-1)*Image_Width+xIndex],
						Input_Image[zIndex*Image_Width*Image_Height +(yIndex-1)*Image_Width+(xIndex+1)],
						Input_Image[zIndex*Image_Width*Image_Height +yIndex*Image_Width+(xIndex-1)],
						Input_Image[zIndex*Image_Width*Image_Height +yIndex*Image_Width+xIndex],
						Input_Image[zIndex*Image_Width*Image_Height +yIndex*Image_Width+(xIndex+1)],
						Input_Image[zIndex*Image_Width*Image_Height +(yIndex+1)*Image_Width+(xIndex-1)],
						Input_Image[zIndex*Image_Width*Image_Height +(yIndex+1)*Image_Width+xIndex],
						Input_Image[zIndex*Image_Width*Image_Height +(yIndex+1)*Image_Width+(xIndex+1)]
						};

		for(int a=0;a<8;a++)
			for(int b=a+1;b<9;b++)
			{
				if(temp[a] > temp[b])
				{
					float z=0;
					z=temp[b];
					temp[b]=temp[a];
					temp[a]=z;
				}					
			}
		
		Output_Image[zIndex*Image_Width*Image_Height + (yIndex*Image_Width) + xIndex]=temp[4];	
	}
}
//--------------------------------------------------------------------------------------
__global__ void MedianFilter_gpu(float *Device_ImageData, int Image_Width, int Image_Height, int numZ)
{
	__shared__ float surround[BLOCK_WIDTH*BLOCK_HEIGHT][MEDIAN_LENGTH];

	int iterator;
	const int Half_Of_MEDIAN_LENGTH = (MEDIAN_LENGTH / 2) + 1;
	int StartPoint = MEDIAN_DIMENSION / 2;
	int EndPoint = StartPoint + 1;

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	const int tid = threadIdx.y*blockDim.y + threadIdx.x;

	if (x >= Image_Width || y >= Image_Height)
		return;

	//Fill surround with pixel value of Image in Matrix Pettern of MEDIAN_DIMENSION x MEDIAN_DIMENSION
	if (x == 0 || x == Image_Width - StartPoint || y == 0 || y == Image_Height - StartPoint)
	{
	}
	else
	{
		iterator = 0;
		for (int r = x - StartPoint; r < x + (EndPoint); r++)
		{
			for (int c = y - StartPoint; c < y + (EndPoint); c++)
			{
				surround[tid][iterator] = *(Device_ImageData + (c*Image_Width) + r +numZ*Image_Width*Image_Height);
				iterator++;
			}
		}
		//Sort the Surround Array to Find Median. Use Bubble Short  if Matrix oF 3 x 3 Matrix
		//You can use Insertion commented below to Short Bigger Dimension Matrix  

		////      bubble short //

		for (int i = 0; i<Half_Of_MEDIAN_LENGTH; ++i)
		{
			// Find position of minimum element
			int min = i;
			for (int l = i + 1; l<MEDIAN_LENGTH; ++l)
			if (surround[tid][l] <surround[tid][min])
				min = l;
			// Put found minimum element in its place
			float temp = surround[tid][i];
			surround[tid][i] = surround[tid][min];
			surround[tid][min] = temp;
		}//bubble short  end

		// it will give value of surround[tid][4] as Median Value if use 3 x 3 matrix
		*(Device_ImageData + (y*Image_Width) + x + numZ*Image_Width*Image_Height) = surround[tid][Half_Of_MEDIAN_LENGTH - 1];

		__syncthreads();
	}
}
//--------------------------------------------------------------------------------------
__global__ void checkMaps(float *Phase_img, float *Amp_img, int Image_Width, int Image_Height, int numZ)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;

	if (i <  Image_Width && j < Image_Height)
	{
		idx = i + j*Image_Width + numZ *Image_Width*Image_Height;
		if (isnan(Phase_img[idx]) == true || isinf(Phase_img[idx]) == true)
			Phase_img[idx] = 0;

		if (isnan(Amp_img[idx]) == true || isinf(Amp_img[idx]) == true)
			Amp_img[idx] = 1.0;
		else if (Amp_img[idx] == 0)
			Amp_img[idx] = 0.01;
		else
			Amp_img[idx] = fabs(Amp_img[idx]);
	}

}
//--------------------------------------------------------------------------------------
void CopyDataArray(cufftComplex *u_sp, float *Phase, float *Amp, int sizeX, int sizeY, int sizeZ, int numZ)
{
	int blocksInX = (sizeX + 32 - 1) / 32;
	int blocksInY = (sizeY + 32 - 1) / 32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);

	cuCopyArray <<<grid, block >>>(u_sp, Phase, Amp, sizeX, sizeY, numZ);
}
//--------------------------------------------------------------------------------------
__global__ void cuCopyArray(cufftComplex *u_sp, float *phase, float *amp, int Image_Width, int Image_Height, int numZ)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx,idx2;

	if (i <  Image_Width && j < Image_Height)
	{
		idx2 = j + i*Image_Height;
		idx  = i + j*Image_Width +numZ *Image_Width*Image_Height;
		
		u_sp[idx2].x = log(amp[idx]);
		u_sp[idx2].y = phase[idx];
	}

}
//--------------------------------------------------------------------------------------
void EdwardSphere(cufftComplex *u_sp, cufftComplex *F, float * C, float fm0, float df, float AngX, float AngY
	, int sizeX, int sizeY, int sizeZ)
{
	bool Fz_err = false;	//check sqrt(less than 0)
	float fx0 = fm0 * sin(AngY);
	float fy0 = fm0 * sin(AngX);
	float fz0 = sqrt(fm0 * fm0 - fx0*fx0 - fy0*fy0);

	int blocksInX = (sizeX + 32 - 1) / 32;
	int blocksInY = (sizeY + 32 - 1) / 32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);

	fillEdwardSphere<<<grid,block>>>(u_sp, F, C, fx0, fy0, fz0, fm0, df, Fz_err, sizeX, sizeY, sizeZ);
}
//--------------------------------------------------------------------------------------
__global__ void fillEdwardSphere(cufftComplex *u_sp, cufftComplex *F, float * C, float fx0, float fy0, float fz0
	, float fm0, float df, bool Fz_err, int sizeX, int sizeY, int sizeZ)
{
	float Fx, Fy, Fz, fx, fy, fz;
	int ii, jj, Nx, Ny, Nz, idx;
	float fm02 = fm0 * fm0;
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < sizeX && j < sizeY)
	{
		ii = i - (sizeX / 2.0f);
		jj = j - (sizeY / 2.0f);
		Fx = ii * df;
		Fy = jj * df;

		fx = Fx + fx0;
		fy = Fy + fy0;
		fz = sqrt(fm02 - (fx*fx + fy*fy));

		if ((fm02 - fx0*fx0 - fy0*fy0)<0 || (fm02 - fx*fx - fy*fy)<0)
			Fz_err = true;
		else
			Fz_err = false;

		Fz = fz - fz0;

		Nx = ii;	//round(Fx/df);
		Ny = jj;	//round(Fy/df);
		Nz;

		if (Fz_err == false && Nx >= -sizeX / 2 && Nx<sizeX / 2 && Ny >= -sizeY / 2 && Ny<sizeY / 2)
		{
			Nx = cuMod(Nx , sizeX);
			Ny = cuMod(Ny , sizeY);
			Nz = cuMod(cuRound(Fz / df), sizeZ);
			idx = Nx + Ny*sizeX + Nz*sizeX*sizeY;
			F[idx].x += (-fz * 4 * M_PI * u_sp[Nx + Ny*sizeX].y);
			F[idx].y += ( fz * 4 * M_PI * u_sp[Nx + Ny*sizeX].x);
			C[idx]   += 1.0;
		}
	}
}
//--------------------------------------------------------------------------------------
__device__ int cuMod(int a, int b)
{
	return (((a < 0) ? ((a % b) + b) : a) % b);
}
//--------------------------------------------------------------------------------------
__device__ float cuRound(float num)
{
	return (num > 0.0) ? floor(num + 0.5) : ceil(num - 0.5);
}
//--------------------------------------------------------------------------------------
void initial_F_and_F2(float *C, cufftComplex *cuF, cufftComplex *cuF2, float recon_dx , int Image_Width, int Image_Height, int Image_Slice)
{
	int blocksInX = (Image_Width+8-1)/8;
	int blocksInY = (Image_Height+8-1)/8;
	int blocksInZ = (Image_Slice+8-1)/8;
	dim3 grid(blocksInX, blocksInY, blocksInZ);
	dim3 block(8, 8, 8);
	
	Initial_F_F2<<<grid,block>>>(C,cuF,cuF2,recon_dx,Image_Width,Image_Height,Image_Slice);
	
}
//--------------------------------------------------------------------------------------
__global__ void Initial_F_F2(float *cu_C, cufftComplex *cu_F, cufftComplex *cu_F2, float cu_dx,
									int Image_Width, int Image_Height, int Image_Slice)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;

	if((xIndex<Image_Width) && (yIndex<Image_Height) && (zIndex<Image_Slice))
	{
		unsigned int index_out = xIndex + Image_Width*yIndex + Image_Width*Image_Height*zIndex;

		if(cu_C[index_out] != 0)
		{
			cu_F[index_out].x = cu_F[index_out].x / cu_C[index_out];
			cu_F[index_out].y = cu_F[index_out].y / cu_C[index_out];
		}
		cu_F2[index_out].x = cu_F[index_out].x / cu_dx;
		cu_F2[index_out].y = cu_F[index_out].y / cu_dx;

		cu_F[index_out].x = cu_F2[index_out].x;
		cu_F[index_out].y = cu_F2[index_out].y;
		
	}
}
//--------------------------------------------------------------------------------------
void est_n_3D(cufftComplex *n, cufftComplex *cu_F, float recon_k2, float recon_med, int sizeX, int sizeY, int sizeZ)
{
	int blocksInX = (sizeX + 8 - 1) / 8;
	int blocksInY = (sizeY + 8 - 1) / 8;
	int blocksInZ = (sizeZ + 8 - 1) / 16;
	dim3 grid(blocksInX, blocksInY, blocksInZ);
	dim3 block(8, 8, 16);
	//printf("Mode: %d", ReconMode);

	switch (ReconMode)
	{
		case 0:{
				   calculate_n3D << <grid, block >> >(n, cu_F, recon_k2, recon_med, sizeX, sizeY, sizeZ);
			   }break;
		case 1:{
					cufftComplex *cu_n;
					cudaMalloc((void **)&cu_n, sizeof(cufftComplex)* sizeX*sizeY*sizeZ);
	
					calculate_n3D << <grid, block >> >(cu_n, cu_F, recon_k2, recon_med, sizeX, sizeY, sizeZ);

					cudaMemcpy(n, cu_n, sizeof(cufftComplex)*sizeX*sizeY*sizeZ, cudaMemcpyDeviceToHost);

					cudaFree(cu_n);
			   }break;
	}	
}
//--------------------------------------------------------------------------------------
__global__ void calculate_n3D(cufftComplex * cu_n, cufftComplex * cu_F, float cu_k2, float cu_med, int sizeX, int sizeY, int sizeZ)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

	if(i<sizeX && j<sizeY && k<sizeZ)
	{
		unsigned int idx = i + j*sizeX + k*sizeY*sizeZ;

		//suppose that z = c+di, and we want to find sqrt(z) = a+bi
		float c, d, a, b;
		c = (cu_F[idx].x / (-cu_k2)) + 1;
		d = (cu_F[idx].y / (-cu_k2));

		a = sqrt((sqrt((c*c)+(d*d))+c)/2);
		if(d!=0)
		{
			b = sqrt((sqrt((c*c)+(d*d))-c)/2) * (d/abs(d));
		}
		else
		{
			b = 0.0;
		}

		cu_n[idx].x = a * cu_med;
		cu_n[idx].y = b * cu_med;

		if(cu_n[idx].x < cu_med)
		{
			cu_n[idx].x = cu_med;
		}
		if(cu_n[idx].y < 0)
		{
			cu_n[idx].y = 0;
		}		
	}
}
//--------------------------------------------------------------------------------------
void modify_F_3D(cufftComplex *n, cufftComplex *cu_F, float recon_k2, float recon_med 
					, int Image_Width, int Image_Height, int Image_Slice)
{
	int blocksInX = (Image_Width+8-1)/8;
	int blocksInY = (Image_Height+8-1)/8;
	int blocksInZ = (Image_Slice+8-1)/8;
	dim3 grid(blocksInX, blocksInY, blocksInZ);
	dim3 block(8, 8, 8);

	switch (ReconMode)
	{
		case 0:{
					Modify_F3D<<<grid,block>>>(n, cu_F, recon_k2,recon_med,Image_Width,Image_Height,Image_Slice);
			   }break;
		case 1:{
					cufftComplex *cu_n;
					cudaMalloc((void **)&cu_n,sizeof(cufftComplex)*Image_Width*Image_Height*Image_Slice);
					cudaMemcpy(cu_n, n, sizeof(cufftComplex)*Image_Width*Image_Height*Image_Slice, cudaMemcpyHostToDevice);
					
					Modify_F3D<<<grid,block>>>(cu_n, cu_F, recon_k2,recon_med,Image_Width,Image_Height,Image_Slice);
					
					cudaFree(cu_n);
			   }break;
	}
}
//--------------------------------------------------------------------------------------
__global__ void Modify_F3D(cufftComplex * cu_n, cufftComplex * cu_F, float cu_k2, float cu_med, int Image_Width, int Image_Height, int Image_Slice)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;

	if((xIndex<Image_Width) && (yIndex<Image_Height) && (zIndex<Image_Slice))
	{
		unsigned int i = xIndex + Image_Width*yIndex + Image_Width*Image_Height*zIndex;
		
		cu_F[i].x = (((cu_n[i].x*cu_n[i].x) - (cu_n[i].y*cu_n[i].y))/(cu_med)-1)*(-cu_k2);
		cu_F[i].y = (cu_n[i].x*cu_n[i].y*2)/(cu_med)*(-cu_k2);
	}
}
//--------------------------------------------------------------------------------------
void check_F_3D(cufftComplex *cu_F, cufftComplex *cu_F2, int Image_Width, int Image_Height, int Image_Slice)
{
	int blocksInX = (Image_Width+8-1)/8;
	int blocksInY = (Image_Height+8-1)/8;
	int blocksInZ = (Image_Slice+8-1)/8;
	dim3 grid(blocksInX, blocksInY, blocksInZ);
	dim3 block(8, 8, 8);
	
	check_F3D<<<grid,block>>>(cu_F, cu_F2, Image_Width,Image_Height,Image_Slice);	
}
//--------------------------------------------------------------------------------------
__global__ void check_F3D(cufftComplex * cu_F, cufftComplex * cu_F2, int Image_Width, int Image_Height, int Image_Slice)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;

	if((xIndex<Image_Width) && (yIndex<Image_Height) && (zIndex<Image_Slice))
	{
		unsigned int i = xIndex + Image_Width*yIndex + Image_Width*Image_Height*zIndex;

		if(cu_F2[i].x != 0.0 && cu_F2[i].y != 0.0)
		{
			cu_F[i].x = cu_F2[i].x;
			cu_F[i].y = cu_F2[i].y;
		}
	}
}
//--------------------------------------------------------------------------------------
void Combine2ComplexStack(cufftComplex *u_sp_stack, float *phase_stack, float *amp_stack, int sizeX, int sizeY, int sizeZ)
{
	int blocksInX = (sizeX + 8 - 1) / 8;
	int blocksInY = (sizeY + 8 - 1) / 8;
	int blocksInZ = (sizeZ + 16 - 1) / 16;
	dim3 grid(blocksInX, blocksInY, blocksInZ);
	dim3 block(8, 8, 8);

	cuCombine2ComplexStack << <grid, block >> >(u_sp_stack, phase_stack, amp_stack, sizeX, sizeY, sizeZ);
}
//--------------------------------------------------------------------------------------
__global__ void cuCombine2ComplexStack(cufftComplex *u_sp_stack, float *phase_stack, float *amp_stack, int sizeX, int sizeY, int sizeZ)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;
	unsigned int idx;

	if (i < sizeX && j < sizeY && k < sizeZ)
	{
		idx = i + j*sizeX + k*sizeX*sizeY;
		u_sp_stack[idx].x = log(phase_stack[idx]);
		u_sp_stack[idx].y = amp_stack[idx];
	}
}
//--------------------------------------------------------------------------------------
void Shift2DonStack(cufftComplex *U, int sizeX, int sizeY, int sizeZ)
{
	int blocksInX = (sizeX + 32 - 1) / 32;
	int blocksInY = (sizeY + 32 - 1) / 32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);

	for (int i = 0; i < sizeZ; i++)
		cuShift2DonStack<<<grid,block>>>(U, i, sizeX, sizeY, sizeZ);
}
//--------------------------------------------------------------------------------------
__global__ void cuShift2DonStack(cufftComplex *U, int z, int sizeX, int sizeY, int sizeZ)
{
	cufftComplex tmp13, tmp24;

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int offset = z*sizeX*sizeY;

	if (i<sizeX/2 && j<sizeY/2)
	{
		// interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
		tmp13 = U[i + j*sizeX + offset];
		U[i + j*sizeX + offset] = U[(i + sizeX / 2) + (j + sizeY / 2)*sizeX + offset];
		U[(i + sizeX / 2) + (j + sizeY / 2)*sizeX + offset] = tmp13;
		tmp24 = U[(i + sizeX / 2) + j*sizeX + offset];
		U[(i + sizeX / 2) + j*sizeX + offset] = U[i + (j + sizeY / 2)*sizeX + offset];
		U[i + (j + sizeY / 2)*sizeX + offset] = tmp24;
	}
}
//--------------------------------------------------------------------------------------
void FrquencyInterpolation(cufftComplex *F, float * C, cufftComplex *U, float *AngleX, float *AngleY
	, float fm0, float df, float dx, float n_med, int frameNum, int FrameSize, int sizeX, int sizeY, int sizeZ)
{
	float *cuAngX, *cuAngY;
	cudaMalloc((void **)&cuAngX, sizeof(float)*FrameSize);
	cudaMalloc((void **)&cuAngY, sizeof(float)*FrameSize);
	cudaMemcpy(cuAngX, AngleX, sizeof(float)*FrameSize, cudaMemcpyHostToDevice);
	cudaMemcpy(cuAngY, AngleY, sizeof(float)*FrameSize, cudaMemcpyHostToDevice);

	float AngX = AngleX[frameNum];
	float AngY = AngleY[frameNum];
	float fx0 = fm0*sin(AngX);// fm0 * sin(AngY);
	float fy0 = 0;// fm0 * sin(AngX);
	float fz0 = fm0*cos(AngX);// sqrt(fm0 * fm0 - fx0*fx0 - fy0*fy0);

	int blocksInX = (sizeX + 32 - 1) / 32;
	int blocksInY = (sizeY + 32 - 1) / 32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);

	FI_kernel << <grid, block >> >(F, C, U, cuAngX, cuAngY, fx0, fy0, fz0, fm0, df, dx, n_med, frameNum, FrameSize, sizeX, sizeY, sizeZ);

	cudaFree(cuAngX);
	cudaFree(cuAngY);
}
//--------------------------------------------------------------------------------------
__global__ void FI_kernel(cufftComplex *F, float * C, cufftComplex *U, float *AngleX, float *AngleY
	, float fx0, float fy0, float fz0, float fm0, float df, float dx, float n_med, int frameNum, int frameSize, int sizeX, int sizeY, int sizeZ)
{
	bool Fz_err = false;	//check sqrt(less than 0)
	float Fx, Fy, Fz, fx, fy, fz;
	float Fxp, Fxp1, Fxp2, Fyp, Fyp1, Fyp2, Fzp, fzp;
	int ii, jj, Nx, Ny, Nz, idx, rr, tt, ss;
	cufftComplex a1, a2, a3, a4, b1, b2, b3, b4, val;
	float fm02 = fm0 * fm0;
	float anglep, angle1, angle2;
	float nume, denom;
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx1, idx2, idx3, idx4;

	if (i < sizeX && j < sizeY)
	{
		ii = i - (sizeX / 2.0f);
		jj = j - (sizeY / 2.0f);
		Fx = ii * df;
		Fy = jj * df;

		fx = Fx + fx0;
		fy = Fy + fy0;
		fz = sqrt(fm02 - (fx*fx + fy*fy));

		if ((fm02 - fx0*fx0 - fy0*fy0)<0 || (fm02 - fx*fx - fy*fy)<0)
			Fz_err = true;
		else
			Fz_err = false;

		Fz = fz - fz0;

		Nx = round(Fx / df);
		Ny = round(Fy / df);
		//Nz;

		if (Fz_err == false && Nx >= -sizeX / 2 && Nx < sizeX / 2 && Ny >= -sizeY / 2 && Ny < sizeY / 2)
		{
			Fz = round(Fz / df)*df;
			calF(Fxp, Fyp, Fzp, fzp, anglep, Fx, Fy, Fz, fx, fy, fz, n_med, fm0, AngleX[frameNum]);
			for (int kk = 0; kk < frameSize - 1; kk++) {
				if (anglep >= AngleX[kk] && anglep <= AngleX[kk + 1]) {
					angle1 = AngleX[kk];
					angle2 = AngleX[kk + 1];
					ss = kk;
				}
				else if (anglep>AngleX[frameSize - 1]) {
					anglep = AngleX[frameSize - 1];
					angle1 = AngleX[frameSize - 1];
					angle2 = AngleX[frameSize - 1];
					ss = frameSize - 1;
				}
				else if (anglep <= AngleX[0]) {
					anglep = AngleX[0];
					angle1 = AngleX[0];
					angle2 = AngleX[0];
					ss = 0;
				}
			}

			for (int kk = -sizeX / 2; kk<sizeX / 2; kk++) {
				if (Fxp >= kk*df && Fxp <= (kk + 1)*df){
					Fxp1 = kk*df;
					Fxp2 = (kk + 1)*df;
					rr = kk + sizeX / 2 + 1;
				}
				else if (Fxp>(sizeX / 2 - 1)*df) {
					rr = sizeX / 2 - 1 + sizeX / 2 + 1;
				}
			}

			for (int kk = -sizeY / 2; kk<sizeY / 2; kk++) {
				if (Fyp >= kk*df && Fyp <= (kk + 1)*df) {
					Fyp1 = kk*df;
					Fyp2 = (kk + 1)*df;
					tt = kk + sizeY / 2 + 1;
				}
				else if (Fyp >(sizeY / 2 - 1)*df) {
					tt = sizeY / 2 - 1 + sizeY / 2 + 1;
				}
			}

			if (rr < sizeX - 1 && tt < sizeY - 1) {
				idx1 = rr + tt      *sizeX + ss*sizeX*sizeY;
				idx2 = rr + (tt + 1)*sizeX + ss*sizeX*sizeY;
				idx3 = (rr + 1) + tt      *sizeX + ss*sizeX*sizeY;
				idx4 = (rr + 1) + (tt + 1)*sizeX + ss*sizeX*sizeY;

				a1 = U[idx1];
				a2 = U[idx2];
				a3 = U[idx3];
				a4 = U[idx4];

				b1 = U[idx1 + sizeX*sizeY];
				b2 = U[idx2 + sizeX*sizeY];
				b3 = U[idx3 + sizeX*sizeY];
				b4 = U[idx4 + sizeX*sizeY];
			}
			else if (rr == sizeX - 1 && tt < sizeY - 1) {
				idx1 = rr + tt      *sizeX + ss*sizeX*sizeY;
				idx2 = rr + (tt + 1)*sizeX + ss*sizeX*sizeY;
				idx3 = rr + tt      *sizeX + ss*sizeX*sizeY;
				idx4 = rr + (tt + 1)*sizeX + ss*sizeX*sizeY;

				a1 = U[idx1];
				a2 = U[idx2];
				a3 = U[idx3];
				a4 = U[idx4];

				b1 = U[idx1 + sizeX*sizeY];
				b2 = U[idx2 + sizeX*sizeY];
				b3 = U[idx3 + sizeX*sizeY];
				b4 = U[idx4 + sizeX*sizeY];
			}
			else if (rr < sizeX - 1 && tt == sizeY - 1) {
				idx1 = rr + tt*sizeX + ss*sizeX*sizeY;
				idx2 = rr + tt*sizeX + ss*sizeX*sizeY;
				idx3 = (rr + 1) + tt*sizeX + ss*sizeX*sizeY;
				idx4 = (rr + 1) + tt*sizeX + ss*sizeX*sizeY;

				a1 = U[idx1];
				a2 = U[idx2];
				a3 = U[idx3];
				a4 = U[idx4];

				b1 = U[idx1 + sizeX*sizeY];
				b2 = U[idx2 + sizeX*sizeY];
				b3 = U[idx3 + sizeX*sizeY];
				b4 = U[idx4 + sizeX*sizeY];
			}

			val = TrilinearFrequency(a1, a2, a3, a4, b1, b2, b3, b4, Fxp1, Fxp2, Fyp1, Fyp2, angle1, angle2, Fxp, Fyp, anglep);

			Nx = cuMod(round(Fx / df), sizeX);
			Ny = cuMod(round(Fy / df), sizeY);
			Nz = cuMod(round(Fz / df), sizeZ);

			/*%---------------------------------------------- -
			%{
			% 不使用Weighted - average
			F_3D(Nx, Ny, Nz) = (fzp*j * 4 * pi) * val;
			%}


			% 使用Weighted - average*/
			idx = Nx + Ny*sizeX + Nz*sizeX*sizeY;
			cufftComplex temp;
			temp.x = 0;
			temp.y = fzp * 4 * M_PI;
			if (C[idx] == 0) {
				C[idx] = 1;				
				temp.x = 0;
				temp.y = fzp * 4 * M_PI;
				F[idx] = ComplexMultiplication(temp, val);
			}
			else{
				//% Note_3D(Nx, Ny, Nz)~= 0
				nume = C[idx];   //%分子
				//Note_3D(Nx, Ny, Nz) = Note_3D(Nx, Ny, Nz) + 1;
				denom = C[idx] + 1; //%分母

				//% 如果每個被overlap 3次，頻率成分 = 1 / 3[a] + 1 / 3[b] + 1 / 3[c]
				temp.x = 0;
				temp.y = fzp * 4 * M_PI;
				F[idx] = ComplexAddition(ComplexProduction(F[idx], nume / denom), ComplexProduction(ComplexMultiplication(temp, val), 1 / denom));
			}
		}
		F[idx].x = F[idx].x / dx;
		F[idx].y = F[idx].y / dx;
	}
}
//--------------------------------------------------------------------------------------
__device__ void calF(float &Fxp, float &Fyp, float &Fzp, float &fzp, float &anglep,
	float Fx, float Fy, float Fz, float fx, float fy, float fz, float n0, float f0, float angle)
{
	float C = -(fy*fy) + Fx*Fx + Fz*Fz;
	float A1 = 4 * (Fx*Fx + Fz*Fz);
	float B1 = -4 * C*Fx;
	float C1 = C*C - 4 * Fz*Fz * n0*n0 * f0*f0 + 4 * Fz*Fz * fy*fy;
	float fxTemp1, fxTemp2, fxp, fyp;

	if (A1 != 0) {
		fxTemp1 = (-B1 + sqrtf(B1*B1 - 4 * A1*C1)) / 2 / A1;
		fxTemp2 = (-B1 - sqrtf(B1*B1 - 4 * A1*C1)) / 2 / A1;

		if ((!isnan(fxTemp1) || !isinf(fxTemp1)) && abs(fx - fxTemp1) <= abs(fx - fxTemp2)){
			fxp = fxTemp1;
			fyp = fy;
			fzp = sqrtf(n0*n0 * f0*f0 - fxp*fxp - fy*fy);

			if (isnan(fzp) || isinf(fzp)){
				fxp = fx;
				fzp = fz;
			}
			anglep = atan2(fxp - Fx, sqrt(n0*n0 * f0*f0 - fxp*fxp - fyp*fyp) - Fz);
		}
		else if ((!isnan(fxTemp2) || !isinf(fxTemp2)) && abs(fx - fxTemp1) > abs(fx - fxTemp2)){
			fxp = fxTemp2;
			fyp = fy;
			fzp = sqrt(n0*n0 * f0*f0 - fxp*fxp - fy*fy);

			if (isnan(fzp) || isinf(fzp)){
				fxp = fx;
				fzp = fz;
			}
			anglep = atan2(fxp - Fx, sqrt(n0*n0 * f0*f0 - fxp*fxp - fyp*fyp) - Fz);
		}
		else{
			fxp = fx;
			fyp = fy;
			fzp = sqrt(n0*n0 * f0*f0 - fxp*fxp - fyp*fyp);
			anglep = angle;
		}
	}
	else if (A1 == 0) {
		fxp = fx;
		fyp = fy;
		fzp = sqrt(n0*n0 * f0*f0 - fxp*fxp - fyp*fyp);
		anglep = atan2(fxp - Fx, sqrt(n0*n0 * f0*f0 - fxp*fxp - fyp*fyp) - Fz);
	}


	if (!isnan(fzp) || !isinf(fzp)) {
		Fxp = fxp - f0*n0*sin(anglep);
		Fyp = fyp;
		Fzp = fzp - f0*n0*cos(anglep);
	}
	else{

		Fxp = fx - f0*n0*sin(anglep);
		Fyp = fy;
		Fzp = fz - f0*n0*cos(anglep);
		fzp = fz;
	}
}
//--------------------------------------------------------------------------------------
__device__ cufftComplex TrilinearFrequency(cufftComplex a1, cufftComplex a2, cufftComplex a3, cufftComplex a4
	, cufftComplex b1, cufftComplex b2, cufftComplex b3, cufftComplex b4, float Fxp, float Fxp1, float Fxp2
	, float Fyp, float Fyp1, float Fyp2, float angle1, float angle2, float anglep)
{
	float V, U, W;
	cufftComplex  m1, m2, m3, m4, n1, n2, val;
	if (angle2 != angle1) {
		V = (anglep - angle1) / (angle2 - angle1);

		m1 = ComplexAddition(ComplexProduction(b1, V), ComplexProduction(a1, (1 - V)));
		m2 = ComplexAddition(ComplexProduction(b2, V), ComplexProduction(a2, (1 - V)));
		m3 = ComplexAddition(ComplexProduction(b3, V), ComplexProduction(a3, (1 - V)));
		m4 = ComplexAddition(ComplexProduction(b4, V), ComplexProduction(a4, (1 - V)));

		U = (Fxp - Fxp1) / (Fxp2 - Fxp1);

		n1 = ComplexAddition(ComplexProduction(m3, U), ComplexProduction(m1, (1 - U)));
		n2 = ComplexAddition(ComplexProduction(m4, U), ComplexProduction(m2, (1 - U)));

		W = (Fyp - Fyp1) / (Fyp2 - Fyp1);

		val = ComplexAddition(ComplexProduction(n2, W), ComplexProduction(n1, (1 - W)));
	}
	else if (angle2 == angle1) {
		val.x = 0;
		val.y = 0;
	}

	return val;
}
//--------------------------------------------------------------------------------------
__device__ cufftComplex ComplexProduction(cufftComplex V, float F)
{
	float a = V.x;
	float b = V.y;
	cufftComplex val;
	val.x = a*F;
	val.y = b*F;

	return val;
}
//--------------------------------------------------------------------------------------
__device__ cufftComplex ComplexMultiplication(cufftComplex V1, cufftComplex V2)
{
	float a = V1.x;
	float b = V1.y;
	float c = V2.x;
	float d = V2.y;
	cufftComplex val;
	val.x = a*c - b*d;
	val.y = a*d + b*c;

	return val;
}
//--------------------------------------------------------------------------------------
__device__ cufftComplex ComplexDivision(cufftComplex V1, cufftComplex V2)
{
	float a = V1.x;
	float b = V1.y;
	float c = V2.x;
	float d = V2.y;
	cufftComplex val;
	val.x = (a*c + b*d) / (c*c + d*d);
	val.y = (b*c - a*d) / (c*c + d*d);

	return val;
}
//--------------------------------------------------------------------------------------
__device__ cufftComplex ComplexAddition(cufftComplex V1, cufftComplex V2)
{
	float a = V1.x;
	float b = V1.y;
	float c = V2.x;
	float d = V2.y;
	cufftComplex val;
	val.x = a + c;
	val.y = b + d;

	return val;
}
//--------------------------------------------------------------------------------------
__device__ cufftComplex ComplexSubtraction(cufftComplex V1, cufftComplex V2)
{
	float a = V1.x;
	float b = V1.y;
	float c = V2.x;
	float d = V2.y;
	cufftComplex val;
	val.x = a - c;
	val.y = b - d;

	return val;
}
//--------------------------------------------------------------------------------------
__device__ cufftComplex ComplexSqrt(cufftComplex V)
{
	float c = V.x;
	float d = V.y;
	float a, b;
	cufftComplex val;
	val.x = sqrtf(c + sqrtf(c*c + d*d) / 2.0);
	val.y = (d/fabs(d)) * sqrtf(-c + sqrtf(c*c + d*d) / 2.0);

	return val;
}
//--------------------------------------------------------------------------------------
__device__ float ComplexABS(cufftComplex V)
{
	float a = V.x;
	float b = V.y;
	float val = sqrtf(a*a + b*b);
	return val;
}
//--------------------------------------------------------------------------------------
void Est_n_3D_POCS(cufftComplex *cu_n, cufftComplex *cu_n2, cufftComplex *cu_F, float recon_k2, float recon_med, int sizeX, int sizeY, int sizeZ)
{
	int blocksInX = (sizeX + 8 - 1) / 8;
	int blocksInY = (sizeY + 8 - 1) / 8;
	int blocksInZ = (sizeZ + 8 - 1) / 16;
	dim3 grid(blocksInX, blocksInY, blocksInZ);
	dim3 block(8, 8, 16);
	//printf("Mode: %d", ReconMode);

	cuEst_n_3D_POCS << <grid, block >> >(cu_n, cu_n2, cu_F, recon_k2, recon_med, sizeX, sizeY, sizeZ);

}
//--------------------------------------------------------------------------------------
__global__ void cuEst_n_3D_POCS(cufftComplex * cu_n, cufftComplex *cu_n2, cufftComplex * cu_F, float cu_k2, float cu_med, int sizeX, int sizeY, int sizeZ)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

	if (i<sizeX && j<sizeY && k<sizeZ)
	{
		unsigned int idx = i + j*sizeX + k*sizeY*sizeZ;
		cufftComplex temp;
		//suppose that z = c+di, and we want to find sqrt(z) = a+bi
		/*float c, d, a, b;
		c = (cu_F[idx].x / (-cu_k2)) + 1;
		d = (cu_F[idx].y / (-cu_k2));

		a = sqrt((sqrt((c*c) + (d*d)) + c) / 2);
		if (d != 0)
		{
			b = sqrt((sqrt((c*c) + (d*d)) - c) / 2) * (d / abs(d));
		}
		else
		{
			b = 0.0;
		}

		cu_n[idx].x = a * cu_med;
		cu_n[idx].y = b * cu_med;*/
		temp = ComplexProduction(cu_F[idx], -cu_k2);
		temp.x = temp.x + 1;
		temp = ComplexSqrt(temp);
		cu_n[idx] = ComplexProduction(temp, cu_med);


		if (cu_n[idx].x < cu_med)
		{
			cu_n2[idx].x = cu_med;
			cu_n2[idx].y = 0.f;
		}
		else{
			cu_n2[idx].x = cu_n[idx].x;
			cu_n2[idx].y = 0.f;
		}
	}
}
//--------------------------------------------------------------------------------------
float Est_Dp_Dd_POCS(cufftComplex *cu_F, cufftComplex *cu_F2, int sizeX, int sizeY, int sizeZ)
{
	int blocksInX = (sizeX + 8 - 1) / 8;
	int blocksInY = (sizeY + 8 - 1) / 8;
	int blocksInZ = (sizeZ + 8 - 1) / 16;
	dim3 grid(blocksInX, blocksInY, blocksInZ);
	dim3 block(8, 8, 16);

	float *temp;
	cudaMalloc((void **)&temp, sizeof(float)*sizeX*sizeY*sizeZ);
	cuEst_Dp_Dd_POCS << <grid, block >> >(temp, cu_F, cu_F2, sizeX, sizeY, sizeZ);

	//estimate the value 
	thrust::device_ptr<float> sum_ptr = thrust::device_pointer_cast(temp);
	double sum = sqrt(thrust::reduce(sum_ptr, sum_ptr + sizeX*sizeY*sizeZ, 0, thrust::plus<float>()));
	cudaFree(temp);

	return sum;
}
//--------------------------------------------------------------------------------------
__global__ void cuEst_Dp_Dd_POCS(float *temp, cufftComplex *cu_F, cufftComplex *cu_F2, int sizeX, int sizeY, int sizeZ)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

	if (i < sizeX && j < sizeY && k < sizeZ)
	{
		unsigned int idx = i + j*sizeX + k*sizeX*sizeY;
		temp[idx] =ComplexABS( ComplexSubtraction(cu_F2[idx], cu_F[idx])) * ComplexABS(ComplexSubtraction(cu_F2[idx], cu_F[idx]) );
	}
}
//--------------------------------------------------------------------------------------
void ConvertN2F(cufftComplex *cuF, cufftComplex *cuF2, cufftComplex *cuN, cufftComplex *cuN2, float recon_k2, float recon_med
	, int sizeX, int sizeY, int sizeZ)
{
	int blocksInX = (sizeX + 8 - 1) / 8;
	int blocksInY = (sizeY + 8 - 1) / 8;
	int blocksInZ = (sizeZ + 8 - 1) / 8;
	dim3 grid(blocksInX, blocksInY, blocksInZ);
	dim3 block(8, 8, 8);

	cuConvertN2F << <grid, block >> >(cuF, cuF2, cuN, cuN2, recon_k2, recon_med, sizeX, sizeY, sizeZ);
}
//--------------------------------------------------------------------------------------
__global__ void cuConvertN2F(cufftComplex *cuF, cufftComplex *cuF2, cufftComplex *cuN, cufftComplex *cuN2, float cu_k2, float cu_med
	, int sizeX, int sizeY, int sizeZ)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

	if (i < sizeX && j < sizeY && k < sizeZ)
	{
		unsigned int idx = i + j*sizeX + k*sizeX*sizeY;

		cuF[idx].x = (((cuN[idx].x*cuN[idx].x) - (cuN[idx].y*cuN[idx].y)) / (cu_med)-1)*(-cu_k2);
		cuF[idx].y = (cuN[idx].x*cuN[idx].y * 2) / (cu_med)*(-cu_k2);

		cuF2[idx].x = (((cuN2[idx].x*cuN2[idx].x) - (cuN2[idx].y*cuN2[idx].y)) / (cu_med)-1)*(-cu_k2);
		cuF2[idx].y = (cuN2[idx].x*cuN2[idx].y * 2) / (cu_med)*(-cu_k2);
	}
}
//--------------------------------------------------------------------------------------
void GradientDescentTV(cufftComplex *out, cufftComplex *cuF2, int sizeX, int sizeY, int sizeZ)
{
	int blocksInX = (sizeX + 8 - 1) / 8;
	int blocksInY = (sizeY + 8 - 1) / 8;
	int blocksInZ = (sizeZ + 8 - 1) / 8;
	dim3 grid(blocksInX, blocksInY, blocksInZ);
	dim3 block(8, 8, 8);

	cuGradientDescentTV << <grid, block >> >(out, cuF2, sizeX, sizeY, sizeZ);
}
//--------------------------------------------------------------------------------------
__global__ void cuGradientDescentTV(cufftComplex *out, cufftComplex *cuF2, int sizeX, int sizeY, int sizeZ)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;
	unsigned int idx1, idx2, idx3, idx4;
	cufftComplex eps;
	eps.x = 1e-8, eps.y = 0;
	
	if (i < sizeX && j < sizeY && k < sizeZ && i>1 && j>1 && k>1)
	{
		cufftComplex term1_deno, term2_deno, term3_deno, term4_deno;
		cufftComplex term1_num, term2_num, term3_num, term4_num;
		cufftComplex a, b, c, d;

		//term1
		idx1 = i + j*sizeX + k*sizeX*sizeY;
		a = ComplexSubtraction(cuF2[idx1], cuF2[idx1 - sizeX]);
		b = ComplexSubtraction(cuF2[idx1], cuF2[idx1 - 1]);
		c = ComplexSubtraction(cuF2[idx1], cuF2[idx1 - sizeX*sizeY]);
		a = ComplexMultiplication(a, a);
		b = ComplexMultiplication(b, b);
		c = ComplexMultiplication(c, c);
		term1_deno = ComplexSqrt(ComplexAddition(ComplexAddition(ComplexAddition(a, b), c), eps));
		

		//term2
		idx2 = idx1 + sizeX;
		a = ComplexSubtraction(cuF2[idx2], cuF2[idx2 - sizeX]);
		b = ComplexSubtraction(cuF2[idx2], cuF2[idx2 - 1]);
		c = ComplexSubtraction(cuF2[idx2], cuF2[idx2 - sizeX*sizeY]);
		a = ComplexMultiplication(a, a);
		b = ComplexMultiplication(b, b);
		c = ComplexMultiplication(c, c);
		term2_deno = ComplexSqrt(ComplexAddition(ComplexAddition(ComplexAddition(a, b), c), eps));
		
		//term3
		idx3 = idx1 + 1;
		a = ComplexSubtraction(cuF2[idx3], cuF2[idx3 - sizeX]);
		b = ComplexSubtraction(cuF2[idx3], cuF2[idx3 - 1]);
		c = ComplexSubtraction(cuF2[idx3], cuF2[idx3 - sizeX*sizeY]);
		a = ComplexMultiplication(a, a);
		b = ComplexMultiplication(b, b);
		c = ComplexMultiplication(c, c);
		term3_deno = ComplexSqrt(ComplexAddition(ComplexAddition(ComplexAddition(a, b), c), eps));

		//term4
		idx4 = idx1 + 1;
		a = ComplexSubtraction(cuF2[idx4], cuF2[idx4 - sizeX]);
		b = ComplexSubtraction(cuF2[idx4], cuF2[idx4 - 1]);
		c = ComplexSubtraction(cuF2[idx4], cuF2[idx4 - sizeX*sizeY]);
		a = ComplexMultiplication(a, a);
		b = ComplexMultiplication(b, b);
		c = ComplexMultiplication(c, c);
		term4_deno = ComplexSqrt(ComplexAddition(ComplexAddition(ComplexAddition(a, b), c), eps));


		term1_num = ComplexProduction( ComplexSubtraction(ComplexSubtraction(ComplexSubtraction(cuF2[idx1], cuF2[idx1-sizeX])
			, cuF2[idx1-1]), cuF2[idx1-sizeX*sizeY]),3.0);
			//3 * f_3D_2(ss, tt, kk) - f_3D_2(ss, tt - 1, kk) - f_3D_2(ss - 1, tt, kk) - f_3D_2(ss, tt, kk - 1);
		term2_num = ComplexSubtraction(cuF2[idx2], cuF2[idx1]); //f_3D_2(ss, tt + 1, kk) - f_3D_2(ss, tt, kk);
		term3_num = ComplexSubtraction(cuF2[idx3], cuF2[idx1]); //f_3D_2(ss + 1, tt, kk) - f_3D_2(ss, tt, kk);
		term4_num = ComplexSubtraction(cuF2[idx4], cuF2[idx1]); //f_3D_2(ss, tt, kk + 1) - f_3D_2(ss, tt, kk);

		a = ComplexSubtraction(term1_num, term1_deno);
		b = ComplexSubtraction(term2_num, term2_deno);
		c = ComplexSubtraction(term3_num, term3_deno);
		d = ComplexSubtraction(term4_num, term4_deno);

		out[idx1] = ComplexSubtraction(ComplexSubtraction(ComplexSubtraction(a, b), c), d);

		out[idx1].x = out[idx1].x / ComplexABS(out[idx1]);
		out[idx1].y = out[idx1].y / ComplexABS(out[idx1]);
	}
}
//--------------------------------------------------------------------------------------
void EstF_TV(cufftComplex *cuF2, cufftComplex *cuN, float dtvg, int sizeX, int sizeY, int sizeZ)
{
	int blocksInX = (sizeX + 8 - 1) / 8;
	int blocksInY = (sizeY + 8 - 1) / 8;
	int blocksInZ = (sizeZ + 8 - 1) / 8;
	dim3 grid(blocksInX, blocksInY, blocksInZ);
	dim3 block(8, 8, 8);

	cuEstF_TV << <grid, block >> >(cuF2, cuN, dtvg, sizeX, sizeY, sizeZ);
}
//--------------------------------------------------------------------------------------
__global__ void cuEstF_TV(cufftComplex *cuF2, cufftComplex *cuN, float dtvg, int sizeX, int sizeY, int sizeZ)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

	if (i < sizeX && j < sizeY && k < sizeZ)
	{
		unsigned int idx = i + j*sizeX + k*sizeX*sizeY;
		cufftComplex temp = ComplexSubtraction(cuF2[idx], ComplexProduction(cuN[idx], dtvg));
		cuF2[idx] = temp;
	}
}
//--------------------------------------------------------------------------------------
void EstF_Beta(cufftComplex *cuF, cufftComplex *cuF2, float beta, int sizeX, int sizeY, int sizeZ)
{
	int blocksInX = (sizeX + 8 - 1) / 8;
	int blocksInY = (sizeY + 8 - 1) / 8;
	int blocksInZ = (sizeZ + 8 - 1) / 8;
	dim3 grid(blocksInX, blocksInY, blocksInZ);
	dim3 block(8, 8, 8);

	cuEstF_Beta<<<grid,block>>>(cuF, cuF2, beta, sizeX, sizeY, sizeZ);
}
//--------------------------------------------------------------------------------------
__global__ void cuEstF_Beta(cufftComplex *cuF, cufftComplex *cuF2, float beta, int sizeX, int sizeY, int sizeZ)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

	if (i < sizeX && j < sizeY && k < sizeZ)
	{
		unsigned int idx = i + j*sizeX + k*sizeX*sizeY;
		if (ComplexABS(cuF2[idx]) != 0)
			cuF[idx] = ComplexAddition(ComplexProduction(cuF2[idx], beta), ComplexProduction(cuF2[idx], 1.0 - beta));
	}
}
//--------------------------------------------------------------------------------------
