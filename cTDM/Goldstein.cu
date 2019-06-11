#include "Goldstein.cuh"


//--------------------------------------------------------------------------------------
void cudaGoldsteinUnwrap2D(float *cu_phi, float *cu_UnWrapPhase, int sizeX, int sizeY,int frameNumber)
{
	int MaxBoxRadius = 901;

	float *IM_Mask;
	cudaMalloc((void **)&IM_Mask,sizeof(float)*sizeX*sizeY);
	cudaMemcpy(IM_Mask, d_OnesArray, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToDevice);
	
	float *ResiduesCharge;
	cudaMalloc((void **)&ResiduesCharge,sizeof(float)*sizeX*sizeY);
	cudaMemcpy(ResiduesCharge, d_ZerosArray, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToDevice);

    float *BranchCut;
	cudaMalloc((void **)&BranchCut,sizeof(float)*sizeX*sizeY);
	cudaMemcpy(BranchCut, d_ZerosArray, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToDevice);
	
	//------------------------------------------------------------------------	
	cudaPhaseResidues<<<gridGold,blockGold>>>(cu_phi,sizeX,sizeY,ResiduesCharge);

	cuBranchCuts(ResiduesCharge, BranchCut, IM_Mask, sizeX, sizeY, MaxBoxRadius);
	cudaFree(ResiduesCharge);
	
	cudaCheckIMMask<<<gridGold,blockGold>>>(BranchCut,IM_Mask,sizeX,sizeY);
	cudaFloodFill(BranchCut,cu_phi,cu_UnWrapPhase,IM_Mask,sizeX,sizeY);

	/*float *h_BranchCut = (float *)malloc(sizeX*sizeY*sizeof(float));
	cudaMemcpy(h_BranchCut, BranchCut, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToHost);
	float *h_phi = (float *)malloc(sizeX*sizeY*sizeof(float));
	cudaMemcpy(h_phi, cu_phi, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToHost);
	float *h_UnWrapPhase = (float *)malloc(sizeX*sizeY*sizeof(float));
	cudaMemcpy(h_UnWrapPhase, cu_UnWrapPhase, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToHost);
	float *h_IM_Mask = (float *)malloc(sizeX*sizeY*sizeof(float));
	cudaMemcpy(h_IM_Mask, IM_Mask, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToHost);
	
	FloodFill(h_BranchCut,sizeX, sizeY, h_phi, h_UnWrapPhase, h_IM_Mask);

	cudaMemcpy(cu_UnWrapPhase, h_UnWrapPhase, sizeX*sizeY*sizeof(float),cudaMemcpyHostToDevice);
	
	free(h_BranchCut);
	free(h_phi);
	free(h_UnWrapPhase);
	free(h_IM_Mask);*/

	cudaFree(BranchCut);
	cudaFree(IM_Mask);
}
//--------------------------------------------------------------------------------------
__global__ void cudaPhaseResidues(float *phi,int sizeX, int sizeY, float *ResiduesCharge)
{
    float res1,res2,res3,res4;
	float tempRes;

	unsigned int r = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int c = blockDim.y * blockIdx.y + threadIdx.y;

	if(r<sizeX && c<sizeY)
	{
		ResiduesCharge[r+c*sizeX] = 0.0;

		if(r!=sizeX-1 && c!=sizeY-1)
		{
			res1 = cudaMod( phi[r+c*sizeX]         - phi[(r+1)+c*sizeX]     + M_PI , M_PI*2) - M_PI;
			res2 = cudaMod( phi[(r+1)+c*sizeX]     - phi[(r+1)+(c+1)*sizeX] + M_PI , M_PI*2) - M_PI;
			res3 = cudaMod( phi[(r+1)+(c+1)*sizeX] - phi[r+(c+1)*sizeX]     + M_PI , M_PI*2) - M_PI;
			res4 = cudaMod( phi[r+(c+1)*sizeX]     - phi[r+c*sizeX]         + M_PI , M_PI*2) - M_PI;
		}
		else if(r == sizeX-1 && c != sizeY-1)
		{
			res1 = cudaMod( phi[r+c*sizeX]         - 0                      + M_PI , M_PI*2) - M_PI;
			res2 = cudaMod( 0                      - 0                      + M_PI , M_PI*2) - M_PI;
			res3 = cudaMod( 0                      - phi[r+(c+1)*sizeX]     + M_PI , M_PI*2) - M_PI;
			res4 = cudaMod( phi[r+(c+1)*sizeX]     - phi[r+c*sizeX] + M_PI         , M_PI*2) - M_PI;
		}
		else if(r != sizeX-1 && c == sizeY-1)
		{
			res1 = cudaMod( phi[r+c*sizeX]         - phi[(r+1)+c*sizeX]     + M_PI , M_PI*2) - M_PI;
			res2 = cudaMod( phi[(r+1)+c*sizeX]     - 0                      + M_PI , M_PI*2) - M_PI;
			res3 = cudaMod( 0                      - 0                      + M_PI , M_PI*2) - M_PI;
			res4 = cudaMod( 0                      - phi[r+c*sizeX]         + M_PI , M_PI*2) - M_PI;
		}
		else if(r == sizeX-1 && c == sizeY-1)
		{
			res1 = cudaMod( phi[r+c*sizeX]         - 0                      + M_PI , M_PI*2) - M_PI;
			res2 = cudaMod( 0                      - 0                      + M_PI , M_PI*2) - M_PI;
			res3 = cudaMod( 0                      - 0                      + M_PI , M_PI*2) - M_PI;
			res4 = cudaMod( 0                      - phi[r+c*sizeX]         + M_PI , M_PI*2) - M_PI;
		}

		tempRes=res1+res2+res3+res4;

		if(tempRes >= 6.0)
			ResiduesCharge[r+c*sizeX] = 1;
		else if (tempRes <= -6.0)
			ResiduesCharge[r+c*sizeX] =-1;

		if(r==sizeX-1 || c==sizeY-1)
			ResiduesCharge[r+c*sizeX] = 0;
	}
}
//--------------------------------------------------------------------------------------
void cuBranchCuts(float *ResiduesCharge, float *BranchCut, float *IM_Mask, int sizeX, int sizeY, int MaxBoxRadius)
{
	//memory setting
	float *ResiduesChargeMasked;
	cudaMalloc((void **)&ResiduesChargeMasked,sizeof(float)*sizeX*sizeY);
	cudaMemcpy(ResiduesChargeMasked,ResiduesCharge,sizeof(float)*sizeX*sizeY,cudaMemcpyDeviceToDevice);

	float *ResidueBalanced;
	cudaMalloc((void **)&ResidueBalanced,sizeof(float)*sizeX*sizeY);
	cudaMemcpy(ResidueBalanced, d_ZerosArray, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToDevice);

	//make IM_mask
	inverseLogic<<<gridGold,blockGold>>>(IM_Mask,BranchCut,sizeX,sizeY);	

	//decide the radius size
	MaxBoxRadius = int(min(float(MaxBoxRadius),floor(float(sizeX/2))));

	cudaBC<<<gridGold,blockGold>>>(BranchCut, ResidueBalanced, ResiduesChargeMasked, MaxBoxRadius,sizeX,sizeY);

	cudaFree(ResidueBalanced);
	cudaFree(ResiduesChargeMasked);
}
//--------------------------------------------------------------------------------------
__global__	void cudaBC(float *BranchCut, float *ResidueBalanced, float *ResiduesChargeMasked, int MaxBoxRadius, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx, m1, n1, m2, n2, mm, nn, ndel;

	if(i<sizeX && j<sizeY)
	{
		idx = i+sizeX*j;
		float chargeCounter = ResiduesChargeMasked[idx];
		if(chargeCounter != 0)
		{
			if(i<=0 || i>=sizeX-1 || j<=0 || j>=sizeY-1)
			{
				BranchCut		    [idx]=1;  // Make this point a branchcut to the edge
				ResidueBalanced     [idx]=1;  // Mark this residue as balanced
				ResiduesChargeMasked[idx]=0;  // Remove from the set of unmatched residues
			}

			unsigned int MaxRadius = min(MaxBoxRadius, __float2int_rn(sizeX*0.5));

			for(unsigned int r=1; r<MaxRadius; r++)
			{
				m1 = max(i-r,0);
				m2 = min(i+r,sizeX-1);
				n1 = max(j-r,0);
				n2 = min(j+r,sizeY-1);
				
				for(unsigned int mm=m1;mm<=m2;mm++)
				{
					if(chargeCounter != 0)
					{
						if(mm==m1 || mm==m2)
							ndel = 1;
						else
							ndel = n2-n1;
				
						for(unsigned int nn=n1;nn<=n2;nn=nn+ndel)
						{
							if(mm<=0 || mm>=sizeX-1)
							{						
								cuPlaceBranchCutInternal(BranchCut,sizeX,sizeY,i,j,mm,j);

								chargeCounter = 0;
								ResidueBalanced     [idx       ]=1;
								ResiduesChargeMasked[idx       ]=0;
								ResiduesChargeMasked[i+sizeX*nn]=0;
							}
							else if(nn<=0 || nn>=sizeY-1)
							{							
								cuPlaceBranchCutInternal(BranchCut,sizeX,sizeY,i,j,i,nn);

								chargeCounter = 0;
								ResidueBalanced     [idx       ]=1;
								ResiduesChargeMasked[idx       ]=0;
								ResiduesChargeMasked[mm+sizeX*j]=0;
							}
							else if(ResiduesChargeMasked[idx]*ResiduesChargeMasked[mm+sizeX*nn] == -1.0)
							{							
								cuPlaceBranchCutInternal(BranchCut,sizeX,sizeY,i,j,mm,nn);

								chargeCounter = 0;
								ResidueBalanced     [idx        ]=1;
								ResidueBalanced     [mm+sizeX*nn]=1;
								ResiduesChargeMasked[idx        ]=0;
								ResiduesChargeMasked[mm+sizeX*nn]=0;
							}
						}
					}//end-for-nn
				}//end-for-mm
			}
		}
	}
}
//--------------------------------------------------------------------------------------
__global__ void changeElement(float *arr, int idx, float val)
{
    arr[idx] = val;
}
//--------------------------------------------------------------------------------------
__global__ void inverseLogic(float* IM_mask, float *BranchCut, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int index;

	if(i<sizeX && j<sizeY)
	{
		index = i+ j*sizeX;

		BranchCut[index] = 1;
		if(IM_mask[index] == 1)
			BranchCut[index] = 0;			
	}
}
//--------------------------------------------------------------------------------------
__device__ void cuPlaceBranchCutInternal(float *BranchCut,int sizeX,int sizeY,
										unsigned int r1, unsigned int c1, unsigned int r2, unsigned int c2)
{
	BranchCut[r1+sizeX*c1]=1;
	BranchCut[r2+sizeX*c2]=1;

	float dR = __int2float_rn(r2-r1);
	float dC = __int2float_rn(c2-c1);

	float radius   = sqrtf(dR*dR+dC*dC);
	float costheta = dR/radius;
	float sintheta = dC/radius;

	unsigned int rFill, cFill;

	for(unsigned int ii=1;ii<=__float2int_rn(radius); ii++)
	{
		 rFill = r1 + __float2int_rn(ii*costheta);
		 cFill = c1 + __float2int_rn(ii*sintheta);

		 //if(rFill+sizeX*cFill>=0 && rFill+sizeX*cFill<sizeX*sizeY)
		 if(rFill>min(r1,r2)-1 && rFill<max(r1,r2)+1 && cFill>min(c1,c2)-1 && cFill<max(c1,c2)+1)
			BranchCut[rFill+sizeX*cFill]=1;
	}
}
//--------------------------------------------------------------------------------------
__global__ void cudaCheckIMMask(float *BranchCut, float *IM_Mask, int sizeX, int sizeY)
{
	unsigned int r = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int c = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;
	if(r<sizeX && c<sizeY)
	{
		idx = r+c*sizeX;
		if(BranchCut[idx]==1)
			IM_Mask[idx]=0;
	}
}
//--------------------------------------------------------------------------------------
void cudaFloodFill(float *BranchCut, float *phimap, float *IM_Unwrapped, float *IM_Mask, int sizeX, int sizeY)
{
	//memory setting
	float *UnwrappedBinary;
	cudaMalloc((void **)&UnwrappedBinary,sizeof(float)*sizeX*sizeY);
	cudaMemcpy(UnwrappedBinary, d_ZerosArray, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToDevice);
	
	float *Adjoin;
	cudaMalloc((void **)&Adjoin         ,sizeof(float)*sizeX*sizeY);
	cudaMemcpy(Adjoin         , d_ZerosArray, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToDevice);

	float *AdjoinTemp;
	cudaMalloc((void **)&AdjoinTemp     ,sizeof(float)*sizeX*sizeY);
	cudaMemcpy(AdjoinTemp     , d_ZerosArray, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToDevice);

	float *AdjoinStuck;
	cudaMalloc((void **)&AdjoinStuck    ,sizeof(float)*sizeX*sizeY);
	cudaMemcpy(AdjoinStuck    , d_ZerosArray, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToDevice);

	cudaMemcpy(IM_Unwrapped   , d_ZerosArray, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToDevice);

	float *flagArray;
	cudaMalloc((void **)&flagArray      ,sizeof(float)*sizeX*sizeY);
	
	//make seed points for Flood Fill Alg.
	cuFF_SeedPoints<<<gridGold,blockGold>>>(BranchCut, Adjoin, phimap, IM_Unwrapped, UnwrappedBinary, sizeX, sizeY);
	
	//count all of the non-zero elements
	int NumberOfBranchCuts = countNonZero(BranchCut,sizeX,sizeY);
	unsigned int CountLimit = 0;	
	int L1=0, L2=0;	

	while(cudaSumAdjoin(Adjoin,sizeX,sizeY)!=0)
	{		
		unsigned int counterTerminate=0;
				
		while(CountLimit<2)
		{
			counterTerminate++;
			if(counterTerminate>5000)
			{
				cout<<"失敗; Counter time: "<<counterTerminate<<endl;
				return;
			}
			
			cudaMemcpy(AdjoinStuck, AdjoinTemp, sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToDevice);
			cudaMemcpy(AdjoinTemp , Adjoin    , sizeX*sizeY*sizeof(float),cudaMemcpyDeviceToDevice);
			L1 = countNonZero(AdjoinStuck,sizeX,sizeY);			
			L2 = countNonZero(Adjoin     ,sizeX,sizeY);
			//cout<<"L1:"<<L1<<"   L2:"<<L2<<endl;
			//system("pause");
			if(L1 == L2)
			{
				cuFF_MatchArray<<<gridGold,blockGold>>>(Adjoin, AdjoinStuck, flagArray, sizeX, sizeY);
				if(countNonZero(flagArray,sizeX,sizeY)==0)	CountLimit++;
			}
			else
			{
				CountLimit=0;
			}
			cuFF_Internal_1<<<gridGold,blockGold>>>(Adjoin, BranchCut, UnwrappedBinary, IM_Unwrapped, phimap, sizeX, sizeY);
			
			/*if(counterTerminate==1000)
			{
				//char *Save_phase = (char *)malloc(150*sizeof(char));										//output path
				//sprintf(Save_phase,"IM_Unwrapped%d.1024.1024.raw",counterTerminate);
				//DeviceMemOut("IM_Unwrapped.1024.1024.raw",IM_Unwrapped,sizeX,sizeY);
				//system("pause");
			}*/

		}//while(CountLimit<2)	
	}//while

	cudaFree(AdjoinTemp);
	cudaFree(AdjoinStuck);
	cudaFree(flagArray);

	//***************************************************************************
	// Finally, fill in the branch cut pixels that adjoin the unwrapped pixels.
	// This can be done because the branch cuts actually lie between the pixels,
	// and not on top of them.
	//***************************************************************************	
	if(NumberOfBranchCuts>0)
	{	
		cuFFcheckArray<<<gridGold,blockGold>>>(BranchCut, Adjoin, sizeX, sizeY);
		cuFF_Internal_2<<<gridGold,blockGold>>>(Adjoin, UnwrappedBinary, IM_Unwrapped, phimap, sizeX, sizeY);
	}

	cudaFree(UnwrappedBinary);
	cudaFree(Adjoin);
}
//--------------------------------------------------------------------------------------
__global__ void cuFF_MatchArray(float* Adjoin, float *AdjoinStuck, float *flagArray, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;

	if(i<sizeX && j<sizeY)
	{
		idx = i+sizeX*j;
		flagArray[idx] = 0;
		if(AdjoinStuck[idx]!=0 && Adjoin[idx]!=AdjoinStuck[idx])
		{
			flagArray[idx] = 1;
		}
	}
}
//--------------------------------------------------------------------------------------
__global__ void cuFF_SeedPoints(float *BranchCut, float *Adjoin, float *phimap, float *IM_Unwrapped, float *UnwrappedBinary, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;
	bool flag;
	
	if(i<sizeX && j<sizeY)
	{
		if(i==sizeX/2 && j==20)
		{
			idx = i+sizeX*j;
			flag = true;
			do{
				if(BranchCut[idx]!=1)
				{
					Adjoin[idx-1    ]    = 1;
					Adjoin[idx+1    ]    = 1;
					Adjoin[idx-sizeX]    = 1;
					Adjoin[idx+sizeX]    = 1;
					IM_Unwrapped[idx]    = phimap[idx];
					UnwrappedBinary[idx] = 1;

					flag = false;
				}
				else
				{
					if(sizeX/2>i)
						idx += 1;
					else
						idx -= 1;

					if(sizeY/2>j)
						idx += sizeX;
					else
						idx -= sizeX;
				}
			}while(flag);			
			
		}
	}
}
//--------------------------------------------------------------------------------------
__global__ void cuFF_Internal_1(float* Adjoin, float *BranchCut, float *UnwrappedBinary, float *IM_Unwrapped, float *phimap, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;	
	unsigned int caseNo, checkidxDel;

	if(i<sizeX && j<sizeY)
	{		
		idx = i+sizeX*j;
		if(Adjoin[idx]!=0)
		{			
			checkidxDel = 0;
			caseNo      = 5;
			//First search on the right for an adjoining unwrapped phase pixel
			if((i+1)<sizeX)
			{
				if(BranchCut[idx+1]==0)
				{
					if(UnwrappedBinary[idx+1]==1)
					{
						caseNo = 1;
					}
					else
					{
						Adjoin[idx+1] = 1;
						checkidxDel++;
					}
				}
			}
				 
			//Then search on the left
			if(i>0)
			{
				if(BranchCut[idx-1]==0)
				{
					if(UnwrappedBinary[idx-1]==1)
					{
						caseNo = min(caseNo,2);
					}
					else
					{
						Adjoin[idx-1] = 1;
						checkidxDel++;
					}
				}
			}

			//Then search below
			if((j+1)<sizeY)
			{
				if(BranchCut[idx+sizeX]==0)
				{
					if(UnwrappedBinary[idx+sizeX]==1)
					{
						caseNo = min(caseNo,3);
					}
					else
					{
						Adjoin[idx+sizeX] = 1;
						checkidxDel++;
					}
				}
			}

			//Finally search above
			if(j>0)
			{
				if(BranchCut[idx-sizeX]==0)
				{
					if(UnwrappedBinary[idx-sizeX]==1)
					{
						caseNo = min(caseNo,4);
					}
					else
					{	
						Adjoin[idx-sizeX] = 1;
						checkidxDel++;
					}
				}
			}
				
			
			if(checkidxDel>0)
			{
				float phaseRef, D, deltap;

				switch (caseNo){
					case 1:
						phaseRef  = IM_Unwrapped[idx+1];
					break;
					case 2:
						phaseRef  = IM_Unwrapped[idx-1];
					break;
					case 3:
						phaseRef  = IM_Unwrapped[idx+sizeX];
					break;
					case 4:
						phaseRef  = IM_Unwrapped[idx-sizeX];						
					break;
				}
				D                 = phimap[idx]-phaseRef;
				deltap            = atan2(sin(D),cos(D));
				IM_Unwrapped[idx] = phaseRef + deltap;
				UnwrappedBinary[idx] = 1; 
			}
			Adjoin[idx] = 0;
		}
	}

}
//--------------------------------------------------------------------------------------
__global__ void cuFF_Internal_2(float* Adjoin, float *UnwrappedBinary, float *IM_Unwrapped, float *phimap, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;	
	unsigned int caseNo, checkidxDel;

	if(i<sizeX && j<sizeY)
	{
		idx = i+sizeX*j;
		if(Adjoin[idx]!=0)
		{
			checkidxDel = 0;
			caseNo      = 5;
			//First search below for an adjoining unwrapped phase pixel
			if((i+1)<sizeX)
			{
				if(UnwrappedBinary[idx+1] == 1)
				{
					caseNo = 1;
				}
				else
				{
					Adjoin[idx+1]=1;
					checkidxDel++;
				}
			}

			//Then search above
			if(i>0)
			{
				if(UnwrappedBinary[idx-1] == 1)
				{
					caseNo = min(caseNo,2);
				}
				else
				{
					Adjoin[idx-1]=1;
					checkidxDel++;
				}
			}

			//Then search on the right
			if((j+1)<sizeY)
			{
				if(UnwrappedBinary[idx+sizeX] == 1)
				{
					caseNo = min(caseNo,3);
				}
				else
				{
					Adjoin[idx+sizeX]=1;
					checkidxDel++;
				}				
			}
			
			//Finally search on the left
			if(j>0)
			{
				if(UnwrappedBinary[idx-sizeX] == 1)
				{
					caseNo = min(caseNo,4);
				}
				else
				{	
					Adjoin[idx-sizeX]=1;
					checkidxDel++;
				}
			}	

			if(checkidxDel>0)
			{
				float phaseRef, D, deltap;

				switch (caseNo){
					case 1:
						phaseRef  = IM_Unwrapped[idx+1];
					break;
					case 2:
						phaseRef  = IM_Unwrapped[idx-1];
					break;
					case 3:
						phaseRef  = IM_Unwrapped[idx+sizeX];
					break;
					case 4:
						phaseRef  = IM_Unwrapped[idx-sizeX];						
					break;
				}
				D                 = phimap[idx]-phaseRef;
				deltap            = atan2(sin(D),cos(D));
				IM_Unwrapped[idx] = phaseRef + deltap;
				UnwrappedBinary[idx] = 1;
			}
			Adjoin[idx] = 0;
		}
	}
}
//--------------------------------------------------------------------------------------
__global__ void cuFFcheckArray(float *BranchCut, float *Adjoin, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;

	if(i>0 && i<sizeX-1 && j>0 && j<sizeY-1)
	{
		idx = i+sizeX*j;
		if(BranchCut[idx] == 1 &&
			( BranchCut[idx+1    ] == 0 || 
			  BranchCut[idx-1    ] == 0 ||
			  BranchCut[idx+sizeX] == 0 ||
			  BranchCut[idx-sizeX] == 0 ))
				  Adjoin[idx]=1;
	}
}
//--------------------------------------------------------------------------------------
void FloodFill(float *BranchCut,int sizeX, int sizeY, float *phi, float *IMUnwrapped,float *IMMask)
{	
	int xRef,yRef,rowRef,colRef;

	int CountLimit;

	float *UnwrappedBinary = (float *)malloc(sizeX*sizeY*sizeof(float));
	//zeros_f(UnwrappedBinary,sizeX*sizeY);
	for(int i=0;i<sizeX*sizeY;i++)	UnwrappedBinary[i] = 0.0;
	
	float *Adjoin = (float *)malloc(sizeX*sizeY*sizeof(float));
	//zeros_f(Adjoin,sizeX*sizeY);
	for(int i=0;i<sizeX*sizeY;i++)	Adjoin[i] = 0.0;
	
	//--------------------------------------
	xRef   = sizeX/2;	    yRef   = 2;	
	rowRef = yRef;  colRef = xRef;
	
	while(BranchCut[xRef+sizeX*yRef]==1)
	{
		xRef=xRef+1;
		yRef=yRef+1;
	}

	int NumberOfBranchCuts=0;

	//#pragma omp parallel for
	for(int cc=0;cc<sizeY;cc++)
		for(int rr=0;rr<sizeX;rr++)
		{
			if(BranchCut[rr+sizeX*cc]!=0)
				NumberOfBranchCuts++;
		}
	//#pragma omp barrier


	if(rowRef > 0)
	  Adjoin[(rowRef-1)+sizeX*colRef] = 1.0;

	if(rowRef < (sizeX-1))
	  Adjoin[(rowRef+1)+sizeX*colRef] = 1.0;
	
	if(colRef > 0)
	  Adjoin[rowRef+sizeX*(colRef-1)] = 1.0;
	
	if(colRef < (sizeY-1))
	  Adjoin[rowRef+sizeX*(colRef+1)] = 1.0;

	IMUnwrapped[rowRef+sizeX*colRef] = phi[rowRef+sizeX*colRef]; 
	UnwrappedBinary[rowRef+sizeX*colRef] = 1; 
	//-----------------------------------------------------
	int flagL1L2 = 1;
	CountLimit = 0;	
	
	int2 *coordinateAdjoinStuck = (int2 *)malloc(sizeof(int2)); int L1 = 0;
	int2 *coordinateAdjoin      = (int2 *)malloc(sizeof(int2)); int L2 = 0;
	int counter2;

	while(sumAdjoin(Adjoin,sizeX,sizeY)!=0)
	{
		int counterTerminate=0;
		
		while(CountLimit<2)
		{
			counterTerminate++;
			if(counterTerminate>5000)
			{
				cout<<"失敗; Counter time: "<<counterTerminate<<endl;
				return;
			}
			
			if(flagL1L2==0)
			{
				L1 = L2;
				coordinateAdjoinStuck = (int2 *)realloc(coordinateAdjoinStuck, L2*sizeof(int2));
				
				for(int ii=0;ii<L2;ii++)
				{
					coordinateAdjoinStuck[ii].x = coordinateAdjoin[ii].x;
					coordinateAdjoinStuck[ii].y = coordinateAdjoin[ii].y;
				}
			}
			flagL1L2=0;
		

			L2 = 0;
			//#pragma omp parallel for
			for(int i=0;i<sizeX*sizeY;i++)
			{
				if(Adjoin[i]!=0)
					L2++;
			}
			//#pragma omp barrier

			coordinateAdjoin = (int2 *)realloc(coordinateAdjoin, L2*sizeof(int2));

			counter2 = 0;
			//#pragma omp parallel for
			for(int cc=0;cc<sizeY;cc++)
				for(int rr=0;rr<sizeX;rr++)
				{
					if(Adjoin[rr+sizeX*cc]!=0)
					{
						coordinateAdjoin[counter2].x = rr;
						coordinateAdjoin[counter2].y = cc;
						//cout<<"L: "<<L2<<"Order: "<<counter2<<"\t (r,c): "<<coordinateAdjoin[counter2].x<<" ,"<<coordinateAdjoin[counter2].y<<endl;
						counter2 ++ ;						
						if(counter2 == L2) break;
					}
				}
			//#pragma omp barrier

			//system("pause");
			
			if(L1==L2)
			{
				int flagT=1;
				for(int ii=0;ii<L1;ii++)
				{
					if((coordinateAdjoinStuck[ii].x != coordinateAdjoin[ii].x) ||
					   (coordinateAdjoinStuck[ii].y != coordinateAdjoin[ii].y) )
						flagT = 0;
				}
				
				if(flagT==1)
					CountLimit=CountLimit+1;
			}
			else
			{
				CountLimit=0;
			}
			
			//#pragma omp parallel for
			for(int ii=0;ii<L2;ii++)
			{
				int rrActive = coordinateAdjoin[ii].x;
				int ccActive = coordinateAdjoin[ii].y;

				float phaseRef,D,deltap;
				float phasev [4];//  = {999999.0};
				float IM_magv[4];//  = {999999.0};
				float idxDel [4];//  = {0};
				int   checkidxDel;
				for(int kk=0;kk<4;kk++)
				{
					phasev [kk] = 999999;
					IM_magv[kk] = 999999;
					idxDel [kk] = 0;
				}
				
				//First search below for an adjoining unwrapped phase pixel
				if((rrActive+1)<=(sizeX-1))
				{
					//cout<<rrActive+1<<"\t"<<ccActive<<endl;
					if(BranchCut[(rrActive+1)+sizeX*ccActive]==0)
					{
						if(UnwrappedBinary[(rrActive+1)+sizeX*ccActive]==1)
						{

							phaseRef   = IMUnwrapped[(rrActive+1)+sizeX*ccActive];    
							D          = phi[rrActive+sizeX*ccActive]-phaseRef;
							deltap     = atan2(sin(D),cos(D));   
							phasev [0] = phaseRef + deltap; 
							IM_magv[0] = 1;
						}
						else
						{
							Adjoin[(rrActive+1)+sizeX*ccActive] = 1;
						}
					}
				}
				 
				//Then search above
				if((rrActive-1)>=0)
				{
					if(BranchCut[(rrActive-1)+sizeX*ccActive]==0)
					{
						if(UnwrappedBinary[(rrActive-1)+sizeX*ccActive]==1)
						{

							phaseRef   = IMUnwrapped[(rrActive-1)+sizeX*ccActive];                        
							D          = phi[rrActive+sizeX*ccActive]-phaseRef;
							deltap     = atan2(sin(D),cos(D));  
							phasev [1] = phaseRef + deltap;  
							IM_magv[1] = 1;
						}
						else
						{
							Adjoin[(rrActive-1)+sizeX*ccActive] = 1;
						}
					}
				}

				//Then search on the right
				if((ccActive+1)<=(sizeY-1))
				{
					if(BranchCut[rrActive+sizeX*(ccActive+1)]==0)
					{
						if(UnwrappedBinary[rrActive+sizeX*(ccActive+1)]==1)
						{
							phaseRef   = IMUnwrapped[rrActive+sizeX*(ccActive+1)];
							D          = phi[rrActive+sizeX*ccActive]-phaseRef;
							deltap     = atan2(sin(D),cos(D));
							phasev [2] = phaseRef + deltap;
							IM_magv[2] = 1;
						}
						else
						{
							Adjoin[rrActive+sizeX*(ccActive+1)] = 1;
						}
					}
				}

				//Finally search on the left
				if((ccActive-1)>=0)
				{
					if(BranchCut[rrActive+sizeX*(ccActive-1)]==0)
					{
						if(UnwrappedBinary[rrActive+sizeX*(ccActive-1)]==1)
						{
							phaseRef   = IMUnwrapped[rrActive+sizeX*(ccActive-1)];
							D          = phi[rrActive+sizeX*ccActive]-phaseRef;
							deltap     = atan2(sin(D),cos(D));
							phasev [3] = phaseRef + deltap;
							IM_magv[3] = 1;
						}
						else
						{	
							Adjoin[rrActive+sizeX*(ccActive-1)] = 1;
						}
					}
				}
				
				checkidxDel = 0;
				  
				for(int ii=0;ii<4;ii++)
				{
					idxDel[ii] = 0;
					if(phasev[ii]!=999999.0)
					{
						checkidxDel++;
						idxDel[ii] = 1;
					}
				}
				
				int flagSearch  = 0;	
				int pointRecord;
				if(checkidxDel>0)
				{
					flagSearch =  0;
					pointRecord = 0;
					for(int kk=0;kk<4;kk++)
					{
						if(idxDel[kk]==1 && flagSearch==0)
						{
							pointRecord =kk;
							flagSearch  =1;
						}
					}

					IMUnwrapped    [rrActive+sizeX*ccActive] = (float)phasev[pointRecord]; //要再check  
					UnwrappedBinary[rrActive+sizeX*ccActive] = 1;  
					Adjoin         [rrActive+sizeX*ccActive] = 0;
				}
				else
				{
					Adjoin         [rrActive+sizeX*ccActive] = 0;
				}
				
			}//end-for-ii 			
			//#pragma omp barrier
		}//while(CountLimit<2)		

	}//while
	free(coordinateAdjoin);
	free(coordinateAdjoinStuck);
	
	//***************************************************************************
	// Finally, fill in the branch cut pixels that adjoin the unwrapped pixels.
	// This can be done because the branch cuts actually lie between the pixels,
	// and not on top of them.
	//***************************************************************************
	
	int2 *coordinateAdjoin2 = (int2 *)malloc(sizeof(int2));
	if(NumberOfBranchCuts>0)
	{	
		//#pragma omp parallel for
		for(int cc=1;cc<sizeY-1;cc++)
			for(int rr=1;rr<sizeX-1;rr++)			
			{
			  if(BranchCut[rr+sizeX*cc] == 1 &&
				  ((BranchCut[(rr+1)+sizeX* cc   ] == 0 || BranchCut[(rr-1)+sizeX* cc   ] == 0 || 
				    BranchCut[ rr   +sizeX*(cc-1)] == 0 || BranchCut[ rr   +sizeX*(cc+1)] == 0)  ))
					Adjoin[rr+sizeX*cc] = 1;
			}
		//#pragma omp barrier
	
		L2 = 0;		
		//#pragma omp parallel for
		for(int i=0;i<sizeX*sizeY;i++)
		{
			if(Adjoin[i]!=0)
				L2++;
		}
		//#pragma omp barrier
		coordinateAdjoin2 = (int2 *)realloc(coordinateAdjoin2, L2*sizeof(int2));

		int counter2 = 0;
		//#pragma omp parallel for
		for(int cc=0; cc<sizeY; cc++)
			for(int rr=0; rr<sizeX; rr++)
			{
				if(Adjoin[rr+sizeX*cc] != 0)
				{
					coordinateAdjoin2[counter2].x = rr;
					coordinateAdjoin2[counter2].y = cc;
					counter2 ++ ;
					if(counter2 == L2) break;
				}
			}
		//#pragma omp barrier

		//#pragma omp parallel for
		for(int ii=0;ii<L2;ii++)
		{
			int rrActive = coordinateAdjoin2[ii].x;
			int ccActive = coordinateAdjoin2[ii].y;

			float phaseRef, D, deltap;	
			float phasev [4];//  = {999999.0};
			float IM_magv[4];//  = {999999.0};
			float idxDel [4];//  = {0};
			int   checkidxDel;
			
			for(int kk=0;kk<4;kk++)
			{
				phasev [kk] = 999999;
				IM_magv[kk] = 999999;
				idxDel [kk] = 0;
			}
		
			//First search below for an adjoining unwrapped phase pixel
			if((rrActive+1)<=(sizeX-1))
			{
				if(UnwrappedBinary[(rrActive+1)+sizeX*ccActive]==1)
				{
					phaseRef   = IMUnwrapped[(rrActive+1)+sizeX*ccActive];
					D          = phi[rrActive+sizeX*ccActive]-phaseRef;
					deltap     = atan2(sin(D),cos(D));
					phasev [0] = phaseRef + deltap;
					IM_magv[0] = 1;
				}
				else
				{
					Adjoin[(rrActive+1)+sizeX*ccActive] = 1;
				}
			}

			//Then search above
			if((rrActive-1)>=0)
			{
				if(UnwrappedBinary[(rrActive-1)+sizeX*ccActive]==1)
				{
					phaseRef   = IMUnwrapped[(rrActive-1)+sizeX*ccActive];
					D          = phi[rrActive+sizeX*ccActive]-phaseRef;
					deltap     = atan2(sin(D),cos(D));
					phasev [1] = phaseRef + deltap;
					IM_magv[1] = 1;
				}
				else
				{
					Adjoin[(rrActive-1)+sizeX*ccActive] = 1; 
				}
			}

			//Then search on the right
			if((ccActive+1)<=(sizeY-1))
			{
				if(UnwrappedBinary[rrActive+sizeX*(ccActive+1)]==1)
				{
					phaseRef   = IMUnwrapped[rrActive+sizeX*(ccActive+1)];
					D          = phi[rrActive+sizeX*ccActive]-phaseRef;
					deltap     = atan2(sin(D),cos(D));
					phasev[2]  = phaseRef + deltap;
					IM_magv[2] = 1;
				}
				else 
				{
					Adjoin[rrActive+sizeX*(ccActive+1)] = 1;
				}
			}


			//Finally search on the left
			if((ccActive-1)>=0)
			{
				if(UnwrappedBinary[rrActive+sizeX*(ccActive-1)]==1)
				{
					phaseRef   = IMUnwrapped[rrActive+sizeX*(ccActive-1)];
					D          = phi[rrActive+sizeX*ccActive]-phaseRef;
					deltap     = atan2(sin(D),cos(D));
					phasev[3]  = phaseRef + deltap;
					IM_magv[3] = 1;
				}
				else
				{	
					Adjoin[rrActive+sizeX*(ccActive-1)] = 1;
				}
			}
			  
			checkidxDel = 0;
			for(int ii=0;ii<4;ii++)
			{
				idxDel[ii] = 0;
				if(phasev[ii]!=999999.0)
				{
					checkidxDel++;
					idxDel[ii] = 1;
				}
			}

			int flagSearch  = 0;	
			int pointRecord;
			if(checkidxDel>0)
			{
				flagSearch =  0;
				pointRecord = 0;
				for(int kk=0;kk<4;kk++)
				{
					if(idxDel[kk]==1 && flagSearch==0)
					{
						pointRecord =kk;
						flagSearch  =1;
					}
				}

				IMUnwrapped    [rrActive+sizeX*ccActive] = (float)phasev[pointRecord]; //要再check  
				UnwrappedBinary[rrActive+sizeX*ccActive] = 1;  
				Adjoin         [rrActive+sizeX*ccActive] = 0;
			}
			else  
			{
				Adjoin         [rrActive+sizeX*ccActive] = 0;  
			}

		}//for(int ii=0;ii<L2;ii++)
		//#pragma omp barrier
	}//if(NumberOfBranchCuts>0)

	free(UnwrappedBinary);
	free(Adjoin);
	free(coordinateAdjoin2);
}
//--------------------------------------------------------------------------------------
int sumAdjoin(float *Adjoin,int rowSize,int colSize)
{	
	int counter = 0;
	for(int i=0;i<rowSize*colSize;i++)
		if(Adjoin[i]==1)
			counter++;
	
	return counter;
}
//--------------------------------------------------------------------------------------
float getCudaVariable(float *arr, int idx)
{
	float h_value;
	float *d_value;
	cudaMalloc((void **)&d_value,sizeof(float));

	getCudaValue<<<1,1>>>(arr,idx,d_value);

	cudaMemcpy(&h_value, &d_value[0], sizeof(float), cudaMemcpyDeviceToHost); 

	float value = h_value;
	cudaFree(d_value);

	return value;
}
__global__ void getCudaValue(float *arr, int idx, float *returnValue)
{
	returnValue[0] = arr[idx];
}
//--------------------------------------------------------------------------------------
int countNonZero(float *arr, int sizeX, int sizeY)
{
	int size = sizeX*sizeY;
	int nBlock = size/512;
	checkZero<<<gridGold,blockGold>>>(arr,countNonZeroArray,sizeX,sizeY);

	for(int range = size/2; range>0; range/=2)
	{
		nBlock = (range-1)/512+1;
		reduce<<<nBlock, 512>>>(countNonZeroArray, range, size);
	}
	int result;
	cudaMemcpy(&result, countNonZeroArray, sizeof(int), cudaMemcpyDeviceToHost);
	return result;
}
__global__ void checkZero(float* arr, int* buf, int sizeX, int sizeY)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;

	if(i<sizeX && j<sizeY)
	{
		idx = i+sizeX*j;
		buf[idx] = 1;
		if(arr[idx] == 0)
			buf[idx] = 0;			
	}    
}
__global__ void reduce(int* buf, int range, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= range || idx+range >= size)
        return;

    buf[idx] += buf[idx+range];
}
//--------------------------------------------------------------------------------------
int cudaSumAdjoin(float *arr, int sizeX, int sizeY)
{
	int size = sizeX*sizeY;
	int nBlock = size/512;
	checkValue<<<gridGold,blockGold>>>(arr,cudaSumAdjoinArray,sizeX,sizeY);

	for(int range = size/2; range>0; range/=2)
	{
		nBlock = (range-1)/512+1;
		reduce<<<nBlock, 512>>>(cudaSumAdjoinArray, range, size);
	}
	int result;
    cudaMemcpy(&result, cudaSumAdjoinArray, sizeof(int), cudaMemcpyDeviceToHost);
	return result;
}
__global__ void checkValue(float* arr, int* buf, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;

	if(i<sizeX && j<sizeY)
	{
		idx = i+sizeX*j;
		buf[idx] = __float2int_rn(arr[idx]);
	}    
}
//--------------------------------------------------------------------------------------
__device__ float cudaMod(float a, float b)
{
	float result = fmod(a,b);

	if(result<0)
		result = result + b;
	
	return result;
}
//--------------------------------------------------------------------------------------
__device__ float cudaRound(float number)
{
    return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}
//--------------------------------------------------------------------------------------