/*********************************************************************************************/
/*                                                                                           */
/*                                                    Yang-Hsien Lin                         */
/*                                                    BOSI, BEBI, National Taiwan University */
/*                                                                                2015.01.24 */
/*********************************************************************************************/
//http://www.pudn.com/downloads132/sourcecode/graph/texture_mapping/detail563408.html
#include "HilbertTransform.cuh"

/**********************************************************************************/
int totalFrame; 
int rowSize = 1024, colSize = 1024;
int ReconX = 256, ReconY = 256, ReconZ = 256;	//default image size
double CameraPixelSize = 5.5;
double wavelength = 532;
double criteriaRange = 3*M_PI, criteriaSTD = 0.5;
double Mag = 85.0;
double Nmed = 1.333;//1.495;//1.494;
double df = 0;
double dx = 5.5;//65*1e-9;
int CameraPosition = 1; //CameraPosition = 0; 表示為舊的架設
/**********************************************************************************/
size_t freeDeviceMemory;
size_t totalDeviceMemory;

int SpeedTestFlag = 0;
int QPI_Method = 0;
int ResizeFlag = 1;
int ReconFlag = 1;
int ReconSave;
int SavePhaseStack = 0;
int SaveAmpStack = 0;
int SizeType = 0;
extern int ReconMode;
int IterTime = 100;
/**********************************************************************************/
char *SPDir   = (char *)malloc(256 * sizeof(char));
char *BGDir   = (char *)malloc(256 * sizeof(char));
char *AngDir  = (char *)malloc(256 * sizeof(char));
char *SaveDir = (char *)malloc(256 * sizeof(char));
/**********************************************************************************/
cufftHandle plan_1D_C2C_FORWARD;
cufftHandle plan_1D_C2C_INVERSE;
cufftHandle plan_1D_C2C_FORWARD_FT;
cufftHandle plan_1D_C2C_INVERSE_FT;
cufftHandle plan_2D_C2C_FORWARD_s1;
cufftHandle plan_2D_C2C_FORWARD_s2;
cufftHandle plan_2D_C2C_INVERSE_s1;
cufftHandle plan_2D_C2C_INVERSE_s2;
cufftHandle plan_2D_C2C_FORWARD_FTUP;
cufftHandle plan_2D_C2C_INVERSE_FTUP;
/**********************************************************************************/
void main(int argc, char *argv[])
{
	if (scriptFileRead("JobList.txt") == 0)
	{
		char *InputPath = (char *)malloc(150 * sizeof(char));
		printf("Plz enter the folder path: \n");
		scanf("%s", InputPath);

		total_time = 0;
		scriptFileRead(InputPath);

		free(InputPath);
	}

	system("pause");
}
//--------------------------------------------------------------------------------------
int scriptFileRead(char* script)
{
	int N = 512;

	FILE *fp = fopen(script, "rb");

	if (fp != NULL)
	{
		char *HilbertSet = (char *)malloc(256 * sizeof(char));
		while (fscanf(fp, "%s", HilbertSet) != EOF)
		{
			/*char *IngPathDirSave = (char *)malloc(256 * sizeof(char));	//for original folder
			char *IngPathDirSP = (char *)malloc(256 * sizeof(char));	//for sample folder
			char *IngPathDirBG = (char *)malloc(256 * sizeof(char));	//for background folder
			char *IngPathDirAng = (char *)malloc(256 * sizeof(char));	//for angle-files folder
			char *IngFilePath = (char *)malloc(256 * sizeof(char));	//for temple path*/

			char *StrLine = (char *)malloc(250 * sizeof(char));
			int rowPts[2], colPts[2];
			int StartNo = 1, EndNo = 1, SpaceNo = 1;

			FILE *fp2 = fopen(HilbertSet, "rb");
			char buf[250];
			int len = 0;
			if (fp2 != NULL)
			{
				fscanf(fp2, "%*[^=]%s", StrLine);	SPDir = obtainParameter(StrLine);						//cout << SPDir << endl;
				fscanf(fp2, "%*[^=]%s", StrLine);	BGDir = obtainParameter(StrLine);						//cout << BGDir << endl;
				fscanf(fp2, "%*[^=]%s", StrLine);	AngDir = obtainParameter(StrLine);						//cout << AngDir << endl;
				fscanf(fp2, "%*[^=]%s", StrLine);	SaveDir = obtainParameter(StrLine);						//cout << SaveDir << endl;
			
				fscanf(fp2, "%*[^=]%s", StrLine);	rowPts[0] = atoi(obtainParameter(StrLine));				//cout << rowPts[0] << endl;
				fscanf(fp2, "%*[^=]%s", StrLine);	rowPts[1] = atoi(obtainParameter(StrLine));				//cout << rowPts[1] << endl;
				fscanf(fp2, "%*[^=]%s", StrLine);	colPts[0] = atoi(obtainParameter(StrLine));				//cout << colPts[0] << endl;
				fscanf(fp2, "%*[^=]%s", StrLine);	colPts[1] = atoi(obtainParameter(StrLine));				//cout << colPts[1] << endl;

				fscanf(fp2, "%*[^=]%s", StrLine);	ReconFlag       = atoi(obtainParameter(StrLine));		//cout << ReconFlag << endl;
				fscanf(fp2, "%*[^=]%s", StrLine);	ReconSave       = atoi(obtainParameter(StrLine));		//cout << ReconSave << endl;
				fscanf(fp2, "%*[^=]%s", StrLine);	SizeType        = atoi(obtainParameter(StrLine));		//cout << SizeType << endl;
				fscanf(fp2, "%*[^=]%s", StrLine);	ResizeFlag      = atoi(obtainParameter(StrLine));		//cout << ResizeFlag << endl;

				fscanf(fp2, "%*[^=]%s", StrLine);	Nmed            = atof(obtainParameter(StrLine));		//cout << Nmed << endl;
				fscanf(fp2, "%*[^=]%s", StrLine);	Mag             = atof(obtainParameter(StrLine));		//cout << Mag << endl;
				fscanf(fp2, "%*[^=]%s", StrLine);	CameraPixelSize = atof(obtainParameter(StrLine));		//cout << CameraPixelSize << endl;
				fscanf(fp2, "%*[^=]%s", StrLine);	wavelength      = atof(obtainParameter(StrLine));		//cout << wavelength << endl;	

				fscanf(fp2, "%*[^=]%s", StrLine);	SavePhaseStack	= atof(obtainParameter(StrLine));
				fscanf(fp2, "%*[^=]%s", StrLine);	SaveAmpStack	= atof(obtainParameter(StrLine));

				fscanf(fp2, "%*[^=]%s", StrLine);	criteriaRange   = atof(obtainParameter(StrLine));
				fscanf(fp2, "%*[^=]%s", StrLine);	criteriaSTD     = atof(obtainParameter(StrLine));
			}
			fclose(fp2);

			
			rowSize = rowPts[1] - rowPts[0] + 1;
			colSize = colPts[1] - colPts[0] + 1;

			//CameraPixelSize *= 1e-6;
			//Wavelength *= 1e-9;

			//generate "0 & 1" device array on float
			/*h_ZerosArray = (float *)realloc(h_ZerosArray, rowSize*colSize*sizeof(float));
			h_OnesArray  = (float *)realloc(h_OnesArray , rowSize*colSize*sizeof(float));
			cudaMalloc((void **)&d_ZerosArray,sizeof(float)*rowSize*colSize);
			cudaMalloc((void **)&d_OnesArray ,sizeof(float)*rowSize*colSize);
			#pragma omp parallel for
			for(int i=0; i<rowSize*colSize; i++)
			{
			h_ZerosArray[i] = 0;
			h_OnesArray [i] = 1;
			}
			#pragma omp barrier
			cudaMemcpy(d_ZerosArray,h_ZerosArray, rowSize*colSize*sizeof(float),cudaMemcpyHostToDevice);
			cudaMemcpy(d_OnesArray,h_OnesArray  , rowSize*colSize*sizeof(float),cudaMemcpyHostToDevice);
			//free(h_ZerosArray);
			//free(h_OnesArray);

			//allocate the device array for calculating
			cudaMalloc((void**)&countNonZeroArray , sizeof(int)*rowSize*colSize);
			cudaMalloc((void**)&cudaSumAdjoinArray, sizeof(int)*rowSize*colSize);*/

			HilbertTransform(SPDir, BGDir, AngDir, SaveDir, rowPts, colPts, rowSize, colSize, StartNo, SpaceNo, EndNo);

			/*free(IngPathDirSave);
			free(IngPathDirSP);
			free(IngPathDirBG);
			free(IngPathDirAng);
			free(IngFilePath);*/
		}
		//cudaFree(d_ZerosArray      );
		//cudaFree(d_OnesArray       );
		//cudaFree(countNonZeroArray );
		//cudaFree(cudaSumAdjoinArray);
		//system("pause");
		free(HilbertSet);

		return 1;
	}
	else
	{
		return 0;
	}
	fclose(fp);

}
//--------------------------------------------------------------------------------------
char *strim(char * str)
{
	char * tail = str;
	char * next = str;

	while (*next)
	{
		if (*next != ' ' && *next != '#')
		{
			if (tail < next)
				*tail = *next;
			tail++;
		}
		next++;
	}
	*tail = '\0';
	
	cout << "tail:" << tail << endl;
	cout << "next:" << next << endl;
	cout << "str:" << str<< endl;
	return str;
}
//--------------------------------------------------------------------------------------
char *obtainParameter(char *strLine)
{
	char *strimLine = strdup(strLine);
	char delim[] = "=";
	//char *pch[2];
	char *token = NULL;
	char *output;
	char *context = NULL;
	int i = 0;
	token = strtok_s(strimLine, delim, &context);
	
	while (token != NULL)
	{
		//printf("%d-->%s\n", i, token);
		//if (i == 0) pch[0] = token;
		//if (i == 1) pch[1] = token;
		i++;
		output = token;
		token = strtok_s(NULL, delim, &context);
	}
	return output;
}
//--------------------------------------------------------------------------------------
// structure used to accumulate the moments and other 
// statistical properties encountered so far.
template <typename T>
struct summary_stats_data
{
	T n;
	T min;
	T max;
	T mean;
	T M2;
	T M3;
	T M4;

	// initialize to the identity element
	void initialize()
	{
		n = mean = M2 = M3 = M4 = 0;
		min = std::numeric_limits<T>::max();
		max = std::numeric_limits<T>::min();
	}

	T variance() { return M2 / (n - 1); }
	T variance_n() { return M2 / n; }
	T skewness() { return std::sqrt(n) * M3 / std::pow(M2, (T) 1.5); }
	T kurtosis() { return n * M4 / (M2 * M2); }
};

// stats_unary_op is a functor that takes in a value x and
// returns a variace_data whose mean value is initialized to x.
template <typename T>
struct summary_stats_unary_op
{
	__host__ __device__
		summary_stats_data<T> operator()(const T& x) const
	{
		summary_stats_data<T> result;
		result.n = 1;
		result.min = x;
		result.max = x;
		result.mean = x;
		result.M2 = 0;
		result.M3 = 0;
		result.M4 = 0;

		return result;
	}
};

// summary_stats_binary_op is a functor that accepts two summary_stats_data 
// structs and returns a new summary_stats_data which are an
// approximation to the summary_stats for 
// all values that have been agregated so far
template <typename T>
struct summary_stats_binary_op
	: public thrust::binary_function<const summary_stats_data<T>&,
	const summary_stats_data<T>&,
	summary_stats_data<T> >
{
	__host__ __device__
		summary_stats_data<T> operator()(const summary_stats_data<T>& x, const summary_stats_data <T>& y) const
	{
		summary_stats_data<T> result;

		// precompute some common subexpressions
		T n = x.n + y.n;
		T n2 = n  * n;
		T n3 = n2 * n;

		T delta = y.mean - x.mean;
		T delta2 = delta  * delta;
		T delta3 = delta2 * delta;
		T delta4 = delta3 * delta;

		//Basic number of samples (n), min, and max
		result.n = n;
		result.min = thrust::min(x.min, y.min);
		result.max = thrust::max(x.max, y.max);

		result.mean = x.mean + delta * y.n / n;

		result.M2 = x.M2 + y.M2;
		result.M2 += delta2 * x.n * y.n / n;

		result.M3 = x.M3 + y.M3;
		result.M3 += delta3 * x.n * y.n * (x.n - y.n) / n2;
		result.M3 += (T) 3.0 * delta * (x.n * y.M2 - y.n * x.M2) / n;

		result.M4 = x.M4 + y.M4;
		result.M4 += delta4 * x.n * y.n * (x.n * x.n - x.n * y.n + y.n * y.n) / n3;
		result.M4 += (T) 6.0 * delta2 * (x.n * x.n * y.M2 + y.n * y.n * x.M2) / n2;
		result.M4 += (T) 4.0 * delta * (x.n * y.M3 - y.n * x.M3) / n;

		return result;
	}
};

char * CombineStr(char * str1, char * str2)
{
	static char strOut[256];                 //scope!
	if ((strlen(str1) + strlen(str2)) < 256)
		sprintf(strOut, "%s%s", str1, str2); //plain quotes!
	return strOut;
}
//--------------------------------------------------------------------------------------
void HilbertTransform(char *SPDir, char *BGDir, char *AngDir, char *SaveDir,
	int rowPts[], int colPts[], int rowSize, int colSize, int StartNo, int SpaceNo, int EndNo)
{
	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("CUDA Device Query...\n");
	printf("There are %d CUDA devices.\n", devCount);

	for (int i = 0; i < devCount; ++i)
	{
		// Get device properties
		printf("\nCUDA Device #%d\n", i);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		printDevProp(devProp);
	}

	cudaMemGetInfo(&freeDeviceMemory, &totalDeviceMemory);
	cout << "GPU memory usage: free = " << freeDeviceMemory / 1024 / 1024 << " MB, total = " << totalDeviceMemory / 1024 / 1024 << " MB" << endl;

	totalFrame = 0;
	int total_SP_Frame = count_file_num(SPDir,"Buffer");
	int total_BG_Frame = count_file_num(BGDir,"Buffer");
	
	if (total_SP_Frame == total_BG_Frame)
		totalFrame = total_SP_Frame;
	else
		totalFrame = min(total_SP_Frame, total_BG_Frame);	
	
	//產生儲存phase的資料夾
	if (_mkdir(SaveDir) == 0)
		printf("成功產生儲存相位影像資料夾!\n");
	else
		printf("警告:產生儲存相位影像資料夾失敗!\n");


	char *SP_img_Path    = (char *)malloc(256 * sizeof(char));							//SP path
	char *BG_img_Path    = (char *)malloc(256 * sizeof(char));							//BG path
	char *Ang_img_Path_X = (char *)malloc(256 * sizeof(char));							//AngX path
	char *Ang_img_Path_Y = (char *)malloc(256 * sizeof(char));							//AngY path
	char *Save_img_Path  = (char *)malloc(256 * sizeof(char));							//output path

	//determine the number of matrix size for calculating
	sprintf(SP_img_Path, "%s\\Buffer1.bmp", SPDir);
	sprintf(BG_img_Path, "%s\\Buffer1.bmp", BGDir);

	int nrSP, ncSP, nrBG, ncBG, nrAng, ncAng;
	int Nx, Ny, Nx2, Ny2;
	bmp_header(SP_img_Path, nrSP, ncSP);
	bmp_header(BG_img_Path, nrBG, ncBG);

	if (nrSP == nrBG && ncSP == ncBG)// && nrAng == ncAng)
	{
		Nx = ceil(log((double)nrSP) / log(2.0));
		Ny = ceil(log((double)ncSP) / log(2.0));
		Nx = pow(2.0, Nx);
		Ny = pow(2.0, Ny);
		Nx2 = Nx >> 2;
		Ny2 = Ny >> 2;

		Nx >= Ny2 ? Ny2 = Nx2 : Nx2 = Ny2;
		Nx2 >= Ny2 ? Ny2 = Nx2 : Nx2 = Ny2;
	}
	else
	{
		printf("Wrong size!");
		return;
	}

	//defined the size of 'grid' and 'block' on device
	//for original size
	int blocksInX = (Nx + 32 - 1) / 32;
	int blocksInY = (Ny + 32 - 1) / 32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);
	//for original size/2
	int blocksInX2 = (Nx/2 + 32 - 1) / 32;
	int blocksInY2 = (Ny/2 + 32 - 1) / 32;
	dim3 grid2(blocksInX2, blocksInY2);
	dim3 block2(32, 32);
	//for original size/4
	int blocksInX3 = (Nx2 + 32 - 1) / 32;
	int blocksInY3 = (Ny2 + 32 - 1) / 32;
	dim3 grid3(blocksInX3, blocksInY3);
	dim3 block3(32, 32);
	//for original size/8
	int blocksInX4 = (Nx2/2 + 32 - 1) / 32;
	int blocksInY4 = (Ny2/2 + 32 - 1) / 32;
	dim3 grid4(blocksInX4, blocksInY4);
	dim3 block4(32, 32);

	//host memory
	unsigned char *SPImgTemp   = (unsigned char *)malloc(Nx*Ny*sizeof(unsigned char));	//SP image temp array
	unsigned char *BGImgTemp   = (unsigned char *)malloc(Nx*Ny*sizeof(unsigned char));	//BG image temp array
	unsigned char *AngImgTempX = (unsigned char *)malloc(Nx*Ny*sizeof(unsigned char));	//angle information on X axis
	unsigned char *AngImgTempY = (unsigned char *)malloc(Nx*Ny*sizeof(unsigned char));	//angle information on Y axis

	cufftComplex* cuSP_temp	= (cufftComplex *)malloc(Nx*Ny*sizeof(cufftComplex));
	cufftComplex* cuBG_temp = (cufftComplex *)malloc(Nx*Ny*sizeof(cufftComplex));
	float *cuSP_float		= (float *)malloc(Nx*Ny*sizeof(float));
	float *cuBG_float		= (float *)malloc(Nx*Ny*sizeof(float));
	float *AngImgX			= (float *)malloc(Nx*Ny*sizeof(float));
	float *AngImgY			= (float *)malloc(Nx*Ny*sizeof(float));

	float **SP_float_All, **BG_float_All, *SP_float, *BG_float;
	cufftComplex **SP_cu_All, **BG_cu_All, *SP_cu, *BG_cu;

	if (SpeedTestFlag == 1) {		
		switch (QPI_Method){
		case 0:
			SP_float_All = (float **)malloc(totalFrame*sizeof(float *));
			BG_float_All = (float **)malloc(totalFrame*sizeof(float *));
			for (int i = 0; i < totalFrame; i++)
			{
				SP_float_All[i] = (float *)malloc(Nx*Ny * sizeof(float));
				BG_float_All[i] = (float *)malloc(Nx*Ny * sizeof(float));
				if (SP_float_All[i] == NULL || BG_float_All[i] == NULL)
				{
					fprintf(stderr, "out of memory\n");
				}
			}
			break;
		case 1:
		case 2:
			SP_cu_All = (cufftComplex **)malloc(totalFrame*sizeof(cufftComplex *));
			BG_cu_All = (cufftComplex **)malloc(totalFrame*sizeof(cufftComplex *));
			for (int i = 0; i < totalFrame; i++)
			{
				SP_cu_All[i] = (cufftComplex *)malloc(Nx*Ny * sizeof(cufftComplex));
				BG_cu_All[i] = (cufftComplex *)malloc(Nx*Ny * sizeof(cufftComplex));
				if (SP_cu_All[i] == NULL || BG_cu_All[i] == NULL)
				{
					fprintf(stderr, "out of memory\n");
				}
			}
			break;
		}
	}
	else {
		switch (QPI_Method){
		case 0:
			SP_float = (float *)malloc(Nx*Ny * sizeof(float));
			BG_float = (float *)malloc(Nx*Ny * sizeof(float));
			break;
		case 1:
		case 2:
			SP_cu = (cufftComplex *)malloc(Nx*Ny * sizeof(cufftComplex));
			BG_cu = (cufftComplex *)malloc(Nx*Ny * sizeof(cufftComplex));
			break;
		}
		
	}
	

	//device memory
	cufftComplex *cuSP, *cuBG, *cuSP2, *cuBG2, *selectSP, *selectBG;
	float *SPWrapPhase, *BGWrapPhase, *SPWrapPhase2, *BGWrapPhase2;
	float *UnWrapPhaseSP, *UnWrapPhaseBG, *UnWrapPhaseSP2, *UnWrapPhaseBG2;
	int *circleImg;
	float *cuPhaseMap, *cuAmpMap, *cuPhaseMap2, *cuAmpMap2;
	cudaMalloc((void **)&cuPhaseMap, sizeof(float)*Nx *Ny);
	cudaMalloc((void **)&cuAmpMap, sizeof(float)*Nx *Ny);

	/*switch (QPI_Method){
	case 0:
		cudaMalloc((void **)&cuSP2, sizeof(cufftComplex)*Nx2*Ny2);
		cudaMalloc((void **)&cuBG2, sizeof(cufftComplex)*Nx2*Ny2);
		cudaMalloc((void **)&SPWrapPhase2, sizeof(float)*Nx2*Ny2);
		cudaMalloc((void **)&BGWrapPhase2, sizeof(float)*Nx2*Ny2);
		cudaMalloc((void **)&UnWrapPhaseSP2, sizeof(float)*Nx2*Ny2);
		cudaMalloc((void **)&UnWrapPhaseBG2, sizeof(float)*Nx2*Ny2);
		cudaMalloc((void **)&cuPhaseMap2, sizeof(float)*Nx2*Ny2);
		cudaMalloc((void **)&cuAmpMap2, sizeof(float)*Nx2*Ny2);
		break;
	case 1:
		cudaMalloc((void **)&cuSP, sizeof(cufftComplex)*Nx *Ny);
		cudaMalloc((void **)&cuBG, sizeof(cufftComplex)*Nx *Ny);
		cudaMalloc((void **)&cuSP2, sizeof(cufftComplex)*Nx2*Ny2);
		cudaMalloc((void **)&cuBG2, sizeof(cufftComplex)*Nx2*Ny2);
		cudaMalloc((void **)&SPWrapPhase2, sizeof(float)*Nx2*Ny2);
		cudaMalloc((void **)&BGWrapPhase2, sizeof(float)*Nx2*Ny2);
		cudaMalloc((void **)&UnWrapPhaseSP2, sizeof(float)*Nx2*Ny2);
		cudaMalloc((void **)&UnWrapPhaseBG2, sizeof(float)*Nx2*Ny2);
		cudaMalloc((void **)&cuPhaseMap2, sizeof(float)*Nx2*Ny2);
		cudaMalloc((void **)&cuAmpMap2, sizeof(float)*Nx2*Ny2);
		break;
	case 2:
		cudaMalloc((void **)&cuSP, sizeof(cufftComplex)*Nx *Ny);
		cudaMalloc((void **)&cuBG, sizeof(cufftComplex)*Nx *Ny);
		cudaMalloc((void **)&selectSP, sizeof(cufftComplex)*Nx *Ny);
		cudaMalloc((void **)&selectBG, sizeof(cufftComplex)*Nx *Ny);		
		cudaMalloc((void **)&circleImg, sizeof(int)*Nx*Ny);

		cudaMalloc((void **)&SPWrapPhase, sizeof(float)*Nx *Ny);
		cudaMalloc((void **)&BGWrapPhase, sizeof(float)*Nx *Ny);
		cudaMalloc((void **)&UnWrapPhaseSP, sizeof(float)*Nx *Ny);
		cudaMalloc((void **)&UnWrapPhaseBG, sizeof(float)*Nx *Ny);
		break;
	}*/
	
	//case 0
	cudaMalloc((void **)&cuSP2, sizeof(cufftComplex)*Nx2*Ny2);		//0&1
	cudaMalloc((void **)&cuBG2, sizeof(cufftComplex)*Nx2*Ny2);		//0&1
	cudaMalloc((void **)&SPWrapPhase2, sizeof(float)*Nx2*Ny2);		//0&1
	cudaMalloc((void **)&BGWrapPhase2, sizeof(float)*Nx2*Ny2);		//0&1
	cudaMalloc((void **)&UnWrapPhaseSP2, sizeof(float)*Nx2*Ny2);	//0&1
	cudaMalloc((void **)&UnWrapPhaseBG2, sizeof(float)*Nx2*Ny2);	//0&1
	cudaMalloc((void **)&cuPhaseMap2, sizeof(float)*Nx2*Ny2);		//0&1
	cudaMalloc((void **)&cuAmpMap2, sizeof(float)*Nx2*Ny2);			//0&1

	//case 1
	cudaMalloc((void **)&cuSP, sizeof(cufftComplex)*Nx *Ny);		//1&2
	cudaMalloc((void **)&cuBG, sizeof(cufftComplex)*Nx *Ny);		//1&2

	//case 2
	cudaMalloc((void **)&selectSP, sizeof(cufftComplex)*Nx *Ny);
	cudaMalloc((void **)&selectBG, sizeof(cufftComplex)*Nx *Ny);
	cudaMalloc((void **)&circleImg, sizeof(int)*Nx*Ny);

	cudaMalloc((void **)&SPWrapPhase, sizeof(float)*Nx *Ny);
	cudaMalloc((void **)&BGWrapPhase, sizeof(float)*Nx *Ny);
	cudaMalloc((void **)&UnWrapPhaseSP, sizeof(float)*Nx *Ny);
	cudaMalloc((void **)&UnWrapPhaseBG, sizeof(float)*Nx *Ny);

	// create the plans for Forward/Inverse cuFFT
	cufftPlan1d(&plan_1D_C2C_FORWARD, Ny2 * 2, CUFFT_C2C, Nx2);
	cufftPlan1d(&plan_1D_C2C_INVERSE, Nx2 * 2, CUFFT_C2C, Ny2);
	cufftPlan1d(&plan_1D_C2C_FORWARD_FT, Nx, CUFFT_C2C, Ny / 4);
	cufftPlan1d(&plan_1D_C2C_INVERSE_FT, Nx / 4, CUFFT_C2C, Ny / 4);

	cufftPlan2d(&plan_2D_C2C_FORWARD_s1, Nx, Ny, CUFFT_C2C);
	cufftPlan2d(&plan_2D_C2C_FORWARD_s2, Nx2, Ny2, CUFFT_C2C);
	cufftPlan2d(&plan_2D_C2C_INVERSE_s1, Nx, Ny, CUFFT_C2C);
	cufftPlan2d(&plan_2D_C2C_INVERSE_s2, Nx2, Ny2, CUFFT_C2C);

	cufftPlan2d(&plan_2D_C2C_FORWARD_FTUP, Nx2/2, Ny2/2, CUFFT_C2C);
	cufftPlan2d(&plan_2D_C2C_INVERSE_FTUP, Nx2 / 2, Ny2 / 2, CUFFT_C2C);
	

	//host memory for output
	float *PhaseMap   = (float *)malloc(Nx*Ny*sizeof(float));
	float *AmpMap     = (float *)malloc(Nx*Ny*sizeof(float));
	float *FinalPhase = (float *)malloc(Nx*Ny*sizeof(float));
	float *FinalAmp   = (float *)malloc(Nx*Ny*sizeof(float));
	microImg *ResultImg = (microImg *)malloc(Nx*Ny*sizeof(microImg));

	//alloc memory for Reconstruction if 'ReconFlag' is TRUE
	int totalFrame_temp = totalFrame;
	float *PhaseStack, *AmpStack;
	bool *status_series;
	float *sampleAngleRadX_Stack, *sampleAngleRadY_Stack;
	status_series = (bool   *)malloc(totalFrame_temp*sizeof(bool));
	sampleAngleRadX_Stack = (float *)malloc(totalFrame_temp*sizeof(float));
	sampleAngleRadY_Stack = (float *)malloc(totalFrame_temp*sizeof(float));

	int deleteCount = 0;


	if (ReconFlag != 2)
	{
		if (ReconFlag > 0)
		{
			if (SizeType) {
				ReconX = 512, ReconY = 512, ReconZ = 512;
			}
			else {
				ReconX = 256, ReconY = 256, ReconZ = 256;
			}
			
			PhaseStack = (float  *)malloc(ReconX*ReconY*totalFrame_temp * sizeof(float));
			AmpStack = (float  *)malloc(ReconX*ReconY*totalFrame_temp * sizeof(float));			
		}
		

		//Read All Images
		if (SpeedTestFlag == 1) {
			for (int frame = 1; frame <= totalFrame; frame++)
			{
				double SampleAngleDegX = 0.0, SampleAngleDegY = 0.0;

				sprintf(SP_img_Path, "%s\\Buffer%d.bmp", SPDir, frame);
				sprintf(BG_img_Path, "%s\\Buffer%d.bmp", BGDir, frame);

				sprintf(Ang_img_Path_X, "%s\\X", AngDir);	read_bmp(SP_img_Path, Nx, Ny, SPImgTemp);
				sprintf(Ang_img_Path_Y, "%s\\Y", AngDir);	read_bmp(BG_img_Path, Nx, Ny, BGImgTemp);
				int exsit_AngX = -1, exsit_AngY = -1;
#ifdef _WIN32
				exsit_AngX = _access(Ang_img_Path_X, 0);
				exsit_AngY = _access(Ang_img_Path_Y, 0);
#elif __linux__ 
				struct stat stX, stY;
				if (stat(Ang_img_Path_X, &stX) == 0)
					if (stX.st_mode & S_IFDIR != 0)
						exsit_AngX = 1;

				if (stat(Ang_img_Path_Y, &stY) == 0)
					if (stY.st_mode & S_IFDIR != 0)
						exsit_AngY = 1;
#endif
				if (exsit_AngX != -1 && exsit_AngY != -1)
				{
					sprintf(Ang_img_Path_X, "%s\\X\\Buffer%d.bmp", AngDir, frame);
					read_bmp(Ang_img_Path_X, Nx, Ny, AngImgTempX);

					sprintf(Ang_img_Path_Y, "%s\\Y\\Buffer%d.bmp", AngDir, frame);
					read_bmp(Ang_img_Path_Y, Nx, Ny, AngImgTempY);
				}
				else {
					sprintf(Ang_img_Path_X, "%s\\Buffer%d.bmp", AngDir, frame);
					read_bmp(Ang_img_Path_X, Nx, Ny, AngImgTempX);
				}

				dx = CameraPixelSize;

				switch (QPI_Method) {
				case 0:
					if (nrSP == Nx && ncSP == Ny)
					{

						for (int j = 0; j < Ny; j++)
							for (int i = 0; i < Nx; i++) {
								SP_float_All[frame - 1][j + i*Ny] = (float)SPImgTemp[j + i*Ny];
								BG_float_All[frame - 1][j + i*Ny] = (float)BGImgTemp[j + i*Ny];

								AngImgX[i + j*Nx] = (float)AngImgTempX[j + i*Ny];
							}

						if (exsit_AngX != -1 && exsit_AngY != -1) {
							for (int j = 0; j < colSize; j++)
								for (int i = 0; i < rowSize; i++)
									AngImgY[j + i*colSize] = (float)AngImgTempY[i + j*rowSize];
						}

						//計算df 
						//if(frame==StartNo)
						df = 1 / (Nx*dx);
					}
					else if (nrSP == Nx / 2 && ncSP == Ny / 2)//影像大小邊長皆除以2
					{
						for (int j = 0; j < Ny; j++)
							for (int i = 0; i < Nx; i++) {
								SP_float_All[frame - 1][j + i*Ny] = (float)SPImgTemp[j * 2 + i * 2 * Ny];
								BG_float_All[frame - 1][j + i*Ny] = (float)BGImgTemp[j * 2 + i * 2 * Ny];

								AngImgX[i + j*Nx] = (float)AngImgTempX[j * 2 + i * 2 * Ny];
							}

						if (exsit_AngX != -1 && exsit_AngY != -1) {
							for (int j = colPts[0]; j < colPts[1]; j++)
								for (int i = rowPts[0]; i < rowPts[1]; i++)
									AngImgY[i + j*Nx] = (float)AngImgTempY[j * 2 + i * 2 * Ny];
						}

						//計算df 
						if (frame == StartNo) {
							dx = dx * 2;
							df = 1 / (nrSP*dx);
						}
					}
					else {
						printf("影像大小設定嚴重錯誤!");
						system("pause");
					}
					break;

				case 1:
				case 2:
					//影像轉移
					if (nrSP == Nx && ncSP == Ny)
					{
						for (int j = 0; j < Ny; j++)
							for (int i = 0; i < Nx; i++)
							{
								SP_cu_All[frame - 1][i + j*Nx].x = (float)SPImgTemp[j + i*Ny];
								SP_cu_All[frame - 1][i + j*Nx].y = 0.0;

								BG_cu_All[frame - 1][i + j*Nx].x = (float)BGImgTemp[j + i*Ny];
								BG_cu_All[frame - 1][i + j*Nx].y = 0.0;

								AngImgX[i + j*Nx] = (float)AngImgTempX[j + i*Ny];
							}

						if (exsit_AngX != -1 && exsit_AngY != -1)
						{
							for (int j = 0; j < colSize; j++)
								for (int i = 0; i < rowSize; i++)
									AngImgY[j + i*colSize] = (float)AngImgTempY[i + j*rowSize];
						}

						//計算df 
						//if(frame==StartNo)
						df = 1 / (Nx*dx);
					}
					else if (nrSP == Nx / 2 && ncSP == Ny / 2)//影像大小邊長皆除以2
					{
						for (int j = 0; j < Ny; j++)
							for (int i = 0; i < Nx; i++)
							{
								SP_cu_All[frame - 1][i + j*Nx].x = (float)SPImgTemp[j * 2 + i * 2 * Ny];
								SP_cu_All[frame - 1][i + j*Nx].y = 0.0;

								BG_cu_All[frame - 1][i + j*Nx].x = (float)BGImgTemp[j * 2 + i * 2 * Ny];
								BG_cu_All[frame - 1][i + j*Nx].y = 0.0;

								AngImgX[i + j*Nx] = (float)AngImgTempX[j * 2 + i * 2 * Ny];
							}

						if (exsit_AngX != -1 && exsit_AngY != -1)
						{
							for (int j = colPts[0]; j < colPts[1]; j++)
								for (int i = rowPts[0]; i < rowPts[1]; i++)
									AngImgY[i + j*Nx] = (float)AngImgTempY[j * 2 + i * 2 * Ny];

						}

						//計算df 
						if (frame == StartNo) {
							dx = dx * 2;
							df = 1 / (nrSP*dx);
						}
					}
					else {
						printf("影像大小設定嚴重錯誤!");
						system("pause");
					}
					break;
				}

				//estimate the angle
				sampleAngleRadX_Stack[frame - 1] = AngCal_GPU(AngImgX, Nx, Ny, frame, totalFrame);
				sampleAngleRadY_Stack[frame - 1] = exsit_AngY != -1 ? AngCal_GPU(AngImgY, Nx, Ny, frame, totalFrame) : 0;
			}
		}

		//Start Reconstruction
		AccumFrame = 0;
		start_time = clock();
		wrap_time = 0;
		unwrap_time = 0;
		extract_time = 0;
		dataTransfer_time = 0;
		PrintProcess(0, SPDir, BGDir, AngDir);

		//for(int frame=1;frame<=1;frame++)
		int rCircle = 0, cCircle = 0, radiusCircle = 0;
		for (int frame = 1; frame <= totalFrame; frame++)
		{
			if (!SpeedTestFlag) {
				double SampleAngleDegX = 0.0, SampleAngleDegY = 0.0;

				sprintf(SP_img_Path, "%s\\Buffer%d.bmp", SPDir, frame);
				sprintf(BG_img_Path, "%s\\Buffer%d.bmp", BGDir, frame);

				sprintf(Ang_img_Path_X, "%s\\X", AngDir);	read_bmp(SP_img_Path, Nx, Ny, SPImgTemp);
				sprintf(Ang_img_Path_Y, "%s\\Y", AngDir);	read_bmp(BG_img_Path, Nx, Ny, BGImgTemp);
				int exsit_AngX = _access(Ang_img_Path_X, 0);
				int exsit_AngY = _access(Ang_img_Path_Y, 0);
				if (exsit_AngX != -1 && exsit_AngY != -1)
				{
					sprintf(Ang_img_Path_X, "%s\\X\\Buffer%d.bmp", AngDir, frame);
					read_bmp(Ang_img_Path_X, Nx, Ny, AngImgTempX);

					sprintf(Ang_img_Path_Y, "%s\\Y\\Buffer%d.bmp", AngDir, frame);
					read_bmp(Ang_img_Path_Y, Nx, Ny, AngImgTempY);
				}
				else {
					sprintf(Ang_img_Path_X, "%s\\Buffer%d.bmp", AngDir, frame);
					read_bmp(Ang_img_Path_X, Nx, Ny, AngImgTempX);
				}

				dx = CameraPixelSize;

				switch (QPI_Method) {
				case 0:
					if (nrSP == Nx && ncSP == Ny)
					{

						for (int j = 0; j < Ny; j++)
							for (int i = 0; i < Nx; i++) {
								SP_float[j + i*Ny] = (float)SPImgTemp[j + i*Ny];
								BG_float[j + i*Ny] = (float)BGImgTemp[j + i*Ny];

								AngImgX[i + j*Nx] = (float)AngImgTempX[j + i*Ny];
							}

						if (exsit_AngX != -1 && exsit_AngY != -1) {
							for (int j = 0; j < colSize; j++)
								for (int i = 0; i < rowSize; i++)
									AngImgY[j + i*colSize] = (float)AngImgTempY[i + j*rowSize];
						}

						//計算df 
						//if(frame==StartNo)
						df = 1 / (Nx*dx);
					}
					else if (nrSP == Nx / 2 && ncSP == Ny / 2)//影像大小邊長皆除以2
					{
						for (int j = 0; j < Ny; j++)
							for (int i = 0; i < Nx; i++) {
								SP_float[j + i*Ny] = (float)SPImgTemp[j * 2 + i * 2 * Ny];
								BG_float[j + i*Ny] = (float)BGImgTemp[j * 2 + i * 2 * Ny];

								AngImgX[i + j*Nx] = (float)AngImgTempX[j * 2 + i * 2 * Ny];
							}

						if (exsit_AngX != -1 && exsit_AngY != -1) {
							for (int j = colPts[0]; j < colPts[1]; j++)
								for (int i = rowPts[0]; i < rowPts[1]; i++)
									AngImgY[i + j*Nx] = (float)AngImgTempY[j * 2 + i * 2 * Ny];
						}

						//計算df 
						if (frame == StartNo) {
							dx = dx * 2;
							df = 1 / (nrSP*dx);
						}
					}
					else {
						printf("影像大小設定嚴重錯誤!");
						system("pause");
					}
					break;

				case 1:
				case 2:
					//影像轉移
					if (nrSP == Nx && ncSP == Ny)
					{
						for (int j = 0; j < Ny; j++)
							for (int i = 0; i < Nx; i++)
							{
								SP_cu[i + j*Nx].x = (float)SPImgTemp[j + i*Ny];
								SP_cu[i + j*Nx].y = 0.0;

								BG_cu[i + j*Nx].x = (float)BGImgTemp[j + i*Ny];
								BG_cu[i + j*Nx].y = 0.0;

								AngImgX[i + j*Nx] = (float)AngImgTempX[j + i*Ny];
							}

						if (exsit_AngX != -1 && exsit_AngY != -1)
						{
							for (int j = 0; j < colSize; j++)
								for (int i = 0; i < rowSize; i++)
									AngImgY[j + i*colSize] = (float)AngImgTempY[i + j*rowSize];
						}

						//計算df 
						//if(frame==StartNo)
						df = 1 / (Nx*dx);
					}
					else if (nrSP == Nx / 2 && ncSP == Ny / 2)//影像大小邊長皆除以2
					{
						for (int j = 0; j < Ny; j++)
							for (int i = 0; i < Nx; i++)
							{
								SP_cu[i + j*Nx].x = (float)SPImgTemp[j * 2 + i * 2 * Ny];
								SP_cu[i + j*Nx].y = 0.0;

								BG_cu[i + j*Nx].x = (float)BGImgTemp[j * 2 + i * 2 * Ny];
								BG_cu[i + j*Nx].y = 0.0;

								AngImgX[i + j*Nx] = (float)AngImgTempX[j * 2 + i * 2 * Ny];
							}

						if (exsit_AngX != -1 && exsit_AngY != -1)
						{
							for (int j = colPts[0]; j < colPts[1]; j++)
								for (int i = rowPts[0]; i < rowPts[1]; i++)
									AngImgY[i + j*Nx] = (float)AngImgTempY[j * 2 + i * 2 * Ny];

						}

						//計算df 
						if (frame == StartNo) {
							dx = dx * 2;
							df = 1 / (nrSP*dx);
						}
					}
					else {
						printf("影像大小設定嚴重錯誤!");
						system("pause");
					}
					break;
				}

				//estimate the angle
				sampleAngleRadX_Stack[frame - 1] = AngCal_GPU(AngImgX, Nx, Ny, frame, totalFrame);
				//sampleAngleRadX_Stack[frame - 1] = (-40.0 + 80.0/499.0 * float(frame-1))*M_PI/180;
				//system("pause");
				sampleAngleRadY_Stack[frame - 1] = exsit_AngY != -1 ? AngCal_GPU(AngImgY, Nx, Ny, frame, totalFrame) : 0;
			}

			sE_time = clock();

			//Start QPI extraction			
			switch (QPI_Method) {
			case 0:
				if (SpeedTestFlag == 1)
				{
					extractQPI(SP_float_All[frame - 1], BG_float_All[frame - 1], cuSP2, cuBG2, Nx, Ny);
				}
				else
				{
					sF_time = clock();
					extractQPI(SP_float, BG_float, cuSP2, cuBG2, Nx, Ny);
				}
				break;
			case 1:
				if (SpeedTestFlag == 1) {
					cudaMemcpy(cuSP, SP_cu_All[frame - 1], sizeof(cufftComplex)*Nx*Ny, cudaMemcpyHostToDevice);
					cudaMemcpy(cuBG, BG_cu_All[frame - 1], sizeof(cufftComplex)*Nx*Ny, cudaMemcpyHostToDevice);
				}
				else {
					cudaMemcpy(cuSP, SP_cu, sizeof(cufftComplex)*Nx*Ny, cudaMemcpyHostToDevice);
					cudaMemcpy(cuBG, BG_cu, sizeof(cufftComplex)*Nx*Ny, cudaMemcpyHostToDevice);
				}

				s_wrap = clock();
				//foreward FFT 2D 				
				//cuFFT2D(cuSP, Nx, Ny, -1);
				//cuFFT2D(cuBG, Nx, Ny, -1);
				cufftExecC2C(plan_2D_C2C_FORWARD_s1, cuSP, cuSP, CUFFT_FORWARD);
				cufftExecC2C(plan_2D_C2C_FORWARD_s1, cuBG, cuBG, CUFFT_FORWARD);

				//shift FFT 2D
				cuFFT2Dshift << <grid2, block2 >> > (cuSP, Nx, Ny);
				cuFFT2Dshift << <grid2, block2 >> > (cuBG, Nx, Ny);
				e_wrap = clock();	wrap_time += e_wrap - s_wrap;

				//estimate the circle center and radius
				if (frame == 1) obtainRadius(cuBG, radiusCircle, rCircle, cCircle, Nx, Ny);

				s_wrap = clock();
				//withod zero-padding method <--> (N/4)*(N/4)
				get1stOrder_new << <grid3, block3 >> > (cuSP2, cuSP, radiusCircle, rCircle, cCircle, Nx, Ny);	//Notice: selectSP/cuSP
				get1stOrder_new << <grid3, block3 >> > (cuBG2, cuBG, radiusCircle, rCircle, cCircle, Nx, Ny);
				cuFFT2Dshift << <grid4, block4 >> > (cuSP2, Nx2, Ny2);
				cuFFT2Dshift << <grid4, block4 >> > (cuBG2, Nx2, Ny2);
				//cuFFT2D(cuSP2, Nx2, Ny2, 1);
				//cuFFT2D(cuBG2, Nx2, Ny2, 1);
				cufftExecC2C(plan_2D_C2C_INVERSE_s2, cuSP2, cuSP2, CUFFT_INVERSE);
				cufftExecC2C(plan_2D_C2C_INVERSE_s2, cuBG2, cuBG2, CUFFT_INVERSE);
				scaleFFT2D << <grid3, block3 >> >(cuSP2, Nx2, Ny2, 1.f / (Nx2 * Ny2));
				scaleFFT2D << <grid3, block3 >> >(cuBG2, Nx2, Ny2, 1.f / (Nx2 * Ny2));
				e_wrap = clock();	wrap_time += e_wrap - s_wrap;
				break;

			case 2:
				if (SpeedTestFlag == 1) {
					cudaMemcpy(cuSP, SP_cu_All[frame - 1], sizeof(cufftComplex)*Nx*Ny, cudaMemcpyHostToDevice);
					cudaMemcpy(cuBG, BG_cu_All[frame - 1], sizeof(cufftComplex)*Nx*Ny, cudaMemcpyHostToDevice);
				}
				else {
					cudaMemcpy(cuSP, SP_cu, sizeof(cufftComplex)*Nx*Ny, cudaMemcpyHostToDevice);
					cudaMemcpy(cuBG, BG_cu, sizeof(cufftComplex)*Nx*Ny, cudaMemcpyHostToDevice);
				}
				s_wrap = clock();
				//foreward FFT 2D 
				//cuFFT2D(cuSP, Nx, Ny, -1);
				//cuFFT2D(cuBG, Nx, Ny, -1);
				cufftExecC2C(plan_2D_C2C_FORWARD_s1, cuSP, cuSP, CUFFT_FORWARD);
				cufftExecC2C(plan_2D_C2C_FORWARD_s1, cuBG, cuBG, CUFFT_FORWARD);

				//shift FFT 2D
				cuFFT2Dshift << <grid2, block2 >> > (cuSP, Nx, Ny);
				cuFFT2Dshift << <grid2, block2 >> > (cuBG, Nx, Ny);
				e_wrap = clock();	wrap_time += e_wrap - s_wrap;

				//estimate the circle center and radius
				if (frame == 1) obtainRadius(cuBG, radiusCircle, rCircle, cCircle, Nx, Ny);

				s_wrap = clock();
				//generate the circle image		
				cudaMemset(circleImg, 0, sizeof(int)*Nx*Ny);
				circleImgGenerate << <grid, block >> > (circleImg, Nx, Ny, rCircle, cCircle, radiusCircle);

				//cudaFree(circleImg);

				cudaMemcpy(selectSP, cuSP, sizeof(cufftComplex)*Nx*Ny, cudaMemcpyDeviceToDevice);
				cudaMemcpy(selectBG, cuBG, sizeof(cufftComplex)*Nx*Ny, cudaMemcpyDeviceToDevice);

				cudaMemset(cuSP, 0, sizeof(cufftComplex)*Nx*Ny);
				cudaMemset(cuBG, 0, sizeof(cufftComplex)*Nx*Ny);

				//move the 1st order singnal to the center of image
				get1stOrder << <grid, block >> > (cuSP, selectSP, radiusCircle, rCircle, cCircle, Nx, Ny);
				get1stOrder << <grid, block >> > (cuBG, selectBG, radiusCircle, rCircle, cCircle, Nx, Ny);

				//FFT shift
				cuFFT2Dshift << <grid2, block2 >> > (cuSP, Nx, Ny);
				cuFFT2Dshift << <grid2, block2 >> > (cuBG, Nx, Ny);

				//inverse FFT 2D
				//cuFFT2D(cuSP, Nx, Ny, 1);
				//cuFFT2D(cuBG, Nx, Ny, 1);
				cufftExecC2C(plan_2D_C2C_INVERSE_s1, cuSP, cuSP, CUFFT_INVERSE);
				cufftExecC2C(plan_2D_C2C_INVERSE_s1, cuBG, cuBG, CUFFT_INVERSE);
				scaleFFT2D << <grid, block >> >(cuSP, Nx, Ny, 1.f / (Nx * Ny));
				scaleFFT2D << <grid, block >> >(cuBG, Nx, Ny, 1.f / (Nx * Ny));

				//estimating the warpped phase and amp images
				estimateWrapPhase << <grid, block >> > (SPWrapPhase, BGWrapPhase, cuSP, cuBG, Nx, Ny);
				estimateAmp << <grid, block >> > (cuAmpMap, cuSP, cuBG, Nx, Ny);
				e_wrap = clock();	wrap_time += e_wrap - s_wrap;
				break;

			}

			switch (QPI_Method) {
			case 0:
			case 1:
				
				s_wrap = clock();
				calcWrapPhase << <grid3, block3 >> > (UnWrapPhaseSP2, cuAmpMap2, cuSP2, cuBG2, Nx2, Ny2);
				e_wrap = clock();	wrap_time += e_wrap - s_wrap;							

				//UWLS				
				//FastUnwrapping(UnWrapPhaseSP2, cuPhaseMap2, Nx2, Ny2);
				DCT_UWLS_Unwrapped(cuPhaseMap2, UnWrapPhaseSP2, Nx2, Ny2);

				//resize the Phase & Amp maps
				bilinear_interpolation_kernel << <grid, block >> > (cuPhaseMap, cuPhaseMap2, cuAmpMap, cuAmpMap2, Nx2, Ny2, Nx, Ny);
				break;
			
			case 2:
				estimateWrapPhase << <grid, block >> > (SPWrapPhase, BGWrapPhase, cuSP, cuBG, Nx, Ny);
				estimateAmp << <grid, block >> > (cuAmpMap, cuSP, cuBG, Nx, Ny);
				e_wrap = clock();	wrap_time += e_wrap - s_wrap;

				//FFT-based unwrapping
				s_unwrap = clock();
				FastUnwrapping(BGWrapPhase, UnWrapPhaseBG, Nx, Ny);
				FastUnwrapping(SPWrapPhase, UnWrapPhaseSP, Nx, Ny);
				e_unwrap = clock();	unwrap_time += e_unwrap - s_unwrap;

				estimatePhase << <grid, block >> > (cuPhaseMap, UnWrapPhaseSP, UnWrapPhaseBG, Nx, Ny);
				break;
			}

			//DeviceMemOut("C:\\HilbertImg\\Data\\SP.256.256.raw", UnWrapPhaseSP2, Nx2, Ny2);
			//Goldstein's Unwarpping
			//cudaGoldsteinUnwrap2D(SPWrapPhase,UnWrapPhaseSP,Nx,Ny,frame);
			//cudaGoldsteinUnwrap2D(BGWrapPhase,UnWrapPhaseBG,Nx,Ny,frame);


			//do median filter on phase and amp imgs
			//MedianFilter_gpu<<<grid,block>>>(cuPhaseMap, Nx, Ny);
			//MedianFilter_gpu<<<grid,block>>>(cuAmpMap  , Nx, Ny);

			eE_time = clock();	extract_time += (eE_time - sE_time);

			cudaMemcpy(FinalPhase, cuPhaseMap, sizeof(float)*Nx*Ny, cudaMemcpyDeviceToHost);
			cudaMemcpy(FinalAmp, cuAmpMap, sizeof(float)*Nx*Ny, cudaMemcpyDeviceToHost);
			//DCT_UWLS_Unwrapped(UnWrapPhaseSP2, SPWrapPhase2, Nx2, Ny2);
			/*DeviceMemOut("test.256.256.raw", UnWrapPhaseSP2, Nx2, Ny2);*/
			//Calibration
			//phaseCalibration(FinalPhase, Nx, Ny);
			//ampCalibration(FinalAmp, Nx, Ny);

			bool status = true;
			float *IdentifyArea;
			if (ResizeFlag == 0)
			{
				IdentifyArea = (float *)malloc(Nx*Ny * sizeof(float));
				for (int i = 0; i < Nx*Ny; i++)
					IdentifyArea[i] = FinalPhase[i];

				if (checkArray(IdentifyArea, 0.5, 5 * M_PI, Nx*Ny) == true)
				{
					status = false;
					deleteCount++;
				}
			}
			else
			{
				IdentifyArea = (float *)malloc(Nx*Ny / 4 * sizeof(float));
				for (int j = Ny / 4; j < Ny * 3 / 4; j++)
					for (int i = Nx / 4; i < Nx * 3 / 4; i++)
					{
						IdentifyArea[(i - Nx / 4) + (j - Ny / 4) * Nx / 2] = FinalPhase[i + j*Nx];
					}
				/*
				// setup arguments
				typedef float T;
				summary_stats_unary_op<T>  unary_op;
				summary_stats_binary_op<T> binary_op;
				summary_stats_data<T> init;
				summary_stats_data<T> result;

				// part2 (if Mask[i]>10 and src[i]>threshold)
				thrust::device_vector<float> d_SP(cuPhaseMap, cuPhaseMap + Nx2*Ny2);
				init.initialize();
				result = thrust::transform_reduce(d_SP.begin(), d_SP.end(), unary_op, init, binary_op);
				float sp_mean = result.mean;
				float sp_std = sqrtf(result.variance_n());
				cout << "SP_MAX:" << result.max << endl;
				cout << "SP_MIN:" << result.min << endl;
				thrust::device_vector<float> d_BG(UnWrapPhaseBG2, UnWrapPhaseBG2 + Nx2*Ny2);
				init.initialize();
				result = thrust::transform_reduce(d_BG.begin(), d_BG.end(), unary_op, init, binary_op);
				float bg_mean = result.mean;
				float bg_std = sqrtf(result.variance_n());

				cout << "SP  mean: " << sp_mean << "std: "<< sp_std<< endl;
				cout << "BG  mean: " << bg_mean << "std: " << bg_std << endl;
				system("pause");
				*/
				if (checkArray(IdentifyArea, criteriaSTD, criteriaRange, Nx*Ny / 4) == true)
				{
					status = false;
					deleteCount++;
				}
			}
			free(IdentifyArea);



			//export the final data
#pragma omp parallel for
			for (int i = 0; i < Nx*Ny; i++)
			{
				ResultImg[i].phase = FinalPhase[i];
				ResultImg[i].amp = FinalAmp[i];
			}
#pragma omp barrier


			sprintf(Save_img_Path, "%s\\buffer%03d.phimap", SaveDir, frame);
			outputImg(Save_img_Path, ResultImg, status, sampleAngleRadX_Stack[frame - 1], sampleAngleRadY_Stack[frame - 1], Nx, Ny);

			AccumFrame++;
			PrintProcess(frame, SPDir, BGDir, AngDir);

			//copy to ReconStack
			if (ReconFlag == 1)
			{
				status_series[frame - 1] = status;
				//sampleAngleRadX_Stack[frame - 1] = SampleAngleDegX;
				//sampleAngleRadY_Stack[frame - 1] = SampleAngleDegY;
				Combine2Stack(PhaseStack, AmpStack, ResultImg, Nx, Ny, frame - 1);
			}

			if (SavePhaseStack == 1)
			{
				sprintf(Save_img_Path, "%s\\Phase%03d_%dx%d.raw", SaveDir, frame, Nx, Ny);
				FILE *fp = fopen(Save_img_Path, "wb");
				fwrite(FinalPhase, sizeof(float), Nx*Ny, fp);
				fclose(fp);
			}

			if (SaveAmpStack == 1)
			{
				sprintf(Save_img_Path, "%s\\Amp%03d_%dx%d.raw", SaveDir, frame, Nx, Ny);
				FILE *fp = fopen(Save_img_Path, "wb");
				fwrite(FinalAmp, sizeof(float), Nx*Ny, fp);
				fclose(fp);
			}

			
		}
		system("pause");

		PrintProcess(totalFrame, SPDir, BGDir, AngDir);

		if (SpeedTestFlag) {
			switch (QPI_Method) {
			case 0:
				for (int i = 0; i < totalFrame; i++)
				{
					free(SP_float_All[i]);
					free(BG_float_All[i]);
				}
				free(SP_float_All);
				free(BG_float_All);
				break;
			case 1:
			case 2:
				for (int i = 0; i < totalFrame; i++)
				{
					free(SP_cu_All[i]);
					free(BG_cu_All[i]);
				}
				free(SP_cu_All);
				free(BG_cu_All);
				break;
			}
		}
		else {
			switch (QPI_Method) {
			case 0:
				free(SP_float);
				free(BG_float);
				break;
			case 1:
			case 2:
				free(SP_cu);
				free(BG_cu);
				break;
			}
		}
	}
	else if(ReconFlag ==2)
	{
		if (SizeType) {
			ReconX = 512, ReconY = 512, ReconZ = 512;
		}
		else {
			ReconX = 256, ReconY = 256, ReconZ = 256;
		}

		totalFrame = count_file_num(SaveDir, "buffer");
		totalFrame_temp = totalFrame;

		PhaseStack = (float  *)malloc(ReconX*ReconY*totalFrame_temp * sizeof(float));
		AmpStack = (float  *)malloc(ReconX*ReconY*totalFrame_temp * sizeof(float));

		status_series = (bool   *)realloc(status_series, totalFrame_temp * sizeof(bool));
		sampleAngleRadX_Stack = (float *)realloc(sampleAngleRadX_Stack, totalFrame_temp * sizeof(float));
		sampleAngleRadY_Stack = (float *)realloc(sampleAngleRadY_Stack, totalFrame_temp * sizeof(float));

		LoadDateFromDir(PhaseStack, AmpStack, status_series, sampleAngleRadX_Stack, sampleAngleRadY_Stack, totalFrame, deleteCount, SaveDir);
	}

	
	//3D-RI reconstruction
	int ReconFrame = totalFrame - deleteCount;
	if ((ReconFlag==1 || ReconFlag == 2) && ReconFrame > 1)
	{		
		RefreshStack(PhaseStack, AmpStack, status_series, sampleAngleRadX_Stack, sampleAngleRadY_Stack, ReconX, ReconY, ReconFrame, deleteCount);
		free(status_series);

		/*FILE *tmpP = fopen("tmp.raw", "wb");
		fwrite(PhaseStack, sizeof(float), ReconX * ReconY * ReconFrame, tmpP);
		fclose(tmpP);*/
		
		
		//BatchReconPOCS(PhaseStack, AmpStack, sampleAngleRadX_Stack, sampleAngleRadY_Stack, ReconFrame);
		BatchRecon(PhaseStack, AmpStack, sampleAngleRadX_Stack, sampleAngleRadY_Stack, ReconFrame);
	}
	
	

	free(SP_img_Path);
	free(BG_img_Path);
	free(Ang_img_Path_X);
	free(Ang_img_Path_Y);
	free(Save_img_Path);

	free(SPImgTemp);
	free(BGImgTemp);
	free(AngImgTempX);
	free(AngImgTempY);
	free(AngImgX);
	free(AngImgY);

	free(cuSP_temp);
	free(cuBG_temp);
	free(PhaseMap);
	free(AmpMap);
	free(FinalPhase);
	free(FinalAmp);
	free(ResultImg);

	free(sampleAngleRadX_Stack);
	free(sampleAngleRadY_Stack);

	//release device memory
	cudaFree(circleImg);
	cudaFree(cuSP);
	cudaFree(cuBG);
	cudaFree(cuSP2);
	cudaFree(cuBG2);
	cudaFree(selectSP);
	cudaFree(selectBG);
	cudaFree(SPWrapPhase);
	cudaFree(BGWrapPhase);
	cudaFree(SPWrapPhase2);
	cudaFree(BGWrapPhase2);
	cudaFree(UnWrapPhaseSP);
	cudaFree(UnWrapPhaseBG);
	cudaFree(UnWrapPhaseSP2);
	cudaFree(UnWrapPhaseBG2);
	cudaFree(cuPhaseMap);
	cudaFree(cuAmpMap);
	cudaFree(cuPhaseMap2);
	cudaFree(cuAmpMap2);

	cudaDeviceReset();

}
//--------------------------------------------------------------------------------------
void Combine2Stack(float *Phase3D, float *Amp3D, microImg *Img,	int Nx, int Ny, int frameNum)
{
	switch (ResizeFlag){

	case 0:{
			   float *In_phase  = (float *)malloc(Nx*Ny        *sizeof(float));
			   float *In_amp    = (float *)malloc(Nx*Ny        *sizeof(float));
			   float *Out_phase = (float *)malloc(ReconX*ReconY*sizeof(float));
			   float *Out_amp   = (float *)malloc(ReconX*ReconY*sizeof(float));

			   if (Nx == ReconX && Ny == ReconY)
			   {
				   for (int i = 0; i < Nx*Ny; i++)
				   {
					   Out_phase[i] = Img[i].phase;
					   Out_amp[i] = Img[i].amp;
				   }
			   }
			   else
			   {
				   for (int i = 0; i < Nx*Ny; i++)
				   {
					   In_phase[i] = Img[i].phase;
					   In_amp[i] = Img[i].amp;
				   }
				   bilinear(In_phase, Out_phase, Nx, Ny, ReconX, ReconY);
				   bilinear(In_amp, Out_amp, Nx, Ny, ReconX, ReconY);
			   }
			   free(In_phase);
			   free(In_amp);

			   int idx1 = 0, idx2 = 0;
			   for (int j = 0; j < ReconY; j++)
			   for (int i = 0; i < ReconX; i++)
			   {
				   idx1 = i + j*ReconX;
				   idx2 = i + j*ReconX + frameNum*ReconX*ReconY;
				   //cut the center area (512*512) from orginal image (1024*1024) to 3-D 'Phase' & 'Amp'				
				   Phase3D[idx2] = Out_phase[idx1];
				   //PhaseCal_2D[idx1] = Phase_temp[idx2];

				   Amp3D[idx2] = Out_amp[idx1];

				   //check and modify the value
				   if (is_nan(Phase3D[idx2]) == true || is_inf(Phase3D[idx2]) == true)
					   Phase3D[idx2] = 0;
				   if (is_nan(Amp3D[idx2]) == true || is_inf(Amp3D[idx2]) == true)
					   Amp3D[idx2] = 1;
			   }
			   free(Out_phase);
			   free(Out_amp);

			   /*
			   for(int j=0; j<colSize;j++)
			   for(int i=0; i<rowSize;i++)
			   {
			   //cut the center area (512*512) from orginal image (1024*1024) to 3-D 'Phase' & 'Amp'
			   Phase_temp[i+j*rowSize+frameNum*rowSize*colSize]
			   = Img[int((i+org_rowSize*0.25)+(j+org_colSize*0.25)*org_rowSize)].phase;
			   PhaseCal_2D[i+j*rowSize] = Phase_temp[i+j*rowSize+frameNum*rowSize*colSize];

			   Amp_temp[i+j*rowSize+frameNum*rowSize*colSize]
			   = Img[int((i+org_rowSize*0.25)+(j+org_colSize*0.25)*org_rowSize)].amp;

			   //check and modify the value
			   if(is_nan(Phase_temp[i+j*rowSize+frameNum*rowSize*colSize])==true || is_inf(Phase_temp[i+j*rowSize+frameNum*rowSize*colSize])==true)
			   Phase_temp[i+j*rowSize+frameNum*rowSize*colSize] = 0;
			   if(is_nan(Amp_temp[i+j*rowSize+frameNum*rowSize*colSize])==true || is_inf(Amp_temp[i+j*rowSize+frameNum*rowSize*colSize])==true)
			   Amp_temp[i+j*rowSize+frameNum*rowSize*colSize] = 1;
			   }*/

			   /*if(checkArray(PhaseCal_2D, 0.6, 30)==true)
			   {
			   status = false;
			   }*/

	}break;

	case 1: {
		int idx1, idx2;
		if (Nx == ReconX && Ny == ReconY)
		{
			for (int i = 0; i < ReconX*ReconY; i++)
			{
				idx1 = i + frameNum*ReconX*ReconY;
				Phase3D[idx1] = Img[i].phase;
				Amp3D[idx1] = Img[i].amp;

				//check and modify the value
				if (is_nan(Phase3D[idx1]) == true || is_inf(Phase3D[idx1]) == true)
					Phase3D[idx1] = 0;
				if (is_nan(Amp3D[idx1]) == true || is_inf(Amp3D[idx1]) == true)
					Amp3D[idx1] = 1;
			}
		}
		else
		{
			for (int j = (Nx - ReconX) / 2; j < (Nx + ReconX) / 2; j++)
				for (int i = (Ny - ReconY) / 2; i < (Ny + ReconY) / 2; i++)
				{
					idx2 = (i - (Nx - ReconX) / 2) + (j - (Ny - ReconY) / 2) * ReconX + frameNum*ReconX*ReconY;
					idx1 = i + j * Nx;
					Phase3D[idx2] = Img[idx1].phase;
					Amp3D[idx2] = Img[idx1].amp;

					//check and modify the value
					if (isnan(Phase3D[idx2]) == true || isinf(Phase3D[idx2]) == true)
						Phase3D[idx2] = 0;
					if (isnan(Amp3D[idx2]) == true || isinf(Amp3D[idx2]) == true)
						Amp3D[idx2] = 1;
				}
		}
	}break;
	}
}
//--------------------------------------------------------------------------------------
void openImage(char * fpath, microImg * buf, bool &statu, double &angleX, double &angleY, int &sizeX, int &sizeY)
{
	FILE *fp;
	fp = fopen(fpath, "rb");
	if (!fp)
	{
		printf("\nCannot open the image: %s", fpath);
	}
	else
	{
		fread(&statu, 1, sizeof(bool), fp);
		fread(&angleX, 1, sizeof(double), fp);
		fread(&angleY, 1, sizeof(double), fp);
		fread(&sizeX, 1, sizeof(int), fp);
		fread(&sizeY, 1, sizeof(int), fp);
		buf = (microImg  *)realloc(buf, sizeX*sizeY * sizeof(microImg));
		fread(buf, sizeX*sizeY, sizeof(microImg), fp);
	}
	fclose(fp);
}
//--------------------------------------------------------------------------------------
void LoadDateFromDir(float *Phase3D, float *Amp3D, bool *status_series, float *sampleAngleRadX_Stack, float *sampleAngleRadY_Stack, int &totalFrame, int &deleteCount, char *dir)
{
	deleteCount = 0;		

	//read image & angle data	
	for (int order = 1; order <= totalFrame; order++)
	{
		end_time = clock();
		total_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;
		system("cls");
		printf("Loading Data from:%s\n"
			"Time      : %f sec", SaveDir, total_time);

		int frameNum = order - 1;
		char *PathImg = (char *)malloc(250 * sizeof(char));
		bool status;
		int Nx = 1024, Ny = 1024;
		microImg *Img = (microImg *)malloc(Nx*Ny * sizeof(microImg));	//input image
		double RadX, RadY;

		sprintf(PathImg, "%s\\buffer%03d.phimap", dir, order);
		openImage(PathImg, Img, status, RadX, RadY, Nx, Ny);

		if (!status)	deleteCount++;
		status_series[order - 1] = status;
		sampleAngleRadX_Stack[order - 1] = RadX;
		sampleAngleRadY_Stack[order - 1] = RadX;
		Combine2Stack(Phase3D, Amp3D, Img, Nx, Ny, order-1);

		free(Img);
		free(PathImg);
	}
}
//--------------------------------------------------------------------------------------
void RefreshStack(float *PhaseStack, float *AmpStack, bool *status_series, float *sampleAngleRadX, float *sampleAngleRadY, int ReconX, int ReconY, int ReconFrame, int deleteCount)
{
	PrintProcess(totalFrame, SPDir, BGDir, AngDir);
	printf("Delete %d slices from memory space...\n", deleteCount);
	printf("Total %d slices for Reconstruction.\n", ReconFrame);

	float *Phase_temp = (float  *)calloc(ReconX*ReconY*ReconFrame, sizeof(float));
	float *Amp_temp = (float  *)calloc(ReconX*ReconY*ReconFrame, sizeof(float));
	float *sampleAngleRadX_temp = (float *)calloc(ReconFrame, sizeof(float));
	float *sampleAngleRadY_temp = (float *)calloc(ReconFrame, sizeof(float));

	int count = 0;
	for (int z = 0; z < ReconFrame + deleteCount; z++)
	{
		if (status_series[z] == true)
		{
			for (int i = 0; i<ReconX*ReconY; i++)
			{
				Phase_temp[i + count*ReconX*ReconY] = PhaseStack[i + z*ReconX*ReconY];
				Amp_temp[i + count*ReconX*ReconY] = AmpStack[i + z*ReconX*ReconY];
	
			}
			sampleAngleRadX_temp[count] = sampleAngleRadX[z];
			sampleAngleRadY_temp[count] = sampleAngleRadY[z];
			count++;
		}
	}

	PhaseStack = (float *)realloc(PhaseStack, ReconX*ReconY*ReconFrame*sizeof(float));
	AmpStack = (float *)realloc(AmpStack, ReconX*ReconY*ReconFrame*sizeof(float));
	sampleAngleRadX = (float *)realloc(sampleAngleRadX, ReconFrame*sizeof(float));
	sampleAngleRadY = (float *)realloc(sampleAngleRadY, ReconFrame*sizeof(float));

	int ReconXY = ReconX * ReconY;
	for (int i = 0; i < ReconFrame; i++)
	{
		sampleAngleRadX[i] = sampleAngleRadX_temp[i];
		sampleAngleRadY[i] = sampleAngleRadY_temp[i];

		for (int t = 0; t < ReconXY; t++)
		{
			PhaseStack[t + i * ReconXY] = Phase_temp[t + i * ReconXY];
			AmpStack[t + i * ReconXY] = Amp_temp[t + i * ReconXY];
		}
	}

	cout << "ReconFrame: " << ReconFrame << endl;
	cout << "deleteCount: " << count << endl;
	/*memcpy(PhaseStack, Phase_temp, sizeof(Phase_temp));
	memcpy(AmpStack, Amp_temp, sizeof(Amp_temp));*/
	//memcpy(sampleAngleRadX, sampleAngleRadX_temp, ReconFrame * sizeof(float));
	//memcpy(sampleAngleRadY, sampleAngleRadY_temp, ReconFrame * sizeof(float));

	free(Phase_temp);
	free(Amp_temp);
	free(sampleAngleRadX_temp);
	free(sampleAngleRadY_temp);	
}
//--------------------------------------------------------------------------------------
void extractQPI(float *SP, float *BG, cufftComplex *cuSP_FFT, cufftComplex *cuBG_FFT, int Nx, int Ny)
{
	int blocksInX = (Nx + 32 - 1) / 32;
	int blocksInY = (Ny/4 + 32 - 1) / 32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);

	float *cuSP_temp, *cuBG_temp, *cuSP_resample, *cuBG_resample;
	cudaMalloc((void **)&cuSP_temp, sizeof(float)*Nx *Ny);
	cudaMalloc((void **)&cuBG_temp, sizeof(float)*Nx *Ny);
	cudaMalloc((void **)&cuSP_resample, sizeof(float)*Nx*(Ny / 4) );
	cudaMalloc((void **)&cuBG_resample, sizeof(float)*Nx*(Ny / 4) );
	s_datatransfer = clock();
	cudaMemcpy(cuSP_temp, SP, sizeof(float)*Nx*Ny, cudaMemcpyHostToDevice);
	cudaMemcpy(cuBG_temp, BG, sizeof(float)*Nx*Ny, cudaMemcpyHostToDevice);
	e_datatransfer = clock();
	dataTransfer_time += e_datatransfer - s_datatransfer;
	
	s_wrap = clock();
	bilinear_interpolation_kernel << <grid, block >> >(cuSP_resample, cuSP_temp, cuBG_resample, cuBG_temp, Nx, Ny, Nx, Ny/4);
	//bilinear_interpolation_kernel << <grid, block >> >(cuBG_resample, cuBG_temp, Nx, Ny, Nx, Ny/4);
	e_wrap = clock();	wrap_time += e_wrap - s_wrap;

	sequence1DFFT(cuSP_resample, cuSP_FFT, Nx, Ny);
	sequence1DFFT(cuBG_resample, cuBG_FFT, Nx, Ny);	
	//DeviceMemOut("cuSP_resample.1024.256.raw", cuSP_resample, Nx, Ny / 4);
	

	cudaFree(cuSP_temp);
	cudaFree(cuBG_temp);
	cudaFree(cuSP_resample);
	cudaFree(cuBG_resample);
}
//--------------------------------------------------------------------------------------
void sequence1DFFT(float *ResampleArray, cufftComplex *out_array, int Nx, int Ny)
{
	int blocksInX = (Nx / 2 + 32 - 1) / 32;
	int blocksInY = (Ny / 4 + 32 - 1) / 32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);

	//host memory
	//cufftComplex *host_FFT = (cufftComplex *)malloc(Nx * (Ny / 4) * sizeof(cufftComplex));
	//cufftComplex *host_out = (cufftComplex *)malloc((Nx / 4)*(Ny / 4) *sizeof(cufftComplex));
	//device memory
	cufftComplex *device_FFT, *out_FFT;
	float *sumFFT_1D;
	cudaMalloc((void **)&device_FFT, sizeof(cufftComplex)*Nx*(Ny / 4));
	cudaMalloc((void **)&out_FFT, sizeof(cufftComplex)*(Nx / 4)*(Ny / 4));
	cudaMalloc((void **)&sumFFT_1D, sizeof(float)*Nx);
	cudaMemset(sumFFT_1D, 0, Nx*sizeof(float));

	//copy the floating array to cufftComplex
	dim3 dimGrid(Nx / TILE_DIM, Ny /4 / TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
	real2cufft << <dimGrid, dimBlock >> >(device_FFT, ResampleArray);

	s_wrap = clock();
	//1D FFT
	//cuFFT1D(device_FFT, Nx, Ny / 4, -1);	
	cufftExecC2C(plan_1D_C2C_FORWARD_FT, device_FFT, device_FFT, CUFFT_FORWARD);
	//DeviceMemOutFFT("D:\\device_FFT.1024.256.raw", device_FFT, Nx, Ny / 4);
	
	//crop the component from FT domain
	

	s_wrap = clock();
	shiftArray << <grid, block >> >(device_FFT, Nx, Ny / 4);
	HistogramFT << <(Nx + 1024 - 1) / 1024, 1024 >> >(sumFFT_1D,device_FFT,Nx, Ny/4);
	e_wrap = clock();	wrap_time += e_wrap - s_wrap;

	//DeviceMemOut("D:\\sumFFT_1D.1024.1.raw", sumFFT_1D, Nx, 1);
	//find out the maximum and its index
	thrust::device_ptr<float> max_ptr = thrust::device_pointer_cast(sumFFT_1D);
	thrust::device_ptr<float> result_offset = thrust::max_element(max_ptr + int(Nx*0.6), max_ptr + Nx);

	//float max_value = result_offset[0];
	int max_idx = &result_offset[0] - &max_ptr[0];
	//printf("\nMininum value = %f\n", max_value);
	//printf("Position = %i\n", &result_offset[0] - &max_ptr[0]);

	int blocksX2 = (Nx / 4 + 32 - 1) / 32;
	int blocksY2 = (Ny / 4 + 32 - 1) / 32;
	dim3 grid2(blocksX2, blocksY2);
	dim3 block2(32, 32);

	int blocksX3 = (Nx / 8 + 32 - 1) / 32;
	int blocksY3 = (Ny / 4 + 32 - 1) / 32;
	dim3 grid3(blocksX3, blocksY3);
	dim3 block3(32, 32);

	s_wrap = clock();
	CropFTdomain << <grid2, block2 >> >(device_FFT, out_FFT, Nx, Ny, max_idx);
	//DeviceMemOutFFT("out_FFT.256.256.raw", out_FFT, (Nx / 4), (Ny / 4));
	shiftArray << <grid3, block3 >> >(out_FFT, Nx / 4, Ny / 4);
	//inverse FFT
	//cuFFT1D(out_FFT, Nx / 4, Ny / 4, 1);
	cufftExecC2C(plan_1D_C2C_INVERSE_FT, out_FFT, out_FFT, CUFFT_INVERSE);
	e_wrap = clock();	wrap_time += e_wrap - s_wrap;

	//DeviceMemOutFFT("out_iFFT.256.256.raw", out_FFT, (Nx / 4), (Ny / 4));
	dim3 dimGrid2(Nx / 4 / TILE_DIM, Ny / 4 / TILE_DIM, 1);
	dim3 dimBlock2(TILE_DIM, BLOCK_ROWS, 1);

	s_wrap = clock();
	copySharedMem << <dimGrid2, dimBlock2 >> >(out_array, out_FFT, Nx/4);
	e_wrap = clock();	wrap_time += e_wrap - s_wrap;

	cudaFree(device_FFT);
	cudaFree(sumFFT_1D);
	cudaFree(out_FFT);
	//free(host_out);
	//free(host_FFT);
}
//--------------------------------------------------------------------------------------
__global__ void real2cufft(cufftComplex *odata, const float *idata)
{
	__shared__ float tile[TILE_DIM * TILE_DIM];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[(threadIdx.y + j)*TILE_DIM + threadIdx.x] = idata[(y + j)*width + x];

	__syncthreads();

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
	{
		odata[(y + j)*width + x].x = tile[(threadIdx.y + j)*TILE_DIM + threadIdx.x];
		odata[(y + j)*width + x].y = 0;
	}

	/*__shared__ float tile[TILE_DIM][TILE_DIM];

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
		odata[(y + j)*width + x].x = tile[threadIdx.x][threadIdx.y + j];
		odata[(y + j)*width + x].y = 0;
	}*/	
}
//--------------------------------------------------------------------------------------
__global__ void copySharedMem(cufftComplex *odata, const cufftComplex *idata, const float scale)
{
	__shared__ cufftComplex tile[TILE_DIM ][ TILE_DIM];

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
		odata[(y + j)*width + x].x = tile[threadIdx.x][threadIdx.y + j].x * (1 / scale);
		odata[(y + j)*width + x].y = tile[threadIdx.x][threadIdx.y + j].y * (-1 / scale);
	}
		
}
//--------------------------------------------------------------------------------------
__global__ void HistogramFT(float *sumFFT_1D, cufftComplex *device_FFT, int Nx, int Ny)
{
	/*unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	if ((xIndex < Nx) && (yIndex < Ny))
	{		
		unsigned int idx = xIndex + Nx*yIndex;
		sumFFT_1D[xIndex] += (sqrtf(device_FFT[idx].x*device_FFT[idx].x + device_FFT[idx].y*device_FFT[idx].y));
	}*/
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int idx;
	if (i >= Nx) return;
	float sum = 0;
	for (int j = 0; j < Ny; j++)
	{
		idx = i + j*Nx;
		sum += (sqrtf(device_FFT[idx].x*device_FFT[idx].x + device_FFT[idx].y*device_FFT[idx].y));
	}
	sumFFT_1D[i] = sum;
}
//--------------------------------------------------------------------------------------
__global__ void shiftArray(cufftComplex *device_FFT, int Nx, int Ny)
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
/*__global__ void getmax(float *in1, float *out1, int *index)
{
	// Declare arrays to be in shared memory.
	__shared__ float max[1024];

	int nTotalThreads = blockDim.x;    // Total number of active threads
	float temp;
	float max_val;
	int max_index;
	int arrayIndex;

	// Calculate which element this thread reads from memory
	arrayIndex = gridDim.x*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
	max[threadIdx.x] = in1[arrayIndex];
	max_val = max[threadIdx.x];
	max_index = blockDim.x*blockIdx.x + threadIdx.x;
	__syncthreads();

	while (nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);
		if (threadIdx.x < halfPoint)
		{
			temp = max[threadIdx.x + halfPoint];
			if (temp > max[threadIdx.x])
			{
				max[threadIdx.x] = temp;
				max_val = max[threadIdx.x];
			}
		}
		__syncthreads();

		nTotalThreads = (nTotalThreads >> 1);    // divide by two.
	}

	if (threadIdx.x == 0)
	{
		out1[num_blocks*blockIdx.y + blockIdx.x] = max[threadIdx.x];
	}

	if (max[blockIdx.x] == max_val)
	{
		index[blockIdx.x] = max_index;
	}
}*/
//--------------------------------------------------------------------------------------
__global__ void CropFTdomain(cufftComplex *device_FFT, cufftComplex *device_crop, int Nx, int Ny, int center)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx1, idx2;

	if (xIndex < Nx / 4 && yIndex < Ny / 4)
	{
		idx1 = xIndex + (Nx / 4)*yIndex;
		idx2 = (xIndex - (Nx / 8) + center )+ Nx*yIndex;
		device_crop[idx1] = device_FFT[idx2];
	}	
}
//--------------------------------------------------------------------------------------
void BatchRecon(float *PhaseStack, float *AmpStack, float *sampleAngleRadX, float *sampleAngleRadY, int ReconFrame)
{
	//defined Grid & Block for GPU
	int blocksInX2D = (ReconX / 2 + 32 - 1) / 32;
	int blocksInY2D = (ReconY / 2 + 32 - 1) / 32;
	dim3 grid2D(blocksInX2D, blocksInY2D);
	dim3 block2D(32, 32);

	int blocksInX3D = (ReconX / 2 + 8 - 1) / 8;
	int blocksInY3D = (ReconY / 2 + 8 - 1) / 8;
	int blocksInZ3D = (ReconZ / 2 + 16 - 1) / 16;
	dim3 grid3D(blocksInX3D, blocksInY3D, blocksInZ3D);
	dim3 block3D(8, 8, 16);

	// Refractive Index of Medium
	float n_med = Nmed;
	float n_med2 = Nmed * Nmed;

	// Wavelength of Laser
	float lambda = wavelength *1e-9;

	// Frequency of Laser
	float f0 = 1 / lambda;
	float fm0 = f0 * n_med;
	float fm02 = fm0 * fm0;


	// Spatial Resolution of Grid
	//dx = 512/ffsize;	//need to check; original formula: dx*512/ffsize, where ffsize presents 512
	dx = (CameraPixelSize*1e-6) / Mag * 1024 / ReconZ;
	//dx = (pixelsizecamera*1e-6)/mag;

	// Wave Vector
	float k = 2 * M_PI * n_med / lambda;
	float k2 = k * k;

	// Frequency Resolution
	float df = 1 / ((ReconZ)*dx);
	int size_3D = ReconX*ReconY*ReconZ;	//total elements of the output stack

	//Filtering
	end_time = clock();
	total_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;
	
	PrintProcess(totalFrame, SPDir, BGDir, AngDir);
	printf("Filter Processing...\n"
		"Time      : %f sec", total_time);

	// Denoise using median filter (3 by 3)
	float *CUDA_Phase;
	float *CUDA_Amp;
	cudaMalloc((void **)&CUDA_Phase, sizeof(float)*ReconX*ReconY*ReconFrame);
	cudaMalloc((void **)&CUDA_Amp, sizeof(float)*ReconX*ReconY*ReconFrame);
	MedianFilter(CUDA_Phase, CUDA_Amp, PhaseStack, AmpStack, ReconX, ReconY, ReconFrame);
	free(PhaseStack);
	free(AmpStack);

	PrintProcess(totalFrame, SPDir, BGDir, AngDir);


	//2D FFT
	PrintProcess(totalFrame, SPDir, BGDir, AngDir);
	printf("Fast Fourier Transform...\n");

	cufftComplex *CUDA_F;
	cufftComplex *u_sp_device;
	float *CUDA_C;
	cudaMalloc((void **)&CUDA_F, sizeof(cufftComplex)*size_3D);
	cudaMalloc((void **)&u_sp_device, sizeof(cufftComplex)*ReconX*ReconY);
	cudaMalloc((void **)&CUDA_C, sizeof(float)*size_3D);
	cudaMemset(CUDA_C, 0, size_3D*sizeof(float));

	cufftComplex *iniComplex = (cufftComplex *)malloc(size_3D*sizeof(cufftComplex));
	for (int i = 0; i < size_3D; i++)	{ iniComplex[i].x = 0, iniComplex[i].y = 0; }
	cudaMemcpy(CUDA_F, iniComplex, sizeof(cufftComplex)*size_3D, cudaMemcpyHostToDevice);
	free(iniComplex);

	PrintProcess(totalFrame, SPDir, BGDir, AngDir);


	for (int z = 0; z<ReconFrame; z++)
	{
		PrintProcess(totalFrame, SPDir, BGDir, AngDir);
		printf("Fast Fourier Transform...(%d/%d)\n", z + 1, ReconFrame);

		CopyDataArray(u_sp_device, CUDA_Phase, CUDA_Amp, ReconX, ReconY, ReconZ, z);

		cuFFT2Dshift << <grid2D, block2D >> >(u_sp_device, ReconX, ReconY);
		cuFFT2D(u_sp_device, ReconX, ReconY, -1);	//FFT

		EdwardSphere(u_sp_device, CUDA_F, CUDA_C, fm0, df, sampleAngleRadX[z], sampleAngleRadY[z], ReconX, ReconY, ReconZ);
	}
	cudaFree(u_sp_device);
	cudaFree(CUDA_Phase);
	cudaFree(CUDA_Amp);
	//free(sampleAngleRadX);
	//free(sampleAngleRadY);
	//cuFFT3Dshift << <grid3D, block3D >> >(CUDA_F, ReconX, ReconY, ReconZ);
	//DeviceMemOutFFT("out.raw", CUDA_F, ReconX*ReconY, ReconZ);
	//system("pause");

	cufftComplex *n_3D = (cufftComplex *)malloc(size_3D*sizeof(cufftComplex));

	cufftComplex *CUDA_F2;
	cudaMalloc((void **)&CUDA_F2, sizeof(cufftComplex)*size_3D);
	PrintProcess(totalFrame, SPDir, BGDir, AngDir);

	//int ReconMode = 0;
	int AvailSize = freeDeviceMemory / 1024 / 1024;
	int RequireSize = size_3D / 1024 / 1024 * 4 * 2;
	if (AvailSize > RequireSize * 2)
		ReconMode = 0;
	else if (AvailSize > RequireSize)
		ReconMode = 1;
	//printf("\nAVAI: %d; size: %d\nChoose Type: %d\n", AvailSize, RequireSize * 2, ReconMode);
	//system("pause");

	//initial the F- and F2-fields
	initial_F_and_F2(CUDA_C, CUDA_F, CUDA_F2, dx, ReconX, ReconY, ReconZ);
	cudaFree(CUDA_C);

	cuFFT3D(CUDA_F, ReconX, ReconY, ReconZ, 1);	//inverse FFT
	//PrintProcess(SPdir, BGdir, Savedir);

	switch (ReconMode) {
	case 0:{
			   //estimate the RI from f-field
			   cufftComplex *CUDA_n;
			   cudaMalloc((void **)&CUDA_n, sizeof(cufftComplex)*size_3D);
			   est_n_3D(CUDA_n, CUDA_F, k2, n_med, ReconX, ReconY, ReconZ);

			   //Iteration
			   for (int iter = 0; iter <= IterTime; iter++)
			   {
				   PrintProcess(totalFrame, SPDir, BGDir, AngDir);
				   printf("Iterative time: %d\n", iter);

				   modify_F_3D(CUDA_n, CUDA_F, k2, n_med2, ReconX, ReconY, ReconZ);

				   cuFFT3D(CUDA_F, ReconX, ReconY, ReconZ, -1);	//FFT

				   check_F_3D(CUDA_F, CUDA_F2, ReconX, ReconY, ReconZ);

				   cuFFT3D(CUDA_F, ReconX, ReconY, ReconZ, 1);	//IFFT

				   est_n_3D(CUDA_n, CUDA_F, k2, n_med, ReconX, ReconY, ReconZ);

			   }//end of iteration

			   cuFFT3Dshift << <grid3D, block3D >> >(CUDA_n, ReconX, ReconY, ReconZ);
			   cudaMemcpy(n_3D, CUDA_n, sizeof(cufftComplex)*size_3D, cudaMemcpyDeviceToHost);
			   cudaFree(CUDA_n);
	}break;

	case 1:{
			   //estimate the N-value from f-field
			   est_n_3D(n_3D, CUDA_F, k2, n_med, ReconX, ReconY, ReconZ);

			   //Iteration
			   for (int iter = 0; iter <= IterTime; iter++)
			   {
				   PrintProcess(totalFrame, SPDir, BGDir, AngDir);
				   printf("Iterative time: %d\n", iter);

				   modify_F_3D(n_3D, CUDA_F, k2, n_med2, ReconX, ReconY, ReconZ);

				   cuFFT3D(CUDA_F, ReconX, ReconY, ReconZ, -1);	//FFT

				   check_F_3D(CUDA_F, CUDA_F2, ReconX, ReconY, ReconZ);

				   cuFFT3D(CUDA_F, ReconX, ReconY, ReconZ, 1);	//IFFT

				   est_n_3D(n_3D, CUDA_F, k2, n_med, ReconX, ReconY, ReconZ);

			   }//end of iteration


			   FFT3Dshift_cufftComplex(n_3D, ReconX, ReconY, ReconZ);
	}break;
	}

	// export the final results
	PrintProcess(totalFrame, SPDir, BGDir, AngDir);
	printf("Export the final results...\n"
		   "Time      : %f sec\n", total_time);

	//export the 3-dimensional reconstructure stack	
	char *SavePath = (char *)malloc(250 * sizeof(char));
	switch (ReconSave){
	case 0:
		sprintf(SavePath, "%s\\3D-RI-Complex_%dx%dx%d.rimap", SaveDir, ReconX, ReconY, ReconZ);
		exportComplex(SavePath, n_3D, size_3D);
		break;
	case 1: {
		sprintf(SavePath, "%s\\3D-RI-Real_%dx%dx%d.rimap", SaveDir, ReconX, ReconY, ReconZ);
		float *real_3D = (float*)malloc(size_3D*sizeof(float));

#pragma omp parallel for
		for (int i = 0; i<size_3D; i++)	real_3D[i] = n_3D[i].x;
#pragma omp barrier

		exportRAW_F(SavePath, real_3D, size_3D);
		free(real_3D);
	} break;
	case 2: {
		sprintf(SavePath, "%s\\3D-RI-Imag_%dx%dx%d.rimap", SaveDir, ReconX, ReconY, ReconZ);
		float *imag_3D = (float*)malloc(size_3D*sizeof(float));

#pragma omp parallel for
		for (int i = 0; i<size_3D; i++)	imag_3D[i] = n_3D[i].y;
#pragma omp barrier

		exportRAW_F(SavePath, imag_3D, size_3D);
		free(imag_3D);		
	} break;
		
	}

	free(SavePath);
	free(n_3D);
	cudaFree(CUDA_F);
	cudaFree(CUDA_F2);
}
//--------------------------------------------------------------------------------------
void BatchReconPOCS(float *PhaseStack, float *AmpStack, float *sampleAngleRadX, float *sampleAngleRadY, int ReconFrame)
{
	//defined Grid & Block for GPU
	int blocksInX2D = (ReconX / 2 + 32 - 1) / 32;
	int blocksInY2D = (ReconY / 2 + 32 - 1) / 32;
	dim3 grid2D(blocksInX2D, blocksInY2D);
	dim3 block2D(32, 32);

	int blocksInX3D = (ReconX / 2 + 8 - 1) / 8;
	int blocksInY3D = (ReconY / 2 + 8 - 1) / 8;
	int blocksInZ3D = (ReconZ / 2 + 8 - 1) / 16;
	dim3 grid3D(blocksInX3D, blocksInY3D, blocksInZ3D);
	dim3 block3D(8, 8, 16);

	// Refractive Index of Medium
	float n_med = Nmed;
	float n_med2 = Nmed * Nmed;

	// Wavelength of Laser
	float lambda = wavelength *1e-9;

	// Frequency of Laser
	float f0 = 1 / lambda;
	float fm0 = f0 * n_med;
	float fm02 = fm0 * fm0;


	// Spatial Resolution of Grid
	//dx = 512/ffsize;	//need to check; original formula: dx*512/ffsize, where ffsize presents 512
	dx = (CameraPixelSize*1e-6) / Mag * 1024 / ReconZ;
	//dx = (pixelsizecamera*1e-6)/mag;

	// Wave Vector
	float k = 2 * M_PI * n_med / lambda;
	float k2 = k * k;

	// Frequency Resolution
	float df = 1 / ((ReconZ)*dx);
	int size_3D = ReconX*ReconY*ReconZ;	//total elements of the output stack

	//Filtering
	end_time = clock();
	total_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;

	PrintProcess(totalFrame, SPDir, BGDir, AngDir);
	printf("Copy data from HOST to DEVICE...\n"
		"Time      : %f sec", total_time);

	// Denoise using median filter (3 by 3)
	float *CUDA_Phase;
	float *CUDA_Amp;
	cufftComplex *CUDA_Rytov_stack;		//used for record the complex information
	cudaMalloc((void **)&CUDA_Phase, sizeof(float)*ReconX*ReconY*ReconFrame);
	cudaMalloc((void **)&CUDA_Amp, sizeof(float)*ReconX*ReconY*ReconFrame);
	cudaMalloc((void **)&CUDA_Rytov_stack, sizeof(cufftComplex)*ReconX*ReconY*ReconFrame);
	//copy the phase and amp info. to the Device Array
	cudaMemcpy(CUDA_Phase, PhaseStack, sizeof(float)*ReconX*ReconY*ReconFrame, cudaMemcpyHostToDevice);
	cudaMemcpy(CUDA_Amp, AmpStack, sizeof(float)*ReconX*ReconY*ReconFrame, cudaMemcpyHostToDevice);
	Combine2ComplexStack(CUDA_Rytov_stack, CUDA_Phase, CUDA_Amp, ReconX, ReconY, ReconFrame);
	cudaFree(CUDA_Phase);
	cudaFree(CUDA_Amp);
	free(PhaseStack);
	free(AmpStack);

	//2D FFT
	PrintProcess(totalFrame, SPDir, BGDir, AngDir);
	printf("Perform the BatchMode Fourier Transform...\n");
	//do the batch Fourier transform
	Shift2DonStack(CUDA_Rytov_stack, ReconX, ReconY, ReconFrame);
	cuFFT2D_Batch(CUDA_Rytov_stack, ReconX, ReconY, ReconFrame, -1);
	Shift2DonStack(CUDA_Rytov_stack, ReconX, ReconY, ReconFrame);

	//Frequency interpolation
	cufftComplex *CUDA_F;
	float *CUDA_C;
	cudaMalloc((void **)&CUDA_F, sizeof(cufftComplex)*size_3D);
	cudaMalloc((void **)&CUDA_C, sizeof(float)*size_3D);
	cudaMemset(CUDA_C, 0.f, size_3D*sizeof(float));

	/*cufftComplex *iniComplex = (cufftComplex *)malloc(size_3D*sizeof(cufftComplex));
	for (int i = 0; i < size_3D; i++)	{ iniComplex[i].x = 0, iniComplex[i].y = 0; }
	cudaMemcpy(CUDA_F, iniComplex, sizeof(cufftComplex)*size_3D, cudaMemcpyHostToDevice);
	free(iniComplex);*/

	for (int i = 0; i < ReconFrame; i++)
	{
		PrintProcess(totalFrame, SPDir, BGDir, AngDir);
		printf("Perform the frequency interpolation in the Fourier domain... (%d/%d)\n", i + 1, ReconFrame);
		FrquencyInterpolation(CUDA_F, CUDA_C, CUDA_Rytov_stack, sampleAngleRadX, sampleAngleRadY, fm0, df, dx, n_med, i, ReconFrame, ReconX, ReconY, ReconZ);
	}
	free(sampleAngleRadX);
	free(sampleAngleRadY);
	cudaFree(CUDA_Rytov_stack);
	cudaFree(CUDA_C);

	cufftComplex *n_3D = (cufftComplex *)malloc(size_3D*sizeof(cufftComplex));

	cufftComplex *CUDA_N, *CUDA_N2;
	cufftComplex *CUDA_f2, *CUDA_F2;
	cudaMalloc((void **)&CUDA_N, sizeof(cufftComplex)*size_3D);
	cudaMalloc((void **)&CUDA_N2, sizeof(cufftComplex)*size_3D);
	cudaMalloc((void **)&CUDA_f2, sizeof(cufftComplex)*size_3D);
	cudaMalloc((void **)&CUDA_F2, sizeof(cufftComplex)*size_3D);
	PrintProcess(totalFrame, SPDir, BGDir, AngDir);

	//int ReconMode = 0;
	int AvailSize = freeDeviceMemory / 1024 / 1024;
	int RequireSize = size_3D / 1024 / 1024 * 4 * 2;
	if (AvailSize > RequireSize * 2)
		ReconMode = 0;
	else if (AvailSize > RequireSize)
		ReconMode = 1;
	


	//Steadily Decreasing Parameter(SDP參數)
	float beta = 1.000;
	float beta_red = 0.995;

	float ng = 20; //TV - steepest descent loop 次數
	float alpha = 0.2;
	float rmax = 0.95;
	float alpha_red = 0.95;
	float Dp = 0.f, Dd = 0.f, Dg = 0.f, ddeps;

	//PART 1 :POCS
	//(1) 將 F_3D(measured data的frequency domain) copy 到 F_3D_2
	//     * F_3D   會隨著 iteration 改變
	//     * F_3D_2 維持不改變
	//F_3D_2 = F_3D;
	cudaMemcpy(CUDA_F2, CUDA_F, sizeof(cufftComplex)*size_3D, cudaMemcpyDeviceToDevice);

	//(2) 將 F_3D 作 inverse Fourier transform 得到 f_3D
	//f_3D = ifftn(F_3D);
	cuFFT3D(CUDA_F, ReconX, ReconY, ReconZ, 1);

	//(3) 轉換成折射率分布
	//n_3D = sqrt(-f_3D / k2 + 1) * n_med;		

	//(4) Enforcement(暴力法)
	//n_3D_2 = real(n_3D);  //real condition
	//n_3D_2(n_3D_2<n_med) = n_med;       //positive condition
	Est_n_3D_POCS(CUDA_N, CUDA_N2, CUDA_F, k2, n_med, ReconX, ReconY, ReconZ);

	//(5 - 1) 轉換為f _3D 和 f_3D_2
	//f_3D = -k2*(n_3D  . ^ 2 / n_med ^ 2 - 1); % Old
	//f_3D_2 = -k2*(n_3D_2. ^ 2 / n_med ^ 2 - 1); % New
	ConvertN2F(CUDA_F, CUDA_f2, CUDA_N, CUDA_N2, k2, n_med2, ReconX, ReconY, ReconZ);

	//(5 - 2) 計算data consistency step distance(Dp)
	//Dp = sqrt(sum(abs(f_3D_2 - f_3D).*abs(f_3D_2 - f_3D)));
	Dp = Est_Dp_Dd_POCS(CUDA_f2, CUDA_F, ReconX, ReconY, ReconZ);

	//(6) 計算Dd
	//F_3D = fftn(f_3D_2);
	//Dd = sqrt(sum(abs(F_3D_2 - F_3D).*abs(F_3D_2 - F_3D)));
	cudaMemcpy(CUDA_F, CUDA_f2, sizeof(cufftComplex)*size_3D, cudaMemcpyDeviceToDevice);
	cuFFT3D(CUDA_F, ReconX, ReconY, ReconZ, -1);	//forward FFT
	Dd = Est_Dp_Dd_POCS(CUDA_F2, CUDA_F, ReconX, ReconY, ReconZ);


	//% PART 2 :Gradient descent step(TV)
	//	% == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
	//	f_3D = f_3D_2;
	cudaMemcpy(CUDA_F, CUDA_f2, sizeof(cufftComplex)*size_3D, cudaMemcpyDeviceToDevice);
	//n_3D_2 = zeros(ffsize, ffsize, ffsize);  % n_3D_2用來儲存grad(TV) (*******INPORTANTANT*******)
	float dtvg = 0.f;
	//% (1) TV - steepest descent loop
	for (int TSDloop = 1; TSDloop <= ng; TSDloop++) {
		//% (a)計算 n_3D_2 的 grad(TV)		(b)計算normalized grad(TV)
		GradientDescentTV(CUDA_N2, CUDA_f2, ReconX, ReconY, ReconZ);

		//% (c)
		//% dtvg = alpha * dp(for 1st計算), dtvg下降的斜率
		dtvg = alpha * Dp;
		//f_3D_2 = f_3D_2 - dtvg * n_3D_2;
		EstF_TV(CUDA_f2, CUDA_N2, dtvg, ReconX, ReconY, ReconZ);
	}


	//% (2) 計算 Gradient descent step distance(Dg) : f_3D_2(New), f_3D(Old)
	//Dg = sqrt(sum(abs(f_3D_2 - f_3D).*abs(f_3D_2 - f_3D)));
	Dg = Est_Dp_Dd_POCS(CUDA_f2, CUDA_F, ReconX, ReconY, ReconZ);

	//% == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
	//	%  PART 3. 決定迭代參數
	//	%  計算dtvg給第2次iteration
	//	% == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
	if(Dg > rmax*Dp && Dd>ddeps)
		dtvg = dtvg * alpha_red;

		
		
	//% ***********************************************************************
	//% Iteration(迭代)
	//% ***********************************************************************
	//fprintf('Start interation....\n')

	for (int ITERloop = 1; ITERloop <= 100; ITERloop++) {
		//F_3D = fftn(f_3D);
		cuFFT3D(CUDA_F, ReconX, ReconY, ReconZ, -1);	//forward FFT
		//% == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
		//% PART 1 :POCS
		//% == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
		//% (1) 將 F_3D(measured data的frequency domain) copy 到 F_3D_2
		//%     * F_3D   會隨著 iteration 改變
		//%     * F_3D_2 維持不改變
		//F_3D(F_3D_2~= 0) = beta*F_3D_2(F_3D_2~= 0) + (1 - beta)*F_3D(F_3D_2~= 0);
		EstF_Beta(CUDA_F, CUDA_F2, beta, ReconX, ReconY, ReconZ);
		beta = beta*beta_red;

		//% (2) 將 F_3D 作 inverse Fourier transform 得到 f_3D
			//f_3D = ifftn(F_3D);
		cuFFT3D(CUDA_F, ReconX, ReconY, ReconZ, 1);	//inverse FFT

		//% (3) 轉換成折射率分布
			//n_3D = sqrt(-f_3D / k2 + 1) * n_med;

		//% (4) Enforcement ???
			//n_3D_2(real(n_3D) < n_med) = n_med;       % positive condition
			//n_3D_2 = real(n_3D);  % real
			Est_n_3D_POCS(CUDA_N, CUDA_N2, CUDA_F, k2, n_med, ReconX, ReconY, ReconZ);


		//% (5 - 1) 計算data consistency step distance(Dp)
			//f_3D = -k2*(n_3D. ^ 2 / n_med ^ 2 - 1); % Old
			//f_3D_2 = -k2*(n_3D_2. ^ 2 / n_med ^ 2 - 1); % New
			ConvertN2F(CUDA_F, CUDA_f2, CUDA_N, CUDA_N2, k2, n_med2, ReconX, ReconY, ReconZ);

		//% (5 - 2) 計算data consistency step distance(Dp)
			//Dp = sqrt(sum(abs(f_3D_2 - f_3D).*abs(f_3D_2 - f_3D)));
			Dp = Est_Dp_Dd_POCS(CUDA_f2, CUDA_F, ReconX, ReconY, ReconZ);

		//% (6) 計算Dd
			//F_3D = fftn(f_3D_2);
			//Dd = sqrt(sum(abs(F_3D_2 - F_3D).*abs(F_3D_2 - F_3D)));
			cudaMemcpy(CUDA_F, CUDA_f2, sizeof(cufftComplex)*size_3D, cudaMemcpyDeviceToDevice);
			cuFFT3D(CUDA_F, ReconX, ReconY, ReconZ, -1);	//forward FFT
			Dd = Est_Dp_Dd_POCS(CUDA_F2, CUDA_F, ReconX, ReconY, ReconZ);

		//% == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
		//% PART 2 :Gradient descent step(TV)
		//% == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
			//f_3D = f_3D_2;
			cudaMemcpy(CUDA_F, CUDA_f2, sizeof(cufftComplex)*size_3D, cudaMemcpyDeviceToDevice);

		ng = 20;
		//% (1) TV - steepest descent loop
		for (int TSDloop = 1; TSDloop <= ng; TSDloop++) {

			//% (a)計算 n_3D_2 的 grad(TV)
			/*for ss = 2 : ffsize
			for tt = 2 : ffsize
			for kk = 2 : ffsize
			term1_deno = sqrt((f_3D_2(ss, tt, kk) - f_3D_2(ss, tt - 1, kk)). ^ 2 + (f_3D_2(ss, tt, kk) - f_3D_2(ss - 1, tt, kk)). ^ 2 + (f_3D_2(ss, tt, kk) - f_3D_2(ss, tt, kk - 1)). ^ 2);
			term2_deno = sqrt((f_3D_2(ss, tt + 1, kk) - f_3D_2(ss, tt, kk)). ^ 2 + (f_3D_2(ss, tt + 1, kk) - f_3D_2(ss - 1, tt + 1, kk)). ^ 2 + (f_3D_2(ss, tt + 1, kk) - f_3D_2(ss, tt + 1, kk - 1)). ^ 2);
			term3_deno = sqrt((f_3D_2(ss + 1, tt, kk) - f_3D_2(ss + 1, tt - 1, kk)). ^ 2 + (f_3D_2(ss + 1, tt, kk) - f_3D_2(ss, tt, kk)). ^ 2 + (f_3D_2(ss + 1, tt, kk) - f_3D_2(ss + 1, tt + 1, kk - 1)). ^ 2);
			term4_deno = sqrt((f_3D_2(ss, tt, kk + 1) - f_3D_2(ss, tt - 1, kk + 1)). ^ 2 + (f_3D_2(ss, tt, kk + 1) - f_3D_2(ss - 1, tt, kk + 1)). ^ 2 + (f_3D_2(ss, tt, kk + 1) - f_3D_2(ss, tt, kk)). ^ 2);

			term1_num = 3 * f_3D_2(ss, tt, kk) - f_3D_2(ss, tt - 1, kk) - f_3D_2(ss - 1, tt, kk) - f_3D_2(ss, tt, kk - 1);
			term2_num = f_3D_2(ss, tt + 1, kk) - f_3D_2(ss, tt, kk);
			term3_num = f_3D_2(ss + 1, tt, kk) - f_3D_2(ss, tt, kk);
			term4_num = f_3D_2(ss, tt, kk + 1) - f_3D_2(ss, tt, kk);

			//% 將n_3D_2用來儲存grad(TV) (*******INPORTANTANT*******)
			n_3D_2(ss, tt, kk) = term1_num / term1_deno - term2_num / term2_deno - term3_num / term3_deno - term4_num / term4_deno;

			end
			end
			end

			//% (b)計算normalized grad(TV)
			n_3D_2 = n_3D_2. / abs(n_3D_2);*/
			GradientDescentTV(CUDA_N2, CUDA_f2, ReconX, ReconY, ReconZ);

			//% (c)
			//f_3D_2 = f_3D_2 - dtvg * n_3D_2;
			EstF_TV(CUDA_f2, CUDA_N2, dtvg, ReconX, ReconY, ReconZ);

			//% (2) 計算 Gradient descent step distance(Dg) : f_3D_2(New), f_3D(Old)
			//Dg = sqrt(sum(abs(f_3D_2 - f_3D).*abs(f_3D_2 - f_3D)));
			Dg = Est_Dp_Dd_POCS(CUDA_f2, CUDA_F, ReconX, ReconY, ReconZ);


			//% == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
			//%  PART 3. 決定迭代參數
			//%  計算dtvg給下一次iteration
			//% == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
			if (Dg > rmax*Dp && Dd > ddeps)
				dtvg = dtvg * alpha_red;
		}
	}


	//f_3D = -k2*(n_3D. ^ 2 / n_med ^ 2 - 1);
	modify_F_3D(CUDA_N, CUDA_F, k2, n_med2, ReconX, ReconY, ReconZ);
	//F_3D = fftn(f_3D);
	cuFFT3D(CUDA_F, ReconX, ReconY, ReconZ, -1);	//FFT
	//F_3D(F_3D_2~= 0) = beta*F_3D_2(F_3D_2~= 0) + (1 - beta)*F_3D(F_3D_2~= 0);
	EstF_Beta(CUDA_F, CUDA_F2, beta, ReconX, ReconY, ReconZ);
	//f_3D = ifftn(F_3D);
	cuFFT3D(CUDA_F, ReconX, ReconY, ReconZ, 1);	//IFFT
	//n_3D = sqrt(-f_3D / k2 + 1) *n_med;
	est_n_3D(CUDA_N, CUDA_F, k2, n_med, ReconX, ReconY, ReconZ);


		
	cudaMemcpy(n_3D, CUDA_N, sizeof(cufftComplex)*size_3D, cudaMemcpyDeviceToHost);
	cudaFree(CUDA_N);
		

	// export the final results
	PrintProcess(totalFrame, SPDir, BGDir, AngDir);
	printf("Export the final results...\n"
		"Time      : %f sec\n", total_time);

	//export the 3-dimensional reconstructure stack	
	char *SavePath = (char *)malloc(250 * sizeof(char));
	switch (ReconSave){
	case 0:
		sprintf(SavePath, "%s\\3D-RI-Complex_%dx%dx%d.raw", SaveDir, ReconX, ReconY, ReconZ);
		exportComplex(SavePath, n_3D, size_3D);
		break;
	case 1: {
				sprintf(SavePath, "%s\\3D-RI-Real_%dx%dx%d.raw", SaveDir, ReconX, ReconY, ReconZ);
				float *real_3D = (float*)malloc(size_3D*sizeof(float));

#pragma omp parallel for
				for (int i = 0; i<size_3D; i++)	real_3D[i] = n_3D[i].x;
#pragma omp barrier

				exportRAW_F(SavePath, real_3D, size_3D);
				free(real_3D);
	} break;
	case 2: {
				sprintf(SavePath, "%s\\3D-RI-Imag_%dx%dx%d.raw", SaveDir, ReconX, ReconY, ReconZ);
				float *imag_3D = (float*)malloc(size_3D*sizeof(float));

#pragma omp parallel for
				for (int i = 0; i<size_3D; i++)	imag_3D[i] = n_3D[i].y;
#pragma omp barrier

				exportRAW_F(SavePath, imag_3D, size_3D);
				free(imag_3D);
	} break;

	}

	free(SavePath);
	free(n_3D);
	cudaFree(CUDA_F);
	cudaFree(CUDA_F2);
	cudaFree(CUDA_f2);
}
//--------------------------------------------------------------------------------------
void bilinear(float *input, float *output, int M1, int N1, int M2, int N2)
{
	int x, y, index;
	float A, B, C, D, gray;
	float x_ratio = ((float)(M1 - 1)) / M2;
	float y_ratio = ((float)(N1 - 1)) / N2;
	float x_diff, y_diff;
	int offset = 0;
	for (int i = 0; i<N2; i++) {
		for (int j = 0; j<M2; j++) {
			x = (int)(x_ratio * j);
			y = (int)(y_ratio * i);
			x_diff = (x_ratio * j) - x;
			y_diff = (y_ratio * i) - y;
			index = y*M1 + x;

			// range is 0 to 255 thus bitwise AND with 0xff
			A = input[index];
			B = input[index + 1];
			C = input[index + M1];
			D = input[index + M1 + 1];

			// Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
			gray = (float)(
				A*(1 - x_diff)*(1 - y_diff) + B*(x_diff)*(1 - y_diff) +
				C*(y_diff)*(1 - x_diff) + D*(x_diff*y_diff)
				);

			output[offset++] = gray;
		}
	}
}
//--------------------------------------------------------------------------------------
void outputImg(char *fpath, microImg *output, bool status, double angleX, double angleY, int sizeX, int sizeY)
{
	FILE *fp;
	fp = fopen(fpath, "wb");
	if (!fp)
	{
		printf("\nCannot save the image at: %s", fpath);
	}
	else
	{
		fwrite(&status, sizeof(bool), 1, fp);
		fwrite(&angleX, sizeof(double), 1, fp);
		fwrite(&angleY, sizeof(double), 1, fp);
		fwrite(&sizeX, sizeof(int), 1, fp);
		fwrite(&sizeY, sizeof(int), 1, fp);
		fwrite(output, sizeof(microImg), sizeX*sizeY, fp);
	}
	fclose(fp);
}
//--------------------------------------------------------------------------------------
void openImage(char * fpath, microImg * buf, bool &status, double &angleX, double &angleY)
{
	FILE *fp;
	fp = fopen(fpath, "rb");
	if (!fp)
	{
		printf("\nCannot open the image: %s", fpath);
	}
	else
	{
		int sizeX, sizeY;
		fread(&status, sizeof(bool), 1, fp);
		fread(&angleX, sizeof(double), 1, fp);
		fread(&angleY, sizeof(double), 1, fp);
		fread(&sizeX, sizeof(int), 1, fp);
		fread(&sizeY, sizeof(int), 1, fp);
		buf = (microImg *)calloc(sizeX*sizeY, sizeof(microImg));
		fread(buf, sizeof(microImg), sizeX*sizeY, fp);
	}
	fclose(fp);
}
//--------------------------------------------------------------------------------------
int count_file_num(char* target_folder, char *head_name)
{
	int count = 0;                //檔案的counter

#ifdef _WIN32	
	char *szDir = (char *)malloc(256 * sizeof(char));          //要讀取的資料夾的位址。 
	wchar_t* szDir2 = (wchar_t*)malloc(256 * sizeof(wchar_t));
	WIN32_FIND_DATA FileData;    //指著目前讀取到的File的指標。
	HANDLE hList;                //指著要讀取的資料夾的指標。
	sprintf(szDir, "%s\\%s*", target_folder, head_name);
	mbstowcs(szDir2, szDir, strlen(szDir) + 1);

	if ((hList = FindFirstFile(szDir2, &FileData)) == INVALID_HANDLE_VALUE)
		cout << "No directory can be found." << endl ;
	else {
		while (1) {
			count++;
			if (!FindNextFile(hList, &FileData)) {
				if (GetLastError() == ERROR_NO_MORE_FILES)
					break;
			}
			
		}
	}
	free(szDir);
	free(szDir2);
	FindClose(hList);
#elif __linux__ 
	DIR* dirFile = opendir(target_folder);
	if (dirFile)
	{
		struct dirent* hFile;
		errno = 0;
		while ((hFile = readdir(dirFile)) != NULL)
		{
			if (!strcmp(hFile->d_name, ".")) continue;
			if (!strcmp(hFile->d_name, "..")) continue;

			// in linux hidden files all start with '.'
			if (gIgnoreHidden && (hFile->d_name[0] == '.')) continue;

			// dirFile.name is the name of the file. Do whatever string comparison 
			// you want here. Something like:
			if (strstr(hFile->d_name, head_name))
				count++;
			//printf("found an .txt file: %s", hFile->d_name);
		}
		closedir(dirFile);
	}
#endif
	return count;
}
//--------------------------------------------------------------------------------------
void AngleCalculation(float *UnWrapPhase, double &SampleAngleDegX, double &SampleAngleDegY,
	double &CameraAngleDegX, double &CameraAngleDegY, int rowSize, int colSize)
{
	float endPhase = 0;
	float Mux = 0;
	double SampleAngleRad, CameraAngleRad;
	//計算X 
	endPhase = UnWrapPhase[((int)(rowSize - 3))*colSize + ((int)(colSize / 2))] - UnWrapPhase[((int)(0 + 2))*colSize + ((int)(colSize / 2))];
	Mux = (float)((endPhase / (2 * M_PI)) / ((rowSize - 4)*CameraPixelSize));

	CameraAngleRad = asin((double)(Mux*wavelength* 1e-9));
	SampleAngleRad = asin((double)(Mag*sin(CameraAngleRad) / Nmed));

	CameraAngleDegX = CameraAngleRad * 180 / M_PI;
	SampleAngleDegX = SampleAngleRad * 180 / M_PI;

	//計算Y
	endPhase = UnWrapPhase[((int)(rowSize / 2))*colSize + ((int)(colSize - 3))] - UnWrapPhase[((int)(rowSize / 2))*colSize + ((int)(0 + 2))];
	Mux = (float)((endPhase / (2 * M_PI)) / ((colSize - 4)*CameraPixelSize));
	CameraAngleRad = asin((double)(Mux*wavelength* 1e-9));
	SampleAngleRad = asin((double)(Mag*sin(CameraAngleRad) / Nmed));

	CameraAngleDegY = CameraAngleRad * 180 / M_PI;
	SampleAngleDegY = SampleAngleRad * 180 / M_PI;
}
//--------------------------------------------------------------------------------------
double modulus(double a, double b)
{
	int result = static_cast<int>(a / b);
	return a - static_cast<double>(result)* b;
}
//--------------------------------------------------------------------------------------
double myfmod(double a, double b)
{
	double result = fmod(a, b);

	if (result<0)
		result = result + b;

	return result;
}
//--------------------------------------------------------------------------------------
double myround(double number)
{
	return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}
//--------------------------------------------------------------------------------------
double mymax(double a, double b)
{
	if (a>b)
		return a;
	else
		return b;
}
//--------------------------------------------------------------------------------------
double mymin(double a, double b)
{
	if (a>b)
		return b;
	else
		return a;
}
//--------------------------------------------------------------------------------------
int mod(int a, int b)
{
	return (((a < 0) ? ((a % b) + b) : a) % b);
}
//--------------------------------------------------------------------------------------
double round_MS(double num)
{
	return (num > 0.0) ? floor(num + 0.5) : ceil(num - 0.5);
}
//--------------------------------------------------------------------------------------
bool is_nan(double dVal)
{
	double dNan = std::numeric_limits<double>::quiet_NaN();

	if (dVal == dNan)
		return true;
	return false;
}
//--------------------------------------------------------------------------------------
bool is_inf(double dVal)
{
	double dNan = std::numeric_limits<double>::infinity();

	if (dVal == dNan)
		return true;
	return false;
}
//--------------------------------------------------------------------------------------
__global__ void MedianFilter_gpu(float *Device_ImageData, int Image_Width, int Image_Height)
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
				surround[tid][iterator] = *(Device_ImageData + (c*Image_Width) + r);
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
		*(Device_ImageData + (y*Image_Width) + x) = surround[tid][Half_Of_MEDIAN_LENGTH - 1];

		__syncthreads();
	}
}
//--------------------------------------------------------------------------------------
void bmp_header(char *filename, int &nr, int &nc)
{
	FILE* fp = fopen(filename, "rb");

	if (fp != NULL)
	{
		unsigned char header_info[54];
		fread(header_info, sizeof(unsigned char), 54, fp); // read 54-byte header
		nr = *(int*)&header_info[18];
		nc = *(int*)&header_info[22];
	}
	fclose(fp);
}
//--------------------------------------------------------------------------------------
void read_bmp(char *filename, int nr, int nc, unsigned char *image)
{
	int i, j, linewidth;
	FILE *fp1;
	if ((fp1 = fopen(filename, "rb")) == NULL)
	{
		printf("Cannot open file...<%s>\n", filename);
		system("pause");
	}
	// Skip header 54(BMP Info)+256*4(Color palette)
	for (i = 0; i<1078; i++) fgetc(fp1);

	// Work up the line width
	linewidth = nc;
	if (linewidth & 0x0003)
	{
		linewidth |= 0x0003;
		linewidth++;
	}

	for (i = nr - 1; i >= 0; i--)
	{
		for (j = 0; j<nc; j++)
		{
			//image[i*nc+j] = fgetc(fp1);
			image[j*nr + i] = fgetc(fp1);	//rotate with org progm.
		}

		if (nc != linewidth)
		{
			for (j = 0; j<linewidth - nc; j++)
			{
				fgetc(fp1);
			}
		}
	}
	fclose(fp1);
}
//--------------------------------------------------------------------------------------
__global__ void circleImgGenerate(int *circleImg, int Nx, int Ny, int r, int c, int radius)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i<Nx && j<Ny)
	{
		if ((i - r)*(i - r) + (j - c)*(j - c) < radius*radius)
			circleImg[i*Ny + j] = 1;
		else
			circleImg[i*Ny + j] = 0;
	}
}
//--------------------------------------------------------------------------------------
void zeros_complex(complex<float> *buf, int size)
{
	for (int i = 0; i<size; i++)
		buf[i] = complex<float>(0.0, 0.0);
}
//--------------------------------------------------------------------------------------
void ones(double *A, int size)
{
	for (int i = 0; i<size; i++)
		A[i] = 1.0;
}
//--------------------------------------------------------------------------------------
void zeros(double *A, int size)
{
	for (int i = 0; i<size; i++)
		A[i] = 0.0;
}
//--------------------------------------------------------------------------------------
void ones_f(float *A, int size)
{
	for (int i = 0; i<size; i++)
		A[i] = 1.0;
}
//--------------------------------------------------------------------------------------
void zeros_f(float *A, int size)
{
	for (int i = 0; i<size; i++)
		A[i] = 0.0;
}
//--------------------------------------------------------------------------------------
float AngCal(cufftComplex *src, int sizeX, int sizeY, int frame, int totalFrame)
{
	int blocksInX = (sizeX + 32 - 1) / 32;
	int blocksInY = (sizeY + 32 - 1) / 32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);

	int blocksInX2 = (sizeX/2 + 32 - 1) / 32;
	int blocksInY2 = (sizeY/2 + 32 - 1) / 32;
	dim3 grid2(blocksInX2, blocksInY2);
	dim3 block2(32, 32);

	float angle;
	float fx, sin_theta;
	float mag = Mag;
	float fm = Nmed / wavelength * 1e-9;

	cufftComplex *cuFT_img;
	cudaMalloc((void **)&cuFT_img, sizeof(cufftComplex)*sizeX*sizeY);

	cudaMemcpy(cuFT_img, src, sizeof(cufftComplex)*sizeX*sizeY, cudaMemcpyHostToDevice);

	modifyAngCalArray << <grid, block >> >(cuFT_img, sizeX, sizeY);

	cuFFT2D(cuFT_img, sizeX, sizeY, -1);
	cuFFT2Dshift << <grid2, block2 >> >(cuFT_img, sizeX, sizeY);

	cufftComplex *FT_img = (cufftComplex *)malloc(sizeX*sizeY*sizeof(cufftComplex));
	cudaMemcpy(FT_img, cuFT_img, sizeof(cufftComplex)*sizeX*sizeY, cudaMemcpyDeviceToHost);
	cudaFree(cuFT_img);

	float *pFT = (float *)malloc(sizeX*sizeY*sizeof(float));
	for (int i = 0; i<sizeX*sizeY; i++)
	{
		pFT[i] = log10(sqrt(FT_img[i].x * FT_img[i].x + FT_img[i].y * FT_img[i].y));
	}
	exportRAW_F("angle.raw", pFT, sizeX*sizeY);
	free(FT_img);

	float *IMGFT = (float *)malloc(sizeX / 2 * sizeof(float));
	float maxValue = -99999;
	int sortOrder = 0;
	if (frame >= 1 && frame <= round_MS(totalFrame / 2))
	{
		for (int i = 0; i<rowSize / 2; i++)
		{
			IMGFT[i] = pFT[i + 597 * sizeX];
			if (IMGFT[i] > maxValue)
			{
				maxValue = IMGFT[i];
				sortOrder = i;
			}
		}
		fx = (511 - sortOrder) * 1 / (1024 * dx);

		sin_theta = mag*fx / fm;
		angle = fabs(asin(sin_theta) * 180 / M_PI);
	}
	else if (frame >= round_MS(totalFrame / 2) + 1 && frame <= round_MS(totalFrame))
	{
		for (int i = sizeX / 2; i<sizeY; i++)
		{
			IMGFT[i - sizeX / 2] = pFT[i + 597 * sizeX];
			if (IMGFT[i - sizeX / 2] > maxValue)
			{
				maxValue = IMGFT[i - sizeX / 2];
				sortOrder = i - sizeX / 2;
			}
		}
		fx = abs((sortOrder + 1) * 1 / (1024 * dx* 1e-6));

		sin_theta = mag*fx / fm;
		angle = -fabs(asin(sin_theta) * 180 / M_PI);
	}

	cout << "order2: " << sortOrder << endl;
	cout << "sin(t): " << sin_theta << endl;
	cout << "fm: " << fm << endl;
	cout << "fx: " << fx << endl;
	cout << "dx: " << dx << endl;
	cout << "angle: " << angle << endl;

	free(pFT);
	free(IMGFT);

	return angle;
}
//--------------------------------------------------------------------------------------
float AngCal2(float *src, int sizeX, int sizeY, int frame, int totalFrame)
{
	float angle;
	float fx, sin_theta;
	float mag = Mag;
	float fm = Nmed / (wavelength * 1e-9);	//Nmed/Wavelength
	float tempMax = FLT_MIN;
	int posX = 0, posY = 0;

	complex<float> *FT_img = (complex<float> *)malloc(sizeX*sizeY*sizeof(complex<float>));
	float *pFT = (float *)malloc(sizeX*sizeY*sizeof(float));

	for (int j = 0; j<sizeY; j++)
	for (int i = 0; i<sizeX; i++)
	{
		//if(src[i+j*sizeX]>120)
		//src[i+j*sizeX] = 255;
		FT_img[i + j*sizeX] = complex<float>(src[i + j*sizeX], 0);
	}

	FFT2D(FT_img, sizeX, sizeY, -1);
	FFT2Dshift(FT_img, sizeX, sizeY);

	for (int j = 0; j<sizeY; j++)
	for (int i = 0; i<sizeX; i++)
	{
		pFT[i + j*sizeX] = log10(abs(FT_img[i + j*sizeX]));

		if (pFT[i + j*sizeX]>tempMax && j >520 && j<750 && i>200 && i<1000)
		{
			tempMax = pFT[i + j*sizeX];
			posY = j;
			posX = i;
		}
	}

	fx = (511 - posX) * 1 / (1024 * dx* 1e-6);
	sin_theta = mag*fx / fm;

	if (sin_theta<0)
	{
		angle = fabs(asin(sin_theta));
	}
	else
	{
		angle = -fabs(asin(sin_theta));
	}


	free(FT_img);
	free(pFT);
	
	return angle;
}
//--------------------------------------------------------------------------------------
float AngCal_GPU(float *src, int sizeX, int sizeY, int frame, int totalFrame)
{
	float angle;
	float fx, sin_theta;
	float mag = Mag;
	float fm = Nmed / (wavelength * 1e-9);	//Nmed/Wavelength
	int posX = 0, posY = 0;

	cufftReal *src_img;
	cudaMalloc((void **)&src_img, sizeof(cufftReal)*sizeX*sizeY);
	cudaMemcpy(src_img, src, sizeof(cufftReal)*sizeX*sizeY, cudaMemcpyHostToDevice);
	
	cufftComplex *FT_img;
	cudaMalloc((void **)&FT_img, sizeof(cufftComplex)*sizeX*sizeY);

	
	dim3 dimGrid(sizeX / TILE_DIM, sizeY / TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
	real2cufft << <dimGrid, dimBlock >> >(FT_img, src_img);

	int blocksInX = (sizeX + 32 - 1) / 32;
	int blocksInY = (sizeY + 32 - 1) / 32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);

	int blocksInX2 = (sizeX / 2 + 32 - 1) / 32;
	int blocksInY2 = (sizeY / 2 + 32 - 1) / 32;
	dim3 grid2(blocksInX2, blocksInY2);
	dim3 block2(32, 32);
	
	cuFFT2D(FT_img, sizeX, sizeY, -1);
	cuFFT2Dshift << <grid2, block2 >> >(FT_img, sizeX, sizeY);
	cufftComplex2Real<<<grid,block>>>(src_img, FT_img, sizeX, sizeY);
	//DeviceMemOut("angleFFT.raw", src_img, sizeX, sizeY);
	//find out the maximum and its index	
	thrust::device_ptr<float> max_ptr = thrust::device_pointer_cast(src_img);
	thrust::device_ptr<float> result_offset = thrust::max_element(max_ptr + (int)(sizeX*sizeY*0.53), max_ptr + (int)(sizeX*sizeY*0.79));
	posX = (&result_offset[0] - &max_ptr[0]) % sizeX;
	posY = (&result_offset[0] - &max_ptr[0]) / sizeX;

	
	fx = (511 - posX) * 1 / (1024 * dx* 1e-6);
	sin_theta = mag*fx / fm;

	if (sin_theta<0)
		angle = fabs(asin(sin_theta));
	else
		angle = -fabs(asin(sin_theta));

	/*cout << posX << endl;
	cout << posY << endl;
	cout << mag << endl;
	cout << wavelength << endl;
	cout << Nmed << endl;
	cout << dx << endl;
	cout << angle << endl;
	system("pause"); */
	cudaFree(src_img);
	cudaFree(FT_img);
	//thrust::device_delete(result_offset,1);
	//thrust::device_delete(max_ptr,sizeof(src_img);
	//thrust::device_free(result_offset);
	return angle;
}
//--------------------------------------------------------------------------------------
__global__ void cufftComplex2Real(float *d, cufftComplex *s, int sizeX, int sizeY)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	if (xIndex < sizeX && yIndex < sizeY)
	{
		unsigned int idx = xIndex + sizeX*yIndex;
		d[idx] = log10(sqrt(s[idx].x*s[idx].x + s[idx].y*s[idx].y));
		if (isnan(d[idx]) == true || isinf(d[idx]) == true) d[idx] = 0;
	}
}
//--------------------------------------------------------------------------------------
void AngCal3(float *src, int sizeX, int sizeY, int frame, int totalFrame, double &AngX, double &AngY)
{
	//float angle;
	float fx, fy, sin_theta_x, sin_theta_y;
	float mag = Mag;
	float fm = Nmed / (wavelength * 1e-9);	//Nmed/Wavelength
	float tempMax = FLT_MIN;
	int posX = 0, posY = 0;

	complex<float> *FT_img = (complex<float> *)malloc(sizeX*sizeY*sizeof(complex<float>));
	float *pFT = (float *)malloc(sizeX*sizeY*sizeof(float));

	for (int j = 0; j<sizeY; j++)
	for (int i = 0; i<sizeX; i++)
	{
		//if(src[i+j*sizeX]>120)
		//src[i+j*sizeX] = 255;
		FT_img[i + j*sizeX] = complex<float>(src[i + j*sizeX], 0);
	}

	FFT2D(FT_img, sizeX, sizeY, -1);
	FFT2Dshift(FT_img, sizeX, sizeY);

	for (int j = 0; j<sizeY; j++)
	for (int i = 0; i<sizeX; i++)
	{
		pFT[i + j*sizeX] = log10(abs(FT_img[i + j*sizeX]));

		if (pFT[i + j*sizeX]>tempMax && j >520 && j<750 && i>200 && i<1000)
		{
			tempMax = pFT[i + j*sizeX];
			posY = j;
			posX = i;
		}
	}

	fx = (511 - posX) / (1024 * dx* 1e-6);
	fy = (591 - posY) / (1024 * dx* 1e-6);
	sin_theta_x = mag*fx / fm;
	sin_theta_y = mag*fy / fm;

	if (sin_theta_x<0)
	{
		AngX = fabs(asin(sin_theta_x));
	}
	else
	{
		AngX = -fabs(asin(sin_theta_x));
	}
	AngY = fabs(asin(sin_theta_y));


	free(FT_img);
	free(pFT);


}
//--------------------------------------------------------------------------------------
bool checkArray(float *array2D, float limitSTD, float limitRange, int size)
{
	double mean = 0;
	float max = -99999.0;
	float min = 99999.0;

	for (int i = 0; i < size; i++)
	{
		mean += array2D[i];
		if (array2D[i] > max)
			max = array2D[i];
		if (array2D[i] < min)
			min = array2D[i];
	}
	mean /= size;

	double var = 0;
	double std_dev = 0;

	for (int i = 0; i < size; i++)
	{
		var += ((array2D[i] - mean) * (array2D[i] - mean));
	}

	var /= size;
	std_dev = sqrt(var);

	//cout << "STD:" << std_dev << " ,RAG: " << fabs(max - min) << endl;
	if (std_dev > limitSTD || (float)(fabs(max - min)) > limitRange)
	{
		return true;
	}
	else
	{
		return false;
	}
}
//--------------------------------------------------------------------------------------
void phaseCalibration(float *Phase2D, int sizeX, int sizeY)
{
	/* parameter vector */
	int n_par = 2;                /* number of parameters in model function f */
	double par_row[2] = { 0, 0 };
	double par_col[2] = { 0, 0 };

	int point_size = 3;
	int pRow[6] = { 40, 70, 100, 412, 442, 472 };
	int pCol[6] = { 40, 70, 100, 412, 442, 472 };

	// Row
	for (int z = 0; z<point_size; z++)
	{
		/* arbitrary starting value */
		double par[2] = { 1, 1 };
		/* data points */
		int m_dat = sizeX;
		double *Z = (double *)malloc(m_dat*sizeof(double));
		double *posX = (double *)malloc(m_dat*sizeof(double));

#pragma omp parallel for
		for (int i = 0; i<rowSize; i++)
		{
			Z[i] = (double)Phase2D[pRow[z] * sizeY + i];
			posX[i] = (double)i;
		}
#pragma omp barrier

		data_struct data = { posX, Z, f };

		/* auxiliary parameters */
		lm_status_struct status;
		lm_control_struct control = lm_control_double;

		lmmin(n_par, par, m_dat, (const void*)&data, evaluate_surface, &control, &status);

		for (int k = 0; k<n_par; k++)
		{
			par_row[k] += par[k];
		}

		free(Z);
		free(posX);
	}
	for (int k = 0; k<n_par; k++)
	{
		par_row[k] /= point_size;
	}

#pragma omp parallel for
	for (int i = 0; i<sizeX; i++)
	{
		for (int j = 0; j<sizeX; j++)
		{
			Phase2D[j*sizeX + i] -= (float)(par_row[0] + par_row[1] * (i));
		}
	}
#pragma omp barrier

	///////////////////////////////////////////////////////////////////////////////////
	// Col
	for (int z = 0; z<point_size; z++)
	{
		/* arbitrary starting value */
		double par[2] = { 1, 1 };
		/* data points */
		int m_dat = sizeY;
		double *Z = (double *)malloc(m_dat*sizeof(double));
		double *posX = (double *)malloc(m_dat*sizeof(double));

		int j = 0;

#pragma omp parallel for
		for (int j = 0; j<colSize; j++)
		{
			Z[j] = (double)Phase2D[j*sizeY + pCol[z]];
			posX[j] = (double)j;
		}
#pragma omp barrier

		data_struct data = { posX, Z, f };

		/* auxiliary parameters */
		lm_status_struct status;
		lm_control_struct control = lm_control_double;
		control.verbosity = 10;

		lmmin(n_par, par, m_dat, (const void*)&data, evaluate_surface, &control, &status);

		for (int k = 0; k<n_par; k++)
		{
			par_col[k] += par[k];
		}

		free(Z);
		free(posX);
	}
	for (int k = 0; k<n_par; k++)
	{
		par_col[k] /= point_size;
	}

#pragma omp parallel for
	for (int j = 0; j<sizeY; j++)
	{
		for (int i = 0; i<sizeY; i++)
		{
			Phase2D[j*sizeX + i] -= (float)(par_col[0] + par_col[1] * (j));
		}
	}
#pragma omp barrier

}
//--------------------------------------------------------------------------------------
void ampCalibration(float *Amp2D, int sizeX, int sizeY)
{
	/* parameter vector */
	int n_par = 2;                /* number of parameters in model function f */
	double par_row[2] = { 0, 0 };
	double par_col[2] = { 0, 0 };

	int point_size = 3;
	int pRow[6] = { 40, 70, 100, 412, 442, 472 };
	int pCol[6] = { 40, 70, 100, 412, 442, 472 };

	// Row
	for (int z = 0; z<point_size; z++)
	{
		/* arbitrary starting value */
		double par[2] = { 1, 1 };
		/* data points */
		int m_dat = sizeX;
		double *Z = (double *)malloc(m_dat*sizeof(double));
		double *posX = (double *)malloc(m_dat*sizeof(double));

#pragma omp parallel for
		for (int i = 0; i<sizeX; i++)
		{
			Z[i] = (double)Amp2D[pRow[z] * sizeY + i];
			posX[i] = (double)i;
		}
#pragma omp barrier

		data_struct data = { posX, Z, f };

		/* auxiliary parameters */
		lm_status_struct status;
		lm_control_struct control = lm_control_double;

		lmmin(n_par, par, m_dat, (const void*)&data, evaluate_surface, &control, &status);

		for (int k = 0; k<n_par; k++)
		{
			par_row[k] += par[k];
		}

		free(Z);
		free(posX);
	}
	for (int k = 0; k<n_par; k++)
	{
		par_row[k] /= point_size;
	}

#pragma omp parallel for
	for (int i = 0; i<sizeX; i++)
	{
		for (int j = 0; j<sizeY; j++)
		{
			Amp2D[j*sizeX + i] -= (float)(par_row[0] + par_row[1] * (i));
		}
	}
#pragma omp barrier

	///////////////////////////////////////////////////////////////////////////////////
	// Col
	for (int z = 0; z<point_size; z++)
	{
		/* arbitrary starting value */
		double par[2] = { 1, 1 };
		/* data points */
		int m_dat = sizeY;
		double *Z = (double *)malloc(m_dat*sizeof(double));
		double *posX = (double *)malloc(m_dat*sizeof(double));

#pragma omp parallel for
		for (int j = 0; j<sizeY; j++)
		{
			Z[j] = (double)Amp2D[j*colSize + pCol[z]];
			posX[j] = (double)j;
		}
#pragma omp barrier

		data_struct data = { posX, Z, f };

		/* auxiliary parameters */
		lm_status_struct status;
		lm_control_struct control = lm_control_double;

		lmmin(n_par, par, m_dat, (const void*)&data, evaluate_surface, &control, &status);

		for (int k = 0; k<n_par; k++)
		{
			par_col[k] += par[k];
		}

		free(Z);
		free(posX);
	}
	for (int k = 0; k<n_par; k++)
	{
		par_col[k] /= point_size;
	}

#pragma omp parallel for
	for (int j = 0; j<sizeY; j++)
	{
		for (int i = 0; i<sizeX; i++)
		{
			Amp2D[j*sizeX + i] -= (float)(par_col[0] + par_col[1] * (j));
		}
	}
#pragma omp barrier

#pragma omp parallel for
	for (int i = 0; i<sizeX*sizeY; i++)
	{
		Amp2D[i] += 1.0;
	}
#pragma omp barrier
}
//--------------------------------------------------------------------------------------
void PrintProcess(int counter, char *SPDir, char *BGDir, char *AngDir)
{
	end_time = clock();
	float total_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;

	cudaMemGetInfo(&freeDeviceMemory, &totalDeviceMemory);
	size_t usedMem = totalDeviceMemory - freeDeviceMemory;

	system("cls");
	printf("*****************************************************************\n"
		"*   Hilbert Transform with FFT-based unwrapping                 *\n"
		"*   3D-RI Tomographic Diffractive Reconstruction                *\n"
		"*                               (GPU Verson. CUDA Toolkit 10.0) *\n"
		"*                                                               *\n"
		"*                                              Yang-Hsien Lin   *\n"
		"*                                                     06.2016   *\n"
		"*****************************************************************\n\n"
		"CPU Cores and Threads  : %d; %d \n"
		"GPU Memory Usage       : %d / %d (MB)\n"
		"Progression            : %2d (%%) <--> (%d/%d)\n"
		"Sample Folder          : %s \n"
		"Background Folder      : %s \n"
		"Angle Folder           : %s \n"		
		"Reconstruction         : %s \n"
		"Rescale Phase Images   : %s \n"
		"Extraction Time        : %f sec (%.2f FPS)\n"
		"Unwrapping Time        : %f sec (%.2f FPS)\n"
		"Total Time             : %f sec \n"
		, omp_get_num_procs(), omp_get_num_threads()
		, usedMem / 1024 / 1024, totalDeviceMemory / 1024 / 1024
		, counter * 100 / totalFrame, counter, totalFrame
		, SPDir, BGDir, AngDir		
		, ReconFlag ? "Yes" : "No"
		, ResizeFlag ? "Yes" : "No"
		, wrap_time / CLOCKS_PER_SEC, 1/((wrap_time / CLOCKS_PER_SEC)/AccumFrame)
		, unwrap_time / CLOCKS_PER_SEC, 1/(((unwrap_time+ wrap_time) / CLOCKS_PER_SEC) / AccumFrame)
		, total_time);

}
//--------------------------------------------------------------------------------------
void printDevProp(cudaDeviceProp devProp)
{
	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Name:                          %s\n", devProp.name);
	printf("Total global memory:           %lu\n", devProp.totalGlobalMem);
	printf("Total shared memory per block: %lu\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %lu\n", devProp.memPitch);
	printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
	printf("Maximum dimension (x,y,z) of block:(%d,%d,%d)\n", devProp.maxThreadsDim[0]
		, devProp.maxThreadsDim[1]
		, devProp.maxThreadsDim[2]);
	printf("Maximum dimension (x,y,z) of grid :(%4d,%4d,%4d)\n", devProp.maxGridSize[0]
		, devProp.maxGridSize[1]
		, devProp.maxGridSize[2]);
	printf("Clock rate:                    %d\n", devProp.clockRate);
	printf("Total constant memory:         %lu\n", devProp.totalConstMem);
	printf("Texture alignment:             %lu\n", devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	//return;
}
//--------------------------------------------------------------------------------------
/* fit model: a plane p0 + p1*tx + p2*tz - y*/
double f(double x, double y, const double *p)
{
	return p[0] + p[1] * x - y;
}
//--------------------------------------------------------------------------------------
/* fit model: a plane p0 + p1*tx + p2*tz - y*/
double f1(double tx, double tz, double y, const double *p)
{
	return p[0] + p[1] * tx + p[2] * tz - y;
}
//--------------------------------------------------------------------------------------
/* function evaluation, determination of residues */
void evaluate_surface(const double *par, int m_dat, const void *data,
	double *fvec, int *info)
{
	/* for readability, explicit type conversion */
	data_struct *D;
	D = (data_struct*)data;

	int i;
	for (i = 0; i < m_dat; i++)
		fvec[i] = D->y[i] - D->f(D->tx[i], D->y[i], par);
}
//--------------------------------------------------------------------------------------
void obtainRadius(cufftComplex *input, int &radius, int &r, int &c, int Nx, int Ny)
{
	cufftComplex *input_temp = (cufftComplex *)malloc(Nx*Ny*sizeof(cufftComplex));
	cudaMemcpy(input_temp, input, sizeof(cufftComplex)*Nx*Ny, cudaMemcpyDeviceToHost);

	//copy the memory from device to host and generate the log10-abs-map
	float *temp = (float *)malloc(Nx*Ny*sizeof(float));
	for (int i = 0; i<Nx*Ny; i++)
	{
		temp[i] = log10(sqrt(input_temp[i].x*input_temp[i].x + input_temp[i].y*input_temp[i].y));
	}
	free(input_temp);

	float tempMax = -99999;
	int startY = 600, endY = Ny;
	int startX = 0;
	for (int j = startY; j<endY; j++)
	for (int i = startX; i<Nx; i++)
	{
		if (temp[i + j*Nx]>tempMax)
		{
			tempMax = temp[i + j*Nx];
			r = j;
			c = i;
			radius = abs(r - (Ny/2)) / 3;
		}
	}
	//cout<<"radius:"<<radius<<endl;

	free(temp);
}
//--------------------------------------------------------------------------------------
void exportRAW_F(char * fpath, float* buf, int size)
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
//--------------------------------------------------------------------------------------
void exportComplex(char * fpath, cufftComplex* buf, int size)
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
		fwrite(buf, size, sizeof(cufftComplex), fp);
	}
	fclose(fp);
}
//--------------------------------------------------------------------------------------
__global__ void zeros_cu_int(int *input, int sizeX, int sizeY)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int i = xIndex + sizeX * yIndex;

	if ((xIndex<sizeX) && (yIndex<sizeY))
	{
		input[i] = 0;
	}
}
//--------------------------------------------------------------------------------------
__global__ void zeros_cu_float(float *input, int sizeX, int sizeY)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int i = xIndex + sizeX * yIndex;

	if ((xIndex<sizeX) && (yIndex<sizeY))
	{
		input[i] = 0.0;
	}
}
//--------------------------------------------------------------------------------------
__global__ void ones_cu_float(float *input, int sizeX, int sizeY)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int i = xIndex + sizeX * yIndex;

	if ((xIndex<sizeX) && (yIndex<sizeY))
	{
		input[i] = 1.0;
	}
}
//--------------------------------------------------------------------------------------
__global__ void zeros_cufft(cufftComplex *input, int sizeX, int sizeY)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int i = xIndex + sizeX * yIndex;

	if ((xIndex<sizeX) && (yIndex<sizeY))
	{
		input[i].x = 0.0;
		input[i].y = 0.0;
	}
}
//--------------------------------------------------------------------------------------
__global__ void copy_cuFFT(cufftComplex *input, cufftComplex *output, int *checkArray, int sizeX, int sizeY)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	if ((xIndex<sizeX) && (yIndex<sizeY))
	{
		unsigned int i = xIndex + sizeX * yIndex;

		if (checkArray[i] == 1)
		{
			output[i] = input[i];
		}
	}
}
//--------------------------------------------------------------------------------------
__global__ void get1stOrder(cufftComplex *out, cufftComplex *in, int radius, int r, int c, int sizeX, int sizeY)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	if ((xIndex >= sizeX / 2 - (radius + 3)) && (xIndex <= sizeX / 2 + (radius + 3)) &&
		(yIndex >= sizeY / 2 - (radius + 3)) && (yIndex <= sizeY / 2 + (radius + 3)))
	{
		out[xIndex + yIndex*sizeX] = in[(xIndex + (c - sizeX / 2)) + (yIndex + (r - sizeY / 2))*sizeX];
	}
}
//--------------------------------------------------------------------------------------
__global__ void get1stOrder_new(cufftComplex *out, cufftComplex *in, int radius, int r, int c, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int newX = sizeX / 4;
	unsigned int newY = sizeY / 4;
	unsigned int idx_i, idx_j;

	if ((i<newX) && (j<newY))
	{
		idx_i = i + (c - newX / 2);
		idx_j = j + (r - newY / 2);
		if ((idx_i >= c - (radius )) && (idx_i <= c + (radius )) && (idx_j >= r - (radius )) && (idx_j <= r + (radius )))
			out[i + j*newX] = in[idx_i + idx_j*sizeX];
		else
		{
			out[i + j*newX].x = 0;
			out[i + j*newX].y = 0;
		}
	}
}
//--------------------------------------------------------------------------------------
__global__ void estimateWrapPhase(float *SPWrap, float *BGWrap, cufftComplex *SP, cufftComplex *BG, int sizeX, int sizeY)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int i;

	if ((xIndex<sizeX) && (yIndex<sizeY))
	{
		i = xIndex + sizeX * yIndex;
		//SP[i].y = SP[i].y*(-1);
		//BG[i].y = BG[i].y*(-1);
		SPWrap[i] = atan2(SP[i].y, SP[i].x);
		BGWrap[i] = atan2(BG[i].y, BG[i].x);
	}
}
//--------------------------------------------------------------------------------------
__global__ void estimatePhase(float *Phase, float *UnSPWrap, float *UnBGWrap, int sizeX, int sizeY)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;
	if ((xIndex<sizeX) && (yIndex<sizeY))
	{
		idx = xIndex + sizeX * yIndex;
		Phase[idx] = (float)(UnBGWrap[idx] - UnSPWrap[idx]);

		if (isnan(Phase[idx]) || isinf(Phase[idx]))
			Phase[idx] = 0;
	}
}
__global__ void calcWrapPhase(float *Phase, float *Amp, cufftComplex *SP, cufftComplex *BG, int sizeX, int sizeY)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;// = xIndex + sizeX * yIndex;

	//float SPWrap, BGWrap;
	if ((xIndex<sizeX) && (yIndex<sizeY))
	{
		idx = xIndex + sizeX * yIndex;
		//SPWrap = atan2(SP[idx].y, SP[idx].x);
		//BGWrap = atan2(BG[idx].y, BG[idx].x);
		Phase[idx] = (float)(atan2(BG[idx].y, BG[idx].x) - atan2(SP[idx].y, SP[idx].x));
		if (isnan(Phase[idx]) || isinf(Phase[idx]))	Phase[idx] = 0;

		Amp[idx] = sqrt(SP[idx].y * SP[idx].y + SP[idx].x * SP[idx].x)
			/ sqrt(BG[idx].y * BG[idx].y + BG[idx].x * BG[idx].x);

		if (isnan(Amp[idx]) == true || isinf(Amp[idx]) == true)	Amp[idx] = 0;

	}
}
//--------------------------------------------------------------------------------------
__global__ void estimateAmp(float *Amp, cufftComplex *SP, cufftComplex *BG, int sizeX, int sizeY)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	if ((xIndex<sizeX) && (yIndex<sizeY))
	{
		unsigned int i = xIndex + sizeX * yIndex;

		Amp[i] = sqrt(SP[i].y * SP[i].y + SP[i].x * SP[i].x)
			/ sqrt(BG[i].y * BG[i].y + BG[i].x * BG[i].x);

		if (isnan(Amp[i]) == true || isinf(Amp[i]) == true)
			Amp[i] = 0;
	}
}
//--------------------------------------------------------------------------------------
__global__ void bilinear_interpolation_kernel(float * __restrict__ d_result_1, const float * __restrict__ d_data_1,
	float * __restrict__ d_result_2, const float * __restrict__ d_data_2,
	const int M1, const int N1, const int M2, const int N2)
{
	const int i = threadIdx.x + blockDim.x * blockIdx.x;
	const int j = threadIdx.y + blockDim.y * blockIdx.y;

	const float x_ratio = ((float)(M1 - 1)) / M2;
	const float y_ratio = ((float)(N1 - 1)) / N2;

	if ((i<M2) && (j<N2))
	{
		float result_temp1_1, result_temp2_1;
		float result_temp1_2, result_temp2_2;

		const int    ind_x = (int)(x_ratio * i);
		const float  a = (x_ratio * i) - ind_x;

		const int    ind_y = (int)(y_ratio * j);
		const float  b = (y_ratio * j) - ind_y;

		float d00_1, d01_1, d10_1, d11_1;
		float d00_2, d01_2, d10_2, d11_2;
		if (((ind_x) < M1) && ((ind_y) < N1)) {
			d00_1 = d_data_1[ind_y    *M1 + ind_x];
			d00_2 = d_data_2[ind_y    *M1 + ind_x];
		}
		else {
			d00_1 = 0.f;
			d00_2 = 0.f;
		}
		if (((ind_x + 1) < M1) && ((ind_y) < N1)) {
			d10_1 = d_data_1[ind_y    *M1 + ind_x + 1];
			d10_2 = d_data_2[ind_y    *M1 + ind_x + 1];
		}
		else {
			d10_1 = 0.f;
		}
		if (((ind_x) < M1) && ((ind_y + 1) < N1)) {
			d01_1 = d_data_1[(ind_y + 1)*M1 + ind_x];
			d01_2 = d_data_2[(ind_y + 1)*M1 + ind_x];
		}
		else
		{
			d01_1 = 0.f;
			d01_2 = 0.f;
		}
		if (((ind_x + 1) < M1) && ((ind_y + 1) < N1))
		{
			d11_1 = d_data_1[(ind_y + 1)*M1 + ind_x + 1];
			d11_2 = d_data_2[(ind_y + 1)*M1 + ind_x + 1];
		}
		else {
			d11_1 = 0.f;
			d11_2 = 0.f;
		}

		result_temp1_1 = a * d10_1 + (-d00_1 * a + d00_1);
		result_temp2_1 = a * d11_1 + (-d01_1 * a + d01_1);
		d_result_1[i + M2*j] = b * result_temp2_1 + (-result_temp1_1 * b + result_temp1_1);
		
		result_temp1_2 = a * d10_2 + (-d00_2 * a + d00_2);
		result_temp2_2 = a * d11_2 + (-d01_2 * a + d01_2);
		d_result_2[i + M2*j] = b * result_temp2_2 + (-result_temp1_2 * b + result_temp1_2);
	}
}
//--------------------------------------------------------------------------------------
__global__ void modifyAngCalArray(cufftComplex *input, int sizeX, int sizeY)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx;

	if (i<sizeX && j<sizeY)
	{
		idx = i + sizeX*j;
		if (input[idx].x>120)
			input[idx].x = 255;
	}
}
//--------------------------------------------------------------------------------------
void DeviceMemOut(char *path, float *arr, int sizeX, int sizeY)
{
	int size = sizeX*sizeY;
	float *temp = (float *)malloc(size*sizeof(float));
	cudaMemcpy(temp, arr, size*sizeof(float), cudaMemcpyDeviceToHost);

	FILE *fp;
	fp = fopen(path, "wb");
	fwrite(temp, size, sizeof(float), fp);
	fclose(fp);
	free(temp);
}
//--------------------------------------------------------------------------------------
void DeviceMemOutFFT(char *path, cufftComplex *arr, int sizeX, int sizeY)
{
	int size = sizeX*sizeY;
	cufftComplex *temp = (cufftComplex *)malloc(size*sizeof(cufftComplex));
	float *temp2 = (float *)malloc(size*sizeof(float));
	cudaMemcpy(temp, arr, size*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++)
	{
		temp2[i] = log10(sqrt(temp[i].x*temp[i].x + temp[i].y*temp[i].y));
		if (is_nan(temp2[i]) == true || is_inf(temp2[i]) == true) temp2[i] = 0;
	}

	FILE *fp;
	fp = fopen(path, "wb");
	fwrite(temp2, size, sizeof(float), fp);
	fclose(fp);
	free(temp);
	free(temp2);
}