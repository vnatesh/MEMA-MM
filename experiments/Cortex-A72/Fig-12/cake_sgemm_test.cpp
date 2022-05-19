#include "cake.h"

 
int main( int argc, char** argv ) {

	struct timespec start, end;
	long nanoseconds, seconds;
	double diff_t = 0, ans = 0;

	if(argc < 2) {
		printf("Enter number of Cores\n");
		exit(1);
	}

	int M, K, N, p, write_result, iters;
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    p = atoi(argv[4]);
	write_result = atoi(argv[6]);

	printf("M = %d, K = %d, N = %d\n", M,K,N);

	cake_cntx_t* cake_cntx = cake_query_cntx();

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
	srand(time(NULL));
	rand_init(A, M, K);
	rand_init(B, K, N);

	clock_gettime(CLOCK_REALTIME, &start);

	// ans = cake_sgemm(A, B, C, M, N, K, p, cake_cntx, argv, 0,0,1,0,KMN);
	ans = cake_sgemm(A, B, C, M, N, K, p, cake_cntx, argv);

	clock_gettime(CLOCK_REALTIME, &end);
	seconds = end.tv_sec - start.tv_sec;
	nanoseconds = end.tv_nsec - start.tv_nsec;
	diff_t = seconds + nanoseconds*1e-9;
	printf("sgemm time: %f \n", diff_t); 

	free(A);
	free(B);
	free(C);


	if(write_result) {
	    char fname[50];
	    snprintf(fname, sizeof(fname), "results_dlmc");
	    FILE *fp;
	    fp = fopen(fname, "a");
	    fprintf(fp, "mema,%d,%d,%d,%f\n",M,K,N,ans);
	    fclose(fp);
	}


	return 0;
}


