#include "cake.h"


int main( int argc, char** argv ) {

	sleep(5);

	struct timespec start, end;
	double diff_t;

	int M, K, N;
	int A_sz, B_sz, C_sz;	
	float *A, *B, *A_p, *B_p, *C;

	enum sched sch;
	cake_cntx_t* cake_cntx = cake_query_cntx();
	int p = cake_cntx->ncores; // max # cores
	int iters = 1;

	for(int i = 500; i <= 5000; i += 500) {

		M = i, K = i, N = i;
		printf("M = %d, K = %d, N = %d\n", M,K,N);

		A = (float*) malloc(M * K * sizeof( float ));
		B = (float*) malloc(K * N * sizeof( float ));
		C = (float*) calloc(M * N , sizeof( float ));

		// initialize A and B
		srand(time(NULL));
		rand_init(A, M, K);
		rand_init(B, K, N);

		blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));

		sch = KMN;

		init_block_dims(M, N, K, p, x, cake_cntx, sch);
	
		A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch);
		if(posix_memalign((void**) &A_p, 64, A_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}
		pack_A(A, A_p, M, K, p, x, cake_cntx, sch);

	    B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx);
		if(posix_memalign((void**) &B_p, 64, B_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}
		pack_B(B, B_p, K, N, p, x, cake_cntx, sch);

		free(A);
		free(B);

		clock_gettime(CLOCK_REALTIME, &start);

		for(int j = 0; j < iters; j++) {
			cake_sgemm(A_p, B_p, C, M, N, K, p, cake_cntx, 1,1,1,0,KMN);		
		}

		clock_gettime(CLOCK_REALTIME, &end);
		long seconds = end.tv_sec - start.tv_sec;
		long nanoseconds = end.tv_nsec - start.tv_nsec;
		diff_t = seconds + nanoseconds*1e-9;
		printf("sgemm time: %f \n", diff_t / iters); 

		free(A_p);
		free(B_p);
		free(C);

		sleep(2);
	}



    // char fname[50];
    // snprintf(fname, sizeof(fname), "results_sq");
    // FILE *fp;
    // fp = fopen(fname, "a");
    // fprintf(fp, "cake,%d,%d,%f\n",p,M,diff_t);
    // fclose(fp);


	return 0;
}