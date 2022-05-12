#include "kernels.h"





void print_memory_info() {
   // allocate enough room for every thread's stack statistics
   int cnt = osThreadGetCount();
   mbed_stats_stack_t *stats = (mbed_stats_stack_t*) malloc(cnt * sizeof(mbed_stats_stack_t));

   char buffer[100];
   char buffer1[100];

   cnt = mbed_stats_stack_get_each(stats, cnt);
   for (int i = 0; i < cnt; i++) {
       sprintf(buffer, "Thread: 0x%lX, Stack size: %lu / %lu\r\n", stats[i].thread_id, stats[i].max_size, stats[i].reserved_size);
       Serial.print(buffer);
       Serial.println();

   }
   free(stats);


  arm_matrix_instance_f32 A;      /* Matrix A Instance */
  arm_matrix_instance_f32 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_f32 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_f32 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  unsigned long start1, end1, diff;
  float *A_f32, *B_f32, *C_f32, *C_f32_ref;
  uint32_t M = 8, N = 8, K = 8;

  A_f32 = (float *) malloc( M*K*sizeof( float ));
  B_f32 = (float *) malloc( K*N*sizeof( float ));
  C_f32_ref = (float *) malloc( M*N*sizeof( float ));

  char buf1[100];
  sprintf(buf1, "M = %d, K = %d, N = %d", M, K, N);
  Serial.println(buf1);

  // gettimeofday (&start, NULL);
  srand(time(NULL));
  rand_init(A_f32, M, K);
  rand_init(B_f32, K, N);


  arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);

  /* Initialise Matrix Instance B with numRows, numCols and data array(B_f32) */
  arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);


  /* Initialise C Matrix Instance with numRows, numCols and data array(C_f32) */
  arm_mat_init_f32(&C_ref, M, N, (float32_t *) C_f32_ref);


  start1 = micros();

  status = arm_mat_mult_f32(&A, &B, &C_ref);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm time: "); 
  Serial.println(diff); //prints time since program started

  // print_mat(&C_ref, M, N);

  // print_mat(&C, M, N);


//  for(int i = 0; i < M*N; i++) {
//      sprintf(buffer, "%f ", C.pData[i]);
//      Serial.print(buffer);
//   }
//  Serial.println("");
//  

  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  Serial.println();

  start1 = micros();

  status = inner_fp32_1x4x1(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm inner_fp32_1x4x1 time: "); 
  Serial.println(diff); //prints time since program started



  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = inner_fp32_1x16x1(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm inner_fp32_1x16x1 time: "); 
  Serial.println(diff); //prints time since program started

  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);

  // print_mat(&C, M, N);


  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = test1_arr(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm test1_arr time: "); 
  Serial.println(diff); //prints time since program started

  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = inner_fp32_2x4x2(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm inner_fp32_2x4x2 time: "); 
  Serial.println(diff); //prints time since program started

  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = inner_fp32_2x8x2(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm inner_fp32_2x8x2 time: "); 
  Serial.println(diff); //prints time since program started

  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
  // print_mat(&C, M, N);



  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
  
  start1 = micros();

  status = outer_fp32_2x2_packed(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_fp32_2x2_packed time: "); 
  Serial.println(diff); //prints time since program started



  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
  
  start1 = micros();

  status = outer_fp32_2x2(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_fp32_2x2 time: "); 
  Serial.println(diff); //prints time since program started




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
  
  start1 = micros();

  // status = outer_fp32_3x3(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_fp32_3x3 time: "); 
  Serial.println(diff); //prints time since program started




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
  
  start1 = micros();

  status = outer_fp32_4x4(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_fp32_4x4 time: "); 
  Serial.println(diff); //prints time since program started




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
  
  start1 = micros();

  status = outer_fp32_5x5_packed(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_fp32_5x5_packed time: "); 
  Serial.println(diff); //prints time since program started




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = outer_fp32_5x5_old(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_fp32_5x5_old time: "); 
  Serial.println(diff); //prints time since program started

  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
  // print_mat(&C, M, N);
  




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = outer_fp32_5x5(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_fp32_5x5 time: "); 
  Serial.println(diff); //prints time since program started

  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
  // print_mat(&C, M, N);

  
//
//
//  for(int i = 0; i < M*K; i++) {
//      sprintf(buffer1, "%f ", A.pData[i]);
//      Serial.print(buffer1);
//  }
//  Serial.println("");
//
//    for(int i = 0; i < K*N; i++) {
//      sprintf(buffer1, "%f ", B.pData[i]);
//      Serial.print(buffer1);
//  }
//  Serial.println("");
//  
//
//
//  for(int i = 0; i < M*N; i++) {
//      sprintf(buffer1, "%f ", C.pData[i]);
//      Serial.print(buffer1);
//  }
//  Serial.println("");
  

  Serial.println();
  
  // Grab the heap statistics
  mbed_stats_heap_t heap_stats;
  mbed_stats_heap_get(&heap_stats);
  sprintf(buffer, "Heap size: %lu / %lu bytes\r\n", heap_stats.current_size, heap_stats.reserved_size);
  Serial.println(buffer);
  
  free(A_f32);
  free(B_f32);
  free(C_f32);
  free(C_f32_ref);
}





void print_memory_info1() {
   // allocate enough room for every thread's stack statistics
   int cnt = osThreadGetCount();
   mbed_stats_stack_t *stats = (mbed_stats_stack_t*) malloc(cnt * sizeof(mbed_stats_stack_t));

   char buffer[100];
   char buffer1[100];

   cnt = mbed_stats_stack_get_each(stats, cnt);
   for (int i = 0; i < cnt; i++) {
       sprintf(buffer, "Thread: 0x%lX, Stack size: %lu / %lu\r\n", stats[i].thread_id, stats[i].max_size, stats[i].reserved_size);
       Serial.print(buffer);
       Serial.println();

   }
   free(stats);


  arm_matrix_instance_q15 A;      /* Matrix A Instance */
  arm_matrix_instance_q15 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_q15 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_q15 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  unsigned long start1, end1, diff;
  int16_t *A_16, *B_16, *C_16, *C_16_ref, *B_trans;
  uint32_t M = 56, N = 56, K = 56;

  A_16 = (int16_t *) malloc( M*K*sizeof( int16_t ));
  B_16 = (int16_t *) malloc( K*N*sizeof( int16_t ));
  C_16_ref = (int16_t *) malloc( M*N*sizeof( int16_t ));

  B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));

  printf("M = %d, K = %d, N = %d\n", M, K, N);

  // gettimeofday (&start, NULL);
  srand(time(NULL));
  rand_init_q15(A_16, M, K);
  rand_init_q15(B_16, K, N);

  arm_mat_init_q15(&A, M, K, (q15_t *) A_16);

  /* Initialise Matrix Instance B with numRows, numCols and data array(B_16) */
  arm_mat_init_q15(&B, K, N, (q15_t *) B_16);

  /* Initialise C Matrix Instance with numRows, numCols and data array(C_16) */
  arm_mat_init_q15(&C_ref, M, N, (q15_t *) C_16_ref);

  // print_mat_q15(&A, M, K);
  // print_mat_q15(&B, K, N);


  start1 = micros();

  status = arm_mat_mult_fast_q15(&A, &B, &C_ref, B_trans);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("arm_mat_mult_fast_q15 time: "); 
  Serial.println(diff); //prints time since program started

  // print_mat_q15(&C_ref, M, N);



  free(B_trans);

  C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
  arm_mat_init_q15(&C, M, N, (q15_t *) C_16);

  B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));
  
  Serial.println();

  start1 = micros();

  status = arm_q15_inner_2x4x2(&A, &B, &C, (q15_t *) B_trans);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("arm_q15_inner_2x4x2 time: "); 
  Serial.println(diff); //prints time since program started

  // print_mat_q15(&C, M, N);


  q15_gemm_checker(C_16, C_ref.pData, N, M, K);




  free(B_trans);
  free(C_16);

  C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
  arm_mat_init_q15(&C, M, N, (q15_t *) C_16);

  B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));
  
  Serial.println();

  start1 = micros();

  status = arm_q15_inner_2x2x2(&A, &B, &C, (q15_t *) B_trans);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("arm_q15_inner_2x2x2 time: "); 
  Serial.println(diff); //prints time since program started



  q15_gemm_checker(C_16, C_ref.pData, N, M, K);





  free(B_trans);
  free(C_16);
  
  C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
  arm_mat_init_q15(&C, M, N, (q15_t *) C_16);

  B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));
  
  Serial.println();

  start1 = micros();

  status = outer_q15_4x2(&A, &B, &C, (q15_t *) B_trans);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("outer_q15_4x2 time: "); 
  Serial.println(diff); //prints time since program started

  q15_gemm_checker(C_16, C_ref.pData, N, M, K);






  free(B_trans);
  free(C_16);
  
  C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
  arm_mat_init_q15(&C, M, N, (q15_t *) C_16);

  B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));
  
  Serial.println();

  start1 = micros();

  status = outer_q15_6x2(&A, &B, &C, (q15_t *) B_trans);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("outer_q15_6x2 time: "); 
  Serial.println(diff); //prints time since program started

  q15_gemm_checker(C_16, C_ref.pData, N, M, K);




  // Grab the heap statistics
  mbed_stats_heap_t heap_stats;
  mbed_stats_heap_get(&heap_stats);
  sprintf(buffer, "Heap size: %lu / %lu bytes\r\n", heap_stats.current_size, heap_stats.reserved_size);
  Serial.println(buffer);
  
  free(A_16);
  free(B_16);
  free(C_16);
  free(C_16_ref);
  free(B_trans);
}




void arm_vs_mema_q15() {

  arm_matrix_instance_q15 A;      /* Matrix A Instance */
  arm_matrix_instance_q15 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_q15 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_q15 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  unsigned long start1, end1, diff;
  int16_t *A_16, *B_16, *C_16, *C_16_ref, *B_trans;
  uint32_t M = 90, N = 90, K = 90;
  char buf[100];
  
  A_16 = (int16_t *) malloc( M*K*sizeof( int16_t ));
  B_16 = (int16_t *) malloc( K*N*sizeof( int16_t ));
  C_16_ref = (int16_t *) calloc( M*N,sizeof( int16_t ));
  B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));


  // gettimeofday (&start, NULL);
  srand(time(NULL));
  rand_init_q15(A_16, M, K);
  rand_init_q15(B_16, K, N);

  arm_mat_init_q15(&A, M, K, (q15_t *) A_16);
  arm_mat_init_q15(&B, K, N, (q15_t *) B_16);
  arm_mat_init_q15(&C_ref, M, N, (q15_t *) C_16_ref);

  start1 = micros();
  status = arm_mat_mult_fast_q15(&A, &B, &C_ref, B_trans);
  end1 = micros();
  diff = end1 - start1;
  Serial.print("arm_mat_mult_fast_q15 warmup time: "); 
  Serial.println(diff); //prints time since program started


  free(A_16);
  free(B_16);
  free(C_16_ref);
  free(B_trans);

  Serial.println("\nM,N,K,algo,time");

  for(uint32_t i = 8; i < 111; i+=8) {

    M = i;
    N = i;
    K = i;

    A_16 = (int16_t *) malloc( M*K*sizeof( int16_t ));
    B_16 = (int16_t *) malloc( K*N*sizeof( int16_t ));
    B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));
    C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
    
    srand(time(NULL));
    rand_init_q15(A_16, M, K);
    rand_init_q15(B_16, K, N);

    arm_mat_init_q15(&A, M, K, (q15_t *) A_16);
    arm_mat_init_q15(&B, K, N, (q15_t *) B_16);
    arm_mat_init_q15(&C, M, N, (q15_t *) C_16);



    start1 = micros();
    status = arm_mat_mult_fast_q15(&A, &B, &C, (q15_t *) B_trans);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,inner_2x2x2,%lu", i,i,i,diff);
    Serial.println(buf);



    free(B_trans);
    free(C_16);
    C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
    arm_mat_init_q15(&C, M, N, (q15_t *) C_16);
    B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));
    
    start1 = micros();
    status = arm_q15_inner_2x4x2(&A, &B, &C, (q15_t *) B_trans);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,inner_2x4x2,%lu", i,i,i,diff);
    Serial.println(buf);




    free(B_trans);
    free(C_16);
    C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
    arm_mat_init_q15(&C, M, N, (q15_t *) C_16);
    B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));
    
    start1 = micros();
    status = outer_q15_4x2(&A, &B, &C, (q15_t *) B_trans);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,outer_q15_4x2,%lu", i,i,i,diff);
    Serial.println(buf);

    free(A_16);
    free(B_16);
    free(C_16);
    free(B_trans);
  }
}


void arm_vs_mema_fp32() {


  arm_matrix_instance_f32 A;      /* Matrix A Instance */
  arm_matrix_instance_f32 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_f32 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_f32 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  unsigned long start1, end1, diff;
  float *A_f32, *B_f32, *C_f32, *C_f32_ref;
  uint32_t M = 100, N = 100, K = 100;
  char buf[100];

  A_f32 = (float *) malloc( M*K*sizeof( float ));
  B_f32 = (float *) malloc( K*N*sizeof( float ));
  C_f32_ref = (float *) calloc( M*N, sizeof( float ));

  // gettimeofday (&start, NULL);
  srand(time(NULL));
  rand_init(A_f32, M, K);
  rand_init(B_f32, K, N);

  arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);
  arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);
  arm_mat_init_f32(&C_ref, M, N, (float32_t *) C_f32_ref);


  start1 = micros();
  status = arm_mat_mult_f32(&A, &B, &C_ref);
  end1 = micros();
  diff = end1 - start1;
  Serial.print("warmup sgemm time: "); 
  Serial.println(diff); //prints time since program started

  free(A_f32);
  free(B_f32);
  free(C_f32_ref);

  
  Serial.println("\nM,N,K,algo,time");

  for(int i = 5; i < 111; i+=5) {

    A_f32 = (float *) malloc( i*i*sizeof( float ));
    B_f32 = (float *) malloc( i*i*sizeof( float ));
    C_f32 = (float *) calloc( i*i, sizeof( float ));

    srand(time(NULL));
    rand_init(A_f32, i,i);
    rand_init(B_f32, i,i);

    arm_mat_init_f32(&A, i,i, (float32_t *) A_f32);
    arm_mat_init_f32(&B, i,i, (float32_t *) B_f32);
    arm_mat_init_f32(&C, i, i, (float32_t *) C_f32);

    start1 = micros();
    status = inner_fp32_1x16x1(&A, &B, &C);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,inner_1x16x1,%lu", i,i,i,diff);
    Serial.println(buf);




    free(C_f32);
    C_f32 = (float *) calloc( i*i, sizeof( float ));
    arm_mat_init_f32(&C, i, i, (float32_t *) C_f32);

    start1 = micros();
    status = outer_fp32_5x5(&A, &B, &C);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,mema,%lu", i,i,i,diff);
    Serial.println(buf); //prints time since program started


  
    free(A_f32);
    free(B_f32);
    free(C_f32);
  }




  for(int i = 8; i < 111; i+=8) {

    A_f32 = (float *) malloc( i*i*sizeof( float ));
    B_f32 = (float *) malloc( i*i*sizeof( float ));
    C_f32 = (float *) calloc( i*i, sizeof( float ));

    srand(time(NULL));
    rand_init(A_f32, i,i);
    rand_init(B_f32, i,i);

    arm_mat_init_f32(&A, i,i, (float32_t *) A_f32);
    arm_mat_init_f32(&B, i,i, (float32_t *) B_f32);
    arm_mat_init_f32(&C, i, i, (float32_t *) C_f32);

    start1 = micros();
    status = inner_fp32_2x8x2(&A, &B, &C);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,inner_2x8x2,%lu", i,i,i,diff);
    Serial.println(buf); //prints time since program started

  
    free(A_f32);
    free(B_f32);
    free(C_f32);
  }
}







void power_inner_fp32() {


  arm_matrix_instance_f32 A;      /* Matrix A Instance */
  arm_matrix_instance_f32 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_f32 C;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  float *A_f32, *B_f32, *C_f32;
  
  for(uint32_t j = 10; j <= 50; j+=10) {

    uint32_t M = j, N = j, K = j;
    char buf[100];
  
    A_f32 = (float *) malloc( M*K*sizeof( float ));
    B_f32 = (float *) malloc( K*N*sizeof( float ));
    C_f32 = (float *) calloc( M*N, sizeof( float ));
  
    // gettimeofday (&start, NULL);
    srand(time(NULL));
    rand_init(A_f32, M, K);
    rand_init(B_f32, K, N);
  
    arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);
    arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);
    arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

    int iters = 1000;
    for(int i = 0; i < iters; i++) {
      status = inner_fp32_2x8x2(&A, &B, &C);
    }
  
    free(A_f32);
    free(B_f32);
    free(C_f32);

    delay(1000);
  }


 for(uint32_t j = 60; j <= 110; j+=10) {

    uint32_t M = j, N = j, K = j;
    char buf[100];
  
    A_f32 = (float *) malloc( M*K*sizeof( float ));
    B_f32 = (float *) malloc( K*N*sizeof( float ));
    C_f32 = (float *) calloc( M*N, sizeof( float ));
  
    // gettimeofday (&start, NULL);
    srand(time(NULL));
    rand_init(A_f32, M, K);
    rand_init(B_f32, K, N);
  
    arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);
    arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);
    arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

    int iters = 500;
    for(int i = 0; i < iters; i++) {
      status = inner_fp32_2x8x2(&A, &B, &C);
    }
  
    free(A_f32);
    free(B_f32);
    free(C_f32);

    delay(1000);
  }


  
}



void power_outer_fp32() {


  arm_matrix_instance_f32 A;      /* Matrix A Instance */
  arm_matrix_instance_f32 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_f32 C;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  float *A_f32, *B_f32, *C_f32;
  
  for(uint32_t j = 10; j <= 50; j+=10) {

    uint32_t M = j, N = j, K = j;
    char buf[100];
  
    A_f32 = (float *) malloc( M*K*sizeof( float ));
    B_f32 = (float *) malloc( K*N*sizeof( float ));
    C_f32 = (float *) calloc( M*N, sizeof( float ));
  
    // gettimeofday (&start, NULL);
    srand(time(NULL));
    rand_init(A_f32, M, K);
    rand_init(B_f32, K, N);
  
    arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);
    arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);
    arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

    int iters = 1000;
    for(int i = 0; i < iters; i++) {
      status = outer_fp32_5x5(&A, &B, &C);
    }
  
    free(A_f32);
    free(B_f32);
    free(C_f32);
    
    delay(1000);
  }


 for(uint32_t j = 60; j <= 110; j+=10) {

    uint32_t M = j, N = j, K = j;
    char buf[100];
  
    A_f32 = (float *) malloc( M*K*sizeof( float ));
    B_f32 = (float *) malloc( K*N*sizeof( float ));
    C_f32 = (float *) calloc( M*N, sizeof( float ));
  
    // gettimeofday (&start, NULL);
    srand(time(NULL));
    rand_init(A_f32, M, K);
    rand_init(B_f32, K, N);
  
    arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);
    arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);
    arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

    int iters = 500;
    for(int i = 0; i < iters; i++) {
      status = outer_fp32_5x5(&A, &B, &C);
    }
  
    free(A_f32);
    free(B_f32);
    free(C_f32);
    
    delay(1000);
  }

}






void power_inner_q15() {

  arm_matrix_instance_q15 A;      /* Matrix A Instance */
  arm_matrix_instance_q15 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_q15 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_q15 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  int16_t *A_16, *B_16, *C_16, *B_trans;

  for(uint32_t j = 8; j <= 56; j+=8) {

    uint32_t M = j, N = j, K = j;
    
    A_16 = (int16_t *) malloc( M*K*sizeof( int16_t ));
    B_16 = (int16_t *) malloc( K*N*sizeof( int16_t ));
    B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));
    C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
    
    srand(time(NULL));
    rand_init_q15(A_16, M, K);
    rand_init_q15(B_16, K, N);

    arm_mat_init_q15(&A, M, K, (q15_t *) A_16);
    arm_mat_init_q15(&B, K, N, (q15_t *) B_16);
    arm_mat_init_q15(&C, M, N, (q15_t *) C_16);

    int iters = 1000;
    for(int i = 0; i < iters; i++) {
//      status = arm_mat_mult_fast_q15(&A, &B, &C, (q15_t *) B_trans);
          status = arm_q15_inner_2x4x2(&A, &B, &C, (q15_t *) B_trans);
    }

//    status = arm_q15_inner_2x4x2(&A, &B, &C, (q15_t *) B_trans);

    free(A_16);
    free(B_16);
    free(C_16);
    free(B_trans);
    
    delay(1000);
  }


  for(uint32_t j = 64; j <= 111; j+=8) {

    uint32_t M = j, N = j, K = j;
    
    A_16 = (int16_t *) malloc( M*K*sizeof( int16_t ));
    B_16 = (int16_t *) malloc( K*N*sizeof( int16_t ));
    B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));
    C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
    
    srand(time(NULL));
    rand_init_q15(A_16, M, K);
    rand_init_q15(B_16, K, N);

    arm_mat_init_q15(&A, M, K, (q15_t *) A_16);
    arm_mat_init_q15(&B, K, N, (q15_t *) B_16);
    arm_mat_init_q15(&C, M, N, (q15_t *) C_16);

    int iters = 500;
    for(int i = 0; i < iters; i++) {
      status = arm_mat_mult_fast_q15(&A, &B, &C, (q15_t *) B_trans);
    }

//    status = arm_q15_inner_2x4x2(&A, &B, &C, (q15_t *) B_trans);

    free(A_16);
    free(B_16);
    free(C_16);
    free(B_trans);
    
    delay(1000);
  }
  
}




void power_outer_q15() {

  arm_matrix_instance_q15 A;      /* Matrix A Instance */
  arm_matrix_instance_q15 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_q15 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_q15 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  int16_t *A_16, *B_16, *C_16, *B_trans;

  for(uint32_t j = 8; j <= 56; j+=8) {

    uint32_t M = j, N = j, K = j;
    
    A_16 = (int16_t *) malloc( M*K*sizeof( int16_t ));
    B_16 = (int16_t *) malloc( K*N*sizeof( int16_t ));
    B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));
    C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
    
    srand(time(NULL));
    rand_init_q15(A_16, M, K);
    rand_init_q15(B_16, K, N);

    arm_mat_init_q15(&A, M, K, (q15_t *) A_16);
    arm_mat_init_q15(&B, K, N, (q15_t *) B_16);
    arm_mat_init_q15(&C, M, N, (q15_t *) C_16);

    int iters = 1000;
    for(int i = 0; i < iters; i++) {
      status = outer_q15_4x2(&A, &B, &C, (q15_t *) B_trans);
    }

//    status = arm_q15_inner_2x4x2(&A, &B, &C, (q15_t *) B_trans);

    free(A_16);
    free(B_16);
    free(C_16);
    free(B_trans);
    
    delay(1000);
  }


  for(uint32_t j = 64; j <= 111; j+=8) {

    uint32_t M = j, N = j, K = j;
    
    A_16 = (int16_t *) malloc( M*K*sizeof( int16_t ));
    B_16 = (int16_t *) malloc( K*N*sizeof( int16_t ));
    B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));
    C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
    
    srand(time(NULL));
    rand_init_q15(A_16, M, K);
    rand_init_q15(B_16, K, N);

    arm_mat_init_q15(&A, M, K, (q15_t *) A_16);
    arm_mat_init_q15(&B, K, N, (q15_t *) B_16);
    arm_mat_init_q15(&C, M, N, (q15_t *) C_16);

    int iters = 500;
    for(int i = 0; i < iters; i++) {
      status = outer_q15_4x2(&A, &B, &C, (q15_t *) B_trans);
    }

//    status = arm_q15_inner_2x4x2(&A, &B, &C, (q15_t *) B_trans);

    free(A_16);
    free(B_16);
    free(C_16);
    free(B_trans);
    
    delay(1000);
  }
  
}





void test() {

  arm_matrix_instance_f32 A;      /* Matrix A Instance */
  arm_matrix_instance_f32 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_f32 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_f32 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  unsigned long start1, end1, diff;
  float *A_f32, *B_f32, *C_f32, *C_f32_ref, *A_f32_p;
  uint32_t M = 85, N = 85, K = 85;

  A_f32 = (float *) malloc( M*K*sizeof( float ));
  B_f32 = (float *) malloc( K*N*sizeof( float ));
  C_f32_ref = (float *) calloc( M*N, sizeof( float ));

  char buf1[100];
  sprintf(buf1, "M = %d, K = %d, N = %d", M, K, N);
  Serial.println(buf1);

  // gettimeofday (&start, NULL);
  srand(time(NULL));
  rand_sparse(A_f32, M, K, 0.72);
  rand_init(B_f32, K, N);


  arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);

  /* Initialise Matrix Instance B with numRows, numCols and data array(B_f32) */
  arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);


  /* Initialise C Matrix Instance with numRows, numCols and data array(C_f32) */
  arm_mat_init_f32(&C_ref, M, N, (float32_t *) C_f32_ref);


  // print_mat(&A, M, K);

  start1 = micros();

  status = arm_mat_mult_f32(&A, &B, &C_ref);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("arm sgemm time: "); 
  Serial.println(diff); //prints time since program started


  sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));
  A_f32_p = (float *) calloc( M*K, sizeof( float ));
  pack_A_sp(A_f32, A_f32_p, sp_pack, M, K, K, 5);

  // Serial.print("packed A "); 
  // print_arr(sp_pack->A_sp_p, M*K);
  // Serial.print("nnz_outer "); 
  // print_arr_int(sp_pack->nnz_outer, (M*K) / 5);
  // Serial.print("k_inds "); 
  // print_arr_int(sp_pack->k_inds, (M*K) / 5);
  // Serial.print("loc_m "); 
  // print_arr_int(sp_pack->loc_m, (M*K));
  // Serial.print("nnz_outer_blk "); 
  // print_arr_int(sp_pack->nnz_outer_blk, M / 5);
  // Serial.print("k_cnt "); 
  // print_arr_int(sp_pack->k_cnt, M / 5);




  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = outer_fp32_5x5_sp(sp_pack, &B, &C, M, K, N);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sparse sgemm outer_fp32_5x5_sp time: "); 
  Serial.println(diff); //prints time since program started

  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
  // print_mat(&C, M, N);




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = outer_fp32_5x5_sp_test(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sparse sgemm outer_fp32_5x5_sp_test time: "); 
  Serial.println(diff); //prints time since program started

  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
  // print_mat(&C, M, N);




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = outer_fp32_5x5(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_fp32_5x5 time: "); 
  Serial.println(diff); //prints time since program started

  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
  // print_mat(&C, M, N);






 int cnt = osThreadGetCount();
 mbed_stats_stack_t *stats = (mbed_stats_stack_t*) malloc(cnt * sizeof(mbed_stats_stack_t));

 char buffer[100];

 cnt = mbed_stats_stack_get_each(stats, cnt);
 for (int i = 0; i < cnt; i++) {
     sprintf(buffer, "Thread: 0x%lX, Stack size: %lu / %lu\r\n", stats[i].thread_id, stats[i].max_size, stats[i].reserved_size);
     Serial.print(buffer);
     Serial.println();

 }
 free(stats);

  // Grab the heap statistics
  mbed_stats_heap_t heap_stats;
  mbed_stats_heap_get(&heap_stats);
  sprintf(buffer, "Heap size: %lu / %lu bytes\r\n", heap_stats.current_size, heap_stats.reserved_size);
  Serial.println(buffer);





  
  free(A_f32);
  free(B_f32);
  free(C_f32);
  free(C_f32_ref);
  free(sp_pack->A_sp_p);
  free(sp_pack->loc_m);
  free(sp_pack->nnz_outer);
  free(sp_pack->k_inds);
  free(sp_pack->nnz_outer_blk);
  free(sp_pack->k_cnt);
  free(sp_pack);

}







void arm_vs_mema_fp32_sp() {


  arm_matrix_instance_f32 A;      /* Matrix A Instance */
  arm_matrix_instance_f32 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_f32 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_f32 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  unsigned long start1, end1, diff;
  float *A_f32, *B_f32, *C_f32, *C_f32_ref;
  uint32_t M = 100, N = 100, K = 100;
  char buf[100];

  A_f32 = (float *) malloc( M*K*sizeof( float ));
  B_f32 = (float *) malloc( K*N*sizeof( float ));
  C_f32_ref = (float *) calloc( M*N, sizeof( float ));

  // gettimeofday (&start, NULL);
  srand(time(NULL));
  rand_init(A_f32, M, K);
  rand_init(B_f32, K, N);

  arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);
  arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);
  arm_mat_init_f32(&C_ref, M, N, (float32_t *) C_f32_ref);


  start1 = micros();
  status = arm_mat_mult_f32(&A, &B, &C_ref);
  end1 = micros();
  diff = end1 - start1;
  Serial.print("warmup sgemm time: "); 
  Serial.println(diff); //prints time since program started

  free(A_f32);
  free(B_f32);
  free(C_f32_ref);

  
  Serial.println("\nM,N,K,algo,time");

  for(int i = 5; i < 86; i += 5) {

    A_f32 = (float *) malloc( i*i*sizeof( float ));
    B_f32 = (float *) malloc( i*i*sizeof( float ));
    C_f32 = (float *) calloc( i*i, sizeof( float ));

    srand(time(NULL));
    rand_sparse(A_f32, i,i, 0.7);
    rand_init(B_f32, i,i);

    arm_mat_init_f32(&A, i,i, (float32_t *) A_f32);
    arm_mat_init_f32(&B, i,i, (float32_t *) B_f32);
    arm_mat_init_f32(&C, i, i, (float32_t *) C_f32);

    C_f32_ref = (float *) calloc( i*i, sizeof( float ));
    arm_mat_init_f32(&C_ref, i, i, (float32_t *) C_f32_ref);
    status = arm_mat_mult_f32(&A, &B, &C_ref);




    start1 = micros();
    status = inner_fp32_1x16x1(&A, &B, &C);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,inner_1x16x1,%lu", i,i,i,diff);
    Serial.println(buf);



    free(C_f32);
    C_f32 = (float *) calloc( i*i, sizeof( float ));
    arm_mat_init_f32(&C, i, i, (float32_t *) C_f32);

    start1 = micros();
    status = outer_fp32_5x5(&A, &B, &C);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,mema,%lu", i,i,i,diff);
    Serial.println(buf); //prints time since program started


    free(C_f32);
    C_f32 = (float *) calloc( i*i, sizeof( float ));
    arm_mat_init_f32(&C, i, i, (float32_t *) C_f32);

    start1 = micros();
    status = outer_fp32_5x5_sp_test(&A, &B, &C);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,mema_sp,%lu", i,i,i,diff);
    Serial.println(buf); //prints time since program started
    // f32_gemm_checker(C.pData, C_ref.pData, i, i, i);



    free(C_f32);
    C_f32 = (float *) calloc( i*i, sizeof( float ));
    arm_mat_init_f32(&C, i, i, (float32_t *) C_f32);

    sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));
    float* A_f32_p = (float *) calloc( i*i, sizeof( float ));
    pack_A_sp(A_f32, A_f32_p, sp_pack, i, i, i, 5);
    
    start1 = micros();
    status = outer_fp32_5x5_sp(sp_pack, &B, &C, i, i, i);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,mema_sp_packed,%lu", i,i,i,diff);
    Serial.println(buf); //prints time since program started

    
    free(A_f32);
    free(B_f32);
    free(C_f32);
    free(C_f32_ref);
    free(sp_pack->A_sp_p);
    free(sp_pack->loc_m);
    free(sp_pack->nnz_outer);
    free(sp_pack->k_inds);
    free(sp_pack->nnz_outer_blk);
    free(sp_pack->k_cnt);
    free(sp_pack);
  }





  Serial.println("\nsparsity,algo,time");

  int i = 80;
  for(int j = 50; j < 100; j+=5) {

    A_f32 = (float *) malloc( i*i*sizeof( float ));
    B_f32 = (float *) malloc( i*i*sizeof( float ));
    C_f32 = (float *) calloc( i*i, sizeof( float ));

    srand(time(NULL));
    rand_sparse(A_f32, 80, 80, ((float) j) / 100.0);
    rand_init(B_f32, i,i);

    arm_mat_init_f32(&A, i, i, (float32_t *) A_f32);
    arm_mat_init_f32(&B, i, i, (float32_t *) B_f32);
    arm_mat_init_f32(&C, i, i, (float32_t *) C_f32);



    free(C_f32);
    C_f32 = (float *) calloc( i*i, sizeof( float ));
    arm_mat_init_f32(&C, i, i, (float32_t *) C_f32);

    start1 = micros();
    status = outer_fp32_5x5(&A, &B, &C);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,mema,%lu", j, diff);
    Serial.println(buf); //prints time since program started



    free(C_f32);
    C_f32 = (float *) calloc( i*i, sizeof( float ));
    arm_mat_init_f32(&C, i, i, (float32_t *) C_f32);

    sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));
    float* A_f32_p = (float *) calloc( i*i, sizeof( float ));
    pack_A_sp(A_f32, A_f32_p, sp_pack, i, i, i, 5);
    
    start1 = micros();
    status = outer_fp32_5x5_sp(sp_pack, &B, &C, i, i, i);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,mema_sp_packed,%lu", j, diff);
    Serial.println(buf); //prints time since program started


    free(C_f32);
    C_f32 = (float *) calloc( i*i, sizeof( float ));
    arm_mat_init_f32(&C, i, i, (float32_t *) C_f32);

    start1 = micros();
    status = outer_fp32_5x5_sp_test(&A, &B, &C);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,mema_sp,%lu", j, diff);
    Serial.println(buf); //prints time since program started

    
    free(A_f32);
    free(B_f32);
    free(C_f32);
    free(sp_pack->A_sp_p);
    free(sp_pack->loc_m);
    free(sp_pack->nnz_outer);
    free(sp_pack->k_inds);
    free(sp_pack->nnz_outer_blk);
    free(sp_pack->k_cnt);
    free(sp_pack);
  }





  for(int i = 8; i < 111; i+=8) {

    A_f32 = (float *) malloc( i*i*sizeof( float ));
    B_f32 = (float *) malloc( i*i*sizeof( float ));
    C_f32 = (float *) calloc( i*i, sizeof( float ));

    srand(time(NULL));
    rand_init(A_f32, i,i);
    rand_init(B_f32, i,i);

    arm_mat_init_f32(&A, i,i, (float32_t *) A_f32);
    arm_mat_init_f32(&B, i,i, (float32_t *) B_f32);
    arm_mat_init_f32(&C, i, i, (float32_t *) C_f32);

    start1 = micros();
    status = inner_fp32_2x8x2(&A, &B, &C);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,inner_2x8x2,%lu", i,i,i,diff);
    Serial.println(buf); //prints time since program started

  
    free(A_f32);
    free(B_f32);
    free(C_f32);
  }
}







void arm_vs_mema_fp32_k() {


  arm_matrix_instance_f32 A;      /* Matrix A Instance */
  arm_matrix_instance_f32 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_f32 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_f32 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  unsigned long start1, end1, diff;
  float *A_f32, *B_f32, *C_f32, *C_f32_ref;
  uint32_t M = 100, N = 100, K = 100;
  char buf[100];

  A_f32 = (float *) malloc( M*K*sizeof( float ));
  B_f32 = (float *) malloc( K*N*sizeof( float ));
  C_f32_ref = (float *) calloc( M*N, sizeof( float ));

  // gettimeofday (&start, NULL);
  srand(time(NULL));
  rand_init(A_f32, M, K);
  rand_init(B_f32, K, N);

  arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);
  arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);
  arm_mat_init_f32(&C_ref, M, N, (float32_t *) C_f32_ref);


  start1 = micros();
  status = arm_mat_mult_f32(&A, &B, &C_ref);
  end1 = micros();
  diff = end1 - start1;
  Serial.print("warmup sgemm time: "); 
  Serial.println(diff); //prints time since program started

  free(A_f32);
  free(B_f32);
  free(C_f32_ref);

  
  
  Serial.println("\nM,N,K,algo,time");

  float* tmp_arr = (float *) malloc( 1000*sizeof( float ));
  for(int i = 0; i < 1000; i++) {
    tmp_arr[i] = (float) rand();
  }

  for(int t = 0; t < 4; t++) {

    for(int m = 20; m < 81; m += 20) {

      M = m;
      N = m;

      for(int i = 5; i < 166; i+=5) {

        A_f32 = (float *) malloc( M*i*sizeof( float ));
        B_f32 = (float *) malloc( i*N*sizeof( float ));
        C_f32 = (float *) calloc( M*N, sizeof( float ));

        srand(time(NULL));
        rand_init(A_f32, M,i);
        rand_init(B_f32, i,N);

        arm_mat_init_f32(&A, M,i, (float32_t *) A_f32);
        arm_mat_init_f32(&B, i,N, (float32_t *) B_f32);
        arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

        // start1 = micros();
        // status = inner_fp32_1x16x1_test(&A, &B, &C, tmp_arr);
        // end1 = micros();
        // diff = end1 - start1;
        // sprintf(buf, "%d,%d,%d,inner_1x16x1,%lu", M,N,i,diff);
        // Serial.println(buf);




        free(C_f32);
        C_f32 = (float *) calloc( M*N, sizeof( float ));
        arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);


        start1 = micros();
        status = outer_fp32_5x5_test(&A, &B, &C, tmp_arr, t);
        end1 = micros();
        diff = end1 - start1;
        sprintf(buf, "%d,%d,%d,%d,mema,%lu", M,N,i,t+1,diff);
        Serial.println(buf); //prints time since program started


      
        free(A_f32);
        free(B_f32);
        free(C_f32);
      }



      for(int i = 8; i < 169; i+=8) {

        A_f32 = (float *) malloc( M*i*sizeof( float ));
        B_f32 = (float *) malloc( i*N*sizeof( float ));
        C_f32 = (float *) calloc( M*N, sizeof( float ));

        srand(time(NULL));
        rand_init(A_f32, M,i);
        rand_init(B_f32, i,N);

        arm_mat_init_f32(&A, M,i, (float32_t *) A_f32);
        arm_mat_init_f32(&B, i,N, (float32_t *) B_f32);
        arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

        start1 = micros();
        status = inner_fp32_2x8x2_test(&A, &B, &C, tmp_arr, t);
        end1 = micros();
        diff = end1 - start1;
        sprintf(buf, "%d,%d,%d,%d,inner_2x8x2,%lu", M,N,i,t+1,diff);
        Serial.println(buf); //prints time since program started

      
        free(A_f32);
        free(B_f32);
        free(C_f32);
      }
    }
  }


  free(tmp_arr);
}







void arm_vs_mema_fp32_mk();

void arm_vs_mema_fp32_mk() {


  arm_matrix_instance_f32 A;      /* Matrix A Instance */
  arm_matrix_instance_f32 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_f32 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_f32 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  unsigned long start1, end1, diff;
  float *A_f32, *B_f32, *C_f32, *C_f32_ref;
  uint32_t M = 100, N = 100, K = 100;
  char buf[100];

  A_f32 = (float *) malloc( M*K*sizeof( float ));
  B_f32 = (float *) malloc( K*N*sizeof( float ));
  C_f32_ref = (float *) calloc( M*N, sizeof( float ));

  // gettimeofday (&start, NULL);
  srand(time(NULL));
  rand_init(A_f32, M, K);
  rand_init(B_f32, K, N);

  arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);
  arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);
  arm_mat_init_f32(&C_ref, M, N, (float32_t *) C_f32_ref);


  start1 = micros();
  status = arm_mat_mult_f32(&A, &B, &C_ref);
  end1 = micros();
  diff = end1 - start1;
  Serial.print("warmup sgemm time: "); 
  Serial.println(diff); //prints time since program started

  free(A_f32);
  free(B_f32);
  free(C_f32_ref);

  
  N = 60;
  K = 5;

  start1 = micros();

  if(((M >= N) && K <= ((2*M*5) / (M + 5)))) {

  } else if((N >= M) && (K <= ((2*N*5) / (N + 5)))) {

  }  else {

  }

  end1 = micros();
  diff = end1 - start1;
  sprintf(buf, "runtime,%lu",diff);
  Serial.println(buf); //prints time since program started


  Serial.println("\nM,N,K,algo,time");

  for(int m = 5; m < 300; m += 20) {

    A_f32 = (float *) malloc( m*K*sizeof( float ));
    B_f32 = (float *) malloc( K*N*sizeof( float ));
    C_f32 = (float *) calloc( m*N, sizeof( float ));

    srand(time(NULL));
    rand_init(A_f32, m,K);
    rand_init(B_f32, K,N);

    arm_mat_init_f32(&A, m,K, (float32_t *) A_f32);
    arm_mat_init_f32(&B, K,N, (float32_t *) B_f32);
    arm_mat_init_f32(&C, m, N, (float32_t *) C_f32);


    free(C_f32);
    C_f32 = (float *) calloc( m*N, sizeof( float ));
    arm_mat_init_f32(&C, m, N, (float32_t *) C_f32);


    start1 = micros();
    status = outer_fp32_5x5(&A, &B, &C);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,k first,%lu", m,N,K,diff);
    Serial.println(buf); //prints time since program started


    free(C_f32);
    C_f32 = (float *) calloc( m*N, sizeof( float ));
    arm_mat_init_f32(&C, m, N, (float32_t *) C_f32);


    start1 = micros();
    status = outer_fp32_5x5_m_first(&A, &B, &C);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,m first,%lu", m,N,K,diff);
    Serial.println(buf); //prints time since program started


  
    free(A_f32);
    free(B_f32);
    free(C_f32);
  }



  for(int m = 4; m < 300; m+=16) {

    A_f32 = (float *) malloc( m*K*sizeof( float ));
    B_f32 = (float *) malloc( K*N*sizeof( float ));
    C_f32 = (float *) calloc( m*N, sizeof( float ));

    srand(time(NULL));
    rand_init(A_f32, m,K);
    rand_init(B_f32, K,N);

    arm_mat_init_f32(&A, m,K, (float32_t *) A_f32);
    arm_mat_init_f32(&B, K,N, (float32_t *) B_f32);
    arm_mat_init_f32(&C, m, N, (float32_t *) C_f32);

    start1 = micros();
    status = inner_fp32_2x8x2(&A, &B, &C);
    end1 = micros();
    diff = end1 - start1;
    sprintf(buf, "%d,%d,%d,inner_2x8x2,%lu", m,N,K,diff);
    Serial.println(buf); //prints time since program started

  
    free(A_f32);
    free(B_f32);
    free(C_f32);
  }
  
}










void testing() {

  arm_matrix_instance_f32 A;      /* Matrix A Instance */
  arm_matrix_instance_f32 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_f32 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_f32 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  unsigned long start1, end1, diff;
  float *A_f32, *B_f32, *C_f32, *C_f32_ref;
  uint32_t M = 200, N = 70, K = 5;

  A_f32 = (float *) malloc( M*K*sizeof( float ));
  B_f32 = (float *) malloc( K*N*sizeof( float ));
  C_f32_ref = (float *) malloc( M*N*sizeof( float ));

  char buf1[100];
  sprintf(buf1, "M = %d, K = %d, N = %d", M, K, N);
  Serial.println(buf1);

  // gettimeofday (&start, NULL);
  srand(time(NULL));
  rand_init(A_f32, M, K);
  rand_init(B_f32, K, N);

  arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);
  arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);
  arm_mat_init_f32(&C_ref, M, N, (float32_t *) C_f32_ref);


  start1 = micros();

  status = arm_mat_mult_f32(&A, &B, &C_ref);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("arm sgemm time: "); 
  Serial.println(diff); //prints time since program started



  float* tmp_arr = (float *) malloc( 1000*sizeof( float ));
  for(int i = 0; i < 1000; i++) {
    tmp_arr[i] = (float) rand();
  }



  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = outer_fp32_5x5(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_fp32_5x5_k_first time: "); 
  Serial.println(diff); //prints time since program started
  // print_mat(&C, M, N);
  // print_mat(&C_ref, M, N);
  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);






  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = inner_fp32_2x8x2(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm inner_fp32_5x5 time: "); 
  Serial.println(diff); //prints time since program started
  // print_mat(&C, M, N);
  // print_mat(&C_ref, M, N);
  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);





  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = outer_fp32_5x5_m_first(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_fp32_5x5_m_first time: "); 
  Serial.println(diff); //prints time since program started
  // print_mat(&C, M, N);
  // print_mat(&C_ref, M, N);
  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);






  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = outer_fp32_5x5_m_first_test(&A, &B, &C, tmp_arr, 1);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_fp32_5x5_m_first test time: "); 
  Serial.println(diff); //prints time since program started

  // print_mat(&A, M, K);
  // print_mat(&B, K, N);


  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);



  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = outer_fp32_5x5_test(&A, &B, &C, tmp_arr, 1);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_fp32_5x5_k_first test time: "); 
  Serial.println(diff); //prints time since program started
  // print_mat(&C, M, N);
  // print_mat(&C_ref, M, N);
  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);





  free(tmp_arr);
  free(A_f32);
  free(B_f32);
  free(C_f32);
  free(C_f32_ref);
}







void setup() {
 // put your setup code here, to run once:

}


void loop() {
 // put your main code here, to run repeatedly:
  // print_memory_info();
  // print_memory_info1();
//   arm_vs_mema_fp32();
//  arm_vs_mema_q15();
  // sram_bw_prof();

  delay(10000); 
  // power_inner_q15();   
//  power_outer_q15();  
//  test();
   // arm_vs_mema_fp32_sp();
  // arm_vs_mema_fp32_k();
  arm_vs_mema_fp32_mk();

  // testing();
//  power_inner_fp32();
//  power_outer_fp32();
  delay(10000); 

}
