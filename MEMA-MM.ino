
#include "kernels.h"





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

  uint32_t srcRows, srcColumns;  /* Temporary variables */
  arm_status status;

  unsigned long start1, end1, diff;
  int16_t *A_16, *B_16, *C_16, *C_16_ref, *B_trans;
  uint32_t M = 90, N = 90, K = 90;

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

  status = arm_mat_mult_q15(&A, &B, &C_ref, B_trans);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("arm_mat_mult_q15 time: "); 
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

  uint32_t srcRows, srcColumns;  /* Temporary variables */
  arm_status status;

  unsigned long start1, end1, diff;
  float *A_f32, *B_f32, *C_f32, *C_f32_ref;
  uint32_t M = 100, N = 100, K = 100;

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

  status = inner_fp32_1x8x1(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm inner_fp32_1x8x1 time: "); 
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




void setup() {
 // put your setup code here, to run once:

}


void loop() {
 // put your main code here, to run repeatedly:

     print_memory_info();
//      sram_bw_prof();
      delay(10000); 

}
