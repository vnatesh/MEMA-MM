
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



void arm_vs_mema() {


  arm_matrix_instance_f32 A;      /* Matrix A Instance */
  arm_matrix_instance_f32 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_f32 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_f32 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  unsigned long start1, end1, diff;
  float *A_f32, *B_f32, *C_f32, *C_f32_ref;
  uint32_t M = 100, N = 100, K = 100;
  char buf[100];

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
  free(C_f32_ref);


  C_f32 = (float *) calloc( M*N, sizeof( float ));

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
    sprintf(buf, "%d,%d,%d,arm,%lu", i,i,i,diff);
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


  // f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
}



void setup() {
 // put your setup code here, to run once:

}


void loop() {
 // put your main code here, to run repeatedly:

     arm_vs_mema();
//      sram_bw_prof();
      delay(10000); 

}
