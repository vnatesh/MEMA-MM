
#include <arm_math.h>
#include <stdio.h>

#include "mbed.h"
#include "mbed_mem_trace.h"

#define ARM_MATH_LOOPUNROLL 1
#define CHECK_PRINT 0

void print_memory_info() ;
void rand_init(float* mat, int r, int c) ;
void print_mat(arm_matrix_instance_f32* mat, int rows, int cols);
void sram_bw_prof() ;

arm_status inner_1x4x1(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status inner_1x8x1(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status inner_1x20x1(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status inner_2x4x2(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status inner_2x8x2(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_2x2(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_2x2_unpacked(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_3x3(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_4x4(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_5x5(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_5x5_unpacked(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_5x5_ptr(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_6x6(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);



arm_status test1_arr(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);


bool f32_gemm_checker(float* C, float* C_check, int N, int M, int K) ;







arm_status inner_2x4x2(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t *A1, *A2, *B1, *B2, *C00, *C01, *C10, *C11;    
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t i, j, k, C_ind = 0U;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  float32_t *A_ptr1 = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *A_ptr2 = pSrcA->pData + K;                /* Input data matrix pointer A */

  float32_t sum1, sum2, sum3, sum4;                                 /* Accumulators */


  /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
  /* row loop */
  for(i = 0U; i < M/2; i++) {
    /* Output pointer is set to starting address of row being processed */
    C00 = pDst->pData + C_ind;
    C01 = C00 + 1;
    C10 = C00 + N;
    C11 = C10 + 1;

    /* For every row wise process, B pointer is set to starting address of pSrcB data */
    B1 = pSrcB->pData;
    B2 = pSrcB->pData + 1;

    /* column loop */
    for(j = 0U; j < N/2; j++) {

      /* Set the variable sum, that acts as accumulator, to zero */
      sum1 = 0.0f;
      sum2 = 0.0f;
      sum3 = 0.0f;
      sum4 = 0.0f;

      /* Initialize pointer A to point to starting address of column being processed */
      A1 = A_ptr1;
      A2 = A_ptr2;

      /* Loop unrolling: Compute 8 MACs at a time. */
      k = K >> 2;

      /* matrix multiplication */
      while (k > 0U)
      {
        /* c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) */

        /* Perform the multiply-accumulates */

        sum1 += *A1 * *B1;
        B1 += N;

        sum2 += *A1++ * *B2;
        B2 += N;


        sum1 += *A1 * *B1;
        B1 += N;

        sum2 += *A1++ * *B2;
        B2 += N;


        sum1 += *A1 * *B1;
        B1 += N;

        sum2 += *A1++ * *B2;
        B2 += N;


        sum1 += *A1 * *B1;
        B1 -= 3*N ;

        sum2 += *A1++ * *B2;
        B2 -= 3*N ;





        sum3 += *A2 * *B1;
        B1 += N;

        sum4 += *A2++ * *B2;
        B2 += N;


        sum3 += *A2 * *B1;
        B1 += N;

        sum4 += *A2++ * *B2;
        B2 += N;


        sum3 += *A2 * *B1;
        B1 += N;

        sum4 += *A2++ * *B2;
        B2 += N;


        sum3 += *A2 * *B1;
        B1 += N;

        sum4 += *A2++ * *B2;
        B2 += N;


        /* Decrement loop counter */
        k--;
      }

      // /* Loop unrolling: Compute remaining MACs */
      // k = K % 4U;


      // while (k > 0U)
      // {
      //    // c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) 

      //   /* Perform the multiply-accumulates */
      //   sum += *A++ * *B;
      //   B += N;

      //   /* Decrement loop counter */
      //   k--;
      // }

      /* Store result in destination buffer */
      // *C00++ = sum;
      *C00 = sum1;
      *C01 = sum2;
      *C10 = sum3;
      *C11 = sum4;

      C00 += 2;
      C01 += 2;
      C10 += 2;
      C11 += 2; 

      /* Update pointer B to point to starting address of next column */
      B1 = pSrcB->pData + 2*(j+1);
      B2 = B1 + 1;
    }

    /* Update pointer A_ptr1 to point to starting address of next row */
    C_ind = C_ind + 2*N;
    A_ptr1 = A_ptr1 + 2*K;
    A_ptr2 = A_ptr2 + 2*K;
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}



arm_status inner_2x8x2(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t *A1, *A2, *B1, *B2, *C00, *C01, *C10, *C11;    
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t i, j, k, C_ind = 0U;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  float32_t *A_ptr1 = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *A_ptr2 = pSrcA->pData + K;                /* Input data matrix pointer A */

  float32_t sum1, sum2, sum3, sum4;                                 /* Accumulators */


  /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
  /* row loop */
  for(i = 0U; i < M/2; i++) {
    /* Output pointer is set to starting address of row being processed */
    C00 = pDst->pData + C_ind;
    C01 = C00 + 1;
    C10 = C00 + N;
    C11 = C10 + 1;

    /* For every row wise process, B pointer is set to starting address of pSrcB data */
    B1 = pSrcB->pData;
    B2 = pSrcB->pData + 1;

    /* column loop */
    for(j = 0U; j < N/2; j++) {

      /* Set the variable sum, that acts as accumulator, to zero */
      sum1 = 0.0f;
      sum2 = 0.0f;
      sum3 = 0.0f;
      sum4 = 0.0f;

      /* Initialize pointer A to point to starting address of column being processed */
      A1 = A_ptr1;
      A2 = A_ptr2;

      /* Loop unrolling: Compute 8 MACs at a time. */
      k = K >> 3;

      /* matrix multiplication */
      while (k > 0U)
      {
        /* c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) */

        /* Perform the multiply-accumulates */

        sum1 += *A1 * *B1;
        B1 += N;

        sum2 += *A1++ * *B2;
        B2 += N;


        sum1 += *A1 * *B1;
        B1 += N;

        sum2 += *A1++ * *B2;
        B2 += N;


        sum1 += *A1 * *B1;
        B1 += N;

        sum2 += *A1++ * *B2;
        B2 += N;


        sum1 += *A1 * *B1;
        B1 += N;

        sum2 += *A1++ * *B2;
        B2 += N;


        sum1 += *A1 * *B1;
        B1 += N;

        sum2 += *A1++ * *B2;
        B2 += N;


        sum1 += *A1 * *B1;
        B1 += N;

        sum2 += *A1++ * *B2;
        B2 += N;


        sum1 += *A1 * *B1;
        B1 += N;

        sum2 += *A1++ * *B2;
        B2 += N;


        sum1 += *A1 * *B1;
        B1 -= 7*N ;

        sum2 += *A1++ * *B2;
        B2 -= 7*N ;



        sum3 += *A2 * *B1;
        B1 += N;

        sum4 += *A2++ * *B2;
        B2 += N;


        sum3 += *A2 * *B1;
        B1 += N;

        sum4 += *A2++ * *B2;
        B2 += N;


        sum3 += *A2 * *B1;
        B1 += N;

        sum4 += *A2++ * *B2;
        B2 += N;


        sum3 += *A2 * *B1;
        B1 += N;

        sum4 += *A2++ * *B2;
        B2 += N;


        sum3 += *A2 * *B1;
        B1 += N;

        sum4 += *A2++ * *B2;
        B2 += N;


        sum3 += *A2 * *B1;
        B1 += N;

        sum4 += *A2++ * *B2;
        B2 += N;


        sum3 += *A2 * *B1;
        B1 += N;

        sum4 += *A2++ * *B2;
        B2 += N;


        sum3 += *A2 * *B1;
        B1 += N;

        sum4 += *A2++ * *B2;
        B2 += N;


        /* Decrement loop counter */
        k--;
      }

      // /* Loop unrolling: Compute remaining MACs */
      // k = K % 4U;


      // while (k > 0U)
      // {
      //    // c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) 

      //   /* Perform the multiply-accumulates */
      //   sum += *A++ * *B;
      //   B += N;

      //   /* Decrement loop counter */
      //   k--;
      // }

      /* Store result in destination buffer */
      // *C00++ = sum;
      *C00 = sum1;
      *C01 = sum2;
      *C10 = sum3;
      *C11 = sum4;

      C00 += 2;
      C01 += 2;
      C10 += 2;
      C11 += 2; 

      /* Update pointer B to point to starting address of next column */
      B1 = pSrcB->pData + 2*(j+1);
      B2 = B1 + 1;
    }

    /* Update pointer A_ptr1 to point to starting address of next row */
    C_ind = C_ind + 2*N;
    A_ptr1 = A_ptr1 + 2*K;
    A_ptr2 = A_ptr2 + 2*K;
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}




arm_status outer_6x6(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t C00, C01, C02, C03, C04, C05,
  C10, C11, C12, C13, C14, C15, C20, C21, 
  C22, C23, C24, C25, C30, C31, C32, C33, 
  C34, C35, C40, C41, C42, C43, C44, C45,
  C50, C51, C52, C53, C54, C55;    /* Temporary output data matrix pointer */

  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k, a_ind, b_ind, c_ind;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  uint32_t en = N / 6;
  uint32_t em = M / 6;
  uint32_t ek = K * 6;

  for(n = 0U; n < en; n++) {

    c_ind = 6*n*M;

    for(m = 0U; m < em; m++) {

      a_ind = m*ek;
      b_ind = n*ek;

      C00 = C[c_ind];
      C01 = C[c_ind+1];
      C02 = C[c_ind+2];
      C03 = C[c_ind+3];
      C04 = C[c_ind+4];
      C05 = C[c_ind+5];

      C10 = C[c_ind+6];
      C11 = C[c_ind+7];
      C12 = C[c_ind+8];
      C13 = C[c_ind+9];
      C14 = C[c_ind+10];
      C15 = C[c_ind+11];

      C20 = C[c_ind+12];
      C21 = C[c_ind+13];
      C22 = C[c_ind+14];
      C23 = C[c_ind+15];
      C24 = C[c_ind+16];
      C25 = C[c_ind+17];

      C30 = C[c_ind+18];
      C31 = C[c_ind+19];
      C32 = C[c_ind+20];
      C33 = C[c_ind+21];
      C34 = C[c_ind+22];
      C35 = C[c_ind+23];

      C40 = C[c_ind+24];
      C41 = C[c_ind+25];
      C42 = C[c_ind+26];
      C43 = C[c_ind+27];
      C44 = C[c_ind+28];
      C45 = C[c_ind+29];

      C50 = C[c_ind+30];
      C51 = C[c_ind+31];
      C52 = C[c_ind+32];
      C53 = C[c_ind+33];
      C54 = C[c_ind+34];
      C55 = C[c_ind+35];

      /* Perform tile outer product */
      for(k = 0U; k < K; k++) {

        C00 += A[a_ind] * B[b_ind];
        C01 += A[a_ind] * B[b_ind+1];
        C02 += A[a_ind] * B[b_ind+2];
        C03 += A[a_ind] * B[b_ind+3];
        C04 += A[a_ind] * B[b_ind+4];
        C05 += A[a_ind] * B[b_ind+5];

        C10 += A[a_ind+1] * B[b_ind];
        C11 += A[a_ind+1] * B[b_ind+1];
        C12 += A[a_ind+1] * B[b_ind+2];
        C13 += A[a_ind+1] * B[b_ind+3];
        C14 += A[a_ind+1] * B[b_ind+4];
        C15 += A[a_ind+1] * B[b_ind+5];

        C20 += A[a_ind+2] * B[b_ind];
        C21 += A[a_ind+2] * B[b_ind+1];
        C22 += A[a_ind+2] * B[b_ind+2];
        C23 += A[a_ind+2] * B[b_ind+3];
        C24 += A[a_ind+2] * B[b_ind+4];
        C25 += A[a_ind+2] * B[b_ind+5];

        C30 += A[a_ind+3] * B[b_ind];
        C31 += A[a_ind+3] * B[b_ind+1];
        C32 += A[a_ind+3] * B[b_ind+2];
        C33 += A[a_ind+3] * B[b_ind+3];
        C34 += A[a_ind+3] * B[b_ind+4];
        C35 += A[a_ind+3] * B[b_ind+5];

        C40 += A[a_ind+4] * B[b_ind];
        C41 += A[a_ind+4] * B[b_ind+1];
        C42 += A[a_ind+4] * B[b_ind+2];
        C43 += A[a_ind+4] * B[b_ind+3];
        C44 += A[a_ind+4] * B[b_ind+4];
        C45 += A[a_ind+4] * B[b_ind+5];

        C50 += A[a_ind+5] * B[b_ind];
        C51 += A[a_ind+5] * B[b_ind+1];
        C52 += A[a_ind+5] * B[b_ind+2];
        C53 += A[a_ind+5] * B[b_ind+3];
        C54 += A[a_ind+5] * B[b_ind+4];
        C55 += A[a_ind+5] * B[b_ind+5];

        a_ind += 6;
        b_ind += 6;
      }

      C[c_ind] = C00 ;
      C[c_ind+1] = C01  ;
      C[c_ind+2] = C02  ;
      C[c_ind+3] = C03  ;
      C[c_ind+4] = C04  ;
      C[c_ind+5] = C05  ;

      C[c_ind+6] = C10  ;
      C[c_ind+7] = C11  ;
      C[c_ind+8] = C12  ;
      C[c_ind+9] = C13  ;
      C[c_ind+10] = C14  ;
      C[c_ind+11] = C15  ;

      C[c_ind+12] = C20  ;
      C[c_ind+13] = C21  ;
      C[c_ind+14] = C22  ;
      C[c_ind+15] = C23  ;
      C[c_ind+16] = C24  ;
      C[c_ind+17] = C25  ;

      C[c_ind+18] = C30  ;
      C[c_ind+19] = C31  ;
      C[c_ind+20] = C32  ;
      C[c_ind+21] = C33  ;
      C[c_ind+22] = C34  ;
      C[c_ind+23] = C35  ;

      C[c_ind+24] = C40  ;
      C[c_ind+25] = C41  ;
      C[c_ind+26] = C42  ;
      C[c_ind+27] = C43  ;
      C[c_ind+28] = C44  ;
      C[c_ind+29] = C45  ;

      C[c_ind+30] = C50  ;
      C[c_ind+31] = C51  ;
      C[c_ind+32] = C52  ;
      C[c_ind+33] = C53  ;
      C[c_ind+34] = C54  ;
      C[c_ind+35] = C55  ;

      c_ind += 36;
    }
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}





arm_status outer_5x5(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t C00, C01, C02, C03, C04,
  C10, C11, C12, C13, C14, C20, C21, 
  C22, C23, C24, C30, C31, C32, C33, 
  C34, C40, C41, C42, C43, C44;    /* Temporary output data matrix pointer */

  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k, a_ind, b_ind, c_ind;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  uint32_t en = N / 5;
  uint32_t em = M / 5;
  uint32_t ek = K * 5;

  for(n = 0U; n < en; n++) {

    c_ind = 5*n*M;

    for(m = 0U; m < em; m++) {

      a_ind = m*ek;
      b_ind = n*ek;

      C00 = C[c_ind];
      C01 = C[c_ind+1];
      C02 = C[c_ind+2];
      C03 = C[c_ind+3];
      C04 = C[c_ind+4];

      C10 = C[c_ind+5];
      C11 = C[c_ind+6];
      C12 = C[c_ind+7];
      C13 = C[c_ind+8];
      C14 = C[c_ind+9];

      C20 = C[c_ind+10];
      C21 = C[c_ind+11];
      C22 = C[c_ind+12];
      C23 = C[c_ind+13];
      C24 = C[c_ind+14];

      C30 = C[c_ind+15];
      C31 = C[c_ind+16];
      C32 = C[c_ind+17];
      C33 = C[c_ind+18];
      C34 = C[c_ind+19];

      C40 = C[c_ind+20];
      C41 = C[c_ind+21];
      C42 = C[c_ind+22];
      C43 = C[c_ind+23];
      C44 = C[c_ind+24];

      /* Perform tile outer product */
      for(k = 0U; k < K; k++) {

        C00 += A[a_ind] * B[b_ind];
        C01 += A[a_ind] * B[b_ind+1];
        C02 += A[a_ind] * B[b_ind+2];
        C03 += A[a_ind] * B[b_ind+3];
        C04 += A[a_ind] * B[b_ind+4];

        C10 += A[a_ind+1] * B[b_ind];
        C11 += A[a_ind+1] * B[b_ind+1];
        C12 += A[a_ind+1] * B[b_ind+2];
        C13 += A[a_ind+1] * B[b_ind+3];
        C14 += A[a_ind+1] * B[b_ind+4];

        C20 += A[a_ind+2] * B[b_ind];
        C21 += A[a_ind+2] * B[b_ind+1];
        C22 += A[a_ind+2] * B[b_ind+2];
        C23 += A[a_ind+2] * B[b_ind+3];
        C24 += A[a_ind+2] * B[b_ind+4];

        C30 += A[a_ind+3] * B[b_ind];
        C31 += A[a_ind+3] * B[b_ind+1];
        C32 += A[a_ind+3] * B[b_ind+2];
        C33 += A[a_ind+3] * B[b_ind+3];
        C34 += A[a_ind+3] * B[b_ind+4];

        C40 += A[a_ind+4] * B[b_ind];
        C41 += A[a_ind+4] * B[b_ind+1];
        C42 += A[a_ind+4] * B[b_ind+2];
        C43 += A[a_ind+4] * B[b_ind+3];
        C44 += A[a_ind+4] * B[b_ind+4];

        a_ind += 5;
        b_ind += 5;
      }

        C[c_ind] = C00 ;
        C[c_ind+1] = C01 ;
        C[c_ind+2] = C02 ;
        C[c_ind+3] = C03 ;
        C[c_ind+4] = C04 ;

        C[c_ind+5] = C10 ;
        C[c_ind+6] = C11 ;
        C[c_ind+7] = C12 ;
        C[c_ind+8] = C13 ;
        C[c_ind+9] = C14 ;

        C[c_ind+10] = C20 ;
        C[c_ind+11] = C21 ;
        C[c_ind+12] = C22 ;
        C[c_ind+13] = C23 ;
        C[c_ind+14] = C24 ;

        C[c_ind+15] = C30 ;
        C[c_ind+16] = C31 ;
        C[c_ind+17] = C32 ;
        C[c_ind+18] = C33 ;
        C[c_ind+19] = C34 ;

        C[c_ind+20] = C40 ;
        C[c_ind+21] = C41 ;
        C[c_ind+22] = C42 ;
        C[c_ind+23] = C43 ;
        C[c_ind+24] = C44 ;

      c_ind += 25;
    }
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}



arm_status outer_4x4(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t C00, C01, C02, C03, 
  C10, C11, C12, C13, C20, C21, 
  C22, C23, C30, C31, C32, C33;    /* Temporary output data matrix pointer */

  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k, a_ind, b_ind, c_ind;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  uint32_t en = N >> 2;
  uint32_t em = M >> 2;
  uint32_t ek = K << 2;

  for(n = 0U; n < en; n++) {

    c_ind = 4*n*M;

    for(m = 0U; m < em; m++) {

      a_ind = m*ek;
      b_ind = n*ek;

      C00 = C[c_ind];
      C01 = C[c_ind+1];
      C02 = C[c_ind+2];
      C03 = C[c_ind+3];
      C10 = C[c_ind+4];
      C11 = C[c_ind+5];
      C12 = C[c_ind+6];
      C13 = C[c_ind+7];
      C20 = C[c_ind+8];
      C21 = C[c_ind+9];
      C22 = C[c_ind+10];
      C23 = C[c_ind+11];
      C30 = C[c_ind+12];
      C31 = C[c_ind+13];
      C32 = C[c_ind+14];
      C33 = C[c_ind+15];

      /* Perform tile outer product */
      for(k = 0U; k < K; k++) {

        C00 += A[a_ind] * B[b_ind];
        C01 += A[a_ind] * B[b_ind+1];
        C02 += A[a_ind] * B[b_ind+2];
        C03 += A[a_ind] * B[b_ind+3];

        C10 += A[a_ind+1] * B[b_ind];
        C11 += A[a_ind+1] * B[b_ind+1];
        C12 += A[a_ind+1] * B[b_ind+2];
        C13 += A[a_ind+1] * B[b_ind+3];

        C20 += A[a_ind+2] * B[b_ind];
        C21 += A[a_ind+2] * B[b_ind+1];
        C22 += A[a_ind+2] * B[b_ind+2];
        C23 += A[a_ind+2] * B[b_ind+3];

        C30 += A[a_ind+3] * B[b_ind];
        C31 += A[a_ind+3] * B[b_ind+1];
        C32 += A[a_ind+3] * B[b_ind+2];
        C33 += A[a_ind+3] * B[b_ind+3];

        a_ind += 4;
        b_ind += 4;
      }

      C[c_ind] = C00 ;
      C[c_ind+1] = C01 ;
      C[c_ind+2] = C02 ;
      C[c_ind+3] = C03 ;
      C[c_ind+4] = C10 ;
      C[c_ind+5] = C11 ;
      C[c_ind+6] = C12 ;
      C[c_ind+7] = C13 ;
      C[c_ind+8] = C20 ;
      C[c_ind+9] = C21 ;
      C[c_ind+10] = C22 ;
      C[c_ind+11] = C23 ;
      C[c_ind+12] = C30 ;
      C[c_ind+13] = C31 ;
      C[c_ind+14] = C32 ;
      C[c_ind+15] = C33 ;


      c_ind += 16;
    }
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}


arm_status outer_3x3(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t C00, C01, C02, C10, C11, C12, C20, C21, C22;    /* Temporary output data matrix pointer */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k, a_ind, b_ind, c_ind;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */


  for(n = 0U; n < N/3; n++) {

    c_ind = 3*n*M;

    for(m = 0U; m < M/3; m++) {

      a_ind = m*3*K;
      b_ind = 3*n*K;

      C00 = C[c_ind];
      C01 = C[c_ind+1];
      C02 = C[c_ind+2];
      C10 = C[c_ind+3];
      C11 = C[c_ind+4];
      C12 = C[c_ind+5];
      C20 = C[c_ind+6];
      C21 = C[c_ind+7];
      C22 = C[c_ind+8];
      /* Perform tile outer product */
      for(k = 0U; k < K; k++) {

        C00 += A[a_ind] * B[b_ind];
        C01 += A[a_ind] * B[b_ind+1];
        C02 += A[a_ind] * B[b_ind+2];
        C10 += A[a_ind+1] * B[b_ind];
        C11 += A[a_ind+1] * B[b_ind+1];
        C12 += A[a_ind+1] * B[b_ind+2];
        C20 += A[a_ind+2] * B[b_ind];
        C21 += A[a_ind+2] * B[b_ind+1];
        C22 += A[a_ind+2] * B[b_ind+2];

        a_ind += 3;
        b_ind += 3;
      }

      C[c_ind]   = C00;
      C[c_ind+1] = C01;
      C[c_ind+2] = C02;
      C[c_ind+3] = C10;
      C[c_ind+4] = C11;
      C[c_ind+5] = C12;
      C[c_ind+6] = C20;
      C[c_ind+7] = C21;
      C[c_ind+8] = C22;

      c_ind += 3*3;
    }
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}





void rand_init(float* mat, int r, int c) {
    // int MAX = 65536;
    for(int i = 0; i < r*c; i++) {
        // mat[i] = (double) i;
        // mat[i] = 1.0;
        // mat[i] =  (double) (i%MAX);
        mat[i] =  (float) rand() / RAND_MAX*2.0 - 1.0;
    }   
}



void sram_bw_prof() {

  unsigned long start1, end1, diff;

  // int N = 20000;
  int N = 19968;
  float* a = (float*) malloc(N * sizeof(float));
  float* b = (float*) malloc(N * sizeof(float));
  rand_init(a,N,1);

  start1 = micros();

  // for(int i = 0; i < N; i++) {
  //   b[i] = a[i];
  // }
  for(int i = 0; i < N; i += 64) {
  // for(int i = 0; i < N; i += 48) {
  // for(int i = 0; i < N; i += 32) {
  // for(int i = 0; i < N; i += 16) {
    b[i] = a[i];        b[i+1] = a[i+1];
    b[i+2] = a[i+2];    b[i+3] = a[i+3];
    b[i+4] = a[i+4];    b[i+5] = a[i+5];
    b[i+6] = a[i+6];    b[i+7] = a[i+7];
    b[i+8] = a[i+8];    b[i+9] = a[i+9];
    b[i+10] = a[i+10];  b[i+11] = a[i+11];
    b[i+12] = a[i+12];  b[i+13] = a[i+13];
    b[i+14] = a[i+14];  b[i+15] = a[i+15];
    b[i+16] = a[i+16];  b[i+17] = a[i+17];
    b[i+18] = a[i+18];  b[i+19] = a[i+19];
    b[i+20] = a[i+20];  b[i+21] = a[i+21];
    b[i+22] = a[i+22];  b[i+23] = a[i+23];
    b[i+24] = a[i+24];  b[i+25] = a[i+25];
    b[i+26] = a[i+26];  b[i+27] = a[i+27];
    b[i+28] = a[i+28];  b[i+29] = a[i+29];
    b[i+30] = a[i+30];  b[i+31] = a[i+31];

    b[i+32] = a[i+32];  b[i+33] = a[i+33];
    b[i+34] = a[i+34];  b[i+35] = a[i+35];
    b[i+36] = a[i+36];  b[i+37] = a[i+37];
    b[i+38] = a[i+38];  b[i+39] = a[i+39];
    b[i+40] = a[i+40];  b[i+41] = a[i+41];
    b[i+42] = a[i+42];  b[i+43] = a[i+43];
    b[i+44] = a[i+44];  b[i+45] = a[i+45];
    b[i+46] = a[i+46];  b[i+47] = a[i+47];

    b[i+48] = a[i+48];  b[i+49] = a[i+49];
    b[i+50] = a[i+50];  b[i+51] = a[i+51];
    b[i+52] = a[i+52];  b[i+53] = a[i+53];
    b[i+54] = a[i+54];  b[i+55] = a[i+55];
    b[i+56] = a[i+56];  b[i+57] = a[i+57];
    b[i+58] = a[i+58];  b[i+59] = a[i+59];
    b[i+60] = a[i+60];  b[i+61] = a[i+61];
    b[i+62] = a[i+62];  b[i+63] = a[i+63];

  }

  end1 = micros();
  diff = end1 - start1;
  Serial.print("time: "); 
  Serial.println(diff); //prints time since program started
  Serial.print("sram bw: "); 
  Serial.print(((float) (2*N*sizeof(float))) / ((float) diff)); //prints time since program started
  Serial.print(" MB/sec\n"); 

  free(a);
  free(b);
}


void print_mat(arm_matrix_instance_f32* mat, int rows, int cols) {

  char buffer[100];

  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      sprintf(buffer, "%0.2f ", mat->pData[i*cols + j]);
      Serial.print(buffer);
    }
    Serial.println("");
  }
  Serial.println("");

}



arm_status outer_2x2(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t C00, C01, C10, C11;    /* Temporary output data matrix pointer */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k, a_ind, b_ind, c_ind;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */


  for(n = 0U; n < N/2; n++) {

    c_ind = 2*n*M;

    for(m = 0U; m < M/2; m++) {

      a_ind = m*2*K;
      b_ind = 2*n*K;

      C00 = C[c_ind];
      C01 = C[c_ind+1];
      C10 = C[c_ind+2];
      C11 = C[c_ind+3];

      /* Perform tile outer product */
      for(k = 0U; k < K; k++) {

        C00 += A[a_ind] * B[b_ind];
        C01 += A[a_ind] * B[b_ind+1];
        C10 += A[a_ind+1] * B[b_ind];
        C11 += A[a_ind+1] * B[b_ind+1];
        a_ind += 2;
        b_ind += 2;
      }

      C[c_ind] = C00;
      C[c_ind+1] = C01;
      C[c_ind+2] = C10;
      C[c_ind+3] = C11;

      c_ind += 2*2;
    }
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}




arm_status outer_2x2_unpacked(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t C00, C01, C10, C11;    /* Temporary output data matrix pointer */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k, a_ind, b_ind, c_ind;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */


  for(n = 0U; n < N/2; n++) {

    c_ind = n*2;

    for(m = 0U; m < M/2; m++) {

      // c_ind += 2*m*N;

      a_ind = m*2*K;
      b_ind = 2*n;

      C00 = C[c_ind];
      C01 = C[c_ind+1];
      C10 = C[c_ind+N];
      C11 = C[c_ind+N+1];

      /* Perform tile outer product */
      for(k = 0U; k < K; k++) {

        C00 += A[a_ind] * B[b_ind];
        C01 += A[a_ind] * B[b_ind+1];
        C10 += A[a_ind+K] * B[b_ind];
        C11 += A[a_ind+K] * B[b_ind+1];

        a_ind++;
        b_ind += N;
      }

      C[c_ind] = C00;
      C[c_ind+1] = C01;
      C[c_ind+N] = C10;
      C[c_ind+N+1] = C11;

      c_ind += 2*N;
    }
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}






arm_status outer_5x5_ptr(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t C00, C01, C02, C03, C04,
  C10, C11, C12, C13, C14, C20, C21, 
  C22, C23, C24, C30, C31, C32, C33, 
  C34, C40, C41, C42, C43, C44;    /* Temporary output data  */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k, C_ind = 0U;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  float32_t *A_ptr = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B_ptr, *C_curr;

  uint32_t en = N / 5;
  uint32_t em = M / 5;

  /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
  /* row loop */
  for(m = 0U; m < em; m++) {
    /* Output pointer is set to starting address of row being processed */

    C_ind = m*5*N;

    /* For every row wise process, B pointer is set to starting address of pSrcB data */
    // B = pSrcB->pData;

    /* column loop */
    for(n = 0U; n < en; n++) {

      C00 = 0;
      C01 = 0;
      C02 = 0;
      C03 = 0;
      C04 = 0;

      C10 = 0;
      C11 = 0;
      C12 = 0;
      C13 = 0;
      C14 = 0;

      C20 = 0;
      C21 = 0;
      C22 = 0;
      C23 = 0;
      C24 = 0;

      C30 = 0;
      C31 = 0;
      C32 = 0;
      C33 = 0;
      C34 = 0;

      C40 = 0;
      C41 = 0;
      C42 = 0;
      C43 = 0;
      C44 = 0;

      /* Update pointer B to point to starting address of next column */
      B_ptr = pSrcB->pData + 5*n;

      /* matrix multiplication */
      for(k = 0U; k < K; k++) {
      
        A = A_ptr + k ;
        B = B_ptr + k*N;

        C00 += *A * *B++;
        C01 += *A * *B++;
        C02 += *A * *B++;
        C03 += *A * *B++;
        C04 += *A * *B;

        B -= 4;
        A += K;

        C10 += *A * *B++;
        C11 += *A * *B++;
        C12 += *A * *B++;
        C13 += *A * *B++;
        C14 += *A * *B;

        B -= 4;
        A += K;

        C20 += *A * *B++;
        C21 += *A * *B++;
        C22 += *A * *B++;
        C23 += *A * *B++;
        C24 += *A * *B;

        B -= 4;
        A += K;

        C30 += *A * *B++;
        C31 += *A * *B++;
        C32 += *A * *B++;
        C33 += *A * *B++;
        C34 += *A * *B;

        B -= 4;
        A += K;

        C40 += *A * *B++;
        C41 += *A * *B++;
        C42 += *A * *B++;
        C43 += *A * *B++;
        C44 += *A * *B;

        // B = B - 4 + N;

      }

      /* Store result in destination buffer */

      C_curr = pDst->pData + C_ind;

      *C_curr++ = C00;
      *C_curr++ = C01;
      *C_curr++ = C02;
      *C_curr++ = C03;
      *C_curr = C04;

      C_curr = C_curr - 4 + N;

      *C_curr++ = C10;
      *C_curr++ = C11;
      *C_curr++ = C12;
      *C_curr++ = C13;
      *C_curr = C14;

      C_curr = C_curr - 4 + N;

      *C_curr++ = C20;
      *C_curr++ = C21;
      *C_curr++ = C22;
      *C_curr++ = C23;
      *C_curr = C24;

      C_curr = C_curr - 4 + N;

      *C_curr++ = C30;
      *C_curr++ = C31;
      *C_curr++ = C32;
      *C_curr++ = C33;
      *C_curr = C34;

      C_curr = C_curr - 4 + N;

      *C_curr++ = C40;
      *C_curr++ = C41;
      *C_curr++ = C42;
      *C_curr++ = C43;
      *C_curr = C44;


      C_ind += 5;

    }

    /* Update pointer A_ptr to point to starting address of next row */
    A_ptr += 5*K;

  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}



arm_status outer_5x5_unpacked(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t C00, C01, C02, C03, C04,
  C10, C11, C12, C13, C14, C20, C21, 
  C22, C23, C24, C30, C31, C32, C33, 
  C34, C40, C41, C42, C43, C44;    /* Temporary output data matrix pointer */

  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k, a_ind, b_ind, c_ind;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  uint32_t en = N / 5;
  uint32_t em = M / 5;


  for(n = 0U; n < en; n++) {

    c_ind = n*5;

    for(m = 0U; m < em; m++) {

      // c_ind += 2*m*N;
      a_ind = m*5*K;
      b_ind = 5*n;

      C00 = 0;
      C01 = 0;
      C02 = 0;
      C03 = 0;
      C04 = 0;

      C10 = 0;
      C11 = 0;
      C12 = 0;
      C13 = 0;
      C14 = 0;

      C20 = 0;
      C21 = 0;
      C22 = 0;
      C23 = 0;
      C24 = 0;

      C30 = 0;
      C31 = 0;
      C32 = 0;
      C33 = 0;
      C34 = 0;

      C40 = 0;
      C41 = 0;
      C42 = 0;
      C43 = 0;
      C44 = 0;

      /* Perform tile outer product */
      for(k = 0U; k < K; k++) {

        C00 += A[a_ind] * B[b_ind];
        C01 += A[a_ind] * B[b_ind+1];
        C02 += A[a_ind] * B[b_ind+2];
        C03 += A[a_ind] * B[b_ind+3];
        C04 += A[a_ind] * B[b_ind+4];

        C10 += A[a_ind+K] * B[b_ind];
        C11 += A[a_ind+K] * B[b_ind+1];
        C12 += A[a_ind+K] * B[b_ind+2];
        C13 += A[a_ind+K] * B[b_ind+3];
        C14 += A[a_ind+K] * B[b_ind+4];

        C20 += A[a_ind+2*K] * B[b_ind];
        C21 += A[a_ind+2*K] * B[b_ind+1];
        C22 += A[a_ind+2*K] * B[b_ind+2];
        C23 += A[a_ind+2*K] * B[b_ind+3];
        C24 += A[a_ind+2*K] * B[b_ind+4];

        C30 += A[a_ind+3*K] * B[b_ind];
        C31 += A[a_ind+3*K] * B[b_ind+1];
        C32 += A[a_ind+3*K] * B[b_ind+2];
        C33 += A[a_ind+3*K] * B[b_ind+3];
        C34 += A[a_ind+3*K] * B[b_ind+4];

        C40 += A[a_ind+4*K] * B[b_ind];
        C41 += A[a_ind+4*K] * B[b_ind+1];
        C42 += A[a_ind+4*K] * B[b_ind+2];
        C43 += A[a_ind+4*K] * B[b_ind+3];
        C44 += A[a_ind+4*K] * B[b_ind+4];


        a_ind++;
        b_ind += N;
      }

      C[c_ind] = C00;            
      C[c_ind+1] = C01;          
      C[c_ind+2] = C02;          
      C[c_ind+3] = C03;          
      C[c_ind+4] = C04; 

      C[c_ind+N] = C10;          
      C[c_ind+N+1] = C11;        
      C[c_ind+N+2] = C12;        
      C[c_ind+N+3] = C13;        
      C[c_ind+N+4] = C14;  

      C[c_ind+2*N] = C20;        
      C[c_ind+2*N+1] = C21;      
      C[c_ind+2*N+2] = C22;      
      C[c_ind+2*N+3] = C23;      
      C[c_ind+2*N+4] = C24;  

      C[c_ind+3*N] = C30;        
      C[c_ind+3*N+1] = C31;      
      C[c_ind+3*N+2] = C32;      
      C[c_ind+3*N+3] = C33;      
      C[c_ind+3*N+4] = C34;   

      C[c_ind+4*N] = C40;        
      C[c_ind+4*N+1] = C41;      
      C[c_ind+4*N+2] = C42;      
      C[c_ind+4*N+3] = C43;      
      C[c_ind+4*N+4] = C44;      

      c_ind += 5*N;
    }
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}






arm_status inner_1x8x1(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *A_ptr = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B_ptr = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t *C_curr;                                 /* Temporary output data matrix pointer */
  float32_t sum;                                 /* Accumulator */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t i, j, k, C_ind = 0U;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
  /* row loop */
  for(i = M; i > 0U; i--) {
    /* Output pointer is set to starting address of row being processed */
    C_curr = C + C_ind;

    /* For every row wise process, B pointer is set to starting address of pSrcB data */
    B = pSrcB->pData;

    /* column loop */
    for(j = N; j > 0U; j--) {

     // Set the variable sum, that acts as accumulator, to zero 
      sum = 0.0f;

      /* Initialize pointer A to point to starting address of column being processed */
      A = A_ptr;


      /* Loop unrolling: Compute 8 MACs at a time. */
      // k = K / 20;
      k = K >> 3;

      /* matrix multiplication */
      while (k > 0U)
      {
        /* c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) */

        /* Perform the multiply-accumulates */

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;
        /* Decrement loop counter */
        k--;
      }

      /* Loop unrolling: Compute remaining MACs */
      k = K % 8;


      while (k > 0U)
      {
        /* c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) */

        /* Perform the multiply-accumulates */
        sum += *A++ * *B;
        B += N;

        /* Decrement loop counter */
        k--;
      }

      /* Store result in destination buffer */
      *C_curr++ = sum;

      /* Update pointer B to point to starting address of next column */
      B = B_ptr + (N - j + 1);

    }

    /* Update pointer A_ptr to point to starting address of next row */
    C_ind = C_ind + N;
    A_ptr = A_ptr + K;
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}







arm_status inner_1x20x1(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *A_ptr = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B_ptr = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t *C_curr;                                 /* Temporary output data matrix pointer */
  float32_t sum;                                 /* Accumulator */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t i, j, k, C_ind = 0U;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
  /* row loop */
  for(i = M; i > 0U; i--) {
    /* Output pointer is set to starting address of row being processed */
    C_curr = C + C_ind;

    /* For every row wise process, B pointer is set to starting address of pSrcB data */
    B = pSrcB->pData;

    /* column loop */
    for(j = N; j > 0U; j--) {

      /* Set the variable sum, that acts as accumulator, to zero */
      sum = 0.0f;

      /* Initialize pointer A to point to starting address of column being processed */
      A = A_ptr;


      /* Loop unrolling: Compute 8 MACs at a time. */
      k = K / 20;

      /* matrix multiplication */
      while (k > 0U)
      {
        /* c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) */

        /* Perform the multiply-accumulates */
        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;

        sum += *A++ * *B;
        B += N;
        /* Decrement loop counter */
        k--;
      }

      /* Loop unrolling: Compute remaining MACs */
      k = K % 20U;


      while (k > 0U)
      {
        /* c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) */

        /* Perform the multiply-accumulates */
        sum += *A++ * *B;
        B += N;

        /* Decrement loop counter */
        k--;
      }

      /* Store result in destination buffer */
      *C_curr++ = sum;

      /* Update pointer B to point to starting address of next column */
      B = B_ptr + (N - j + 1);

    }

    /* Update pointer A_ptr to point to starting address of next row */
    C_ind = C_ind + N;
    A_ptr = A_ptr + K;
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}



arm_status test1_arr(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *A_ptr = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B_ptr = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t *C_curr;                                 /* Temporary output data matrix pointer */
  float32_t sum;                                 /* Accumulator */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t i, j, k, ke, a_ind, b_ind, c_ind = 0U;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
  for(i = 0; i < M; i++) {

    for(j = 0; j < N; j++) {

      sum = 0.0f;
      ke = K >> 3;

      /* matrix multiplication */
      for(k = 0; k < ke; k++) {
      
        /* c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) */

        /* Perform the multiply-accumulates */
        a_ind = i*K + k*8;
        b_ind = k*8*N + j;

        sum += A[a_ind] * B[b_ind];
        sum += A[a_ind + 1] * B[b_ind + N];
        sum += A[a_ind + 2] * B[b_ind + 2*N];
        sum += A[a_ind + 3] * B[b_ind + 3*N];
        sum += A[a_ind + 4] * B[b_ind + 4*N];
        sum += A[a_ind + 5] * B[b_ind + 5*N];
        sum += A[a_ind + 6] * B[b_ind + 6*N];
        sum += A[a_ind + 7] * B[b_ind + 7*N];

      }

      /* Loop unrolling: Compute remaining MACs */
      ke = K % 0x8U;

      a_ind = i*K + (K - ke);
      b_ind = (K - ke)*N + j;

      for(k = 0; k < ke; k++) {
        /* c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) */

        /* Perform the multiply-accumulates */
        sum += A[a_ind + k] * B[b_ind + k*N];
      }

      /* Store result in destination buffer */
      C[c_ind++] = sum;
    }
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}






// arm_status inner_1x4x1(
//   const arm_matrix_instance_f32 * pSrcA,
//   const arm_matrix_instance_f32 * pSrcB,
//         arm_matrix_instance_f32 * pDst)
// {
//   float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
//   float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
//   float32_t *A_ptr = pSrcA->pData;                /* Input data matrix pointer A */
//   float32_t *B_ptr = pSrcB->pData;                /* Input data matrix pointer B */
//   float32_t *C = pDst->pData;                 /* Output data matrix pointer */
//   float32_t *C_curr;                                 /* Temporary output data matrix pointer */
//   float32_t sum;                                 /* Accumulator */
//   uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
//   uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
//   uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
//   uint32_t i, j, k, C_ind = 0U;  /* Loop counters */
//   arm_status status;                             /* Status of matrix multiplication */

//   /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
//   /* row loop */
//   for(i = M; i > 0U; i--) {
//     /* Output pointer is set to starting address of row being processed */
//     C_curr = C + C_ind;

//     /* For every row wise process, B pointer is set to starting address of pSrcB data */
//     B = pSrcB->pData;

//     /* column loop */
//     for(j = N; j > 0U; j--) {

//       /* Set the variable sum, that acts as accumulator, to zero */
//       sum = 0.0f;

//       /* Initialize pointer A to point to starting address of column being processed */
//       A = A_ptr;


//       /* Loop unrolling: Compute 4 MACs at a time. */
//       k = K >> 2U;

//       /* matrix multiplication */
//       while (k > 0U)
//       {
//         /* c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) */

//         /* Perform the multiply-accumulates */
//         sum += *A++ * *B;
//         B += N;

//         sum += *A++ * *B;
//         B += N;

//         sum += *A++ * *B;
//         B += N;

//         sum += *A++ * *B;
//         B += N;

//         /* Decrement loop counter */
//         k--;
//       }

//       /* Loop unrolling: Compute remaining MACs */
//       k = K % 0x4U;


//       while (k > 0U)
//       {
//         /* c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) */

//         /* Perform the multiply-accumulates */
//         sum += *A++ * *B;
//         B += N;

//         /* Decrement loop counter */
//         k--;
//       }

//       /* Store result in destination buffer */
//       *C_curr++ = sum;

//       /* Update pointer B to point to starting address of next column */
//       B = B_ptr + (N - j + 1);

//     }

//     /* Update pointer A_ptr to point to starting address of next row */
//     C_ind = C_ind + N;
//     A_ptr = A_ptr + K;
//   }

//   /* Set status as ARM_MATH_SUCCESS */
//   status = ARM_MATH_SUCCESS;


//   /* Return to application */
//   return (status);
// }









arm_status inner_1x4x1(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *A_ptr = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B_ptr = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t *C_curr;                                 /* Temporary output data matrix pointer */
  float32_t sum;                                 /* Accumulator */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t j, C_ind = 0U, i = M, k;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

#ifdef ARM_MATH_MATRIX_CHECK

  /* Check for matrix mismatch condition */
  if ((pSrcA->numCols != pSrcB->numRows) ||
      (pSrcA->numRows != pDst->numRows)  ||
      (pSrcB->numCols != pDst->numCols)    )
  {
    /* Set status as ARM_MATH_SIZE_MISMATCH */
    status = ARM_MATH_SIZE_MISMATCH;
  }
  else

#endif /* #ifdef ARM_MATH_MATRIX_CHECK */

  {
    /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
    /* row loop */
    do
    {
      /* Output pointer is set to starting address of row being processed */
      C_curr = C + C_ind;

      /* For every row wise process, column loop counter is to be initiated */
      j = N;

      /* For every row wise process, B pointer is set to starting address of pSrcB data */
      B = pSrcB->pData;

      /* column loop */
      do
      {
        /* Set the variable sum, that acts as accumulator, to zero */
        sum = 0.0f;

        /* Initialize pointer A to point to starting address of column being processed */
        A = A_ptr;


        /* Loop unrolling: Compute 4 MACs at a time. */
        k = K >> 2U;

        /* matrix multiplication */
        while (k > 0U)
        {
          /* c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) */

          /* Perform the multiply-accumulates */
          sum += *A++ * *B;
          B += N;

          sum += *A++ * *B;
          B += N;

          sum += *A++ * *B;
          B += N;

          sum += *A++ * *B;
          B += N;

          /* Decrement loop counter */
          k--;
        }

        /* Loop unrolling: Compute remaining MACs */
        k = K % 0x4U;


        while (k > 0U)
        {
          /* c(m,p) = a(m,1) * b(1,p) + a(m,2) * b(2,p) + .... + a(m,n) * b(n,p) */

          /* Perform the multiply-accumulates */
          sum += *A++ * *B;
          B += N;

          /* Decrement loop counter */
          k--;
        }

        /* Store result in destination buffer */
        *C_curr++ = sum;

        /* Decrement column loop counter */
        j--;

        /* Update pointer B to point to starting address of next column */
        B = B_ptr + (N - j);

      } while (j > 0U);

      /* Update pointer A_ptr to point to starting address of next row */
      C_ind = C_ind + N;
      A_ptr = A_ptr + K;

      /* Decrement row loop counter */
      i--;

    } while (i > 0U);

    /* Set status as ARM_MATH_SUCCESS */
    status = ARM_MATH_SUCCESS;
  }

  /* Return to application */
  return (status);
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

  status = inner_1x4x1(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm inner_1x4x1 time: "); 
  Serial.println(diff); //prints time since program started



  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = inner_1x8x1(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm inner_1x8x1 time: "); 
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

  status = inner_2x4x2(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm inner_2x4x2 time: "); 
  Serial.println(diff); //prints time since program started

  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = inner_2x8x2(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm inner_2x8x2 time: "); 
  Serial.println(diff); //prints time since program started

  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
  // print_mat(&C, M, N);



  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
  
  start1 = micros();

  status = outer_2x2(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_2x2 time: "); 
  Serial.println(diff); //prints time since program started



  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
  
  start1 = micros();

  status = outer_2x2_unpacked(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_2x2_unpacked time: "); 
  Serial.println(diff); //prints time since program started




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
  
  start1 = micros();

  // status = outer_3x3(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_3x3 time: "); 
  Serial.println(diff); //prints time since program started




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
  
  start1 = micros();

  status = outer_4x4(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_4x4 time: "); 
  Serial.println(diff); //prints time since program started




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
  
  start1 = micros();

  status = outer_5x5(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_5x5 time: "); 
  Serial.println(diff); //prints time since program started




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = outer_5x5_unpacked(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_5x5_unpacked time: "); 
  Serial.println(diff); //prints time since program started

  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
  // print_mat(&C, M, N);
  




  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  start1 = micros();

  status = outer_5x5_ptr(&A, &B, &C);

  end1 = micros();
  diff = end1 - start1;
  Serial.print("sgemm outer_5x5_ptr time: "); 
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










bool f32_gemm_checker(float* C, float* C_check, int N, int M, int K) {

  int CORRECT = 1;
  int cnt = 0;
  int ind1 = 0;
  float eps = 1e-3; // machine precision level

  for(int m = 0; m < M; m++) {
      for(int n = 0; n < N; n++) {
          // if(C_check[m1*N + n1] != C[ind1]) {
          if(fabs(C_check[ind1] - C[ind1]) > eps) {
              cnt++;
              CORRECT = 0;
          }

          if(CHECK_PRINT) printf("%f\t%f\n", C_check[ind1], C[ind1]);
          ind1++; 
        }
    }

    //printf("\n\n");

  if(CORRECT) {
    Serial.println("CORRECT!");
    return 0;
  } else {
    Serial.println("WRONG!");
    // Serial.println("%d\n", cnt);
    return 1;
  }

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
