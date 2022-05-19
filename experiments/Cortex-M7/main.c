#include "stm32f767xx.h"
#include "arm_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void rand_init(float* mat, int r, int c) {
    // int MAX = 65536;
	printf("%d %d\n\n",r,c);
    for(int i = 0; i < r*c; i++) {
        // mat[i] = (double) i;
        // mat[i] = 1.0;
        // mat[i] =  (double) (i%MAX);
        mat[i] =  (float) rand() / RAND_MAX*2.0 - 1.0;
    }
}

void rand_init_q15(int16_t* mat, int r, int c) {
    // int MAX = 65536;
  // char buffer[100];
    for(int i = 0; i < r*c; i++) {
        // mat[i] = (double) i;
        // mat[i] = 1.0;
        // mat[i] =  (double) (i%MAX);
      // sprintf(buffer, "%d ",((rand() % 255) - 128));
      // Serial.print(buffer);
        mat[i] =  ((int16_t) (((rand() % 255) - 128))) << 4;
        // mat[i] =  ((int16_t) (((rand() % 255) - 128)));

        // mat[i] =  ((int16_t) (((rand() % 255) - 128) )) & 0x0000FFFF;

        // int x = ((rand() % RAND_MAX ) - (1  << 30)) & 0x0000FFFF;
        // int16_t a = (x  >> 24) & 0x0000FFFF;
        // mat[i] = a;


    }
}




void print_mat_q15(arm_matrix_instance_q15* mat, int rows, int cols) {


  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      printf("%hi ", mat->pData[i*cols + j]);
    }
    printf("\n");
  }
  printf("\n");

}


arm_status arm_q15_inner_2x4x2(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t                   * pState)
{
        q31_t sum;                                     /* Accumulator */
        q15_t *pSrcBT = pState;                        /* Input data matrix pointer for transpose */
        q15_t *pInA = pSrcA->pData;                    /* Input data matrix pointer A of Q15 type */
        q15_t *pInB = pSrcB->pData;                    /* Input data matrix pointer B of Q15 type */
        q15_t *px;                                     /* Temporary output data matrix pointer */
        uint16_t numRowsA = pSrcA->numRows;            /* Number of rows of input matrix A */
        uint16_t numColsB = pSrcB->numCols;            /* Number of columns of input matrix B */
        uint16_t numColsA = pSrcA->numCols;            /* Number of columns of input matrix A */
        uint16_t numRowsB = pSrcB->numRows;            /* Number of rows of input matrix B */
        uint32_t col, i = 0U, row = numRowsB, colCnt;  /* Loop counters */
        arm_status status;                             /* Status of matrix multiplication */

        q31_t in;                                      /* Temporary variable to hold the input value */
        q31_t inA1, inB1, inA2, inB2;
        q31_t sum2, sum3, sum4;
        q15_t *pInA2, *pInB2, *px2;
        uint32_t j = 0;


  {
    /* Matrix transpose */
    do
    {
      /* The pointer px is set to starting address of column being processed */
      px = pSrcBT + i;

      /* Apply loop unrolling and exchange columns with row elements */
      col = numColsB >> 2U;

      /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.
       ** a second loop below computes the remaining 1 to 3 samples. */
      while (col > 0U)
      {

        /* Read two elements from row */
        in = *__SIMD32(pInB)++;

        /* Unpack and store one element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
        *px = (q15_t) in;
#else
        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

        /* Update pointer px to point to next row of transposed matrix */
        px += numRowsB;

        /* Unpack and store second element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);
#else
        *px = (q15_t) in;
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

        /* Update pointer px to point to next row of transposed matrix */
        px += numRowsB;

        in = *__SIMD32(pInB)++;
#ifndef ARM_MATH_BIG_ENDIAN
        *px = (q15_t) in;
#else
        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */
        px += numRowsB;

#ifndef ARM_MATH_BIG_ENDIAN
        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);
#else
        *px = (q15_t) in;
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */
        px += numRowsB;

        /* Decrement column loop counter */
        col--;
      }

      /* If the columns of pSrcB is not a multiple of 4, compute any remaining output samples here.
       ** No loop unrolling is used. */
      col = numColsB % 0x4U;

      while (col > 0U)
      {
        /* Read and store input element in destination */
        *px = *pInB++;

        /* Update pointer px to point to next row of transposed matrix */
        px += numRowsB;

        /* Decrement column loop counter */
        col--;
      }

      i++;

      /* Decrement row loop counter */
      row--;

    } while (row > 0U);

    /* Reset variables for usage in following multiplication process */
    row = numRowsA;
    i = 0U;
    px = pDst->pData;

    /* Process two rows from matrix A at a time and output two rows at a time */
    row = row >> 1U;
    px2 = px + numColsB;

    /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
    /* row loop */
    while (row > 0U)
    {
      /* For every row wise process, column loop counter is to be initiated */
      col = numColsB;

      /* For every row wise process, pIn2 pointer is set to starting address of transposed pSrcB data */
      pInB = pSrcBT;

      /* Process two (transposed) columns from matrix B at a time */
      col = col >> 1U;
      j = 0;

      /* column loop */
      while (col > 0U)
      {
        /* Set variable sum, that acts as accumulator, to zero */
        sum = 0;

        /* Initiate pointer pInA to point to starting address of column being processed */
        pInA = pSrcA->pData + i;

        sum2 = 0;
        sum3 = 0;
        sum4 = 0;
        pInB  = pSrcBT + j;
        pInA2 = pInA + numColsA;
        pInB2 = pInB + numRowsB;

        /* Read in two elements at once - allows dual MAC instruction */
        colCnt = numColsA >> 2U;


        /* matrix multiplication */
        while (colCnt > 0U)
        {
          /* c(m,n) = a(1,1) * b(1,1) + a(1,2) * b(2,1) + .... + a(m,p) * b(p,n) */

          /* read real and imag values from pSrcA and pSrcB buffer */
          inA1 = *__SIMD32(pInA)++;
          inB1 = *__SIMD32(pInB)++;

          inA2 = *__SIMD32(pInA2)++;
          inB2 = *__SIMD32(pInB2)++;

          /* Multiply and Accumulates */
          sum  = __SMLAD(inA1, inB1, sum);
          sum2 = __SMLAD(inA1, inB2, sum2);
          sum3 = __SMLAD(inA2, inB1, sum3);
          sum4 = __SMLAD(inA2, inB2, sum4);



          inA1 = *__SIMD32(pInA)++;
          inB1 = *__SIMD32(pInB)++;

          inA2 = *__SIMD32(pInA2)++;
          inB2 = *__SIMD32(pInB2)++;

          /* Multiply and Accumulates */
          sum  = __SMLAD(inA1, inB1, sum);
          sum2 = __SMLAD(inA1, inB2, sum2);
          sum3 = __SMLAD(inA2, inB1, sum3);
          sum4 = __SMLAD(inA2, inB2, sum4);

          /* Decrement loop counter */
          colCnt--;
        }

        /* process odd column samples */
        if (numColsA & 1U) {
          inA1 = *pInA++;
          inB1 = *pInB++;
          inA2 = *pInA2++;
          inB2 = *pInB2++;
          sum  += inA1 * inB1;
          sum2 += inA1 * inB2;
          sum3 += inA2 * inB1;
          sum4 += inA2 * inB2;
        }

        /* Saturate and store result in destination buffer */
        *px++  = (q15_t) (sum >> 15);

        *px++  = (q15_t) (sum2 >> 15);
        *px2++ = (q15_t) (sum3 >> 15);
        *px2++ = (q15_t) (sum4 >> 15);
        j += numRowsB * 2;

        /* Decrement column loop counter */
        col--;

      }

      i = i + numColsA;

      i = i + numColsA;
      px = px2 + (numColsB & 1U);
      px2 = px + numColsB;

      /* Decrement row loop counter */
      row--;

    }

    /* Compute any remaining odd row/column below */


    /* Compute remaining output column */
    if (numColsB & 1U) {

      /* Avoid redundant computation of last element */
      row = numRowsA & (~0x1);

      /* Point to remaining unfilled column in output matrix */
      px = pDst->pData + numColsB-1;
      pInA = pSrcA->pData;

      /* row loop */
      while (row > 0)
      {

        /* point to last column in matrix B */
        pInB  = pSrcBT + numRowsB * (numColsB-1);

        /* Set variable sum, that acts as accumulator, to zero */
        sum  = 0;

        /* Compute 4 columns at once */
        colCnt = numColsA >> 2U;

        /* matrix multiplication */
        while (colCnt > 0U)
        {
          inA1 = *__SIMD32(pInA)++;
          inA2 = *__SIMD32(pInA)++;
          inB1 = *__SIMD32(pInB)++;
          inB2 = *__SIMD32(pInB)++;

          sum  = __SMLAD(inA1, inB1, sum);
          sum  = __SMLAD(inA2, inB2, sum);

          /* Decrement loop counter */
          colCnt--;
        }

        colCnt = numColsA & 3U;
        while (colCnt > 0U) {
          sum += (q31_t) (*pInA++) * (*pInB++);
          colCnt--;
        }

        /* Store result in destination buffer */
        *px = (q15_t) (sum  >> 15);
        px += numColsB;

        /* Decrement row loop counter */
        row--;
      }
    }

    /* Compute remaining output row */
    if (numRowsA & 1U) {

      /* point to last row in output matrix */
      px = pDst->pData + (numColsB) * (numRowsA-1);

      pInB  = pSrcBT;
      col = numColsB;
      i = 0U;

      /* col loop */
      while (col > 0)
      {
        /* point to last row in matrix A */
        pInA = pSrcA->pData + (numRowsA-1) * numColsA;

        /* Set variable sum, that acts as accumulator, to zero */
        sum  = 0;

        /* Compute 4 columns at once */
        colCnt = numColsA >> 2U;

        /* matrix multiplication */
        while (colCnt > 0U)
        {
          inA1 = *__SIMD32(pInA)++;
          inA2 = *__SIMD32(pInA)++;
          inB1 = *__SIMD32(pInB)++;
          inB2 = *__SIMD32(pInB)++;

          sum  = __SMLAD(inA1, inB1, sum);
          sum  = __SMLAD(inA2, inB2, sum);

          /* Decrement loop counter */
          colCnt--;
        }

        colCnt = numColsA % 4U;
        while (colCnt > 0U) {
          sum += (q31_t) (*pInA++) * (*pInB++);

          colCnt--;
        }

        /* Store result in destination buffer */
        *px++ = (q15_t) (sum  >> 15);

        /* Decrement column loop counter */
        col--;
      }
    }


    /* Set status as ARM_MATH_SUCCESS */
    status = ARM_MATH_SUCCESS;
  }

  /* Return to application */
  return (status);
}








arm_status inner_fp32_2x8x2(
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

      /* Loop unrolling: Compute remaining MACs */
      k = K % 8U;


      while (k > 0U)
      {

        /* Perform the multiply-accumulates */

        sum1 += *A1 * *B1;
        sum2 += *A1++ * *B2;
        sum3 += *A2 * *B1;
        sum4 += *A2++ * *B2;

        B1 += N;
        B2 += N;

        /* Decrement loop counter */
        k--;
      }

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






arm_status outer_fp32_5x5_k_first(
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
  C34, C40, C41, C42, C43, C44;     /* Temporary output data  */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k, C_ind = 0U;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  float32_t *A_ptr = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B_ptr, *C_curr;

  uint32_t en = N / 5;
  uint32_t em = M / 5;
  uint32_t n_left = N % 5;
  uint32_t m_left = M % 5;

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


    if(n_left) {

      A_ptr = pSrcA->pData + m*5*K;
      B_ptr = pSrcB->pData + 5*en;

      // float32_t c_tmp[5*n_left];

      // for(int q = 0; q < 5*n_left; q++) {
      //   c_tmp[q] = 0;
      // }

      // C_curr = pDst->pData + 5*m*N + 5*en;
      int tmp_ind = 5*m*N + 5*en;

      for(int kk = 0U; kk < K; kk++) {

        A = A_ptr + kk ;
        B = B_ptr + kk*N;
        C_curr = pDst->pData + tmp_ind;

        for(int mm = 0U; mm < 5; mm++) {

          for(int nn = 0U; nn < n_left-1; nn++) {
            // c_tmp[mm*n_left + nn] += *A * *B++;
            *C_curr++ += *A * *B++;
          }

          // c_tmp[mm*n_left + n_left-1] += *A * *B;
          *C_curr += *A * *B;

          B -= (n_left-1);
          A += K;
          C_curr += N - (n_left-1);
        }

        // A++;
        // B += N;
      }

      // for(int mm = 0U; mm < 5; mm++) {

      //   for(int nn = 0U; nn < n_left-1; nn++) {
      //     *C_curr++ += c_tmp[mm*n_left + nn];
      //   }

      //   *C_curr += c_tmp[mm*n_left + n_left-1];
      //   C_curr += N - (n_left-1);
      // }
    }

    /* Update pointer A_ptr to point to starting address of next row */
    A_ptr += 5*K;

  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}







arm_status outer_q15_4x2(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t                   * pState)
{
        q15_t *pSrcBT = pState;                        /* Input data matrix pointer for transpose */
        q15_t *pInA = pSrcA->pData;                    /* Input data matrix pointer A of Q15 type */
        q15_t *pInB = pSrcB->pData;                    /* Input data matrix pointer B of Q15 type */
        q15_t *px;                                     /* Temporary output data matrix pointer */
        uint16_t numRowsA = pSrcA->numRows;            /* Number of rows of input matrix A */
        uint16_t numColsB = pSrcB->numCols;            /* Number of columns of input matrix B */
        uint16_t numColsA = pSrcA->numCols;            /* Number of columns of input matrix A */
        uint16_t numRowsB = pSrcB->numRows;            /* Number of rows of input matrix B */
        uint32_t col, i = 0U, row = numRowsB, colCnt;  /* Loop counters */
        arm_status status;                             /* Status of matrix multiplication */

        q31_t in;                                      /* Temporary variable to hold the input value */
        q31_t inA1, inB1, inA2, inB2, inA3, inA4;
        q31_t sum, sum2, sum3, sum4, sum5, sum6, sum7, sum8;                         /* Accumulator */
        q15_t *pInA2, *pInA3, *pInA4, *pInB2, *px2, *px3, *px4;
        uint32_t j = 0;


  {
    /* Matrix transpose */
    do
    {
      /* Apply loop unrolling and exchange the columns with row elements */
      col = numColsB >> 2;

      /* The pointer px is set to starting address of the column being processed */
      px = pSrcBT + i;

      /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.
       ** a second loop below computes the remaining 1 to 3 samples. */
      while (col > 0U)
      {
#ifndef UNALIGNED_SUPPORT_DISABLE
        /* Read two elements from the row */
        in = *__SIMD32(pInB)++;

        /* Unpack and store one element in the destination */
#ifndef ARM_MATH_BIG_ENDIAN

        *px = (q15_t) in;

#else

        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);

#endif /*    #ifndef ARM_MATH_BIG_ENDIAN    */

        /* Update the pointer px to point to the next row of the transposed matrix */
        px += numRowsB;

        /* Unpack and store the second element in the destination */
#ifndef ARM_MATH_BIG_ENDIAN

        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);

#else

        *px = (q15_t) in;

#endif /*    #ifndef ARM_MATH_BIG_ENDIAN    */

        /* Update the pointer px to point to the next row of the transposed matrix */
        px += numRowsB;

        /* Read two elements from the row */
        in = *__SIMD32(pInB)++;

        /* Unpack and store one element in the destination */
#ifndef ARM_MATH_BIG_ENDIAN

        *px = (q15_t) in;

#else

        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);

#endif /*    #ifndef ARM_MATH_BIG_ENDIAN    */

        /* Update the pointer px to point to the next row of the transposed matrix */
        px += numRowsB;

        /* Unpack and store the second element in the destination */

#ifndef ARM_MATH_BIG_ENDIAN

        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);

#else

        *px = (q15_t) in;

#endif /*    #ifndef ARM_MATH_BIG_ENDIAN    */

#else

        /* Read one element from the row */
        in = *pInB++;

        /* Store one element in the destination */
        *px = in;

        /* Update the pointer px to point to the next row of the transposed matrix */
        px += numRowsB;

        /* Read one element from the row */
        in = *pInB++;

        /* Store one element in the destination */
        *px = in;

        /* Update the pointer px to point to the next row of the transposed matrix */
        px += numRowsB;

        /* Read one element from the row */
        in = *pInB++;

        /* Store one element in the destination */
        *px = in;

        /* Update the pointer px to point to the next row of the transposed matrix */
        px += numRowsB;

        /* Read one element from the row */
        in = *pInB++;

        /* Store one element in the destination */
        *px = in;

#endif /* #ifndef UNALIGNED_SUPPORT_DISABLE */

        /* Update the pointer px to point to the next row of the transposed matrix */
        px += numRowsB;

        /* Decrement the column loop counter */
        col--;
      }

      /* If the columns of pSrcB is not a multiple of 4, compute any remaining output samples here.
       ** No loop unrolling is used. */
      col = numColsB % 0x4U;

      while (col > 0U)
      {
        /* Read and store the input element in the destination */
        *px = *pInB++;

        /* Update the pointer px to point to the next row of the transposed matrix */
        px += numRowsB;

        /* Decrement the column loop counter */
        col--;
      }

      i++;

      /* Decrement the row loop counter */
      row--;

    } while (row > 0U);

    /* Reset variables for usage in following multiplication process */
    row = numRowsA;
    i = 0U;

    /* Process four rows from matrix A at a time and output four rows at a time */
    row = row >> 2U;

    px = pDst->pData;
    px2 = px + numColsB;
    px3 = px2 + numColsB;
    px4 = px3 + numColsB;

    /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
    /* row loop */
    while (row > 0U)
    {
      /* For every row wise process, column loop counter is to be initiated */
      col = numColsB;

      /* For every row wise process, pIn2 pointer is set to starting address of transposed pSrcB data */
      pInB = pSrcBT;

      /* Process two (transposed) columns from matrix B at a time */
      col = col >> 1U;
      j = 0;

      /* column loop */
      while (col > 0U)
      {
        /* Set variable sum, that acts as accumulator, to zero */
        sum = 0;
        sum2 = 0;
        sum3 = 0;
        sum4 = 0;
        sum5 = 0;
        sum6 = 0;
        sum7 = 0;
        sum8 = 0;

        /* Initiate pointer pInA to point to starting address of column being processed */
        pInA = pSrcA->pData + i;
        pInA2 = pInA + numColsA;
        pInA3 = pInA2 + numColsA;
        pInA4 = pInA3 + numColsA;

        pInB  = pSrcBT + j;
        pInB2 = pInB + numRowsB;

        /* Read in two elements at once - allows dual MAC instruction */
        colCnt = numColsA >> 1U;


        /* matrix multiplication */
        while (colCnt > 0U)
        {
          /* c(m,n) = a(1,1) * b(1,1) + a(1,2) * b(2,1) + .... + a(m,p) * b(p,n) */

          /* read real and imag values from pSrcA and pSrcB buffer */
          inA1 = *__SIMD32(pInA)++;
          inB1 = *__SIMD32(pInB)++;

          inA2 = *__SIMD32(pInA2)++;
          inB2 = *__SIMD32(pInB2)++;


          inA3 = *__SIMD32(pInA3)++;
          inA4 = *__SIMD32(pInA4)++;
          /* Multiply and Accumulates */
          sum  = __SMLAD(inA1, inB1, sum);
          sum2 = __SMLAD(inA1, inB2, sum2);
          sum3 = __SMLAD(inA2, inB1, sum3);
          sum4 = __SMLAD(inA2, inB2, sum4);



          /* Multiply and Accumulates */
          sum5  = __SMLAD(inA3, inB1, sum5);
          sum6 = __SMLAD(inA3, inB2, sum6);
          sum7 = __SMLAD(inA4, inB1, sum7);
          sum8 = __SMLAD(inA4, inB2, sum8);

          /* Decrement loop counter */
          colCnt--;
        }

        /* process odd column samples */
        if (numColsA & 1U) {
          inA1 = *pInA++;
          inB1 = *pInB++;
          inA2 = *pInA2++;
          inB2 = *pInB2++;

          inA3 = *pInA3++;
          inA4 = *pInA4++;

          sum  += inA1 * inB1;
          sum2 += inA1 * inB2;
          sum3 += inA2 * inB1;
          sum4 += inA2 * inB2;

          sum5 += inA3 * inB1;
          sum6 += inA3 * inB2;
          sum7 += inA4 * inB1;
          sum8 += inA4 * inB2;

        }

        /* Saturate and store result in destination buffer */
        *px++  = (q15_t) (sum >> 15);
        *px++  = (q15_t) (sum2 >> 15);
        *px2++ = (q15_t) (sum3 >> 15);
        *px2++ = (q15_t) (sum4 >> 15);

        *px3++ = (q15_t) (sum5 >> 15);
        *px3++ = (q15_t) (sum6 >> 15);
        *px4++ = (q15_t) (sum7 >> 15);
        *px4++ = (q15_t) (sum8 >> 15);

        j += numRowsB * 2;

        /* Decrement column loop counter */
        col--;

      }

      i = i + 4*numColsA;
      px = px4 + (numColsB & 1U);
      px2 = px + numColsB;
      px3 = px2 + numColsB;
      px4 = px3 + numColsB;

      /* Decrement row loop counter */
      row--;

    }

    /* Compute any remaining odd row/column below */


    /* Compute remaining output column */
    if (numColsB & 1U) {

      /* Avoid redundant computation of last element */
      row = numRowsA & (~0x1);

      /* Point to remaining unfilled column in output matrix */
      px = pDst->pData + numColsB-1;
      pInA = pSrcA->pData;

      /* row loop */
      while (row > 0)
      {

        /* point to last column in matrix B */
        pInB  = pSrcBT + numRowsB * (numColsB-1);

        /* Set variable sum, that acts as accumulator, to zero */
        sum  = 0;

        /* Compute 4 columns at once */
        colCnt = numColsA >> 2U;

        /* matrix multiplication */
        while (colCnt > 0U)
        {
          inA1 = *__SIMD32(pInA)++;
          inA2 = *__SIMD32(pInA)++;
          inB1 = *__SIMD32(pInB)++;
          inB2 = *__SIMD32(pInB)++;

          sum  = __SMLAD(inA1, inB1, sum);
          sum  = __SMLAD(inA2, inB2, sum);

          /* Decrement loop counter */
          colCnt--;
        }

        colCnt = numColsA & 3U;
        while (colCnt > 0U) {
          sum += (q31_t) (*pInA++) * (*pInB++);
          colCnt--;
        }

        /* Store result in destination buffer */
        *px = (q15_t) (sum  >> 15);
        px += numColsB;

        /* Decrement row loop counter */
        row--;
      }
    }

    /* Compute remaining output row */
    if (numRowsA & 1U) {

      /* point to last row in output matrix */
      px = pDst->pData + (numColsB) * (numRowsA-1);

      pInB  = pSrcBT;
      col = numColsB;
      i = 0U;

      /* col loop */
      while (col > 0)
      {
        /* point to last row in matrix A */
        pInA = pSrcA->pData + (numRowsA-1) * numColsA;

        /* Set variable sum, that acts as accumulator, to zero */
        sum  = 0;

        /* Compute 4 columns at once */
        colCnt = numColsA >> 2U;

        /* matrix multiplication */
        while (colCnt > 0U)
        {
          inA1 = *__SIMD32(pInA)++;
          inA2 = *__SIMD32(pInA)++;
          inB1 = *__SIMD32(pInB)++;
          inB2 = *__SIMD32(pInB)++;

          sum  = __SMLAD(inA1, inB1, sum);
          sum  = __SMLAD(inA2, inB2, sum);

          /* Decrement loop counter */
          colCnt--;
        }

        colCnt = numColsA % 4U;
        while (colCnt > 0U) {
          sum += (q31_t) (*pInA++) * (*pInB++);

          colCnt--;
        }

        /* Store result in destination buffer */
        *px++ = (q15_t) (sum  >> 15);

        /* Decrement column loop counter */
        col--;
      }
    }


    /* Set status as ARM_MATH_SUCCESS */
    status = ARM_MATH_SUCCESS;
  }

  /* Return to application */
  return (status);
}









int f32_gemm_checker(float* C, float* C_check, int N, int M, int K) {

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

//          if(CHECK_PRINT) printf("%f\t%f\n", C_check[ind1], C[ind1]);
          ind1++;
        }
    }

    //printf("\n\n");

  if(CORRECT) {
    printf("CORRECT!\n");
    return 0;
  } else {
	  printf("WRONG!\n");
    // Serial.println("%d\n", cnt);
    return 1;
  }

}




void print_mat(arm_matrix_instance_f32* mat, int rows, int cols) {


  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      printf("%f ", mat->pData[i*cols + j]);
    }
    printf("\n");
  }
  printf("\n");


}



void testing() {

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

  printf("M = %d, K = %d, N = %d", M, K, N);

  // gettimeofday (&start, NULL);
  srand(time(NULL));
  rand_init(A_f32, M, K);
  rand_init(B_f32, K, N);

  arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);
  arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);
  arm_mat_init_f32(&C_ref, M, N, (float32_t *) C_f32_ref);

  unsigned int x, y;
  volatile unsigned int *DWT_CYCCNT  = (volatile unsigned int *)0xE0001004; //address of the register
  volatile unsigned int *DWT_CONTROL = (volatile unsigned int *)0xE0001000; //address of the register
  volatile unsigned int *DWT_LAR   = (volatile unsigned int *)0xE0001FB0; //address of the register
  volatile unsigned int *SCB_DEMCR  = (volatile unsigned int *)0xE000EDFC; //address of the register


   *SCB_DEMCR |= 0x01000000;
   *DWT_LAR = 0xC5ACCE55; // unlock
   *DWT_CYCCNT = 0; // reset the counter
   *DWT_CONTROL |= 1 ; // enable the counter

   x = *DWT_CYCCNT;

   status = arm_mat_mult_f32(&A, &B, &C_ref);

   y = *DWT_CYCCNT;
   x = (y - x); // Elapsed clock ticks, at SystemCoreClock
//   print_mat(&C_ref, M, N);
   printf("arm sgemm time: ");
  printf("%d\n", x); //prints time since program started


//
//  float* tmp_arr = (float *) malloc( 1000*sizeof( float ));
//  for(int i = 0; i < 1000; i++) {
//    tmp_arr[i] = (float) rand();
//  }
//
//
//
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

  *DWT_CYCCNT = 0; // reset the counter


  x = *DWT_CYCCNT;

  status = outer_fp32_5x5_k_first(&A, &B, &C);

  y = *DWT_CYCCNT;
  x = (y - x); // Elapsed clock ticks, at SystemCoreClock
 printf("outer 5x5 time: ");
 printf("%d\n", x); //prints time since program started



//   print_mat(&C, M, N);
  // print_mat(&C_ref, M, N);
  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
//
//
//
//
//  free(C_f32);
//  C_f32 = (float *) calloc( M*N, sizeof( float ));
//  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
//
//  start1 = micros();
//
//  // status = outer_fp32_5x5_k_first_old(&A, &B, &C);
//
//  end1 = micros();
//  diff = end1 - start1;
//  Serial.print("sgemm outer_fp32_5x5_k_first old time: ");
//  Serial.println(diff); //prints time since program started
//  // print_mat(&C, N, M);
//  // print_mat(&C_ref, M, N);
//  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
//
//
//  free(C_f32);
//  C_f32 = (float *) calloc( M*N, sizeof( float ));
//  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
//
//  start1 = micros();
//
//  status = outer_fp32_5x5_n_first(&A, &B, &C);
//
//  end1 = micros();
//  diff = end1 - start1;
//  Serial.print("sgemm outer_fp32_5x5_n_first time: ");
//  Serial.println(diff); //prints time since program started
//  // print_mat(&C, N, M);
//  // print_mat(&C_ref, M, N);
//  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
//
//
//
//
//
//  // free(C_f32);
//  // C_f32 = (float *) calloc( M*N, sizeof( float ));
//  // arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
//
//  // start1 = micros();
//
//  // status = outer_fp32_1x6x4_m_first(&A, &B, &C);
//
//  // end1 = micros();
//  // diff = end1 - start1;
//  // Serial.print("sgemm outer_fp32_1x6x4_m_first time: ");
//  // Serial.println(diff); //prints time since program started
//  // // print_mat(&C, N, M);
//  // // print_mat(&C_ref, M, N);
//  // f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
//
//
  free(C_f32);
  C_f32 = (float *) calloc( M*N, sizeof( float ));
  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);


  *DWT_CYCCNT = 0; // reset the counter
  x = *DWT_CYCCNT;

  status = inner_fp32_2x8x2(&A, &B, &C);

  y = *DWT_CYCCNT;
  x = (y - x); // Elapsed clock ticks, at SystemCoreClock


  printf("sgemm inner_fp32_2x8x2 time: \n");
  printf("%d\n",x); //prints time since program started
  // print_mat(&C, M, N);
  // print_mat(&C_ref, M, N);
//  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);


//
//
//
//  free(C_f32);
//  C_f32 = (float *) calloc( M*N, sizeof( float ));
//  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
//
//  start1 = micros();
//
//  status = outer_fp32_5x5_m_first(&A, &B, &C);
//
//  end1 = micros();
//  diff = end1 - start1;
//  Serial.print("sgemm outer_fp32_5x5_m_first time: ");
//  Serial.println(diff); //prints time since program started
//  // print_mat(&C, M, N);
//  // print_mat(&C_ref, M, N);
//  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
//
//
//
//
//
//
//  free(C_f32);
//  C_f32 = (float *) calloc( M*N, sizeof( float ));
//  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
//
//  start1 = micros();
//
//  status = outer_fp32_5x5_m_first_test(&A, &B, &C, tmp_arr, 1);
//
//  end1 = micros();
//  diff = end1 - start1;
//  Serial.print("sgemm outer_fp32_5x5_m_first test time: ");
//  Serial.println(diff); //prints time since program started
//
//  // print_mat(&A, M, K);
//  // print_mat(&B, K, N);
//
//
//  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
//
//
//
//  free(C_f32);
//  C_f32 = (float *) calloc( M*N, sizeof( float ));
//  arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);
//
//  start1 = micros();
//
//  status = outer_fp32_5x5_test(&A, &B, &C, tmp_arr, 1);
//
//  end1 = micros();
//  diff = end1 - start1;
//  Serial.print("sgemm outer_fp32_5x5_k_first test time: ");
//  Serial.println(diff); //prints time since program started
//  // print_mat(&C, M, N);
//  // print_mat(&C_ref, M, N);
//  f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
//
//
//
//

  free(A_f32);
  free(B_f32);
  free(C_f32);
  free(C_f32_ref);
}






void testing_q15() {
   // allocate enough room for every thread's stack statistics

  arm_matrix_instance_q15 A;      /* Matrix A Instance */
  arm_matrix_instance_q15 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_q15 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_q15 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  unsigned long start1, end1, diff;
  int16_t *A_16, *B_16, *C_16, *C_16_ref, *B_trans;
  // uint32_t M = 400, N = 27, K = 20;
  uint32_t M = 100, N = 100, K = 100;

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
  arm_mat_init_q15(&B, K, N, (q15_t *) B_16);
  arm_mat_init_q15(&C_ref, M, N, (q15_t *) C_16_ref);



  unsigned int x, y;
   volatile unsigned int *DWT_CYCCNT  = (volatile unsigned int *)0xE0001004; //address of the register
   volatile unsigned int *DWT_CONTROL = (volatile unsigned int *)0xE0001000; //address of the register
   volatile unsigned int *DWT_LAR   = (volatile unsigned int *)0xE0001FB0; //address of the register
   volatile unsigned int *SCB_DEMCR  = (volatile unsigned int *)0xE000EDFC; //address of the register


    *SCB_DEMCR |= 0x01000000;
    *DWT_LAR = 0xC5ACCE55; // unlock
    *DWT_CYCCNT = 0; // reset the counter
    *DWT_CONTROL |= 1 ; // enable the counter

    x = *DWT_CYCCNT;

    status = arm_mat_mult_fast_q15(&A, &B, &C_ref, B_trans);

    y = *DWT_CYCCNT;
    x = (y - x); // Elapsed clock ticks, at SystemCoreClock

  printf("arm_mat_mult_fast_q15 time: \n");
  printf("%d\n",x); //prints time since program started

//   print_mat_q15(&C_ref, M, N);



  free(B_trans);

  C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
  arm_mat_init_q15(&C, M, N, (q15_t *) C_16);

  B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));

  printf("\n");

  *DWT_CYCCNT = 0;

  x = *DWT_CYCCNT;

  status = arm_q15_inner_2x4x2(&A, &B, &C, (q15_t *) B_trans);

  y = *DWT_CYCCNT;
  x = (y - x); // Elapsed clock ticks, at SystemCoreClock
  printf("arm_q15_inner_2x4x2 time: \n");
  printf("%d\n",x); //prints time since program started
  // print_mat_q15(&C, M, N);
  // q15_gemm_checker(C_16, C_ref.pData, N, M, K);







  free(B_trans);
  free(C_16);

  C_16 = (int16_t *) calloc( M*N, sizeof( int16_t ));
  arm_mat_init_q15(&C, M, N, (q15_t *) C_16);

  B_trans = (int16_t *) malloc( K*N*sizeof( int16_t ));

  printf("\n");

   *DWT_CYCCNT = 0;

   x = *DWT_CYCCNT;

   status = outer_q15_4x2(&A, &B, &C, (q15_t *) B_trans);

   y = *DWT_CYCCNT;
   x = (y - x); // Elapsed clock ticks, at SystemCoreClock
   printf("outer_q15_4x2 time: \n");
   printf("%d\n",x); //prints time since program started

//   print_mat_q15(&C, M, N);
  // q15_gemm_checker(C_16, C_ref.pData, N, M, K);


  free(A_16);
  free(B_16);
  free(C_16);
  free(C_16_ref);
  free(B_trans);
}








void tiny_ml_benchmark() {


  arm_matrix_instance_f32 A;      /* Matrix A Instance */
  arm_matrix_instance_f32 B;     /* Matrix B(A transpose) instance */
  arm_matrix_instance_f32 C;   /* Matrix C( B multiply with A) instance */
  arm_matrix_instance_f32 C_ref;   /* Matrix C( B multiply with A) instance */

  arm_status status;

  unsigned int x, y;
  volatile unsigned int *DWT_CYCCNT  = (volatile unsigned int *)0xE0001004; //address of the register
  volatile unsigned int *DWT_CONTROL = (volatile unsigned int *)0xE0001000; //address of the register
  volatile unsigned int *DWT_LAR   = (volatile unsigned int *)0xE0001FB0; //address of the register
  volatile unsigned int *SCB_DEMCR  = (volatile unsigned int *)0xE000EDFC; //address of the register


  *SCB_DEMCR |= 0x01000000;
  *DWT_LAR = 0xC5ACCE55; // unlock
  *DWT_CONTROL |= 1 ; // enable the counter  float *A_f32, *B_f32, *C_f32, *C_f32;

  float *A_f32, *B_f32, *C_f32;
  uint32_t M , N , K ;

  int Ms[] = {16,16,32,32,32,64,64,64, 64, 64, 8 ,16 ,32 ,32 ,64 ,64 ,128,128,256,256};
  int Ks[] = {27 ,144,144,288,16 ,288,576,32 , 40, 64, 27 , 8 , 16 , 32 , 32 , 64 , 64 , 128, 128, 256};
  int Ns[] = {1024,1024,256,256,256,64,64,64, 122, 125, 2304,2304,576,576,144,144, 36, 36, 9, 9};

  printf("\nid,M,N,K,algo,time\n");

  for(int i = 1; i < 20; i++) {

    M = Ms[i] + 10 - (Ms[i] % 10);
    K = Ks[i];
    N = Ns[i] + 10 - (Ns[i] % 10);

    A_f32 = (float *) malloc( M*K*sizeof( float ));
    B_f32 = (float *) malloc( K*N*sizeof( float ));
    C_f32 = (float *) malloc( M*N*sizeof( float ));

    srand(time(NULL));
    rand_init(A_f32, M, K);
    rand_init(B_f32, K, N);

    arm_mat_init_f32(&A, M, K, (float32_t *) A_f32);
    arm_mat_init_f32(&B, K, N, (float32_t *) B_f32);
    arm_mat_init_f32(&C_ref, M, N, (float32_t *) C_f32);


    *DWT_CYCCNT = 0; // reset the counter
    x = *DWT_CYCCNT;

    status = arm_mat_mult_f32(&A, &B, &C_ref);

    y = *DWT_CYCCNT;
    x = (y - x); // Elapsed clock ticks, at SystemCoreClock
    printf("%d,%d,%d,%d,arm_mat_mult_f32,%d\n", i, Ms[i],Ns[i],Ks[i],x);
     //prints time since program started




    free(C_f32);
    C_f32 = (float *) calloc( M*N, sizeof( float ));
    arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

    *DWT_CYCCNT = 0; // reset the counter
    x = *DWT_CYCCNT;

    status = outer_fp32_5x5_k_first(&A, &B, &C);

    y = *DWT_CYCCNT;
    x = (y - x); // Elapsed clock ticks, at SystemCoreClock
    printf("%d,%d,%d,%d,mema outer 5x5,%d\n", i, Ms[i],Ns[i],Ks[i],x);
     //prints time since program started
// f32_gemm_checker(C.pData, C_ref.pData, N, M, K);





    free(C_f32);
    C_f32 = (float *) calloc( M*N, sizeof( float ));
    arm_mat_init_f32(&C, M, N, (float32_t *) C_f32);

    *DWT_CYCCNT = 0; // reset the counter
    x = *DWT_CYCCNT;

    status = inner_fp32_2x8x2(&A, &B, &C);

    y = *DWT_CYCCNT;
    x = (y - x); // Elapsed clock ticks, at SystemCoreClock
    printf("%d,%d,%d,%d,arm inner 2x8x2,%d\n", i, Ms[i],Ns[i],Ks[i],x);
     //prints time since program started
// f32_gemm_checker(C.pData, C_ref.pData, N, M, K);
    free(A_f32);
    free(B_f32);
    free(C_f32);
  }
}










int _write(int file, char *ptr, int len)
{
  /* Implement your write code here, this is used by puts and printf for example */
  int i=0;
  for(i=0 ; i<len ; i++)
    ITM_SendChar((*ptr++));
  return len;
}

int main(void) {

	while(1) {
		printf("HEY MANNNN\n");
		printf("HEY MANNNN\n");
		printf("HEY MANNNN\n");
		printf("HEY MANNNN\n");
//		testing_q15();
		tiny_ml_benchmark();
		while(1) {

		}

	}
}
