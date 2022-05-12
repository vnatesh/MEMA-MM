#include "kernels.h"



arm_status outer_fp32_5x5_packed(
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



arm_status outer_fp32_4x4(
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


arm_status outer_fp32_3x3(
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





arm_status outer_fp32_2x2_packed(
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




arm_status outer_fp32_2x2(
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





arm_status outer_fp32_4x4_m_first(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t B00, B01, B02, B03,
  B10, B11, B12, B13, B20, B21, 
  B22, B23, B30, B31, B32, B33; /* Temporary B data  */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  float32_t *A_ptr, *C_ptr;

  uint32_t en = N / 4;
  uint32_t ek = K / 4;

  float32_t C0, C1, C2, C3;

  for(n = 0U; n < en; n++) {

    B = pSrcB->pData + 4*n;
    C_ptr = pDst->pData + 4*n;

    for(k = 0U; k < ek; k++) {

      A_ptr = pSrcA->pData + 4*k;

      B00 = *B++;
      B01 = *B++;
      B02 = *B++;
      B03 = *B;
      
      B += N - 3;

      B10 = *B++;
      B11 = *B++;
      B12 = *B++;
      B13 = *B;

      B += N - 3;

      B20 = *B++;
      B21 = *B++;
      B22 = *B++;
      B23 = *B;

      B += N - 3;

      B30 = *B++;
      B31 = *B++;
      B32 = *B++;
      B33 = *B;

      B += N - 3;
      C = C_ptr;

      /* m-first matrix multiplication */
      for(m = 0U; m < M; m++) {

        A = A_ptr + m*K;
        C = C_ptr + m*N;

        C0 = *C++;
        C1 = *C++;
        C2 = *C++;
        C3 = *C;
        C -= 3;


        C0 += *A * B00;
        C1 += *A * B01;
        C2 += *A * B02;
        C3 += *A * B03;

        A++;

        C0 += *A * B10;
        C1 += *A * B11;
        C2 += *A * B12;
        C3 += *A * B13;

        A++;

        C0 += *A * B20;
        C1 += *A * B21;
        C2 += *A * B22;
        C3 += *A * B23;

        A++;

        C0 += *A * B30;
        C1 += *A * B31;
        C2 += *A * B32;
        C3 += *A * B33;

        *C++ = C0;
        *C++ = C1;
        *C++ = C2;
        *C = C3;
      }
    }
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}



arm_status outer_fp32_5x5(
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






arm_status outer_fp32_5x5_m_first_test(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst,
        float* tmp_arr, int throttle)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t B00, B01, B02, B03, B04,
  B10, B11, B12, B13, B14, B20, B21, 
  B22, B23, B24, B30, B31, B32, B33, 
  B34, B40, B41, B42, B43, B44;    /* Temporary B data  */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  float32_t *A_ptr, *C_ptr;

  uint32_t en = N / 5;
  uint32_t ek = K / 5;

  float32_t C0, C1, C2, C3, C4;
  float tmp_cnt;
  int cntr = 0;

  for(n = 0U; n < en; n++) {

    B = pSrcB->pData + 5*n;
    C_ptr = pDst->pData + 5*n;

    for(k = 0U; k < ek; k++) {

      A = pSrcA->pData + 5*k;

      B00 = *B++;
      B01 = *B++;
      B02 = *B++;
      B03 = *B++;
      B04 = *B;
      
      B += N - 4;

      B10 = *B++;
      B11 = *B++;
      B12 = *B++;
      B13 = *B++;
      B14 = *B;

      B += N - 4;

      B20 = *B++;
      B21 = *B++;
      B22 = *B++;
      B23 = *B++;
      B24 = *B;

      B += N - 4;

      B30 = *B++;
      B31 = *B++;
      B32 = *B++;
      B33 = *B++;
      B34 = *B;

      B += N - 4;

      B40 = *B++;
      B41 = *B++;
      B42 = *B++;
      B43 = *B++;
      B44 = *B;

      B += N - 4;
      C = C_ptr;

      /* m-first matrix multiplication */
      for(m = 0U; m < M; m++) {
        
        C0 = *C++;
        C1 = *C++;
        C2 = *C++;
        C3 = *C++;
        C4 = *C;


        C0 += *A * B00;
        C1 += *A * B01;
        C2 += *A * B02;
        C3 += *A * B03;
        C4 += *A++ * B04;


        C0 += *A * B10;
        C1 += *A * B11;
        C2 += *A * B12;
        C3 += *A * B13;
        C4 += *A++ * B14;


        C0 += *A * B20;
        C1 += *A * B21;
        C2 += *A * B22;
        C3 += *A * B23;
        C4 += *A++ * B24;


        C0 += *A * B30;
        C1 += *A * B31;
        C2 += *A * B32;
        C3 += *A * B33;
        C4 += *A++ * B34;


        C0 += *A * B40;
        C1 += *A * B41;
        C2 += *A * B42;
        C3 += *A * B43;
        C4 += *A * B44;

        *C-- = C4;
        *C-- = C3;
        *C-- = C2;
        *C-- = C1;
        *C = C0;


        A += K - 4;
        C += N;

        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;

        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;

        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;

        tmp_arr[cntr++ % 1000] = cntr;

        // for(int x = 0; x < 25*throttle; x++) {
        //   tmp_arr[cntr++ % 1000] = cntr;
        // }

      }
    }
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}




arm_status outer_fp32_5x5_m_first(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t B00, B01, B02, B03, B04,
  B10, B11, B12, B13, B14, B20, B21, 
  B22, B23, B24, B30, B31, B32, B33, 
  B34, B40, B41, B42, B43, B44;    /* Temporary B data  */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  float32_t *A_ptr, *C_ptr;

  uint32_t en = N / 5;
  uint32_t ek = K / 5;

  float32_t C0, C1, C2, C3, C4;

  for(n = 0U; n < en; n++) {

    B = pSrcB->pData + 5*n;
    C_ptr = pDst->pData + 5*n;

    for(k = 0U; k < ek; k++) {

      A = pSrcA->pData + 5*k;

      B00 = *B++;
      B01 = *B++;
      B02 = *B++;
      B03 = *B++;
      B04 = *B;
      
      B += N - 4;

      B10 = *B++;
      B11 = *B++;
      B12 = *B++;
      B13 = *B++;
      B14 = *B;

      B += N - 4;

      B20 = *B++;
      B21 = *B++;
      B22 = *B++;
      B23 = *B++;
      B24 = *B;

      B += N - 4;

      B30 = *B++;
      B31 = *B++;
      B32 = *B++;
      B33 = *B++;
      B34 = *B;

      B += N - 4;

      B40 = *B++;
      B41 = *B++;
      B42 = *B++;
      B43 = *B++;
      B44 = *B;

      B += N - 4;
      C = C_ptr;

      /* m-first matrix multiplication */
      for(m = 0U; m < M; m++) {
        
        C0 = *C++;
        C1 = *C++;
        C2 = *C++;
        C3 = *C++;
        C4 = *C;


        C0 += *A * B00;
        C1 += *A * B01;
        C2 += *A * B02;
        C3 += *A * B03;
        C4 += *A++ * B04;


        C0 += *A * B10;
        C1 += *A * B11;
        C2 += *A * B12;
        C3 += *A * B13;
        C4 += *A++ * B14;


        C0 += *A * B20;
        C1 += *A * B21;
        C2 += *A * B22;
        C3 += *A * B23;
        C4 += *A++ * B24;


        C0 += *A * B30;
        C1 += *A * B31;
        C2 += *A * B32;
        C3 += *A * B33;
        C4 += *A++ * B34;


        C0 += *A * B40;
        C1 += *A * B41;
        C2 += *A * B42;
        C3 += *A * B43;
        C4 += *A * B44;

        *C-- = C4;
        *C-- = C3;
        *C-- = C2;
        *C-- = C1;
        *C = C0;


        A += K - 4;
        C += N;

      }
    }
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}











arm_status outer_fp32_5x5_test(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst,
        float* tmp_arr, int throttle)
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

  float tmp_cnt;
  int cntr = 0;
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




        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;

        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;

        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;
        tmp_arr[cntr++ % 1000] = cntr;

        // for(int x = 0; x < 25*throttle; x++) {
        //   tmp_arr[cntr++ % 1000] = cntr;
        // }

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

      // for(int x = 0; x < 25*throttle; x++) {
      //   tmp_arr[cntr++ % 1000] = rand();
      // }


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












arm_status outer_fp32_5x5_sp_test(
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

        if(*A != 0) {
          C00 += *A * *B++;
          C01 += *A * *B++;
          C02 += *A * *B++;
          C03 += *A * *B++;
          C04 += *A * *B;
          B -= 4;
        }

        A += K;

        if(*A != 0) {
          C10 += *A * *B++;
          C11 += *A * *B++;
          C12 += *A * *B++;
          C13 += *A * *B++;
          C14 += *A * *B;
          B -= 4;
        }

        A += K;

        if(*A != 0) {
          C20 += *A * *B++;
          C21 += *A * *B++;
          C22 += *A * *B++;
          C23 += *A * *B++;
          C24 += *A * *B;
          B -= 4;
        }

        A += K;

        if(*A != 0) {
          C30 += *A * *B++;
          C31 += *A * *B++;
          C32 += *A * *B++;
          C33 += *A * *B++;
          C34 += *A * *B;
          B -= 4;
        }

        A += K;

        if(*A != 0) {
          C40 += *A * *B++;
          C41 += *A * *B++;
          C42 += *A * *B++;
          C43 += *A * *B++;
          C44 += *A * *B;
        }
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



void pack_A_sp(float* A, float* A_p, sp_pack_t* sp_pack, 
  int M, int K, int k_c, int m_r) {

  int nnz_col, ind_blk, outer_ind = 0, a_ind = 0;
  float a_tmp = 0;

  int mr_bins = m_r + 1;
  int** cnt = (int**) malloc(mr_bins * sizeof(int*));
  int* cnt_inds = (int*) malloc(mr_bins * sizeof(int));


  for(int i = 0; i < mr_bins; i++) {
    cnt[i] = (int*) malloc(k_c * sizeof(int));
  }


  int* nnz_outer = (int*) calloc(((M*K) / m_r) , sizeof(int)); // storing number of nonzeros 
                                                        // in each outer prod col of A

  int* k_inds = (int*) calloc(((M*K) / m_r) , sizeof(int)); // storing kc_ind 
                                                        // of each outer prod col of A

  int* loc_m = (int*) calloc(M*K , sizeof(int)); // array for storing M dim C writeback location for each nnz in A
                                // each value ranges from 0 to mr-1

  int* nnz_outer_blk = (int*) calloc((M / m_r) , sizeof(int)); // storing number of nnz vals
                                                    // in each outer prod block of A

  int* k_cnt = (int*) calloc((M / m_r) , sizeof(int)); // storing number of nnz cols (b/w 0 and k_c) 
                                                    // in each outer prod block of A


  for(int m3 = 0; m3 < M; m3 += m_r) {

     ind_blk = 0;
     memset(cnt_inds, 0, mr_bins*sizeof(int));

     for(int i = 0; i < k_c; i++) {

        nnz_col = 0;

        for(int j = 0; j < m_r; j++) {

           if(A[m3*K + i + j*K] != 0) {
              nnz_col++;
           }
        }

        cnt[nnz_col][cnt_inds[nnz_col]++] = i;
     }


     for(int c = m_r; c > 0; c--) {

        if(!cnt_inds[c]) {
           // ind_blk += 6;
           continue;
        }

        for(int i = 0; i < cnt_inds[c]; i++) {

           for(int j = 0; j < m_r; j++) {
              
              a_tmp = A[m3*K + cnt[c][i] + j*K];
              if(a_tmp != 0) {
                 A_p[a_ind + ind_blk] = a_tmp;
                 loc_m[a_ind + ind_blk++] = j;
              }
           }

           k_inds[outer_ind] = cnt[c][i];
           nnz_outer[outer_ind++] = c;
        }

        k_cnt[m3 / m_r] += cnt_inds[c];  
     }

     // outer_ind += cnt_inds[0]; // skip ahead over cols with 0 nonzeros
     a_ind += ind_blk;
     nnz_outer_blk[m3 / m_r] = ind_blk;
  }


  for(int i = 0; i < mr_bins; i++) {
    free(cnt[i]);
  }

  free(cnt);
  free(cnt_inds);


  sp_pack->A_sp_p = A_p;
  sp_pack->loc_m = loc_m;
  sp_pack->nnz_outer = nnz_outer;
  sp_pack->k_inds = k_inds;
  sp_pack->nnz_outer_blk = nnz_outer_blk;
  sp_pack->k_cnt = k_cnt;
}





arm_status outer_fp32_5x5_sp(
  const sp_pack_t* pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst, 
  uint32_t M, uint32_t K, uint32_t N) {


  float32_t *A = pSrcA->A_sp_p;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t C[5*5];
 
  int* nnz_outer = pSrcA->nnz_outer;
  int* k_inds = pSrcA->k_inds;
  int* loc_m = pSrcA->loc_m;
  int* nnz_outer_blk = pSrcA->nnz_outer_blk;
  int* k_cnt = pSrcA->k_cnt;

  uint32_t m, n, k, kk, m_cnt, locm, C_ind = 0U;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  float32_t *A_ptr = pSrcA->A_sp_p;                /* Input data matrix pointer A */
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

    // number of columns with nnz values
    kk = k_cnt[m];

    /* column loop */

    for(n = 0U; n < en; n++) {

      C[0] = 0;
      C[1] = 0;
      C[2] = 0;
      C[3] = 0;
      C[4] = 0;

      C[5] = 0;
      C[6] = 0;
      C[7] = 0;
      C[8] = 0;
      C[9] = 0;

      C[10] = 0;
      C[11] = 0;
      C[12] = 0;
      C[13] = 0;
      C[14] = 0;

      C[15] = 0;
      C[16] = 0;
      C[17] = 0;
      C[18] = 0;
      C[19] = 0;

      C[20] = 0;
      C[21] = 0;
      C[22] = 0;
      C[23] = 0;
      C[24] = 0;

      A = A_ptr;
      /* Update pointer B to point to starting address of next column */
      B_ptr = pSrcB->pData + 5*n;

      /* matrix multiplication */
      for(k = 0U; k < kk; k++) {
      
        // m_cnt = nnz_outer[k];
        // B = B_ptr + k_inds[k]*N;
        m_cnt = *nnz_outer++;
        B = B_ptr + (*k_inds++ * N);

        for(int j = 0; j < m_cnt; j++) {

          locm = *loc_m * 5;
          C[locm++] +=  *A * *B++;
          C[locm++] +=  *A * *B++;
          C[locm++] +=  *A * *B++;
          C[locm++] +=  *A * *B++;
          C[locm] +=  *A++ * *B;

          B -= 4;
          loc_m++;
        }

      }

      nnz_outer -= kk;
      k_inds -= kk;
      loc_m -= nnz_outer_blk[m];
      /* Store result in destination buffer */

      C_curr = pDst->pData + C_ind;

      *C_curr++ = C[0];
      *C_curr++ = C[1];
      *C_curr++ = C[2];
      *C_curr++ = C[3];
      *C_curr = C[4];

      C_curr = C_curr - 4 + N;

      *C_curr++ = C[5];
      *C_curr++ = C[6];
      *C_curr++ = C[7];
      *C_curr++ = C[8];
      *C_curr = C[9];

      C_curr = C_curr - 4 + N;

      *C_curr++ = C[10];
      *C_curr++ = C[11];
      *C_curr++ = C[12];
      *C_curr++ = C[13];
      *C_curr = C[14];

      C_curr = C_curr - 4 + N;

      *C_curr++ = C[15];
      *C_curr++ = C[16];
      *C_curr++ = C[17];
      *C_curr++ = C[18];
      *C_curr = C[19];

      C_curr = C_curr - 4 + N;

      *C_curr++ = C[20];
      *C_curr++ = C[21];
      *C_curr++ = C[22];
      *C_curr++ = C[23];
      *C_curr = C[24];


      C_ind += 5;
    }

    /* Update pointer A_ptr to point to starting address of next row */
    // A_ptr += 5*K;
    A_ptr += nnz_outer_blk[m];
    loc_m += nnz_outer_blk[m];
    nnz_outer += kk;
    k_inds += kk;
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}



arm_status outer_fp32_5x5_old(
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








// ------------------------- TESTING -----------------------------







arm_status outer_fp32_6x6(
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






arm_status outer_fp32_6x5_m_first(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst)
{
  float32_t *A = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B = pSrcB->pData;                /* Input data matrix pointer B */
  float32_t *C = pDst->pData;                 /* Output data matrix pointer */
  float32_t B00, B01, B02, B03, B04,
  B10, B11, B12, B13, B14, B20, B21, 
  B22, B23, B24, B30, B31, B32, B33, 
  B34, B40, B41, B42, B43, B44,
  B50, B51, B52, B53, B54;    /* Temporary B data  */
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */

  float32_t *A_ptr, *C_ptr;

  uint32_t en = N / 5;
  uint32_t ek = K / 6;

  float32_t C0, C1, C2, C3, C4, C5;

  for(n = 0U; n < en; n++) {

    B = pSrcB->pData + 5*n;
    C_ptr = pDst->pData + 5*n;

    for(k = 0U; k < ek; k++) {

      A = pSrcA->pData + 6*k;

      B00 = *B++;
      B01 = *B++;
      B02 = *B++;
      B03 = *B++;
      B04 = *B;
      
      B += N - 4;

      B10 = *B++;
      B11 = *B++;
      B12 = *B++;
      B13 = *B++;
      B14 = *B;

      B += N - 4;

      B20 = *B++;
      B21 = *B++;
      B22 = *B++;
      B23 = *B++;
      B24 = *B;

      B += N - 4;

      B30 = *B++;
      B31 = *B++;
      B32 = *B++;
      B33 = *B++;
      B34 = *B;

      B += N - 4;

      B40 = *B++;
      B41 = *B++;
      B42 = *B++;
      B43 = *B++;
      B44 = *B;

      B += N - 4;

      B50 = *B++;
      B51 = *B++;
      B52 = *B++;
      B53 = *B++;
      B54 = *B;


      B += N - 4;
      C = C_ptr;

      /* m-first matrix multiplication */
      for(m = 0U; m < M; m++) {
        
        C0 = *C++;
        C1 = *C++;
        C2 = *C++;
        C3 = *C++;
        C4 = *C;


        C0 += *A * B00;
        C1 += *A * B01;
        C2 += *A * B02;
        C3 += *A * B03;
        C4 += *A++ * B04;


        C0 += *A * B10;
        C1 += *A * B11;
        C2 += *A * B12;
        C3 += *A * B13;
        C4 += *A++ * B14;


        C0 += *A * B20;
        C1 += *A * B21;
        C2 += *A * B22;
        C3 += *A * B23;
        C4 += *A++ * B24;


        C0 += *A * B30;
        C1 += *A * B31;
        C2 += *A * B32;
        C3 += *A * B33;
        C4 += *A++ * B34;


        C0 += *A * B40;
        C1 += *A * B41;
        C2 += *A * B42;
        C3 += *A * B43;
        C4 += *A++ * B44;


        C0 += *A * B50;
        C1 += *A * B51;
        C2 += *A * B52;
        C3 += *A * B53;
        C4 += *A * B54;

        *C-- = C4;
        *C-- = C3;
        *C-- = C2;
        *C-- = C1;
        *C = C0;


        A += K - 5;
        C += N;

      }
    }
  }

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}
