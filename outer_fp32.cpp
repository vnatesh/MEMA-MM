#include "kernels.h"

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
