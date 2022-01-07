#include "kernels.h"


arm_status inner_fp32_2x4x2(
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








arm_status inner_fp32_1x8x1(
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







arm_status inner_fp32_1x16x1(
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
      k = K >> 4;

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

        /* Decrement loop counter */
        k--;
      }

      /* Loop unrolling: Compute remaining MACs */
      k = K % 16U;


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









arm_status inner_fp32_1x4x1(
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
