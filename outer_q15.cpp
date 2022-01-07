#include "kernels.h"



arm_status outer_q15_6x2(
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
        q31_t inA1, inB1, inA2, inB2, inA3, inA4, inA5, inA6;
        q31_t sum, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12;                         /* Accumulator */
        q15_t *pInA2, *pInA3, *pInA4, *pInA5, *pInA6, *pInB2, *px2, *px3, *px4, *px5, *px6;
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
        in = read_q15x2_ia ((q15_t **) &pInB);

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

        in = read_q15x2_ia ((q15_t **) &pInB);
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

    /* Process six rows from matrix A at a time and output six rows at a time */
    row = row / 6;

    px = pDst->pData;
    px2 = px + numColsB;
    px3 = px2 + numColsB;
    px4 = px3 + numColsB;
    px5 = px4 + numColsB;
    px6 = px5 + numColsB;

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
        sum9 = 0;
        sum10 = 0;
        sum11= 0;
        sum12 = 0;

        /* Initiate pointer pInA to point to starting address of column being processed */
        pInA = pSrcA->pData + i;
        pInA2 = pInA + numColsA;
        pInA3 = pInA2 + numColsA;
        pInA4 = pInA3 + numColsA;
        pInA5 = pInA4 + numColsA;
        pInA6 = pInA5 + numColsA;

        pInB  = pSrcBT + j;
        pInB2 = pInB + numRowsB;

        /* Read in two elements at once - allows dual MAC instruction */
        colCnt = numColsA >> 1U;


        /* matrix multiplication */
        while (colCnt > 0U)
        {
          /* c(m,n) = a(1,1) * b(1,1) + a(1,2) * b(2,1) + .... + a(m,p) * b(p,n) */

          /* read real and imag values from pSrcA and pSrcB buffer */
          inA1 = read_q15x2_ia ((q15_t **) &pInA);
          inB1 = read_q15x2_ia ((q15_t **) &pInB);

          inA2 = read_q15x2_ia ((q15_t **) &pInA2);
          inB2 = read_q15x2_ia ((q15_t **) &pInB2);

          /* Multiply and Accumulates */
          sum  = __SMLAD(inA1, inB1, sum);
          sum2 = __SMLAD(inA1, inB2, sum2);
          sum3 = __SMLAD(inA2, inB1, sum3);
          sum4 = __SMLAD(inA2, inB2, sum4);


          inA3 = read_q15x2_ia ((q15_t **) &pInA3);
          inA4 = read_q15x2_ia ((q15_t **) &pInA4);

          /* Multiply and Accumulates */
          sum5  = __SMLAD(inA3, inB1, sum5);
          sum6 = __SMLAD(inA3, inB2, sum6);
          sum7 = __SMLAD(inA4, inB1, sum7);
          sum8 = __SMLAD(inA4, inB2, sum8);

          inA5 = read_q15x2_ia ((q15_t **) &pInA5);
          inA6 = read_q15x2_ia ((q15_t **) &pInA6);

          /* Multiply and Accumulates */
          sum9  = __SMLAD(inA5, inB1, sum9);
          sum10 = __SMLAD(inA5, inB2, sum10);
          sum11 = __SMLAD(inA6, inB1, sum11);
          sum12 = __SMLAD(inA6, inB2, sum12);

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
          inA5 = *pInA5++;
          inA6 = *pInA6++;

          sum  += inA1 * inB1;
          sum2 += inA1 * inB2;
          sum3 += inA2 * inB1;
          sum4 += inA2 * inB2;

          sum5 += inA3 * inB1;
          sum6 += inA3 * inB2;
          sum7 += inA4 * inB1;
          sum8 += inA4 * inB2;

          sum9 += inA5 * inB1;
          sum10 += inA5 * inB2;
          sum11 += inA6 * inB1;
          sum12 += inA6 * inB2;
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

        *px5++ = (q15_t) (sum9 >> 15);
        *px5++ = (q15_t) (sum10 >> 15);
        *px6++ = (q15_t) (sum11 >> 15);
        *px6++ = (q15_t) (sum12 >> 15);

        j += numRowsB * 2;

        /* Decrement column loop counter */
        col--;

      }

      i = i + 6*numColsA;
      px = px6 + (numColsB & 1U);
      px2 = px + numColsB;
      px3 = px2 + numColsB;
      px4 = px3 + numColsB;
      px5 = px4 + numColsB;
      px6 = px5 + numColsB;

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
          inA1 = read_q15x2_ia ((q15_t **) &pInA);
          inA2 = read_q15x2_ia ((q15_t **) &pInA);
          inB1 = read_q15x2_ia ((q15_t **) &pInB);
          inB2 = read_q15x2_ia ((q15_t **) &pInB);

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
          inA1 = read_q15x2_ia ((q15_t **) &pInA);
          inA2 = read_q15x2_ia ((q15_t **) &pInA);
          inB1 = read_q15x2_ia ((q15_t **) &pInB);
          inB2 = read_q15x2_ia ((q15_t **) &pInB);

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
      /* The pointer px is set to starting address of column being processed */
      px = pSrcBT + i;

      /* Apply loop unrolling and exchange columns with row elements */
      col = numColsB >> 2U;

      /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.
       ** a second loop below computes the remaining 1 to 3 samples. */
      while (col > 0U)
      {

        /* Read two elements from row */
        in = read_q15x2_ia ((q15_t **) &pInB);

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

        in = read_q15x2_ia ((q15_t **) &pInB);
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
          inA1 = read_q15x2_ia ((q15_t **) &pInA);
          inB1 = read_q15x2_ia ((q15_t **) &pInB);

          inA2 = read_q15x2_ia ((q15_t **) &pInA2);
          inB2 = read_q15x2_ia ((q15_t **) &pInB2);

          /* Multiply and Accumulates */
          sum  = __SMLAD(inA1, inB1, sum);
          sum2 = __SMLAD(inA1, inB2, sum2);
          sum3 = __SMLAD(inA2, inB1, sum3);
          sum4 = __SMLAD(inA2, inB2, sum4);


          inA3 = read_q15x2_ia ((q15_t **) &pInA3);
          inA4 = read_q15x2_ia ((q15_t **) &pInA4);

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
          inA1 = read_q15x2_ia ((q15_t **) &pInA);
          inA2 = read_q15x2_ia ((q15_t **) &pInA);
          inB1 = read_q15x2_ia ((q15_t **) &pInB);
          inB2 = read_q15x2_ia ((q15_t **) &pInB);

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
          inA1 = read_q15x2_ia ((q15_t **) &pInA);
          inA2 = read_q15x2_ia ((q15_t **) &pInA);
          inB1 = read_q15x2_ia ((q15_t **) &pInB);
          inB2 = read_q15x2_ia ((q15_t **) &pInB);

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





