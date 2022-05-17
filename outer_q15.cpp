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



















arm_status outer_q15_1x4x3(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t                   * pState)
{
        q15_t *pSrcBT = pState;                        /* Input data matrix pointer for transpose */
        q15_t *A = pSrcA->pData;                    /* Input data matrix pointer A of Q15 type */
        q15_t *B = pSrcB->pData;                    /* Input data matrix pointer B of Q15 type */
        q15_t *px;                                     /* Temporary output data matrix pointer */
        uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
        uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
        uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
        uint32_t em,ek,en,col, i = 0U, row = K;  /* Loop counters */
        arm_status status;                             /* Status of matrix multiplication */

        int m,n,k;
        q31_t in;                                      /* Temporary variable to hold the input value */
        q31_t inA00, inA01,
          inB00, inB01, inB10, inB11, inB20, inB21;
        q15_t *A0, 
        *B0, *B1, *B2, *C, *C_ptr;
        q31_t C00, C01, C02;                         /* Accumulator */

  {
    /* Matrix transpose */
    do
    {
      /* The pointer px is set to starting address of column being processed */
      px = pSrcBT + i;

      /* Apply loop unrolling and exchange columns with row elements */
      col = N >> 2U;

      /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.
       ** a second loop below computes the remaining 1 to 3 samples. */
      while (col > 0U)
      {

        /* Read two elements from row */
        in = read_q15x2_ia ((q15_t **) &B);

        /* Unpack and store one element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
        *px = (q15_t) in;
#else
        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

        /* Update pointer px to point to next row of transposed matrix */
        px += K;

        /* Unpack and store second element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);
#else
        *px = (q15_t) in;
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

        /* Update pointer px to point to next row of transposed matrix */
        px += K;

        in = read_q15x2_ia ((q15_t **) &B);
#ifndef ARM_MATH_BIG_ENDIAN
        *px = (q15_t) in;
#else
        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */
        px += K;

#ifndef ARM_MATH_BIG_ENDIAN
        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);
#else
        *px = (q15_t) in;
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */
        px += K;

        /* Decrement column loop counter */
        col--;
      }

      /* If the columns of pSrcB is not a multiple of 4, compute any remaining output samples here.
       ** No loop unrolling is used. */
      col = N % 0x4U;

      while (col > 0U)
      {
        /* Read and store input element in destination */
        *px = *B++;

        /* Update pointer px to point to next row of transposed matrix */
        px += K;

        /* Decrement column loop counter */
        col--;
      }

      i++;

      /* Decrement row loop counter */
      row--;

    } while (row > 0U);





    C = pDst->pData;
    // C01 = C + N;
    // C02 = C01 + N;
    // C4 = C02 + N;

    em = M / 1;
    ek = K / 4;
    en = N / 3;

    for(n = 0U; n < en; n++) {

      B0 = pSrcBT + 3*n*K;
      B1 = B0 + K;
      B2 = B1 + K;

      C_ptr = pDst->pData + 3*n;

      for(k = 0U; k < ek; k++) {
      
        A0 = pSrcA->pData + 4*k;
        // A1 = A0 + K;
        // A2 = A1 + K;
        // A3 = A2 + K;

        inB00 = read_q15x2_ia ((q15_t **) &B0);
        inB01 = read_q15x2_ia ((q15_t **) &B0);

        inB10 = read_q15x2_ia ((q15_t **) &B1);
        inB11 = read_q15x2_ia ((q15_t **) &B1);

        inB20 = read_q15x2_ia ((q15_t **) &B2);
        inB21 = read_q15x2_ia ((q15_t **) &B2);

        C = C_ptr;

        for(m = 0; m < em; m++) {

          C00 = (q31_t) (*C++ << 15);
          C01 = (q31_t) (*C++ << 15);
          C02 = (q31_t) (*C << 15);

          inA00 = read_q15x2_ia ((q15_t **) &A0);
          inA01 = read_q15x2_ia ((q15_t **) &A0);

          /* Multiply and Accumulates */
          C00  = __SMLAD(inA00, inB00, C00);
          C00 = __SMLAD(inA01, inB01, C00);

          C01  = __SMLAD(inA00, inB10, C01);
          C01 = __SMLAD(inA01, inB11, C01);

          C02  = __SMLAD(inA00, inB20, C02);
          C02 = __SMLAD(inA01, inB21, C02);

          *C--  = (q15_t) (C00 >> 15);
          *C--  = (q15_t) (C01 >> 15);
          *C  = (q15_t) (C02 >> 15);

          A0 += K - 4;
          C += N;
        }
      }
    }

    /* Set status as ARM_MATH_SUCCESS */
    status = ARM_MATH_SUCCESS;
  }

  /* Return to application */
  return (status);
}
