
#include <arm_math.h>
#include <stdio.h>

#include "mbed.h"
#include "mbed_mem_trace.h"

#define ARM_MATH_LOOPUNROLL 1
#define CHECK_PRINT 0

void print_memory_info();
void print_memory_info1();

void rand_init_q15(int16_t* mat, int r, int c) ;
void rand_init(float* mat, int r, int c) ;

void print_mat(arm_matrix_instance_f32* mat, int rows, int cols);
void print_mat_q15(arm_matrix_instance_q15* mat, int rows, int cols);

bool q15_gemm_checker(int16_t* C, int16_t* C_check, int N, int M, int K) ;
bool f32_gemm_checker(float* C, float* C_check, int N, int M, int K) ;
void sram_bw_prof() ;
void arm_vs_mema_fp32();
void arm_vs_mema_q15();

arm_status inner_fp32_1x4x1(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status inner_fp32_1x8x1(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status inner_fp32_1x16x1(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status inner_fp32_2x4x2(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status inner_fp32_2x8x2(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_fp32_2x2_packed(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_fp32_2x2(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_fp32_3x3(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_fp32_4x4(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_fp32_5x5_packed(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_fp32_5x5_old(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_fp32_5x5(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_fp32_6x6(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status test1_arr(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);


arm_status dsp_test(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t                   * pState);


arm_status arm_q15_inner_2x2x2(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t                   * pState);

arm_status arm_q15_inner_2x4x2(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t                   * pState);

arm_status outer_q15_4x2(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t                   * pState);

arm_status outer_q15_6x2(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t                   * pState);

