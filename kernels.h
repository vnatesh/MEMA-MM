
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



typedef struct sp_pack_t {
   int* loc_m; // M dim C writeback location for each nnz value in A
   int* nnz_outer; // number of nnz in every outer prod col vec (with len m_r) of A;
   int* k_inds; // density-based reorder indices of A cols within a mrxkcxnr tile
   int* nnz_outer_blk; // number of nonzeros in each mrxkcxnr outer product blk
   int* k_cnt; // number of nnz cols (b/w 0 and k_c) in each outer prod block of A
   float* A_sp_p; //sparse packed A (only storing nonzeros)
} sp_pack_t;


void pack_A_sp(float* A, float* A_p, sp_pack_t* sp_pack, 
  int M, int K, int k_c, int m_r);

// packing without density-based reordering of columns
void pack_A_sp_no_reorder(float* A, float* A_p, sp_pack_t* sp_pack, 
  int M, int K, int k_c, int m_r) ;


void rand_sparse(float* mat, int r, int c, float sparsity);
void print_mat1(float* mat, int rows, int cols);
void print_arr(float* mat, int len);
void print_arr_int(int* mat, int len);

arm_status outer_fp32_5x5_sp(
  const sp_pack_t* pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst, 
  uint32_t M, uint32_t K, uint32_t N);

arm_status outer_fp32_5x5_sp_test(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_fp32_5x5_test(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst,
        float* tmp_arr, int throttle);

arm_status inner_fp32_1x16x1_test(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst, float* tmp_arr, int throttle);

arm_status inner_fp32_2x8x2_test(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst, float* tmp_arr, int throttle);

arm_status outer_fp32_5x5_m_first(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

void testing() ;

arm_status outer_fp32_4x4_m_first(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_fp32_5x5_m_first_test(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst,
        float* tmp_arr, int throttle);

arm_status outer_fp32_6x5_m_first(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_fp32_5x5_k_first_old(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_q15_1x4x3(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t                   * pState);


arm_status outer_q31_5x5_k_first(
   int* pSrcA,
   int* pSrcB,
        int* pDst, int M, int K, int N);

void print_mat_q31(arm_matrix_instance_q31* mat, int rows, int cols) ;
bool q31_gemm_checker(int32_t* C, int32_t* C_check, int N, int M, int K) ;
void rand_init_int32(int32_t* mat, int r, int c) ;
void testing_q31() ;
void print_mat_int(int* mat, int rows, int cols) ;









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

arm_status outer_fp32_5x5_k_first(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);

arm_status outer_fp32_5x5_n_first(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst);


arm_status outer_fp32_1x6x4_m_first(
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
