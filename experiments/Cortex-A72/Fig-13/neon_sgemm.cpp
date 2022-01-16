/*
 * Copyright (c) 2018-2019 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/Utils.h"
#include <omp.h>
#include <cstdlib>
#include <unistd.h>

using namespace arm_compute;
using namespace utils;

class NESGEMMExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        sleep(5);

        NPYLoader npy0;
        NPYLoader npy1;
        NPYLoader npy2;
        alpha = 1.0f;
        beta  = 0.0f;
        int p = 4;
        int iters = 1;

        for(int i = 500; i <= 5000; i += 500) {

            M = i, K = i, N = i;
            printf("M = %d, K = %d, N = %d\n", M,K,N);
            
            src0.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
            src1.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
            src2.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));


            init_sgemm_output(dst, src0, src1, DataType::F32);

            // Configure function
            sgemm.configure(&src0, &src1, nullptr, &dst, alpha, beta);

            // Allocate all the images
            src0.allocator()->allocate();
            src1.allocator()->allocate();
            dst.allocator()->allocate();

            src2.allocator()->allocate();

            fill_random_tensor(src0, -1.f, 1.f);
            fill_random_tensor(src1, -1.f, 1.f);
            fill_random_tensor(src2, -1.f, 1.f);
        
            struct timespec start, end;
            double diff_t;

            clock_gettime(CLOCK_REALTIME, &start);

            // use p cores for experiment
            NEScheduler::get().set_num_threads(p);

            for(int j = 0; j < iters; j++) {
                sgemm.run();
            }

            clock_gettime(CLOCK_REALTIME, &end);
            long seconds = end.tv_sec - start.tv_sec;
            long nanoseconds = end.tv_nsec - start.tv_nsec;
            diff_t = seconds + nanoseconds*1e-9;
            printf("sgemm time: %f \n", diff_t / iters); 


            sleep(2);
        }

        sleep(10);
        return true;
    }

    void do_run() override
    {

    }
    
    void do_teardown() override
    {
        if(!output_filename.empty()) /* Save to .npy file */
        {
            save_to_npy(dst, output_filename, is_fortran);
        }
    }

private:
    Tensor      src0{}, src1{}, src2{}, dst{};
    NEGEMM      sgemm{};
    float       alpha{}, beta{};
    size_t      M;
    size_t      N;
    size_t      K;
    int         p;
    bool        is_fortran{};
    std::string output_filename{};
};

/** Main program for sgemm test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Matrix A, [optional] Matrix B, [optional] Matrix C, [optional] alpha, [optional] beta )
 */
int main(int argc, char **argv)
{
    return utils::run_example<NESGEMMExample>(argc, argv);
}