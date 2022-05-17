#include "kernels.h"



void rand_init(float* mat, int r, int c) {
    // int MAX = 65536;
    for(int i = 0; i < r*c; i++) {
        // mat[i] = (double) i;
        // mat[i] = 1.0;
        // mat[i] =  (double) (i%MAX);
        mat[i] =  (float) rand() / RAND_MAX*2.0 - 1.0;
    }   
}


void rand_init_int32(int32_t* mat, int r, int c) {
    // int MAX = 65536;
    for(int i = 0; i < r*c; i++) {
        // mat[i] = (double) i;
        // mat[i] = 1.0;
        // mat[i] =  (double) (i%MAX);
        mat[i] = (int) (((float) rand() / RAND_MAX*2.0 - 1.0) * 128);
    }   
}


// randomized sparse matrix with sparsity % of values that are zero
// threshold pruning
void rand_sparse(float* mat, int r, int c, float sparsity) {

  for(int i = 0; i < r*c; i++) {
    int x = rand();
    if(x <= ((float) RAND_MAX)*sparsity) {
      mat[i] = 0;
    } else {
      mat[i] =  (float) x / ((float) RAND_MAX)*2.0 - 1.0;
    }
  } 
}


void print_mat1(float* mat, int rows, int cols) {

  char buffer[100];

  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      sprintf(buffer, "%0.2f ", mat[i*cols + j]);
      Serial.print(buffer);
    }
    Serial.println("");
  }
  Serial.println("");

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


void print_arr(float* mat, int len) {

  char buffer[100];

  for(int i = 0; i < len; i++) {
      sprintf(buffer, "%0.2f ", mat[i]);
      Serial.print(buffer);
  }
  Serial.println("");
}


void print_arr_int(int* mat, int len) {

  char buffer[100];

  for(int i = 0; i < len; i++) {
      sprintf(buffer, "%d ", mat[i]);
      Serial.print(buffer);
  }
  Serial.println("");
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





bool q15_gemm_checker(int16_t* C, int16_t* C_check, int N, int M, int K) {

  int CORRECT = 1;
  int cnt = 0;
  int ind1 = 0;

  for(int m = 0; m < M; m++) {
      for(int n = 0; n < N; n++) {
          if(C[ind1] != C_check[ind1]) {
              cnt++;
              CORRECT = 0;
          }

          if(CHECK_PRINT) printf("%d\t%d\n", C_check[ind1], C[ind1]);
          ind1++; 
        }
    }

  if(CORRECT) {
    Serial.println("CORRECT!");
    return 0;
  } else {
    Serial.println("WRONG!");
    // Serial.println("%d\n", cnt);
    return 1;
  }


}




bool q31_gemm_checker(int32_t* C, int32_t* C_check, int N, int M, int K) {

  int CORRECT = 1;
  int cnt = 0;
  int ind1 = 0;

  for(int m = 0; m < M; m++) {
      for(int n = 0; n < N; n++) {
          if(C[ind1] != C_check[ind1]) {
              cnt++;
              CORRECT = 0;
          }

          if(CHECK_PRINT) printf("%d\t%d\n", C_check[ind1], C[ind1]);
          ind1++; 
        }
    }

  if(CORRECT) {
    Serial.println("CORRECT!");
    return 0;
  } else {
    Serial.println("WRONG!");
    // Serial.println("%d\n", cnt);
    return 1;
  }


}



void print_mat_q31(arm_matrix_instance_q31* mat, int rows, int cols) {

  char buffer[100];

  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      sprintf(buffer, "%d ", mat->pData[i*cols + j]);
      Serial.print(buffer);
    }
    Serial.println("");
  }
  Serial.println("");

}



void print_mat_q15(arm_matrix_instance_q15* mat, int rows, int cols) {

  char buffer[100];

  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      sprintf(buffer, "%hi ", mat->pData[i*cols + j]);
      Serial.print(buffer);
    }
    Serial.println("");
  }
  Serial.println("");

}

void print_mat_int(int* mat, int rows, int cols) {

  char buffer[100];

  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      sprintf(buffer, "%d ", mat[i*cols + j]);
      Serial.print(buffer);
    }
    Serial.println("");
  }
  Serial.println("");

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



// measure the bandwidth from core to SRAM 

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
