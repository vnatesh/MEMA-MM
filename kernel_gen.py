
x = 0
LM = 25600
for i in range(1,LM+1):
  for j in range(1,LM+1):
    mem = i+j+i*j
    if mem < 25600:
      x+=1
      # print (i,j)



# def gen_n_leftover_k(off,M,N,m,n):
#   n_left = '''
#     if(n_left) {
#       A_ptr = pSrcA->pData + %s*%d*K;
#       B_ptr = pSrcB->pData + %d*en;
#       int tmp_ind = %d*%s*N + %d*en; 
#       ''' % (off,M,N,M,off,N)
#   #
#   xl = []
#   for ns in range(1,n):
#     x = '''
#       if(n_left == %d) {
#       ''' % ns
#     x += gen_c_load(m,ns)
#     x += '''
#         for(int kk = 0U; kk < K; kk++) {
#           A = A_ptr + kk ;
#           B = B_ptr + kk*N;
#           C_curr = pDst->pData + tmp_ind;
#     '''
#     x += gen_mac_ops_k(m,ns)
#     x += gen_c_write_k(m,ns)
#     xl.append(x)
#   return n_left + 'else '.join(xl) + '''
#     }
#     '''


# def gen_m_leftover_k(m,n):
#   m_left = '''
#   if(m_left) {
#     C_ind = em*%d*N; 
#       ''' % (m)
#   #
#   xl = []
#   for ms in range(1,m):
#     x = '''
#     if(m_left == %d) {
#       ''' % ms
#     x += '''
#       for(n = 0U; n < en; n++) {
#       '''
#     x += gen_c_load(ms,n) + '''
#         B_ptr = pSrcB->pData + %d*n;\n''' % n
#     x += '''
#         for(k = 0U; k < K; k++) {

#           A = A_ptr + k ;
#           B = B_ptr + k*N;
#         '''
#     x += gen_mac_ops_k(ms,n)
#     x += gen_c_write_k(ms,n)
#     x += gen_n_leftover_k('em',m,n,ms,n)
#     xl.append(x)
#   return m_left + '} else '.join(xl) + '''
#     }
#   }
#     '''



def gen_m_leftover_k(m,n,ms,ns):
  m_left = '''
  C_ind = em*%d*N; 
      ''' % (m)
  #
  x = '''
  for(n = 0U; n < en; n++) {
    '''
  x += gen_c_load(ms,n) + '''
    B_ptr = pSrcB->pData + %d*n;\n''' % n
  x += '''
    for(k = 0U; k < K; k++) {

      A = A_ptr + k ;
      B = B_ptr + k*N;
      '''
  x += gen_mac_ops_k(ms,n)
  x += gen_c_write_k(ms,n) + '''
  }
  '''
  x += gen_n_leftover_k('em',m,n,ms,ns)
  return m_left + x


def gen_n_leftover_k(off,m_max,n_max,m,ns):
  n_left = '''
      A_ptr = pSrcA->pData + %s*%d*K;
      B_ptr = pSrcB->pData + %d*en;
      int tmp_ind = %d*%s*N + %d*en; 
      ''' % (off,m_max,n_max,m_max,off,n_max)
  #
  ns = N % n_max;
  x = gen_c_load(m,ns)
  x += '''
      for(int kk = 0U; kk < K; kk++) {
        A = A_ptr + kk ;
        B = B_ptr + kk*N;
        C_curr = pDst->pData + tmp_ind;
  '''
  x += gen_mac_ops_k(m,ns)
  x += gen_c_write_k(m,ns)
  return n_left + x


# gen_n_leftover_k('m',5,5,5,3)


def gen_mac_ops_k(m,n):
  mac_ops = ''
  for i in range(m-1):
    for j in range(n-1):
      mac_ops += '''
        C%d%d += *A * *B++;''' % (i,j)
    mac_ops += '''
        C%d%d += *A * *B;\n
        B -= %d;
        A += K;
        ''' % (i,n-1,n-1)
  for j in range(n-1):
    mac_ops += '''
        C%d%d += *A * *B++;''' % (m-1,j)
  mac_ops += '''
        C%d%d += *A * *B;
      }
      ''' % (m-1,n-1)
  return mac_ops



def gen_c_load(m,n):
  C_load = []
  for i in range(m):
    for j in range(n):
      C_load.append('''
      C%d%d = 0;''' % (i,j))
    C_load.append('''
      ''')
  return ''.join(C_load)


def gen_c_write_k(m,n):
  C_write = '''
      C_curr = pDst->pData + C_ind;
      '''
  for i in range(m-1):
    for j in range(n-1):
      C_write += '''
      *C_curr++ = C%d%d;''' % (i,j)
    C_write += '''
      *C_curr = C%d%d;
      C_curr = C_curr - %d + N;
      ''' % (i,n-1,n-1)
  #
  for j in range(n-1):
    C_write += '''
      *C_curr++ = C%d%d;''' % (m-1,j)
  # C_write += '''
  #     *C_curr = C%d%d;
  #     C_ind += %d;
  #   }
  #   ''' % (m-1,n-1,n)
  C_write += '''
      *C_curr = C%d%d;
      C_ind += %d;
    ''' % (m-1,n-1,n)
  return C_write


def gen_func_def(m,k,n,sched):
  return '''
arm_status outer_fp32_%dx%dx%d_%s_first(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
        arm_matrix_instance_f32 * pDst) 
{
  ''' % (m,k,n,sched)
  #
  #

def gen_var_decls():
  return '''  
  float32_t *A = pSrcA->pData; 
  float32_t *B = pSrcB->pData;  
  float32_t *C = pDst->pData; 
  uint16_t M = pSrcA->numRows;            /* Number of rows of input matrix A */
  uint16_t N = pSrcB->numCols;            /* Number of columns of input matrix B */
  uint16_t K = pSrcA->numCols;            /* Number of columns of input matrix A */
  uint32_t m, n, k;  /* Loop counters */
  arm_status status;                             /* Status of matrix multiplication */
  '''

def gen_end():
  return '''

  /* Set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;


  /* Return to application */
  return (status);
}'''


def gen_mema_kernel(m,k,n,sched):
  gen_m_first(m,k,n,sched)
  gen_n_first(m,k,n,sched)
  gen_k_first(m,k,n,sched)


def gen_k_first(M,K,N,m,k,n,sched):
  #
  func_def = gen_func_def(m,k,n,sched)
  var_decls = gen_var_decls()
  end = gen_end()
  var_decls += '''
  uint32_t C_ind = 0U;
  float32_t *A_ptr = pSrcA->pData;                /* Input data matrix pointer A */
  float32_t *B_ptr, *C_curr;
  '''
  #
  C_decls = ','.join(["C%d%d" % (i,j) for i in range(m) for j in range(n)])
  var_decls += ''' 
  float32_t ''' + C_decls + ';\n'
  #
  bounds = '''
  uint32_t en = N / %d;
  uint32_t em = M / %d;

  uint32_t n_left = N %% %d;
  uint32_t m_left = M %% %d;
  ''' % (n,m,n,m)
  #
  #
  outer_loops = '''
  for(m = 0U; m < em; m++) {

    C_ind = m*%d*N;

    for(n = 0U; n < en; n++) {
  ''' % m
  #
  #
  C_load = gen_c_load(m,n) + '''
      B_ptr = pSrcB->pData + %d*n;\n''' % n
  #
  #
  mm_sched = '''
      for(k = 0U; k < K; k++) {

        A = A_ptr + k ;
        B = B_ptr + k*N;
        '''
  #
  #
  mac_ops = gen_mac_ops_k(m,n)
  C_write = gen_c_write_k(m,n) + '''
  }
  '''
  # n_left = gen_n_leftover_k('m',m,n,m,n)
  # m_left = gen_m_leftover_k(m,n)
  n_left = gen_n_leftover_k('m',m,n,m,n)
  m_left = gen_m_leftover_k(m,n,M % m, N % n)
  #
  mm_sched += mac_ops + C_write + n_left + '''
    A_ptr += %d*K;
  }
  ''' % (m) + m_left
  prog = func_def + var_decls + bounds + outer_loops + C_load + mm_sched + end
  print(prog)


# gen_k_first(5,1,5,'k')
M = N = K = 22
gen_k_first(M,K,N,5,1,5,'k')






def gen_m_first(m,k,n,sched):
  #
  func_def = gen_func_def(m,k,n,sched)
  var_decls = gen_var_decls()
  end = gen_end()
  #
  var_decls += '''
  float32_t *A_ptr, *C_ptr;
    '''
  #
  B_decls = ','.join(["B%d%d" % (i,j) for i in range(k) for j in range(n)])
  C_decls = ','.join(["C%d" % (i) for i in range(n)])
  var_decls += '''
  float32_t ''' + B_decls + ';\n'
  var_decls += ''' 
  float32_t ''' + C_decls + ';\n'
  #
  bounds = '''
  uint32_t en = N / %d; 
  uint32_t ek = K / %d;
  ''' % (n,k)
  #
  #
  outer_loops = '''
  for(n = 0U; n < en; n++) {

    B = pSrcB->pData + %d*n;
    C_ptr = pDst->pData + %d*n;

    for(k = 0U; k < ek; k++) {

      A = pSrcA->pData + %d*k;
  ''' % (n,n,k)
  #
  #
  B_init = []
  for i in range(k):
    for j in range(n-1):
      B_init.append('''
      B%d%d = *B++;''' % (i,j))
    B_init.append('''
      B%d%d = *B;\n''' % (i,n-1))
    B_init.append('''
      B += N - %d;\n''' % (n-1))
  #
  #
  B_init.append('''
      C = C_ptr;\n''')
  B_init = ''.join(B_init)
  #
  #
  mm_sched = '''
      for(m = 0U; m < M; m++) {
        ''' + \
        ''.join(['''
        C%d = *C++;''' % (i) \
          for i in range(n-1)]) + \
        '''
        C%d = *C;\n''' % (n-1) 
  #
  #
  mac_ops = []
  for i in range(k-1):
    for j in range(n-1):
      mac_ops.append('''
        C%d += *A * B%d%d;''' % (j,i,j))
    mac_ops.append('''
        C%d += *A++ * B%d%d;\n''' % (n-1,i,n-1))
  #
  #
  for j in range(n):
    mac_ops.append('''
        C%d += *A * B%d%d;''' % (j,k-1,j))
  #
  C_write = '\n' + ''.join(['''
        *C-- = C%d;''' % (i) \
            for i in range(n-1,0,-1)]) + \
          '''
        *C = C0;\n\n''' 
  #
  #
  mm_sched += ''.join(mac_ops) + C_write + '''\
        A += K - %d;
        C += N;
      }
    }
  }''' % (k-1)
  #
  #
  prog = func_def + var_decls + bounds + outer_loops + B_init + mm_sched + end
  print(prog)






gen_m_first(1,6,4,'m')
