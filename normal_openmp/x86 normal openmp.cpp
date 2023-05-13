#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <immintrin.h> // SSE
#include <pthread.h>   // pthread
#include <semaphore.h>
#include <omp.h>
#include <windows.h>
#include <cstdlib>
#define _PRINT
// #define _TEST
#define NUM_THREADS 8
using namespace std;
#pragma comment(lib, "pthreadVC2.lib")

// ============================================== pthread 线程控制变量 ==============================================
typedef struct
{
    int t_id;
} threadParam_t;

sem_t sem_Division;
pthread_barrier_t barrier;
// ============================================== 运算变量 ==============================================
#define N 1000
const int L = 100;//N<1000,L=100;N>=1000,L=500
const int LOOP = 20;
float** A;
float** matrix = NULL;

ofstream res_stream;

void init_data();
void init_matrix();
void calculate_serial();
void calculate_SIMD();
void calculate_openmp_single_SIMD();
void calculate_pthread();
void calculate_openmp_schedule_static();
void calculate_openmp_schedule_dynamic();
void calculate_openmp_schedule_guided();
void calculate_openmp_schedule_guided_nowait();
void calculate_openmp_schedule_guided_SIMD();
void calculate_openmp_static_thread();
void calculate_openmp_dynamic_thread();
void calculate_openmp_row();
void calculate_openmp_column();
void print_matrix();
void test(int);
void print_result(int);

int main()
{
#ifdef _TEST
    res_stream.open("result.csv", ios::out);
    for (int i = 1000; i <= 1000; i += 100)
        test(i);
    for (int i = 1000; i <= 3000; i += 500)
        test(i);
    res_stream.close();
#endif
#ifdef _PRINT
    test(N);
#endif
    system("pause");
    return 0;
}

// 初始化数据
void init_data()
{
    A = new float* [N], matrix = new float* [N];
    for (int i = 0; i < N; i++)
        A[i] = new float[N], matrix[i] = new float[N];
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++)
            A [i][j] = rand() * 1.0 / RAND_MAX * L;
    for (int i = 0; i < N - 1; i++)
        for (int j = i + 1; j < N; j++)
            for (int k = 0; k < N; k++)
                A [j][k] += A [i][k];
}

// 用data初始化matrix，保证每次进行计算的数据是一致的
void init_matrix()
{
    if (matrix != NULL)
        for (int i = 0; i < N; i++)
            delete[] matrix[i];
    delete[] matrix;
    matrix = new float* [N];
    for (int i = 0; i < N; i++)
        matrix[i] = new float[N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = A[i][j];
}

// 串行算法
void calculate_serial()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// SSE并行算法
void calculate_SIMD()
{
    for (int k = 0; k < N; k++)
    {
        // float Akk = matrix[k][k];
        __m128 Akk = _mm_set_ps1(matrix[k][k]);
        int j;
        // 并行处理
        for (j = k + 1; j + 3 < N; j += 4)
        {
            // float Akj = matrix[k][j];
            __m128 Akj = _mm_loadu_ps(matrix[k] + j);
            // Akj = Akj / Akk;
            Akj = _mm_div_ps(Akj, Akk);
            // Akj = matrix[k][j];
            _mm_storeu_ps(matrix[k] + j, Akj);
        }
        // 串行处理结尾
        for (; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            // float Aik = matrix[i][k];
            __m128 Aik = _mm_set_ps1(matrix[i][k]);
            for (j = k + 1; j + 3 < N; j += 4)
            {
                // float Akj = matrix[k][j];
                __m128 Akj = _mm_loadu_ps(matrix[k] + j);
                // float Aij = matrix[i][j];
                __m128 Aij = _mm_loadu_ps(matrix[i] + j);
                // AikMulAkj = matrix[i][k] * matrix[k][j];
                __m128 AikMulAkj = _mm_mul_ps(Aik, Akj);
                // Aij = Aij - AikMulAkj;
                Aij = _mm_sub_ps(Aij, AikMulAkj);
                // matrix[i][j] = Aij;
                _mm_storeu_ps(matrix[i] + j, Aij);
            }
            // 串行处理结尾
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 单独使用openmp进行simd优化
void calculate_openmp_single_SIMD()
{
    int i, j, k;
    float tmp;
    int Num = N;
#pragma omp parallel num_threads(1) private(i, j, k, tmp) shared(matrix, Num)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// pthread_discrete 线程函数
void* threadFunc(void* param)
{
    threadParam_t* thread_param_t = (threadParam_t*)param;
    int t_id = thread_param_t->t_id;
    for (int k = 0; k < N; k++)
    {
        // 如果当前是0号线程，则进行除法操作，其余线程处于等待状态
        if (t_id == 0)
        {
            // float Akk = matrix[k][k];
            __m128 Akk = _mm_set_ps1(matrix[k][k]);
            int j;
            //考虑对齐操作
            for (j = k + 1; j + 3 < N; j += 4)
            {
                // float Akj = matrix[k][j];
                __m128 Akj = _mm_loadu_ps(matrix[k] + j);
                // Akj = Akj / Akk;
                Akj = _mm_div_ps(Akj, Akk);
                // Akj = matrix[k][j];
                _mm_storeu_ps(matrix[k] + j, Akj);
            }
            for (; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        else
        {
            sem_wait(&sem_Division);
        }

        // 除法操作完成后，如果是0号线程，则需要唤醒其他线程
        if (t_id == 0)
        {
            for (int i = 1; i < NUM_THREADS; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            // 循环划分任务
            for (int i = k + t_id; i < N; i += (NUM_THREADS - 1))
            {
                // float Aik = matrix[i][k];
                __m128 Aik = _mm_set_ps1(matrix[i][k]);
                int j = k + 1;
                for (; j + 3 < N; j += 4)
                {
                    // float Akj = matrix[k][j];
                    __m128 Akj = _mm_loadu_ps(matrix[k] + j);
                    // float Aij = matrix[i][j];
                    __m128 Aij = _mm_loadu_ps(matrix[i] + j);
                    // AikMulAkj = matrix[i][k] * matrix[k][j];
                    __m128 AikMulAkj = _mm_mul_ps(Aik, Akj);
                    // Aij = Aij - AikMulAkj;
                    Aij = _mm_sub_ps(Aij, AikMulAkj);
                    // matrix[i][j] = Aij;
                    _mm_storeu_ps(matrix[i] + j, Aij);
                }
                for (; j < N; j++)
                {
                    matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
                }
                matrix[i][k] = 0;
            }
        }

        // 所有线程准备进入下一轮
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread_discrete 并行算法
void calculate_pthread()
{
    // 信号量初始化
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    // 创建线程
    pthread_t threads[NUM_THREADS];
    threadParam_t thread_param_t[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc, (void*)(&thread_param_t[i]));
    }

    // 加入执行线程
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // 销毁信号量
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
}

// 静态数据划分
void calculate_openmp_schedule_static()
{
    int i, j, k;
    float tmp;
    int Num = N;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, Num)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(static)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 动态数据划分
void calculate_openmp_schedule_dynamic()
{
    int i, j, k;
    float tmp;
    int Num = N;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, Num)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 自动调整数据划分
void calculate_openmp_schedule_guided()
{
    int i, j, k;
    float tmp;
    int Num = N;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, Num)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 使用nowait
void calculate_openmp_schedule_guided_nowait()
{
    int i, j, k;
    float tmp;
    int Num = N;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, Num)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(guided) nowait
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// openmp + simd
void calculate_openmp_schedule_guided_SIMD()
{
    int i, j, k;
    int Num = N;
    float tmp;
    __m128 Akk, Akj, Aik, Aij, AikMulAkj;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp, Akk, Akj, Aik, Aij, AikMulAkj) shared(matrix, Num)
    for (k = 0; k < N; k++)
    {
        // float Akk = matrix[k][k];
        Akk = _mm_set_ps1(matrix[k][k]);
        int j;
        // 并行处理
        tmp = matrix[k][k];
#pragma omp single
        {
            for (j = k + 1; j + 3 < N; j += 4)
            {
                // float Akj = matrix[k][j];
                Akj = _mm_loadu_ps(matrix[k] + j);
                // Akj = Akj / Akk;
                Akj = _mm_div_ps(Akj, Akk);
                // Akj = matrix[k][j];
                _mm_storeu_ps(matrix[k] + j, Akj);
            }
            // 串行处理结尾
            for (; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1;
        }
#pragma omp for schedule(guided)
        for (int i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
            // float Aik = matrix[i][k];
            Aik = _mm_set_ps1(matrix[i][k]);
            for (j = k + 1; j + 3 < N; j += 4)
            {
                // float Akj = matrix[k][j];
                Akj = _mm_loadu_ps(matrix[k] + j);
                // float Aij = matrix[i][j];
                Aij = _mm_loadu_ps(matrix[i] + j);
                // AikMulAkj = matrix[i][k] * matrix[k][j];
                AikMulAkj = _mm_mul_ps(Aik, Akj);
                // Aij = Aij - AikMulAkj;
                Aij = _mm_sub_ps(Aij, AikMulAkj);
                // matrix[i][j] = Aij;
                _mm_storeu_ps(matrix[i] + j, Aij);
            }
            // 串行处理结尾
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 静态线程版本
void calculate_openmp_static_thread()
{
    int i, j, k;
    float tmp;
    int Num = N;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, Num)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 动态线程版本
void calculate_openmp_dynamic_thread()
{
    int i, j, k;
    float tmp;
    int Num = N;
    for (k = 0; k < N; k++)
    {
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, tmp) shared(k, matrix, Num)
        {
#pragma omp single
            {
                tmp = matrix[k][k];
#pragma omp simd
                for (j = k + 1; j < N; j++)
                {
                    matrix[k][j] = matrix[k][j] / tmp;
                }
                matrix[k][k] = 1.0;
            }
#pragma omp for schedule(guided)
            for (i = k + 1; i < N; i++)
            {
                tmp = matrix[i][k];
#pragma omp simd
                for (j = k + 1; j < N; j++)
                {
                    matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
                }
                matrix[i][k] = 0;
            }
        }
    }
}

// 按行划分
void calculate_openmp_row()
{
    int i, j, k;
    int Num = N;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, Num)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 按列划分
void calculate_openmp_column()
{
    int i, j, k;
    int Num = N;
#pragma omp parallel num_threads(NUM_THREADS), default(none), private(i, j, k), shared(matrix, Num)
    for (k = 0; k < N; k++)
    {
#pragma omp for schedule(guided)
        for (j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
            for (i = k + 1; i < N; i++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
        }
#pragma omp single
        {
            matrix[k][k] = 1;
            for (i = k + 1; i < N; i++)
            {
                matrix[i][k] = 0;
            }
        }
    }
}

// 打印矩阵
void print_matrix()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

// 测试函数
void test(int n)
{

    cout << "=================================== " << N << " ===================================" << endl;
#ifdef _TEST
    res_stream << n;
#endif

    long long head, tail, freq;
    float time ;
    init_data();
    // ====================================== serial ======================================
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_serial();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "serial:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;
    //// ====================================== SIMD ======================================
    
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_SIMD();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "SIMD:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;

    //// ====================================== openmp_single_SIMD ======================================
    
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_openmp_single_SIMD();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "openmp_single_SIMD:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;

    //// ====================================== pthread ======================================
    
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_pthread();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "pthread:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;

    //// ====================================== openmp_schedule_static ======================================
   
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_openmp_schedule_static();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "openmp_schedule_static:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;
    
    //// ====================================== openmp_schedule_dynamic ======================================
    
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_openmp_schedule_dynamic();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "openmp_schedule_dynamic:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;

    //// ====================================== openmp_schedule_guided ======================================
    
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_openmp_schedule_guided();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "openmp_schedule_guided:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;
    
    //// ====================================== openmp_schedule_guided_nowait ======================================
    
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_openmp_schedule_guided_nowait();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "openmp_schedule_guided_nowait:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;
    
    //// ====================================== openmp_schedule_guided_SIMD ======================================
    
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_openmp_schedule_guided_SIMD();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "openmp_schedule_guided_SIMD:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;

    //// ====================================== openmp_static_thread ======================================
    
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_openmp_static_thread();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "openmp_static_thread:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;

    //// ====================================== openmp_dynamic_thread ======================================
    
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_openmp_dynamic_thread();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "openmp_dynamic_thread:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;

    //// ====================================== openmp_row ======================================
    
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_openmp_row();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "openmp_row:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;

    //// ====================================== openmp_column ======================================
    
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_openmp_column();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "openmp_column:" << (tail - head) * 1000.0 / (freq * LOOP) << "ms" << endl;
    
    //// ===========================================================================================

#ifdef _TEST
    res_stream << endl;
#endif
}

// 结果打印
void print_result(int time)
{
#ifdef _TEST
    res_stream << "," << time / LOOP;
#endif
#ifdef _PRINT
    print_matrix();
#endif
}