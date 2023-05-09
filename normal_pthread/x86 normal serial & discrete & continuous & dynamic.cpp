#include <iostream>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
#include <immintrin.h>
#include <windows.h>
#include <cstdlib>
using namespace std;
#pragma comment(lib, "pthreadVC2.lib")

//------------------------------------------ �߳̿��Ʊ��� ------------------------------------------
typedef struct
{
    int t_id;
} threadParam_t;

sem_t sem_Division;
pthread_barrier_t barrier;
pthread_mutex_t task;
int index = 0;

const int THREAD_NUM = 8;

// ------------------------------------------ ȫ�ּ������ ------------------------------------------
#define N 1000
const int LOOP = 20;
float A[N][N];
float matrix[N][N];

// �����㷨
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

// pthread_discrete �̺߳���
void* threadFunc_discrete(void* param)
{
    threadParam_t* thread_param_t = (threadParam_t*)param;
    int t_id = thread_param_t->t_id;
    for (int k = 0; k < N; k++)
    {
        // �����ǰ��0���̣߳�����г��������������̴߳��ڵȴ�״̬
        if (t_id == 0)
        {
            // float Akk = matrix[k][k];
            __m128 Akk = _mm_set_ps1(matrix[k][k]);
            int j;
            //���Ƕ������
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

        // ����������ɺ������0���̣߳�����Ҫ���������߳�
        if (t_id == 0)
        {
            for (int i = 1; i < THREAD_NUM; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            // ѭ����������
            for (int i = k + t_id; i < N; i += (THREAD_NUM - 1))
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

        // �����߳�׼��������һ��
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread_discrete �����㷨
void calculate_pthread_discrete()
{
    // �ź�����ʼ��
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, THREAD_NUM);

    // �����߳�
    pthread_t threads[THREAD_NUM];
    threadParam_t thread_param_t[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc_discrete, (void*)(&thread_param_t[i]));
    }

    // ����ִ���߳�
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // �����ź���
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
}

// pthread_discrete �̺߳���
void* threadFunc_continuous(void* param)
{
    threadParam_t* thread_param_t = (threadParam_t*)param;
    int t_id = thread_param_t->t_id;
    for (int k = 0; k < N; k++)
    {
        // �����ǰ��0���̣߳�����г��������������̴߳��ڵȴ�״̬
        if (t_id == 0)
        {
            // float Akk = matrix[k][k];
            __m128 Akk = _mm_set_ps1(matrix[k][k]);
            int j;
            //���Ƕ������
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

        // ����������ɺ������0���̣߳�����Ҫ���������߳�
        if (t_id == 0)
        {
            for (int i = 1; i < THREAD_NUM; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            int L = ceil((N - k) * 1.0 / (THREAD_NUM - 1));
            // ѭ����������
            for (int i = k + (t_id - 1) * L + 1; i < N && i < k + t_id * L + 1; i++)
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

        // �����߳�׼��������һ��
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread_continuous �����㷨
void calculate_pthread_continuous()
{
    // �ź�����ʼ��
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, THREAD_NUM);

    // �����߳�
    pthread_t threads[THREAD_NUM];
    threadParam_t thread_param_t[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc_continuous, (void*)(&thread_param_t[i]));
    }

    // ����ִ���߳�
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // �����ź���
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
}

// pthread_dynamci �̺߳���
void* threadFunc_dynamic(void* param)
{
    threadParam_t* thread_param_t = (threadParam_t*)param;
    int t_id = thread_param_t->t_id;
    for (int k = 0; k < N; k++)
    {
        // �����ǰ��0���̣߳�����г��������������̴߳��ڵȴ�״̬
        if (t_id == 0)
        {
            // float Akk = matrix[k][k];
            __m128 Akk = _mm_set_ps1(matrix[k][k]);
            int j;
            //���Ƕ������
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
            index = k + 1;
        }
        else
        {
            sem_wait(&sem_Division);
        }

        // ����������ɺ������0���̣߳�����Ҫ���������߳�
        if (t_id == 0)
        {
            for (int i = 1; i < THREAD_NUM; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            while (index < N)
            {
                pthread_mutex_lock(&task);
                int i = index;
                if (i < N)
                {
                    index++;
                    pthread_mutex_unlock(&task);
                }
                else
                {
                    pthread_mutex_unlock(&task);
                    break;
                }
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

        // �����߳�׼��������һ��
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread_dynamic �����㷨
void calculate_pthread_dynamic()
{
    // �ź�����ʼ��
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, THREAD_NUM);
    task = PTHREAD_MUTEX_INITIALIZER;

    // �����߳�
    pthread_t threads[THREAD_NUM];
    threadParam_t thread_param_t[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc_dynamic, (void*)(&thread_param_t[i]));
    }

    // ����ִ���߳�
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // �����ź���
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&task);
}

void init_data()
{
    for (int i = 0; i < N; i++)
    {
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            A[i][j] = rand();
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i][j] += A[k][j];
}

// ��data��ʼ��matrix����֤ÿ�ν��м����������һ�µ�
void init_matrix()
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = A[i][j];
}

int main()
{
    init_data();
    long long head, tail, freq;

    // ====================================== serial ======================================

    //// similar to CLOCKS_PER_SEC
    //QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    //// start time
    //QueryPerformanceCounter((LARGE_INTEGER*)&head);
    //for (int i = 0; i < LOOP; i++)
    //{
    //    init_matrix();
    //    calculate_serial();
    //}
    //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    //cout << (tail - head) * 1000.0 / (freq * LOOP) << endl;

    // ====================================== pthread_discrete ======================================

    //// similar to CLOCKS_PER_SEC
    //QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    //// start time
    //QueryPerformanceCounter((LARGE_INTEGER*)&head);
    //for (int i = 0; i < LOOP; i++)
    //{
    //    init_matrix();
    //    calculate_pthread_discrete();
    //}
    //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    //cout << (tail - head) * 1000.0 / (freq * LOOP) << endl;

    // ====================================== pthread_continuous ======================================
    
    //// similar to CLOCKS_PER_SEC
    //QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    //// start time
    //QueryPerformanceCounter((LARGE_INTEGER*)&head);
    //for (int i = 0; i < LOOP; i++)
    //{
    //    init_matrix();
    //    calculate_pthread_continuous();
    //}
    //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    //cout << (tail - head) * 1000.0 / (freq * LOOP) << endl;

    // ====================================== pthread_dynamic ======================================

        // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        calculate_pthread_dynamic();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << (tail - head) * 1000.0 / (freq * LOOP) << endl;

    //==========================================================================================

    system("pause");
}
