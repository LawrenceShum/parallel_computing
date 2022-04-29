#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <arm_neon.h>

using namespace std;

struct Timer
{
    /* data */
    using clock_t = std::chrono::high_resolution_clock;
    using time_point_t = std::chrono::time_point<clock_t>;
    using elapsed_time_t = std::chrono::duration<double, std::milli>;

    time_point_t StartTime;
    time_point_t StopTime;
    elapsed_time_t ElapsedTime;

    void Start()
    {
        ElapsedTime = elapsed_time_t::zero();
        StartTime = clock_t::now();
    }

    void Stop()
    {
        StopTime = clock_t::now();
        std::chrono::duration<double, std::milli> elapsedTime = StopTime - StartTime;
        std::cout << elapsedTime.count();
    }
};


void divide_neon(float* input, float divisor, int count)
{
    float32x4_t d = vmovq_n_f32(divisor);
    int new_count = count;
    if(count % 4 != 0)
    {
        new_count = 4 * (count / 4);
    }
    for(int i = 0; i < new_count; i+= 4)
    {
        float32x4_t in, out;
        in = vld1q_f32(input);
        out = vdivq_f32(in, d);
        vst1q_f32(input, out);
        input += 4;
    }
    for(int i = 0; i < count - new_count; i++)
    {
        input[i] /= divisor;
    }
}

void sub_neon(float* input1, float* input2, float front, int count)
{
    float32x4_t front_neon = vmovq_n_f32(front);
    int new_count = count;
    if(count % 4 != 0)
    {
        new_count = 4 * (count / 4);
    }
    for(int i = 0; i < new_count; i += 4)
    {
        float32x4_t in1, in2, out1, out2;
        in1 = vld1q_f32(input1);
        in2 = vld1q_f32(input2);
        out1 = vmulq_f32(in2, front_neon);
        out2 = vsubq_f32(in1, out1);
        vst1q_f32(input1, out2);
        input1 += 4;
        input2 += 4;
    }
    for(int i = 0; i < count - new_count; i++)
    {
        input1[i] = input1[i] - front * input2[i];
    }
}

void sequential_gauss_elimination(float* input, int size)
{
    for(int k = 0; k < size; k++)
    {
        float* p = input + k * size;
        float front = p[k];
        for(int i = 0; i < size; i++)
        {
            p[i] /= front;
        }
        for(int k2 = k + 1; k2 < size; k2++)
        {
            float* p2 = input + k2 * size;
            for(int j = k; j < size; j++)
            {
                p2[j] = p2[j] - p2[k] * p[j];
            }
        }
    }
}

int main()
{
    bool if_print = false;
    Timer T;
    const int size = 500;
    //随机生成一些数字
    srand((unsigned)time(NULL));
    int a = 1, b = 50;
    float* matrix = new float[size * size];
    float* matrix1 = new float[size * size];
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            matrix[i*size + j] = (rand() % (b-a) + a);
            matrix1[i*size + j] = matrix[i*size + j];
        }
    }

    T.Start();
    //开始高斯消元
    for(int k = 0; k < size; k++)
    {
        float* p = matrix + k * size;
        divide_neon(p, matrix[k * size + k], size);
        for(int k2 = k + 1; k2 < size; k2++)
        {
            int c = k / 4;
            int kk = (c + 1) * 4;
            float* p2 = p + kk;
            float* p3 = matrix + k2 * size + kk;
            sub_neon(p3, p2, matrix[k2 * size + k], size - kk);
            for(int dd = k; dd < kk; dd ++)
            {
                matrix[k2*size+dd] = matrix[k2*size+dd] - matrix[k2*size+k]*p[dd];
            }
        }
    }
    std::cout << "SIMD computing time = ";
    T.Stop();
    std::cout <<" ms" << std::endl;

    T.Start();
    sequential_gauss_elimination(matrix1, size);
    std::cout << "sequential computing time = ";
    T.Stop();
    std::cout << " ms" <<std::endl;

    if(if_print)
    {
        std::cout << "result of SIMD computing :" << std::endl;
        for(int i = 0; i < size; i++)
        {
            for(int j = 0; j < size; j++)
            {
                std::cout << matrix[i*size + j] << "     ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "result of sequantial computing :" << std::endl;
        for(int i = 0; i < size; i++)
        {
            for(int j = 0; j < size; j++)
            {
                std::cout << matrix1[i*size + j] << "     ";
            }
            std::cout << std::endl;
        }
    }
    return 0;
}