#include <iostream>
#include <fstream>
#include <sstream>
#include<bitset>
#include<immintrin.h>
#include<time.h>
#include <windows.h>
using namespace std;
const int Columnnum = 130;
const int Rnum = 22;
const int Enum = 8;
const int ArrayColumn = 5;
const int leftbit = 30;
unsigned int R[Columnnum][ArrayColumn];
unsigned int E[Enum][ArrayColumn];
int First[Enum];
bitset<32> MyBit(0);
__m256 te, tr;
int Find_First(int index) {
    int j = 0;
    int cnt = 0;
    while (E[index][j] == 0) {
        j++;
        if (j == ArrayColumn) break;
    }
    if (j == ArrayColumn) return -1;
    unsigned int tmp = E[index][j];
    while (tmp != 0) {
        tmp = tmp >> 1;
        cnt++;
    }
    return Columnnum - 1 - ((j + 1) * 32 - cnt - leftbit);
}
void Init_R() {
    unsigned int a;
    ifstream infile("消元子.txt");

    char fin[100000] = { 0 };
    int index;
    while (infile.getline(fin, sizeof(fin))) {
        std::stringstream line(fin);
        bool flag = 0;
        while (line >> a) {
            if (flag == 0) {
                index = a;
                flag = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            R[index][ArrayColumn - 1 - j] += temp;
        }
    }
}
void Init_E() {
    unsigned int a;
    ifstream infile("被消元行.txt");

    char fin[100000] = { 0 };
    int index = 0;
    while (infile.getline(fin, sizeof(fin))) {
        std::stringstream line(fin);
        int flag = 0;
        while (line >> a) {
            if (flag == 0) {
                First[index] = a;
                flag = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            E[index][ArrayColumn - 1 - j] += temp;
        }
        index++;
    }
}
bool Is_NULL(int index) {
    for (int j = 0; j < ArrayColumn; j++) {
        if (R[index][j] != 0) return 0;
    }
    return 1;
}
void Set_R(int eindex, int rindex) {
    for (int j = 0; j < ArrayColumn; j++) {
        R[rindex][j] = E[eindex][j];
    }
}
void XOR(int eindex, int rindex) {//we do parallel programming here.
    int j = 0;
    for (; j + 8 <= ArrayColumn; j += 8) {
        te = _mm256_loadu_ps((float*)&(E[eindex][j]));
        tr = _mm256_loadu_ps((float*)&(R[rindex][j]));
        te = _mm256_xor_ps(te, tr);
        _mm256_storeu_ps((float*)&(E[eindex][j]), te);
    }
    for (; j < ArrayColumn; j++) {
        E[eindex][j] = E[eindex][j] ^ R[rindex][j];
    }

}
void Align_XOR(int eindex, int rindex) {//we do parallel programming here.
    int j = 0;
    while ((eindex * ArrayColumn + j) % 8 != 0) {//we do alignment here.note that AVX supports the align of 32 bits instead of 16 bits.
        E[eindex][j] = E[eindex][j] ^ R[rindex][j];
        j++;
    }
    for (; j + 8 <= ArrayColumn; j += 8) {
        te = _mm256_load_ps((float*)&(E[eindex][j]));
        tr = _mm256_load_ps((float*)&(R[rindex][j]));
        te = _mm256_xor_ps(te, tr);
        _mm256_store_ps((float*)&(E[eindex][j]), te);
    }
    for (; j < ArrayColumn; j++) {
        E[eindex][j] = E[eindex][j] ^ R[rindex][j];
    }

}
void AVX() {
    for (int i = 0; i < Enum; i++) {
        while (First[i] != -1) {
            if (!Is_NULL(First[i])) {
                XOR(i, First[i]);
                //Align_XOR(i,First[i]);
                First[i] = Find_First(i);
            }
            else {
                Set_R(i, First[i]);
                break;
            }
        }
    }
}
void Print() {//Print the answer
    for (int i = 0; i < Enum; i++) {
        if (First[i] == -1) {
            cout << endl;
            continue;
        }
        for (int j = 0; j < ArrayColumn; j++) {
            if (E[i][j] == 0) continue;
            MyBit = E[i][j];//MENTION: bitset manipulates from the reverse direction
            for (int k = 31; k >= 0; k--) {
                if (MyBit.test(k)) {
                    cout << 32 * (ArrayColumn - j - 1) + k << ' ';
                }
            }
        }
        cout << endl;
    }
}
int main()
{

    long long head, tail, freq;
    Init_R();
    Init_E();
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    AVX();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "N: " << Columnnum << " Time: " << (tail - head) * 1000.0 / freq << "ms" << endl;
    //Print();
    return 0;
}