﻿#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>
#include <time.h>
#include <windows.h>
#include <cstdlib>
//#pragma comment(lib, "pthreadVC2.lib")

using namespace std;
const int Columnnum = 3799;
const int Rnum = 2759;
const int Enum = 1953;
const int ArrayColumn = 119;
const int leftbit = 9;//32 - (1 + 43576 % 32) = 7
unsigned int R[Columnnum][ArrayColumn];
unsigned int E[Enum][ArrayColumn];
int First[Enum];
bool IsNULL[Columnnum] = { 1 };
bitset<32> MyBit(0);
char fin[100000] = { 0 };//�ǵø��ݾ��������޸�fin��С
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
void Set_R(int eindex, int rindex) {
    for (int j = 0; j < ArrayColumn; j++) {
        R[rindex][j] = E[eindex][j];
    }
}
void Init_IsNULL() {
    for (int i = 0; i < Columnnum; i++) {
        bool flag = 0;
        for (int j = 0; j < ArrayColumn; j++) {
            if (R[i][j] != 0) {
                flag = 1;
                IsNULL[i] = 0;
                break;
            }
        }
        if (flag == 0) IsNULL[i] = 1;
    }
}
void XOR(int eindex, int rindex) {
    for (int j = 0; j < ArrayColumn; j++) {
        E[eindex][j] = E[eindex][j] ^ R[rindex][j];
    }
}
void Print() {//Print the answer
    for (int i = 0; i < Enum; i++) {
        cout << "*";
        bool isnull = 1;
        for (int j = 0; j < ArrayColumn; j++) {
            if (E[i][j] != 0) {
                isnull = 0;
                break;
            }
        }
        if (isnull) {
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
void Serial() {
    while (1) {
        bool flag = 1;//flag=0 means all EliminatedLines are handled over
        for (int i = 0; i < Enum; i++) {
            while (First[i] != -1) {
                flag = 0;

                if (IsNULL[First[i]] == 0) {
                    XOR(i, First[i]);
                    First[i] = Find_First(i);
                }
                else {
                    // Set_R(i,First[i]);
                    // IsNULL[First[i]]=0;
                    break;
                }
            }
        }
        if (flag == 1) break;
        for (int i = 0; i < Enum; i++) {
            if (First[i] != -1) {
                if (IsNULL[First[i]] == 1) {
                    Set_R(i, First[i]);
                    IsNULL[First[i]] = 0;
                    First[i] = -1;
                    break;
                }
            }
        }
    }

}

int main()
{
    

    Init_R();
    Init_E();
    Init_IsNULL();
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    Serial();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << (tail - head) * 1000.0 / freq  << endl;
    //Print();
    return 0;
}