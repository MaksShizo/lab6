#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <tbb/tbb.h>

using namespace std;
int n;

void FillAArr(double** a, int n) {
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= n; j++) {
			a[i][j] = rand() % 100;
		}
	}
}

void FillBArr(double** a, double* x, int n) {
	for (int i = 1; i <= n; i++) {
		a[i][n + 1] = 0;
		for (int j = 1; j <= n; j++) {
			a[i][n + 1] += a[i][j] * x[j];
		}
	}
}

bool is_equal(double x, double y) {
	return fabs(x - y) < 0, 00000001;
}

bool CheckAnswers(double* x, double* x2, int n) {
	for (int i = 1; i <= n; i++) {
		if (!is_equal(x[i], x2[i]))
			return false;
	}
	return true;
}

void FillXArr(double* x, int n) {
	for (int i = 1; i <= n; i++) {
		x[i] = rand() % 100;
	}
}

void Gauss(double** a, double* x, int n) {
	/* Прямой ход*/
	unsigned int start = clock();
	for (int k = 1; k < n; k++) {
		tbb::parallel_for(tbb::blocked_range<int>(k + 1, n), [&](const tbb::blocked_range<int>& range) {
			for (int j = range.begin(); j < range.end(); j++) {
				double d = a[j][k - 1] / a[k - 1][k - 1];
				for (int i = 0; i <= n; i++) {
					a[j][i] = a[i][i] - d * a[k - 1][i];
				}
			}
			});
	}

	/*Обратный ход*/
	for (int i = n - 1; i >= 0; i--) {
		double sum = 0.0;
		for (int j = i + 1; j < n; j++) {
			sum += a[i][j] * x[j]; // / a[i][i];
		}
		x[i] = (x[i] - sum) / a[i][i];
	}
	cout << "Гаусс через oneTBB " << clock() - start << endl;
}

void GaussOMP(double** a, double* x, int n) {
	/* Прямой ход*/
	unsigned int start = clock();
	for (int k = 1; k < n; k++) {
#pragma omp parallel for
		for (int j = k; j < n; j++) {
			double d = a[j][k - 1] / a[k - 1][k - 1];
			for (int i = 0; i <= n; i++) {
				a[j][i] = a[i][i] - d * a[k - 1][i];
			}
		}
	}

	/*Обратный ход*/
	for (int i = n - 1; i >= 0; i--) {
		double sum = 0.0;
		for (int j = i + 1; j < n; j++) {
			sum += a[i][j] * x[j]; // / a[i][i];
		}
		x[i] = (x[i] - sum) / a[i][i];
	}
	cout << "Гаусс через OpenMP " << clock() - start << endl;
}

int main()
{
	setlocale(LC_ALL, "Russian");
	cout << "Введите размерность матрицы: " << endl;
	cin >> n;
	double** a = new double* [n];
	double* x = new double[n];
	double* x1 = new double[n];
	double** b = new double* [n];
	double* bx = new double[n];
	double* bx1 = new double[n];
	for (int i = 0; i <= n; i++) {
		a[i] = new double[n + 1];
	}
	FillAArr(a, n);
	FillXArr(x1, n);
	FillBArr(a, x1, n);
	b = a;
	bx = x;
	bx1 = x1;
	Gauss(a, x, n);
	cout << "check answers " << CheckAnswers(x1, x, n) << endl;
	GaussOMP(b, bx, n);
	cout << "check answers " << CheckAnswers(bx1,bx, n) << endl;
	system("pause");
	return 0;

}
