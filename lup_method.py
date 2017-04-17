""" LUP метод решения СЛАУ

Created by: Andrew Ushakov
02.03.2017
"""

import numpy as np
import functools as ft
import copy


class LUP:
    """ LUP метод решения СЛАУ """

    def __init__(self, path):

        # Считывание матрицы из файла
        self._a = list()  # матрица коэффициентов
        self._b = list()  # матрица решения

        for i in open(path):
            *lst, b_end = list(map(int, i.split()))
            self._a.append(lst)
            self._b.append(b_end)

        self._u = list()  # Матрица U
        self._l = np.eye(len(self._a)).tolist()  # Матрица L
        self._x = np.empty(len(self._a)).tolist()  # Вектор X
        self._p = 0  # Счетчик числа перестановок
        self._compute_matrix()  # Вычисляем матрицы L, U и число перестановок P
		
    def _compute_matrix(self):
        """
        Вычисляет матрицы L и U и находит решение СЛАУ

        :return:
        """

        self._u = copy.deepcopy(self._a)  # После изменений - необходимая матрица U
        n = len(self._u)

        for i in range(n):  # В этом цикле заполняем L и U

            # Поиск ведущего элемента
            max_el_idx = i
            for row in range(i, n):
                if self._u[row][i] > self._u[max_el_idx][i]:
                    max_el_idx = row

            self._p = self._p + 1 if max_el_idx != i else self._p  # Считаем количество перестановок
            self._u[max_el_idx], self._u[i] = self._u[i], self._u[max_el_idx]  # Перестановка строк в U

            #  Перестановка строк и столбцов в L
            self._l[i], self._l[max_el_idx] = self._l[max_el_idx], self._l[i]
            for j in range(n):
                self._l[j][i], self._l[j][max_el_idx] = self._l[j][max_el_idx], self._l[j][i]

            # Перестановка строк в векторе решения b
            self._b[i], self._b[max_el_idx] = self._b[max_el_idx], self._b[i]

            # Заполняем матрицы L и U
            for j in range(i + 1, n):
                self._l[j][i] = self._u[j][i] / self._u[i][i]
                for k in range(i, n):
                    self._u[j][k] -= self._l[j][i] * self._u[i][k]

        z = np.empty(n).tolist()  # Вектор z

        z[0] = self._b[0]  # Первый элемент копируем

        # Заполняем остальные элементы z
        for i in range(1, n):
            z[i] = self._b[i]
            for j in range(0, i):
                z[i] -= self._l[i][j] * z[j]

        # Заполняем вектор X
        self._x[n - 1] = z[n - 1] / self._u[n - 1][n - 1]  # Последний элемент
        for i in range(n - 2, -1, -1):
            self._x[i] = z[i] / self._u[i][i]
            for j in range(i + 1, n):
                self._x[i] -= self._u[i][j] * self._x[j] / self._u[i][i]

    def solution(self):
        """
        Возвращает результат решения СЛАУ

        :return: list self._x - решение СЛАУ
        """
		
        return self._x

    def determinant(self):
        """
        Расчитывает и возвращает определитель матрицы СЛАУ

        :return: Детерминант матрицы СЛАУ
        """

        return (-1) ** self._p * ft.reduce(lambda x, y: x * y, np.array(self._u).diagonal())

    def inverse_matrix(self):
        """
        Вывод обратной матрицы СЛАУ

        :return:
        """

        n = len(self._a)
        l = np.eye(n).tolist()
        b = np.eye(n).tolist()

        m = copy.deepcopy(self._a)

        for i in range(n):  # В этом цикле приводим матрицу к верхнетреугольному виду

            # Поиск ведущего элемента
            max_el_idx = i
            for row in range(i, n):
                if m[row][i] > m[max_el_idx][i]:
                    max_el_idx = row

            # Перестановка строк в матрице
            m[max_el_idx], m[i] = m[i], m[max_el_idx]
            b[max_el_idx], b[i] = b[i], b[max_el_idx]

            l[i], l[max_el_idx] = l[max_el_idx], l[i]
            for j in range(n):
                l[j][i], l[j][max_el_idx] = l[j][max_el_idx], l[j][i]

            for j in range(i + 1, n):  # Заполняем матрицу m и l
                l[j][i] = m[j][i] / m[i][i]
                for k in range(i, n):
                    m[j][k] -= l[j][i] * m[i][k]
                for k in range(0, n):
                    b[j][k] -= l[j][i] * b[i][k]

        invA = np.empty((n, n)).tolist()  # Обратная матрица размера матрицы А

        for i in range(n):
            invA[n - 1][i] = b[n - 1][i] / m[n - 1][n - 1]  # Последний элемент
            for k in range(n - 2, -1, -1):  # От предпоследнего индекса к нулевому
                invA[k][i] = b[k][i] / m[k][k]
                for j in range(k + 1, n):
                    invA[k][i] -= m[k][j] * invA[j][i] / m[k][k]

        return invA

if __name__ == '__main__':

    cls = LUP("input.txt")

    print("\n\nLUP Algorithm\n", "-" * len("LUP Algorithm"), sep = '', end = '\n\n')
    print("Solution:", list(map(lambda x:round(x, 2), cls.solution())), end = '\n\n')
    print("Determinant:", cls.determinant(), end='\n\n')
    print("Inverse matrix:\n", np.array(cls.inverse_matrix()))
