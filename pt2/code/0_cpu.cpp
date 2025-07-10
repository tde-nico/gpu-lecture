void matrix_mul(float *M, float *N, float *P, int side) {
	for (int i = 0; i < side; ++i) {
		for (int j = 0; j < side; ++j) {
			float sum = 0;
			for (int k = 0; k < side; ++k) {
				float m = M[i * side + k];
				float n = N[k * side + j];
				sum += m * n;
			}
			P[i * side + j] = sum;
		}
	}
}
