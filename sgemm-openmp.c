#include <stdio.h>
#include <emmintrin.h>
#include <omp.h>

void sgemm( int m, int n, float *A, float *C )
{

    __m128 c_vector1, c_vector2, c_vector3, c_vector4, c_vector5, c_vector6, c_vector7;
    __m128 tmp_vector;
    __m128 mult_vector1, mult_vector2, mult_vector3, mult_vector4, mult_vector5, mult_vector6, mult_vector7;
    #pragma omp parallel
#pragma omp for schedule(dynamic, 4) private(c_vector1, c_vector2, c_vector3, c_vector4, c_vector5, c_vector6, c_vector7, tmp_vector, mult_vector1, mult_vector2, mult_vector3, mult_vector4, mult_vector5, mult_vector6, mult_vector7)
    for( int i = 0; i < (m - m%4)*m; i+=m ) {
	float *c_point = C + i;
	for( int k = 0; k < m - 27; k+=28 ) {
	    c_vector1 = _mm_loadu_ps(c_point + k);
	    c_vector2 = _mm_loadu_ps(c_point + k + 4);
	    c_vector3 = _mm_loadu_ps(c_point + k + 8);
	    c_vector4 = _mm_loadu_ps(c_point + k + 12);
	    c_vector5 = _mm_loadu_ps(c_point + k + 16);
	    c_vector6 = _mm_loadu_ps(c_point + k + 20);
	    c_vector7 = _mm_loadu_ps(c_point + k + 24);
	    float *a_point = A + k;
	    int something = i/m;
	    for( int j = 0; j < n*m; j+=m ) {
		tmp_vector = _mm_load1_ps(A + something + j);
		
		mult_vector1 = _mm_mul_ps(tmp_vector, _mm_loadu_ps(a_point + j));
	 	mult_vector2 = _mm_mul_ps(tmp_vector, _mm_loadu_ps(a_point + 4 + j));
		mult_vector3 = _mm_mul_ps(tmp_vector, _mm_loadu_ps(a_point + 8 + j));
		mult_vector4 = _mm_mul_ps(tmp_vector, _mm_loadu_ps(a_point + 12 + j));
		mult_vector5 = _mm_mul_ps(tmp_vector, _mm_loadu_ps(a_point + 16 + j));
		mult_vector6 = _mm_mul_ps(tmp_vector, _mm_loadu_ps(a_point + 20 + j));
		mult_vector7 = _mm_mul_ps(tmp_vector, _mm_loadu_ps(a_point + 24 + j));
	
		c_vector1 = _mm_add_ps(c_vector1, mult_vector1);
		c_vector2 = _mm_add_ps(c_vector2, mult_vector2);
		c_vector3 = _mm_add_ps(c_vector3, mult_vector3);
		c_vector4 = _mm_add_ps(c_vector4, mult_vector4);
		c_vector5 = _mm_add_ps(c_vector5, mult_vector5);
		c_vector6 = _mm_add_ps(c_vector6, mult_vector6);
		c_vector7 = _mm_add_ps(c_vector7, mult_vector7);
	    }
	    _mm_storeu_ps((c_point + k), c_vector1);
	    _mm_storeu_ps((c_point + k + 4), c_vector2);
	    _mm_storeu_ps((c_point + k + 8), c_vector3);
	    _mm_storeu_ps((c_point + k + 12), c_vector4);
	    _mm_storeu_ps((c_point + k + 16), c_vector5);
	    _mm_storeu_ps((c_point + k + 20), c_vector6);
	    _mm_storeu_ps((c_point + k + 24), c_vector7);
	}
    
        
	//edge cases for loop unrolling
	for( int k = m/28*28; k < m - m%4; k+=4 ) {
	    c_vector1 = _mm_loadu_ps(C + k + i);
	    float* a_point = A + k;
	    for( int j = 0; j < n*m; j+=m ) {
		tmp_vector = _mm_load1_ps(A + i/m + j);	 		
		mult_vector1 = _mm_mul_ps(tmp_vector, _mm_loadu_ps(a_point + j));	
		c_vector1 = _mm_add_ps(c_vector1, mult_vector1);
	    }
	    _mm_storeu_ps((C + k + i), c_vector1); 
	}
    }

    //edge cases
    if (m % 4 != 0) {
	#pragma omp parallel for
    	for (int i = 0; i < m; i++) {
    	    for (int j = m - m % 4; j < m; j++) {
    		for (int k = 0; k < n*m; k+=m) {
    		    *(C + i*m + j) += *(A + j + k) * *(A + i + k);
		    if (m != n) {
			*(C + j*m + i) += *(A + i + k) * *(A + j + k);
		    }
    		}
    	    }
    	}
	if (m != n) {
	    for (int i = m - m%4; i < m; i++) {
		for (int j = m - m%4; j < m; j++) {
		    *(C + i*m + j) /= 2;
		}
	    }
	}
    }     
}

/* int main (){ */
/*     /\*  float A[20] = {1, 2, 3, 4, *\/ */
/*     /\* 		   0, 1, 0, 1, *\/ */
/*     /\* 		   1, 0, 1, 0, *\/ */
/*     /\* 		   0, 1, 0, 1, *\/ */
/*     /\* 		   1, 0, 0, 0}; *\/ */
/*     /\* float C[20] = {0, 0, 0, 0, *\/ */
/*     /\* 		   0, 0, 0, 0, *\/ */
/*     /\* 		   0, 0, 0, 0, *\/ */
/*     /\* 		   0, 0, 0, 0, *\/ */
/*     /\* 		   0, 0, 0, 0}; *\/ */
/*     int m = 5; */
/*     int n = 6; */
/*     float A[30] = {1, 2, 3, 4, 1, */
/*     		   0, 1, 0, 1, 1, */
/*     		   1, 0, 1, 0, 1, */
/*     		   0, 1, 0, 1, 1, */
/*     		   1, 1, 1, 1, 1, */
/* 		   1, 0, 0, 0, 0}; */
/*     float C[25] = {0, 0, 0, 0, 0, */
/*     		   0, 0, 0, 0, 0, */
/*     		   0, 0, 0, 0, 0, */
/*     		   0, 0, 0, 0, 0, */
/*     		   0, 0, 0, 0, 0}; */
/*     /\* float A[16] = {1, 2, 3, 4,  *\/ */
/*     /\* 		   0, 1, 0, 1, *\/ */
/*     /\* 		   1, 0, 1, 0, *\/ */
/*     /\* 		   0, 1, 0, 1}; *\/ */
/*     /\* float C[16] = {0, 0, 0, 0, *\/ */
/*     /\* 		   0, 0, 0, 0, *\/ */
/*     /\* 		   0, 0, 0, 0,  *\/ */
/*     /\* 		   0, 0, 0, 0}; *\/ */
/*     for (int i = 0; i < n; i++ ) { */
/* 	for (int j = 0; j < m; j++) { */
/* 	    printf("%f\t", *(A + i*m + j)); */
/* 	} */
/* 	printf("\n"); */
/*     } */
/*     printf("\n"); */
/*     for( int i = 0; i < m; i++ ) */
/*       for( int k = 0; k < n; k++ )  */
/*         for( int j = 0; j < m; j++ )  */
/* 	  C[i+j*m] += A[i+k*m] * A[j+k*m]; */
/*     for( int i = 0; i < m; i++ ) { */
/*       for( int j = 0; j < m; j++ ) { */
/*     	    printf("\t%f", *(C + i*m + j)); */
/*     	} */
/*     	printf("\n"); */
/*     } */
/*     printf("\n"); */
/*        float B[30] = {1, 2, 3, 4, 1, */
/*     		   0, 1, 0, 1, 1, */
/*     		   1, 0, 1, 0, 1, */
/*     		   0, 1, 0, 1, 1, */
/*     		   1, 1, 1, 1, 1, */
/* 		      1, 0, 0, 0, 0}; */
/*     float D[36] = {0, 0, 0, 0, 0, */
/*     		   0, 0, 0, 0, 0, */
/*     		   0, 0, 0, 0, 0, */
/*     		   0, 0, 0, 0, 0, */
/*     		   0, 0, 0, 0, 0}; */
/*     sgemm(m, n, B, D); */
/* } */
