#include <stdio.h>
#define mmax 20

void input(int *n, int *m, double [][mmax], double [][mmax]);
void output(int n, int m, double [][mmax]);
void inverse(int n, double [][mmax]);
void process(int, int, double [][mmax], double[][mmax], double [][mmax]);
void transverse(int, int, double [][mmax], double[][mmax]);
void matrix_plus(int, int, double [][mmax], double [][mmax], double [][mmax], double);
void matrix_product(int, int, int, double [][mmax], double [][mmax], double [][mmax]);

int main()
{
  int n,m;
  double matrix[mmax][mmax], block[mmax][mmax], solution[mmax][mmax];
  input(&n, &m, matrix, block);
  process(n, m, matrix, block, solution);
	output(n, m, solution);
	return 0;
}

void process(int n, int m, double matrix[][mmax], double block[][mmax], double x[][mmax])
{
	double r[mmax][mmax], p[mmax][mmax];
	double t1[mmax][mmax], t2[mmax][mmax], t3[mmax][mmax], t4[mmax][mmax], r_t[mmax][mmax], p_t[mmax][mmax];
	int i,j,k;
	double alpha[mmax][mmax], beta[mmax][mmax];
	for (i=0; i!=n; i++)
	  for (j=0; j!=m; j++) x[i][j]=0;
	matrix_product(n, n, m, matrix, x, t1);
	matrix_plus(n, m, block, t1, r, -1);
	for (i=0; i!=n; i++)
	  for (j=0; j!=m; j++)
	    p[i][j]=r[i][j];
	k=0;
	while (k<n) {
		transverse(n, m, r, r_t);
		transverse(n, m, p, p_t);
		matrix_product(m, n, m, r_t, r, t1);
		matrix_product(n, n, m, matrix, p, t2);
		matrix_product(m, n, m, p_t, t2, t3);
		inverse(m, t3);
		matrix_product(m, m, m, t3, t1, alpha);
		
		matrix_product(n, m, m, t2, alpha, t4);
		matrix_plus(n, m, x, t4, x, 1);
		
		matrix_product(n, m, m, t2, alpha, t4);
		matrix_plus(n, m, r, t4, r, -1);
		
		transverse(n, m, r, r_t);
		matrix_product(m, n, m, r_t, r, t2);
		inverse(m, t1);
		matrix_product(m, m, m, t1, t2, beta);
		matrix_product(n, m, m, p, beta, t3);
		matrix_plus(n, m, r, t3, p, 1);
		k++;
	}
  return;
}

void transverse(int n, int m, double A[][mmax], double B[][mmax])
{
	int i,j;
	for (i=0; i!=n; i++)
		for (j=0; j!=m; j++)
			B[j][i]=A[i][j];
	return;
}

//Input
void input(int *n, int *m, double matrix[][mmax], double block[][mmax]) {

  int i,j;
  FILE *fp;
  fp=fopen("bcg.in","r");
  if (fp==NULL) {fclose(fp); return;}
  fscanf(fp,"%d %d",n,m);
  for (i=0; i!=*n; i++)
    for (j=0; j!=*n; j++)
      fscanf(fp,"%lf",&matrix[i][j]);
  for (i=0; i!=*n; i++) 
	  for (j=0; j!=*m; j++) fscanf(fp,"%lf",&block[i][j]);
  fclose(fp);
  return;
}

//Output
void output(int n, int m, double matrix[][mmax])
{
  int i,j;
  printf("Solution=\n");
  for (i=0; i!=n; i++) {
  	for (j=0; j!=m; j++) printf("%lf ", matrix[i][j]);
  	printf("\n");
  }
  return; 
}

void matrix_product(int m, int n, int k, double A[][mmax], double B[][mmax], double C[][mmax])
{
	int i,j,l;
	for (i=0; i!=m; i++)
    for (l=0; l!=k; l++) C[i][l]=0;
	for (i=0; i!=m; i++)
    for (j=0; j!=n; j++)
      for (l=0; l!=k; l++)
        C[i][l]=C[i][l]+A[i][j]*B[j][l];
  return;
}

void inverse(int n, double a[][mmax]) {
	int i,j,k;
	double c;
	for (k=0; k!=n; k++) {
		c=1.0/a[k][k];
		a[k][k]=c;
		for (i=0; i!=n; i++) if (i!=k) a[i][k]=-c*a[i][k];
		for (i=0; i!=n; i++) 
		  if (i!=k)
			  for (j=0; j!=n; j++) if (j!=k) a[i][j]=a[i][j]+a[i][k]*a[k][j];
		for (j=0; j!=n; j++) if (j!=k) a[k][j]=c*a[k][j];
	}
}

void matrix_plus(int n, int m, double A[][mmax], double B[][mmax], double C[][mmax], double times)
{
	int i,j;
	for (i=0; i!=n; i++)
		for (j=0; j!=m; j++)
		  C[i][j]=A[i][j]+times*B[i][j];
	return;
}
