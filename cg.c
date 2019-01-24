#include <stdio.h>
#include <math.h>
#define mmax 20
#define epsilon 1e-3

double inner_product(int, double [], double []);
void matrix_vector_product(int, int, double [][mmax], double[], double[]);


int main()
{
  int i,j,k,n;
  FILE *fp;
  double matrix[mmax][mmax],vector[mmax];
  double d[mmax],g[mmax],x[mmax],t[mmax];
  double d1,d2,n1,n2,s;
  fp=fopen("cg.in","r");
  if (fp!=NULL) {
    fscanf(fp,"%d",&n);
    for (i=0; i!=n; i++)
      for (j=0; j!=n; j++)
        fscanf(fp,"%lf",&matrix[i][j]);
    for (i=0; i!=n; i++) fscanf(fp,"%lf",&vector[i]);
    
  }
  for (i=0; i!=n; i++) {
      d[i]=0; x[i]=0; g[i]=-vector[i];
  }
  for (j=1; j!=n+1; j++) {
    d1=inner_product(n,g,g);
    matrix_vector_product(n,n,matrix,x,g);
    for (i=0; i!=n; i++) g[i]=g[i]-vector[i];
    n1=inner_product(n,g,g);
    if (fabs(n1<epsilon)) break;
    for (i=0; i!=n; i++) d[i]=-g[i]+n1/d1*d[i];
    n2=inner_product(n,d,g);
    matrix_vector_product(n,n,matrix,d,t);
    d2=inner_product(n,d,t);
    s=-n2/d2;
    for (i=0; i!=n; i++) x[i]=x[i]+s*d[i];
  }
  printf("x=");
  for (i=0; i!=n; i++) printf("%lf ",x[i]);
  printf("\n");
  return 0;
      
}

double inner_product(int m, double g[], double h[])
{
  double product=0;
  int i;
  for (i=0; i!=m; i++) product=product+g[i]*h[i];
  return product;
}

void matrix_vector_product(int m, int n, double A[][mmax], double B[], double C[])
{
  int i,j;
  for (i=0; i!=m; i++)
      C[i]=0;
  for (i=0; i!=m; i++)
    for (j=0; j!=n; j++)
      C[i]=C[i]+A[i][j]*B[j];
}
