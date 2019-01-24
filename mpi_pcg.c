#include "mpi.h"
#include <stdio.h>
#include <math.h>
#define mmax 20 //The maximum size of the matrix
#define pmax 8 //The maximum number of the processes
#define epsilon 1e-5

double inner_product(int, double [], double []);
void matrix_vector_product(int, int, double [][mmax], double[], double[]);
void input(int *, double [][mmax], double[]);
void output(int, double[]);
void process(int, double[][mmax], double[], double[], int, int);
void vector_plus(int, double[], double[], double[], double);

int main(int argc, char *argv[])
{
//Initialize
  int i,j,k,n,myid,np,current;
  double matrix[mmax][mmax],vector[mmax],solution[mmax];
  double d1,d2,n1,n2,s;
  int disp[pmax],counts[pmax];
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&np);

  if (0==myid) input(&n, matrix, vector); //Input data from external file
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  process(n, matrix, vector, solution, np, myid); //Process
  if (0==myid) output(n, solution); //Output

  MPI_Finalize();
  return 0;  
}

void process(int n, double matrix[][mmax], double vector[], double solution[], int np, int myid) 
{
 
  int disp[pmax], counts[pmax];
  int m,i,j;
  double A[mmax][mmax],gp[mmax],p[mmax],r[mmax],x[mmax],w[mmax],t[mmax],z[mmax],myvector[mmax];
  double alpha,beta,rho,rhop,product0,t1;
  MPI_Datatype vectype;

//Define derived datatype : a vector as a row of double data
  MPI_Type_contiguous(mmax,MPI_DOUBLE,&vectype);
  MPI_Type_commit(&vectype);

//fill in values for array disp[] and counts[] which are to be used in
//MPI_Scatterv and MPI_AllgatherV
  for (i=0; i!=np; i++)
  {
     disp[i]=n/np*i;
     counts[i]=(i!=np-1)?(n/np):(n/np+n%np);
  }
  m = counts[myid];
  
  MPI_Scatterv(matrix, counts, disp, vectype, A, counts[myid], vectype, 0, MPI_COMM_WORLD);
  MPI_Scatterv(vector, counts, disp, MPI_DOUBLE, myvector, counts[myid], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (i=0; i!=n; i++) {
      p[i]=0; x[i]=0;  
  }

  matrix_vector_product(m, n, A, x, t);

  vector_plus(m, myvector, t, r, -1);

  for (j=1; j!=n+1; j++) {
    for (i=0; i!=m; i++) z[i]=r[i]/A[i][disp[myid]+i];

    product0=inner_product(n,r,z); rho=0;
    MPI_Allreduce(&product0, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (1==j) for (i=0; i!=m; i++) p[i]=z[i];
    else { beta=rho/rhop; 
      vector_plus(n, z, p, p, beta);
    }

    MPI_Allgatherv(p, m, MPI_DOUBLE, gp, counts, disp, MPI_DOUBLE, MPI_COMM_WORLD);
    matrix_vector_product(m, n, A, gp, w);

    product0=inner_product(n, p, w); t1=0;
    MPI_Allreduce(&product0, &t1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    alpha=rho/t1;

    vector_plus(n, x, p, x, alpha);
    vector_plus(n, r, w, r, -alpha);
    rhop=rho;

    product0=inner_product(n, r, r);
    MPI_Allreduce(&product0, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  
  MPI_Gatherv(x, m, MPI_DOUBLE, solution, counts, disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return;
}

//Input
void input(int *n, double matrix[][mmax], double vector[]) {
  int i,j;
  FILE *fp;
  fp=fopen("cg.in","r");
  if (fp==NULL) {fclose(fp); return;}
  fscanf(fp,"%d",n);
  for (i=0; i!=*n; i++)
    for (j=0; j!=*n; j++)
      fscanf(fp,"%lf",&matrix[i][j]);
  for (i=0; i!=*n; i++) fscanf(fp,"%lf",&vector[i]);
  fclose(fp);
  return;
}

//Output
void output(int n, double solution[])
{
  int i;
  printf("x=");
  for (i=0; i!=n; i++) printf("%lf ", solution[i]);
  printf("\n");
  return; 
}

//Inner product
double inner_product(int m, double g[], double h[])
{
  double product=0;
  int i;
  for (i=0; i!=m; i++) product=product+g[i]*h[i];
  return product;
}

//Matrix-vector product
void matrix_vector_product(int m, int n, double A[][mmax], double B[], double C[])
{
  int i,j;
  for (i=0; i!=m; i++)
      C[i]=0;
  for (i=0; i!=m; i++)
    for (j=0; j!=n; j++) 
      C[i]=C[i]+A[i][j]*B[j];
  return;
}

//A funtion of c[i]=a[i]+times*b[i]
void vector_plus(int n, double a[], double b[], double c[], double times)
{
  int i;
  for (i=0; i!=n; i++)
    c[i]=a[i]+times*b[i];
  return;
}
