#include "mpi.h"
#include <stdio.h>
#include <math.h>
#define t 2002 //The number of grids on x direction
#define pmax 8 //The maximum number of the processes
#define epsilon 1e-9
#define pi 3.1416

double inner_product(int, double [], double []);
void matrix_vector_product(int, int, int, double [][3], double[], double[]);
void output(int, double[]);
void solve(int, double[][3], double[], double[], int, int);
double getelement(int, int, double[][3], int, int);

int main(int argc, char *argv[])
{
//Initialize
  int i,j,k,n,myid,np,current;
  double matrix[t][3], vector[t], solution[t];
  double h,t1,t2;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&np);
  h=1.0/(t-1);

//Construct coefficient matrix: matrix[t-2][3]
  if (myid==0) {
	for (i=0; i!=t; i++)
		for (j=0; j!=3; j++)
			matrix[i][j]=0;
    matrix[0][0]=2;
	  matrix[0][1]=-1;
    matrix[t-3][1]=-1;	
    matrix[t-3][2]=2;
	for (i=1; i!=t-3; i++) {
 		matrix[i][1]=2;
		matrix[i][0]=-1;
		matrix[i][2]=-1;
	}

//construct vector: vector[t-2]
	for (i=0; i!=t-2; i++)
		vector[i]=h*h*pi*pi*sin(pi*(i+1)/(t-1));
  }
  
  if (0==myid) t1=MPI_Wtime();
  solve(t-2, matrix, vector, solution, np, myid);
	
//Calculate the time elapsed
  if (0==myid) { t2=MPI_Wtime(); printf("%lf\n",t2-t1); }

//Output
  if (0==myid) output(t-2, solution); 

  MPI_Finalize();
  return 0;  
}

void solve(int n, double matrix[][3], double vector[], double solution[], int np, int myid) 
{
  int disp[pmax], counts[pmax];
  int m,i,j;
  double A[n][3], myvector[n], g[n], gx[n], x[n], gd[n], d[n], t0[n];
  double n1, n2, d1, d2, product0, s;
  MPI_Datatype vectype;

//Define derived datatype : a vector of 3 double data
  MPI_Type_contiguous(3,MPI_DOUBLE,&vectype);
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

  for (i=0; i!=m; i++) {
    d[i]=0; x[i]=0; t0[i]=0;
  }

  for (i=0; i!=m; i++) g[i]=-myvector[i];

  for (j=1; j!=n+1; j++) {
    product0=inner_product(m, g, g); d1=0;
    MPI_Allreduce(&product0, &d1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   
  //In a matrix-vector product, the vector must be a full one instead of seperate ones,
  //and as a result MPI_Allgatherv is used.
    MPI_Allgatherv(x, m, MPI_DOUBLE, gx, counts, disp, MPI_DOUBLE, MPI_COMM_WORLD);
    matrix_vector_product(m, n, disp[myid], A, gx, g);

    for (i=0; i!=m; i++) g[i]=g[i]-myvector[i];
    product0=inner_product(m, g, g); n1=0;
    MPI_Allreduce(&product0, &n1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (fabs(n1)<epsilon) break;

    for (i=0; i!=m; i++) d[i]=-g[i]+n1/d1*d[i];
    product0=inner_product(m, d, g); n2=0;
    MPI_Allreduce(&product0, &n2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allgatherv(d, m, MPI_DOUBLE, gd, counts, disp, MPI_DOUBLE, MPI_COMM_WORLD);
    matrix_vector_product(m, n, disp[myid], A, gd, t0);

    product0=inner_product(m, d, t0); d2=0;
    MPI_Allreduce(&product0, &d2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    s=-n2/d2;

    for (i=0; i!=m; i++) x[i]=x[i]+s*d[i];
  }
  MPI_Gatherv(x, m, MPI_DOUBLE, solution, counts, disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return;
}

//Output
void output(int n, double solution[])
{
  int i;
	FILE *fp1, *fp2;
	
	fp1=fopen("output.txt","w+");
	fp2=fopen("error.txt","w+");
  for (i=0; i!=n; i++) {
		fprintf(fp1,"%lf\t%lf\n", (i+1.0)/(n+1.0),  solution[i]);
		fprintf(fp2,"%lf\t%lf\n", (i+1.0)/(n+1.0),  solution[i]-sin(pi*(i+1)/(n+1)));
	}
  fclose(fp1);
	fclose(fp2);
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
void matrix_vector_product(int m, int n, int disp, double A[][3], double B[], double C[])
{
  int i,j;
  for (i=0; i!=m; i++)
      C[i]=0;
  for (i=0; i!=m; i++)
    for (j=0; j!=n; j++) 

      C[i]=C[i]+getelement(n,disp,A,i,j)*B[j];
  return;
}

double getelement(int n, int disp, double A[][3], int i, int j)
{
	int j_prime;
  j_prime=j-disp-i+1;
  if ((disp==0) && (i==0))
    if ((j==0) || (j==1)) return A[i][j]; else return 0;
  if (disp+i==n-1) 
    if (j>=n-2) return A[i][j_prime+1]; else return 0;
  if (j_prime<0 || j_prime>2) return 0; else return A[i][j_prime];
}
