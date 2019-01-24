#include "mpi.h"
#include <stdio.h>
#include <math.h>
#define t 202 // The number of grids on either direction
#define pmax 8 //The maximum number of the processes
#define epsilon 1e-9
#define pi 3.1416

double inner_product(int, double [], double []);
void matrix_vector_product(int, int, int, double[], double[]);
void output(int, double[]);
void solve(int, double[], double[], int, int);

int main(int argc, char *argv[])
{
//Initialize
  int i,j,k,n,myid,np,current;
  double vector[t*t], solution[t*t];
	double h,t1,t2;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&np);
  if (0==myid) t1=MPI_Wtime();
  h=1.0/(t-1);
  k=t-2;

//Construct vector: vector[k*k]
  if (myid==0) {
    for (i=0; i!=k; i++)
      for (j=0; j!=k; j++)
        vector[i*k+j]=8*h*h*pi*pi*cos(2.0*pi*(i+1)/(k+1))*sin(2.0*pi*(j+1)/(k+1));
    for (i=0; i!=k; i++) {
      vector[i]=vector[i]+sin(2.0*pi*(i+1)/(k+1));
      vector[k*k-k+i]=vector[k*k-k+i]+sin(2.0*pi*(i+1)/(k+1));
    }
  }
  
  solve(k, vector, solution, np, myid);
  
//Calculate the time elapsed
  if (0==myid) {
    t2=MPI_Wtime();
    printf("%lf \n",t2-t1);
  }

//Output
  if (0==myid)  output(k, solution);

  MPI_Finalize();
  return 0;  
}

void solve(int n, double vector[], double solution[], int np, int myid) 
{
  int disp[pmax], counts[pmax], disp0[pmax], counts0[pmax];
  int m,i,j;
  double myvector[n*n], g[n*n], gx[n*n], x[n*n], myx[n*n], gd[n*n], d[n*n], myd[n*n], t0[n*n];
  double n1, n2, d1, d2, product0, s;
  MPI_Datatype vectype;

//Define derived datatype : a vector as a row of double data
  MPI_Type_contiguous(n,MPI_DOUBLE,&vectype);
  MPI_Type_commit(&vectype);

//The descriptions of arrays disp[], disp0[], counts[], counts0[] can
//be found in the report.
  for (i=0; i!=np; i++) {
     disp0[i]=n/np*i;
     counts0[i]=(i!=np-1)?(n/np):(n/np+n%np);
	}
	
	if (1==np) {counts[0]=counts0[0]; disp[0]=disp0[0]; }
  
	else {
		counts[0]=n/np+1;  disp[0]=0;
		counts[np-1]=n-n/np*(np-1)+1; disp[np-1]=n-counts[np-1];
    for (i=1; i<=np-2; i++)
    {
      counts[i]=n/np*(np-1)+2;
      disp[i]=n/np*i-1;
    }
  }
  

  m = counts0[myid];
  
  MPI_Scatterv(vector, counts0, disp0, vectype, myvector, counts0[myid], vectype, 0, MPI_COMM_WORLD);

  for (i=0; i!=m*n; i++) {
    d[i]=0; x[i]=0; t0[i]=0;
  }

  for (i=0; i!=m*n; i++) g[i]=-myvector[i];

  for (j=0; j!=n*n; j++) {
    product0=inner_product(m*n, g, g); d1=0;
    MPI_Allreduce(&product0, &d1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   
  //In a matrix-vector product, the vector must be a full one instead of seperate ones,
  //and as a result MPI_Allgatherv is used.

    MPI_Gatherv(x, m, vectype, gx, counts0, disp0, vectype, 0, MPI_COMM_WORLD);
		MPI_Scatterv(gx, counts, disp, vectype, myx, counts[myid], vectype, 0, MPI_COMM_WORLD);
//Redistribute the vectors needed in each process
    matrix_vector_product(m, n, disp0[myid], myx, g);

		for (i=0; i!=m*n; i++) g[i]=g[i]-myvector[i];

		product0=inner_product(m*n, g, g); n1=0;
    MPI_Allreduce(&product0, &n1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (fabs(n1)<epsilon) break;

    for (i=0; i!=m*n; i++) d[i]=-g[i]+n1/d1*d[i];
    product0=inner_product(m*n, d, g); n2=0;
    MPI_Allreduce(&product0, &n2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Gatherv(d, m, vectype, gd, counts0, disp0, vectype, 0,  MPI_COMM_WORLD);
		MPI_Scatterv(gd, counts, disp, vectype, myd, counts[myid], vectype, 0, MPI_COMM_WORLD);
    matrix_vector_product(m, n, disp0[myid], myd, t0);

    product0=inner_product(m*n, d, t0); d2=0;
    MPI_Allreduce(&product0, &d2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    s=-n2/d2;

    for (i=0; i!=m*n; i++) x[i]=x[i]+s*d[i];

  }
  MPI_Gatherv(x, m, vectype, solution, counts0, disp0, vectype, 0, MPI_COMM_WORLD);

  return;
}

//Output
void output(int k, double solution[])
{
  FILE *fp1,*fp2;
	fp1=fopen("output.txt","w+");
	fp2=fopen("error.txt","w+");
  int i,j;
  for (i=0; i!=k; i++) { 
	  for (j=0; j!=k; j++) {
		  fprintf(fp1,"%lf\t%lf\t%lf\n",(i+1.0)/(k+1.0),(j+1.0)/(k+1.0),solution[i*k+j]);
		  fprintf(fp2,"%lf\t%lf\t%lf\n",(i+1.0)/(k+1.0),(j+1.0)/(k+1.0),solution[i*k+j]-cos(2.0*pi*(i+1)/(k+1))*sin(2.0*pi*(j+1)/(k+1)));}
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
void matrix_vector_product(int m, int k, int disp, double B[], double C[])
{
  
  int i,j,p;
	
  for (i=0; i!=m*k; i++)
      C[i]=0;
  for (i=0; i!=m; i++)
		for (j=0; j!=k; j++) {
				p=i*k+j;
				
				if (m==k)		{
						if (i==0) {
								if (j==0) 
										C[p]=4*B[p]-B[p+1]-B[p+k];
								else if (j==k-1) 
										C[p]=4*B[p]-B[p-1]-B[p+k];
								else 
										C[p]=4*B[p]-B[p-1]-B[p+1]-B[p+k];}
						else if (i==k-1)  {
								if (j==0)
										C[p]=4*B[p]-B[p+1]-B[p-k];
								else if (j==k-1)
										C[p]=4*B[p]-B[p-1]-B[p-k];
								else
										C[p]=4*B[p]-B[p-1]-B[p+1]-B[p-k]; }
						else if (j==0)
										C[p]=4*B[p]-B[p-k]-B[p+1]-B[p+k];
						else if (j==k-1)
								C[p]=4*B[p]-B[p-k]-B[p-1]-B[p+k];
						else 
								C[p]=4*B[p]-B[p-k]-B[p+k]-B[p-1]-B[p+1];
						
				}
				else if (disp==0) { 
						if (i==0)
								if (j==0) C[p]=4*B[p]-B[p+1]-B[p+k];
								else if (j==k-1) C[p]=4*B[p]-B[p-1]-B[p+k];
								else C[p]=4*B[p]-B[p-1]-B[p+1]-B[p+k];
						else if (j==0)
								C[p]=4*B[p]-B[p-k]-B[p+k]-B[p+1];
						else if (j==k-1)
								C[p]=4*B[p]-B[p-k]-B[p+k]-B[p-1];
						else
								C[p]=4*B[p]-B[p-k]-B[p+k]-B[p-1]-B[p+1];}
				else if (disp+m==k) { 
						if (i==m-1)
								if (j==0)
										C[p]=4*B[p+k]-B[p]-B[p+k+1];
								else if (j==k-1)
										C[p]=4*B[p+k]-B[p]-B[p+k-1];
								else
										C[p]=4*B[p+k]-B[p+k-1]-B[p+k+1]-B[p];
						else if (j==0)
								C[p]=4*B[p+k]-B[p]-B[p+k*2]-B[p+k+1];
						else if (j==k-1)
								C[p]=4*B[p+k]-B[p]-B[p+k*2]-B[p+k-1];
						else
								C[p]=4*B[p+k]-B[p]-B[p+k*2]-B[p+k-1]-B[p+k+1];}
				else {
						if (j==0)
								C[p]=4*B[p+k]-B[p]-B[p+k*2]-B[p+k+1];
						else if (j==k-1)
								C[p]=4*B[p+k]-B[p]-B[p+k*2]-B[p+k-1];
						else
								C[p]=4*B[p+k]-B[p]-B[p+k*2]-B[p+k-1]-B[p+k+1];
				}
		}
  return;
}
