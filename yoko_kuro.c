
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

//--------------------定数--------------------//

#define NUM ( 240 )
#define TMAX ( 3000.0 )
//#define dt ( 0.1 / ( NUM * NUM ) ) // 0.000001
#define dt ( 8.0e-5 ) // 0.000001
//#define sigma ( 2.0e-4 )
#define RANGE_CHECK(x, xmin, xmax) ( x = ( x < xmin ? xmin : ( x < xmax ?  x : xmax)));

//----------定数(結晶)----------//

#define T ( 281.15 ) // 絶対温度[K]
#define p_e ( 1.66e+3 ) // 平衡蒸気圧[dyn/cm^2]
#define v_c ( 3.25e-23 ) // 結晶相での水分子の体積[cm^3]
#define f_0 ( 8.3e-16 ) // 界面において1分子あたりが占める表面積[cm^2]
#define m ( 3.0e-23 ) // 水分子の質量[g]
#define e ( 4.5e-8 ) // ステップの高さ[m]
#define x_s ( 400 * e ) // 吸着分子がステップ滞在中に表面拡散する平均距離[m]
#define k_B ( 1.38e-16 ) // ボルツマン定数[erg/K]
#define energy ( 2.0e-6 ) // ステップの単位長さあたりの自由エネルギー[erg/cm]
#define C ( 0.2 ) // 拡散係数[m^2/s]
#define alpha_1 ( 0.1 ) // 凝縮定数

#define gamma ( v_c * p_e * C / ( k_B * T ) )
#define beta_max ( alpha_1 * v_c * p_e / sqrt(2 * M_PI * m * k_B * T) ) // 0.001995
#define sigma_star ( 9.5 * f_0 * energy / ( k_B * T * x_s ) )

//#define sigma_infty ( 17 ) // 初期値
#define sigma_infty ( 0.9 ) // 初期値
#define delta_beta ( 0.12 )

//--------------------関数--------------------//

double* make_vector( int N ); // 領域の確保(ベクトル)
double** make_matrix( int ROW, int COL ); // 領域の確保(行列)

void connect( double *x ); // 点のつなぎ(1個)
void connect_double( double* x, double *y ); // 点のつなぎ(2個)

double ip( double x1, double x2, double y1, double y2 ); // 内積
double DIST( double x1, double x2, double y1, double y2 ); // |y-x|
double DET( double x1, double x2, double y1, double y2 ); // 行列式

void gauss(int n, double **A, double *b, double *x); // Ax = b(0〜n-1までの行列)

void runge_kutta( double t, double *X1, double *X2 ); // ルンゲクッタ
void euler( double t, double *X1, double *X2 );
void F( double t, double *x1, double *x2, double *F1, double *F2 ); // 右辺
void pre( double t, double *x1, double *x2, double *T1, double *T2, double *N1, double *N2, double *V, double *W ); // 準備

void initial_condition( double *x1, double *x2 );

void quantities( double t, double *x1, double *x2, double *r, double *t1, double *t2, double *n1, double *n2, double *T1, double *T2, double *N1, double *N2, double *nu, double *phi, double *kappa, double *beta ); //x --> t,n,T,N,phi,kappa
void measure( double t, double *x1, double *x2, double *L, double *A ); // x,l --> L,A

double E( double x1, double x2, double y1, double y2, double z1, double z2 ); // 基本解
void grad_E( double x1, double x2, double y1, double y2, double z1, double z2, double *grad_E1, double *grad_E2 ); // 基本解の勾配

//double E( double x1, double x2, double y1, double y2 ); // 基本解
//void grad_E( double x1, double x2, double y1, double y2, double *grad_E1, double *grad_E2 ); // 基本解の勾配

void PP( double t, double *P, double A ); // P
void grad_P( double t, double *x_mid1, double *x_mid2, double *n1, double *n2, double *l, double A, double *P, double *beta, double *u ); // x,n,r,P --> grad_aP

double omega( int n );

void velocity( double t, double *x1, double *x2, double *n1, double *n2, double *l, double *phi, double *beta, double *kappa, double L, double A, double *V, double *W ); // x,n,r,t,ohi,kappa,L --> V,W
void normal_speed( double t, double *x_mid1, double *x_mid2, double *n1, double *n2, double *phi, double *beta, double *u, double *v, double *V );
void tangent_speed( double t, double *l, double *phi, double *kappa, double *v, double *V, double L, double *W );


//--------------------main-------------------//

int main(void){
  
  int i,z;
  double t;
  double *X1,*X2;
  X1 = make_vector(NUM + 2);
  X2 = make_vector(NUM + 2);
  
  double L,A;

  char file[5000];
  FILE *fp,*fp2;
  
  //fp = fopen("mfs.dat", "w");
  fp2 = fopen("L_A.dat", "w");
  
  t = 0.0;
  z = 0;

  initial_condition(X1,X2);

  sprintf(file, "./data/yoko_kuro_mfs%06d.dat", z);
  fp = fopen(file, "w");
  
  for( i = 0; i <= NUM; i++ ){
    
    fprintf(fp, "%f %f %f\n", X1[i], X2[i], t);
    
  }
  fprintf(fp,"\n");
  
  measure(t,X1,X2,&L,&A);
  
  fprintf(fp2, "t = %f, L = %f, A = %f\n", t, L, A );

  fclose(fp2);

  fclose(fp);
  
  while( t < TMAX ){
    
    //runge_kutta(t,X1,X2);
    euler(t,X1,X2);
    
    z++;
    t = z * dt;

    measure(t,X1,X2,&L,&A);
    printf("t = %f A = %f\n", t, A);
    
    if( z % 100000 == 0 ){
      
      sprintf(file, "./data/yoko_kuro_mfs%06d.dat", z / 100000 );
      fp = fopen(file, "w");

      measure(t,X1,X2,&L,&A);
      
      for( i = 0; i <= NUM; i++ ){
	
	fprintf(fp, "%f %f %f\n", X1[i], X2[i], t);

      }
      fprintf(fp,"\n");

      fclose(fp);
      fp2 = fopen("L_A.dat", "a");

      fprintf(fp2, "t = %f, L = %f, A = %f\n", t, L, A );
      fclose(fp2);

    }

  }

  free(X1);
  free(X2);
  
  return 0;
  
}


//--------------------関数--------------------//

// 領域の確保(ベクトル)
double* make_vector( int N ){
  
  double *a;
  
  // メモリの確保
  if( ( a = malloc( sizeof( double ) * N ) ) == NULL )
    {
      printf( "LACK of AVAILABLE MEMORY!" );
      exit(1);
    }
  
  return a;

}

// 領域の確保(行列)
double** make_matrix(int ROW, int COL){
  
  int i;
  double **b;
  
  // メモリの確保
  if( ( b = malloc( sizeof( double* ) * ROW ) ) == NULL ){
    
    printf( "LACK of AVAILABLE MEMORY!" );
    exit(1);

  }
  
  for( i = 0; i < ROW; i++){
    
    if( ( b[i] = malloc( sizeof( double ) * COL ) ) == NULL ){
      
      printf( "LACK of AVAILABLE MEMORY!" );
      free(b);
      exit(1);

    }

  }
  
  return b;

}

// 点のつなぎ(1個)
void connect( double *x ){
  
  x[0] = x[NUM];
  x[NUM + 1] = x[1];

}

// 点のつなぎ(2個)
void connect_double( double *x, double *y ){
  
  connect(x);
  connect(y);

}

// 内積
double ip( double x1, double x2, double y1, double y2 ){
  
  return ( x1 * y1 + x2 * y2 );

}

// |y-x|
double DIST(double x1, double x2, double y1, double y2){
  
  return ( sqrt( (y1 - x1) * (y1 - x1) + (y2 - x2) * (y2 - x2) ) );

}

// 行列式
double DET( double x1, double x2, double y1, double y2 ){
  
  return ( x1 * y2 - y1 * x2 );
  
}

// Ax = b(0〜n-1までの行列)
void gauss( int n, double **A, double *b, double *x ){
  
  int i,j,k,l,row_max;
  double max,temp,c;
  double temp_vec;
  double *temp_mat;
  
  for( i = 0; i < n - 1; i++ ){
    
    row_max = i;
    max = A[i][i];
    
    for( l = i + 1; l < n; l++ ){
      
      if( max < A[l][i] ){
	
	row_max = l;
	max = A[l][i];

      }		    

    }
    
    if( row_max > i ){
      
      temp_mat = A[i];
      A[i] = A[row_max];
      A[row_max] = temp_mat;
      
      temp_vec = b[i];
      b[i] = b[row_max];
      b[row_max] = temp_vec;

    }
      
    for( k = i + 1; k < n; k++ ){
      
      c = A[k][i] / A[i][i];

      for(j = i; j < n; j++){
	
	A[k][j] -= A[i][j] * c;

      }

      b[k] -= b[i]*c;

    }

  }
  
  for( i = n - 1; i >= 0; i--){
    
    temp = b[i];
    
    for( j = n - 1; j > i; j-- ){
      
      temp -= A[i][j] * x[j];

    }

    x[i] = temp/A[i][i];
    
  }

}

// ルンゲクッタ
void runge_kutta( double t, double *X1, double *X2 ){
  
  int i;
  double t_temp;
  double *x_temp1,*x_temp2,*F1,*F2;
  double *k11,*k12,*k21,*k22,*k31,*k32,*k41,*k42;
  
  k11 = make_vector(NUM + 2);
  k12 = make_vector(NUM + 2);
  k21 = make_vector(NUM + 2);
  k22 = make_vector(NUM + 2);
  k31 = make_vector(NUM + 2);
  k32 = make_vector(NUM + 2);
  k41 = make_vector(NUM + 2);
  k42 = make_vector(NUM + 2);

  x_temp1 = make_vector(NUM + 2);
  x_temp2 = make_vector(NUM + 2);
  F1 = make_vector(NUM + 2);
  F2 = make_vector(NUM + 2);

  F(t,X1,X2,F1,F2);
  
  for( i = 1; i <= NUM; i++ ){
    
      k11[i] = F1[i];
      k12[i] = F2[i];
      x_temp1[i] = X1[i] + dt * k11[i] / 2.0;
      x_temp2[i] = X2[i] + dt * k12[i] / 2.0;

  }
  connect_double(x_temp1,x_temp2);

  t_temp = t + dt/2.0;

  F(t_temp,x_temp1,x_temp2,F1,F2);      

  for( i = 1; i <= NUM; i++ ){
    
      k21[i] = F1[i];
      k22[i] = F2[i];
      x_temp1[i] = X1[i] + dt * k21[i] / 2.0;
      x_temp2[i] = X2[i] + dt * k22[i] / 2.0;

  }
  connect_double(x_temp1,x_temp2);

  F(t_temp,x_temp1,x_temp2,F1,F2);
  
  for( i = 1; i <= NUM; i++ ){
    
    k31[i] = F1[i];
    k32[i] = F2[i];
    x_temp1[i] = X1[i] + k31[i] * dt;
    x_temp2[i] = X2[i] + k32[i] * dt;

  }
  connect_double(x_temp1,x_temp2);
  
  t_temp = t + dt;
      
  F(t_temp,x_temp1,x_temp2,F1,F2);
  
  for( i = 1; i <= NUM; i++ ){
    
    k41[i] = F1[i];
    k42[i] = F2[i];
    
    X1[i] = X1[i] + dt * ( k11[i] + 2.0 * k21[i] + 2.0 * k31[i] + k41[i] ) / 6.0;
    X2[i] = X2[i] + dt * ( k12[i] + 2.0 * k22[i] + 2.0 * k32[i] + k42[i] ) / 6.0;

  }
  connect_double(X1,X2);

  free(k11);
  free(k12);
  free(k21);
  free(k22);
  free(k31);
  free(k32);
  free(k41);
  free(k42);
  free(x_temp1);
  free(x_temp2);
  free(F1);
  free(F2);
  
}

void euler( double t, double *X1, double *X2 ){

  int i;
  double *F1,*F2;

  F1 = make_vector(NUM + 2);
  F2 = make_vector(NUM + 2);

  F(t,X1,X2,F1,F2);

  for( i = 1; i <= NUM; i++ ){
    
    X1[i] = X1[i] + dt * F1[i];
    X2[i] = X2[i] + dt * F2[i];

  }
  connect_double(X1,X2);
  
  free(F1);
  free(F2);
  
}

// 右辺
void F( double t, double *x1, double *x2, double *F1, double *F2 ){
  
  int i;
  
  double *T1,*T2,*N1,*N2;
  T1 = make_vector(NUM + 2);
  T2 = make_vector(NUM + 2);
  N1 = make_vector(NUM + 2);
  N2 = make_vector(NUM + 2);

  double *V,*W;
  V = make_vector(NUM + 2);
  W = make_vector(NUM + 2);
  
  pre(t,x1,x2,T1,T2,N1,N2,V,W);
  
  for( i = 1; i <= NUM; i++ ){
    
    F1[i] = V[i] * N1[i] + W[i] * T1[i];
    F2[i] = V[i] * N2[i] + W[i] * T2[i];
    //printf("V = %f\n", V[i]);

  }
  connect_double(F1,F2);

  free(W);
  free(V);
  free(T1);
  free(T2);
  free(N1);
  free(N2);
  
}

//x --> T,N,V,W
void pre( double t, double *x1, double *x2, double *T1, double *T2, double *N1, double *N2, double *V, double *W ){
  
  double *l;
  double *t1,*t2,*n1,*n2;
  double *nu;
  double *phi;
  double *beta;
  double *kappa;
  double L,A;

  l = make_vector(NUM + 2);
  nu = make_vector(NUM + 2);
  phi = make_vector(NUM + 2);
  kappa = make_vector(NUM + 2);
  beta = make_vector(NUM + 2);

  t1 = make_vector(NUM + 2);
  t2 = make_vector(NUM + 2);
  n1 = make_vector(NUM + 2);
  n2 = make_vector(NUM + 2);
  
  // T,N
  quantities(t,x1,x2,l,t1,t2,n1,n2,T1,T2,N1,N2,nu,phi,kappa,beta);
  
  // L,A
  measure(t,x1,x2,&L,&A);
  
  // V,W
  velocity(t,x1,x2,n1,n2,l,phi,beta,kappa,L,A,V,W);
  
  free(t1);
  free(t2);
  free(n1);
  free(n2);

  free(beta);
  free(kappa);
  free(phi);
  free(nu);
  free(l);
  
}

// 初期曲線
void initial_condition( double *x1, double *x2 ){

  int i,k;
  double u;
  double lambda;

  /*
  for( i = 1; i <= NUM; i++ ){
    
    u = i * 2 * M_PI / NUM;
    
    x1[i] = 5.0e-3 * cos(u);
    x2[i] = 5.0e-3 * sin(u);
    
  }
  connect_double(x1,x2);
  */
  
  for (i = 1; i <= NUM; i++)
  {

    u = i * 2 * M_PI / NUM;
  }

  for (k = 0; k < 6; k++)
  {
    for (i = k * (NUM / 6) + 1; i <= (k + 1) * (NUM / 6); i++)
    {
      lambda = (i - k * (NUM / 6)) * 1.0 / (NUM / 6);
      if (k == 0)
      {
        x1[i] = (1.0 - lambda) * 1.0 + lambda * 1.0 / 2.0;
        x2[i] = (1.0 - lambda) * 0.0 + lambda * sqrt(3.0) / 2.0;
      }
      else if (k == 1)
      {
        x1[i] = (1.0 - lambda) * 1.0 / 2.0 + lambda * (-1.0 / 2.0);
        x2[i] = sqrt(3.0) / 2.0;
      }
      else if (k == 2)
      {
        x1[i] = (1.0 - lambda) * (-1.0 / 2.0) + lambda * (-1.0);
        x2[i] = (1.0 - lambda) * (sqrt(3.0) / 2.0) + lambda * 0.0;
      }
      else if (k == 3)
      {
        x1[i] = (1.0 - lambda) * (-1.0) + lambda * (-1.0 / 2.0);
        x2[i] = (1.0 - lambda) * 0.0 + lambda * (-sqrt(3.0) / 2.0);
      }
      else if (k == 4)
      {
        x1[i] = (1.0 - lambda) * (-1.0 / 2.0) + lambda * (1.0 / 2.0);
        x2[i] = -sqrt(3.0) / 2.0;
      }
      else
      {
        x1[i] = (1.0 - lambda) * (1.0 / 2.0) + lambda * (1.0);
        x2[i] = (1.0 - lambda) * (-sqrt(3.0) / 2.0) + lambda * 0.0;
      }

      x1[i] = 1.0e-2 * x1[i];
      x2[i] = 1.0e-2 * x2[i];
    }
  }
  connect_double(x1, x2);
  
}

//x --> t,n,T,N,phi,kappa
void quantities( double t, double *x1, double *x2, double *l, double *t1, double *t2, double *n1, double *n2, double *T1, double *T2, double *N1, double *N2, double *nu, double *phi, double *kappa, double *beta ){
  
  int i;
  double D,I,D_sgn;
  
  for( i = 1; i <= NUM; i++ ){
    
    l[i] = DIST(x1[i - 1],x2[i - 1],x1[i],x2[i]);
    
    t1[i] = ( x1[i] - x1[i - 1] ) / l[i];
    t2[i] = ( x2[i] - x2[i - 1] ) / l[i];
    
    n1[i] = t2[i];
    n2[i] = -t1[i];

  }
  connect(l);
  connect_double(t1,t2);
  connect_double(n1,n2);
  

  RANGE_CHECK(t1[1],-1.0,1.0);
  
  if( t2[1] >= 0 ){
    
    nu[1] = acos(t1[1]);
    
  }
  
  else{
    
    nu[1] = -acos(t1[1]);
    
  }
  
  for( i = 1; i <= NUM; i++){
    
    D = DET(t1[i],t2[i],t1[i + 1],t2[i + 1]);
    I = ip(t1[i],t2[i],t1[i + 1],t2[i + 1]);
    
    RANGE_CHECK(I,-1.0,1.0);
    
    if( D < 0 ){
      
      D_sgn = -1;
      
    }
    
    else if( D > 0 ){
      
      D_sgn = 1;
      
    }
    
    else{
      
      D_sgn = 0;
      
    }
    
    nu[i + 1] = nu[i] + D_sgn * acos(I);
    
  }
  nu[0] = nu[1] - ( nu[NUM + 1] - nu[NUM] );
  
  
  for( i = 1; i <= NUM; i++ ){
    
    phi[i] = nu[i + 1] - nu[i];
    
  }
  connect(phi);
  

  for( i = 1; i <= NUM; i++){
    
    T1[i] = ( t1[i] + t1[i + 1] ) / ( 2.0 * cos(phi[i] / 2.0) );
    T2[i] = ( t2[i] + t2[i + 1] ) / ( 2.0 * cos(phi[i] / 2.0) );

    N1[i] = T2[i];
    N2[i] = -T1[i];
    
    kappa[i] = ( tan(phi[i] / 2.0) + tan(phi[i - 1] / 2.0) ) / l[i];
    
  }
  connect_double(T1,T2);
  connect_double(N1,N2);
  connect(kappa);
  
  for( i = 1; i <= NUM; i++ ){
    
    //beta[i] = beta_max * cos(6 * nu[i]) + beta_max + beta_max;
    //beta[i] = beta_max * 2 * x_s * tan(3 * nu[i]) * tanh(e / ( 2 * x_s * tan(3 * nu[i]) )) / e;
    //beta[i] = beta_max * tan(3 * nu[i]) * tanh(0.2 / tan(3 * nu[i])) / 0.2;
    beta[i] = beta_max * ( sin(( atan(tan(3*nu[i]-M_PI/2.0)) + M_PI/2.0 ) / 3.0) + sin(M_PI/3.0 - ( atan(tan(3*nu[i]-M_PI/2.0)) + M_PI/2.0 ) / 3.0) ) / sin(M_PI/3.0);
    //beta[i] = beta_max * ( 1 + ( 0.99 / 35.0 ) * cos(6 * nu[i]) );
    //beta[i] = beta_max * ( 1 + 0.1 * ( fabs(cos(3 * nu[i] + M_PI / 2.0)) - 0.5 ) );
    //beta[i] = beta_max * cos(4 * ( nu[i] - M_PI / 4.0 )) + beta_max + beta_max;
    //beta[i] = beta_max;
    //printf("beta[%d] = %f\n", i, beta[i]);
  }
  connect(beta);
  
}

// x,l --> L,A
void measure( double t, double *x1, double *x2, double *L, double *A ){
  
  int i;
  
  *L = 0.0;
  *A = 0.0;
  
  for( i = 1; i <= NUM; i++){
    
    *L += DIST(x1[i],x2[i],x1[i - 1],x2[i - 1]);
    *A += DET(x1[i - 1],x2[i - 1],x1[i],x2[i]);

  }
  *A = *A/2.0;
  
}


// 基本解
double E( double x1, double x2, double y1, double y2, double z1, double z2 ){
  
  return ( ( log(DIST(y1,y2,x1,x2)) - log(DIST(z1,z2,x1,x2)) ) / ( 2.0 * M_PI ) );
  
  //return ( log(DIST(y1,y2,x1,x2) / DIST(z1,z2,x1,x2)) / ( 2.0 * M_PI ) );

}

// 基本解の勾配
void grad_E( double x1, double x2, double y1, double y2, double z1, double z2, double *grad_E1, double *grad_E2 ){
  
  double ry = DIST(y1,y2,x1,x2);
  double rz = DIST(z1,z2,x1,x2);
  
  *grad_E1 = ( x1 - y1 ) / ( 2.0 * M_PI * ry * ry ) - ( x1 - z1 ) / ( 2.0 * M_PI * rz * rz );
  *grad_E2 = ( x2 - y2 ) / ( 2.0 * M_PI * ry * ry ) - ( x2 - z2 ) / ( 2.0 * M_PI * rz * rz );
  
}

/*
// 基本解
double E( double x1, double x2, double y1, double y2 ){
  
  return ( log(DIST(y1,y2,x1,x2)) / ( 2.0 * M_PI ) );
  
  //return ( log(DIST(y1,y2,x1,x2) / DIST(z1,z2,x1,x2)) / ( 2.0 * M_PI ) );

}

// 基本解の勾配
void grad_E( double x1, double x2, double y1, double y2, double *grad_E1, double *grad_E2 ){
  
  double ry = DIST(y1,y2,x1,x2);
  
  *grad_E1 = ( x1 - y1 ) / ( 2.0 * M_PI * ry * ry );
  *grad_E2 = ( x2 - y2 ) / ( 2.0 * M_PI * ry * ry );
  
}
*/

// P
void PP( double t, double *P, double A ){
  
  int i;
  double r_c,R;
  
  r_c = sqrt(A / M_PI);
  R = 6.5 * r_c;
  
  P[0] = 2 * M_PI * sigma_infty / log(R / r_c);
  //P[0] = 0.0;
  
  for( i = 1; i <= NUM; i++ ){
    
    P[i] = 0.0;

  }

  for( i = NUM + 1; i <= 2 * NUM; i++ ){
    
    P[i] = sigma_infty;

  }

}

// x,n,r,P --> grad_P
void grad_P( double t, double *x_mid1, double *x_mid2, double *n1, double *n2, double *l, double A, double *P, double *beta, double *u ){
  
  int i,j;
  double r_c,R;
  double theta;
  double amano;
  double *x3,*x4;
  double *ll;
  double *tt1,*tt2;
  double *nn1,*nn2;
  double *x_mid3,*x_mid4;
  double *y1,*y2,*z1,*z2;
  double **G;
  double **H1,**H2;
  double *H;
  double *Q;
  double grad_E1,grad_E2;
  double *amano1,*amano2,*amano3,*amano4;
  
  
  //double d = 1.0 / sqrt(1.0 * NUM);
  
  double d;

  x3 = make_vector(NUM + 2);
  x4 = make_vector(NUM + 2);
  x_mid3 = make_vector(NUM + 2);
  x_mid4 = make_vector(NUM + 2);
  y1 = make_vector(2 * NUM + 2);
  y2 = make_vector(2 * NUM + 2);
  z1 = make_vector(2 * NUM + 2);
  z2 = make_vector(2 * NUM + 2);

  amano1 = make_vector(NUM + 2);
  amano2 = make_vector(NUM + 2);
  amano3 = make_vector(NUM + 2);
  amano4 = make_vector(NUM + 2);
  
  ll = make_vector(NUM + 2);
  tt1 = make_vector(NUM + 2);
  tt2 = make_vector(NUM + 2);
  nn1 = make_vector(NUM + 2);
  nn2 = make_vector(NUM + 2);
  
  H1 = make_matrix(2 * NUM + 1,2 * NUM + 1);
  H2 = make_matrix(2 * NUM + 1,2 * NUM + 1);

  H = make_vector(2 * NUM + 1);
  Q = make_vector(2 * NUM + 1);

  G = make_matrix(2 * NUM + 1,2 * NUM + 1);

  
  r_c = sqrt(A / M_PI);
  R = 6.5 * r_c;
  
  for( i = 1; i <= NUM; i++ ){
    
    theta = i * 2 * M_PI / NUM;
    
    x3[i] = R * cos(theta);
    x4[i] = R * sin(theta);

    //printf("%f %f\n", x3[i], x4[i]);
    
  }
  connect_double(x3,x4);

  for( i = 1; i <= NUM; i++ ){
    
    ll[i] = DIST(x3[i - 1],x4[i - 1],x3[i],x4[i]);
    
    tt1[i] = ( x3[i] - x3[i - 1] ) / ll[i];
    tt2[i] = ( x4[i] - x4[i - 1] ) / ll[i];
    
    nn1[i] = tt2[i];
    nn2[i] = -tt1[i];

  }
  connect(ll);
  connect_double(tt1,tt2);
  connect_double(nn1,nn2);  

  for( i = 1; i <= NUM; i++ ){
    
    x_mid3[i] = ( x3[i] + x3[i - 1] ) / 2.0;
    x_mid4[i] = ( x4[i] + x4[i - 1] ) / 2.0;

  }
  connect_double(x_mid3,x_mid4);

  for( i = 1; i <= NUM; i++ ){
    
    amano1[i] = -( -( x_mid2[i + 1] - x_mid2[i - 1]) / DIST(x_mid1[i - 1],x_mid2[i - 1],x_mid1[i + 1],x_mid2[i + 1]) );
    
    amano2[i] = -( ( x_mid1[i + 1] - x_mid1[i - 1]) / DIST(x_mid1[i - 1],x_mid2[i - 1],x_mid1[i + 1],x_mid2[i + 1]) );
    
    amano3[i] = -( -( x_mid4[i + 1] - x_mid4[i - 1]) / DIST(x_mid3[i - 1],x_mid4[i - 1],x_mid3[i + 1],x_mid4[i + 1]) );
    
    amano4[i] = -( ( x_mid3[i + 1] - x_mid3[i - 1]) / DIST(x_mid3[i - 1],x_mid4[i - 1],x_mid3[i + 1],x_mid4[i + 1]) );

    //printf("%f %f\n", amano1[i], amano2[i]);
    
  }
  connect_double(amano1,amano2);
  connect_double(amano3,amano4);
  
  for( i = 1; i <= NUM; i++ ){

    d = DIST(x_mid1[i - 1],x_mid2[i - 1],x_mid1[i + 1],x_mid2[i + 1]) / 2.0;
    
    //y1[i] = x_mid1[i] - d * n1[i];
    //y2[i] = x_mid2[i] - d * n2[i];

    y1[i] = x_mid1[i] - d * amano1[i]; 
    y2[i] = x_mid2[i] - d * amano2[i];

    z1[i] = 1000.0 * y1[i];
    z2[i] = 1000.0 * y2[i];

    //printf("%f %f\n", y1[i], y2[i]);
    
  }
  
  for( i = NUM + 1; i <= 2 * NUM; i++ ){

    d = DIST(x_mid3[i - 1 - NUM],x_mid4[i - 1 - NUM],x_mid3[i + 1 - NUM],x_mid4[i + 1 - NUM]) / 2.0;

    //y3[i] = x_mid3[i] + d * nn1[i];
    //y4[i] = x_mid4[i] + d * nn2[i];
    
    y1[i] = x_mid3[i - NUM] + d * amano3[i - NUM]; 
    y2[i] = x_mid4[i - NUM] + d * amano4[i - NUM];

    z1[i] = 1000.0 * y1[i];
    z2[i] = 1000.0 * y2[i];

    //printf("%f %f\n", y1[i], y2[i]);
    
  }

  for( i = 1; i <= NUM; i++ ){
    
    for( j = 1; j <= 2 * NUM; j++ ){
      
      grad_E(x_mid1[i],x_mid2[i],y1[j],y2[j],z1[j],z2[j],&grad_E1,&grad_E2);
      //grad_E(x_mid1[i],x_mid2[i],y1[j],y2[j],&grad_E1,&grad_E2);
      
      H1[i][j] = grad_E1;
      H2[i][j] = grad_E2;
      
    }
    
  }

  for( i = NUM + 1; i <= 2 * NUM; i++ ){
    
    for( j = 1; j <= 2 * NUM; j++ ){
      
      grad_E(x_mid3[i - NUM],x_mid4[i - NUM],y1[j],y2[j],z1[j],z2[j],&grad_E1,&grad_E2);
      //grad_E(x_mid3[i - NUM],x_mid4[i - NUM],y1[j],y2[j],&grad_E1,&grad_E2);
      
      H1[i][j] = grad_E1;
      H2[i][j] = grad_E2;
      
    }

  }
  
  for( j = 1; j <= 2 * NUM; j++){
    
    H[j] = 0.0;
    
    for( i = 1; i <= NUM; i++){
      
      H[j] += ip(H1[i][j],H2[i][j],n1[i],n2[i]) * l[i];

    }
    
  }
  
  G[0][0] = 0.0;

  for( j = 1; j <= 2 * NUM; j++ ){
    
    G[0][j] = H[j];
    //G[0][j] = 1.0;
    
  }

  for( i = 1; i <= 2 * NUM; i++ ){
    
    G[i][0] = 1.0;
    
  }
  
  for( i = 1; i <= NUM; i++ ){
    
    for( j = 1; j <= 2 * NUM; j++ ){
      
      G[i][j] = E(x_mid1[i],x_mid2[i],y1[j],y2[j],z1[j],z2[j]) - gamma * ip(H1[i][j],H2[i][j],n1[i],n2[i]) / beta[i];
      //G[i][j] = E(x_mid1[i],x_mid2[i],y1[j],y2[j]) - gamma * ip(H1[i][j],H2[i][j],n1[i],n2[i]) / beta[i];
      
    }
    
  }
  
  for( i = NUM + 1; i <= 2 * NUM; i++ ){
    
    for( j = 1; j <= 2 * NUM; j++ ){
      
      G[i][j] = E(x_mid3[i - NUM],x_mid4[i - NUM],y1[j],y2[j],z1[j],z2[j]);
      //G[i][j] = E(x_mid3[i - NUM],x_mid4[i - NUM],y1[j],y2[j]);
      
    }
    
  }

  /*
  for( i = 0; i <= 2 * NUM; i++ ){
    
    for( j = 0; j <= 2 * NUM; j++ ){
    
    printf("G[%d][%d] = %.30f\n", i, j, G[i][j]);

    }
    
  }
  */
  
  gauss(2 * NUM + 1,G,P,Q);
    
  
  for( i = 1; i <= NUM; i++ ){

    u[i] = Q[0];

    for( j = 1; j <= 2 * NUM; j++ ){

      //u[i] += Q[j] * E(x_mid1[i],x_mid2[i],y1[j],y2[j]);
      u[i] += Q[j] * E(x_mid1[i],x_mid2[i],y1[j],y2[j],z1[j],z2[j]);      
    }

    //printf("u[%d] = %.30f\n", i, u[i]);
    
  }
  connect(u);

  /*
  for( i = 1; i <= NUM; i++ ){
    
    printf("u[%d] = %.15f\n", i, u[i]);
    
  }
  */
  
  for( i = 0; i <= 2 * NUM; i++ ){
    
    free(G[i]);
    
  }
  free(G);

  free(x3);
  free(x4);
  free(x_mid3);
  free(x_mid4);
  free(y1);
  free(y2);
  free(z1);
  free(z2);
  
  free(amano1);
  free(amano2);
  free(amano3);
  free(amano4);

  free(ll);
  free(tt1);
  free(tt2);
  free(nn1);
  free(nn2);
  
  for( i = 0; i <= 2 * NUM; i++ ){
    
    free(H1[i]);
    
  }
  free(H1);
  
  for( i = 0; i <= 2 * NUM; i++ ){
    
    free(H2[i]);
    
  }
  free(H2);

  free(Q);
  free(H);
  
}

double omega( int n ){
  
  return ( 10.0 * n );

}

void velocity( double t, double *x1, double *x2, double *n1, double *n2, double *l, double *phi, double *beta, double *kappa, double L, double A, double *V, double *W  ){
  
  int i;
  double *P;
  double *x_mid1,*x_mid2;
  double *u;
  double *v;
  double grad_E1, grad_E2;
  
  x_mid1 = make_vector(NUM + 2);
  x_mid2 = make_vector(NUM + 2);
  u = make_vector(NUM + 2);
  P = make_vector(2 * NUM + 1);
  v = make_vector(NUM + 2);
  
  for( i = 1; i <= NUM; i++ ){
    
    x_mid1[i] = ( x1[i] + x1[i - 1] ) / 2.0;
    x_mid2[i] = ( x2[i] + x2[i - 1] ) / 2.0;

  }
  connect_double(x_mid1,x_mid2);
  
  PP(t,P,A);
  grad_P(t,x_mid1,x_mid2,n1,n2,l,A,P,beta,u);
  normal_speed(t,x_mid1,x_mid2,n1,n2,phi,beta,u,v,V);
  tangent_speed(t,l,phi,kappa,v,V,L,W);

  free(x_mid1);
  free(x_mid2);
  free(v);
  free(P);
  free(u);
  
}

void normal_speed( double t, double *x_mid1, double *x_mid2, double *n1, double *n2, double *phi, double *beta, double *u, double *v, double *V ){
  
  int i;
  
  for( i = 1; i <= NUM; i++ ){
    
    v[i] = beta[i] * u[i];

  }
  connect(v);

  for( i = 1; i <= NUM; i++ ){
    
    V[i] = ( v[i] + v[i + 1] ) / ( 2.0 * cos(phi[i] / 2.0) );
  }
  connect(V);
  
}

void tangent_speed( double t, double *l, double *phi, double *kappa, double *v, double *V, double L, double *W ){
  
  int i;
  double *psi,*PSI;
  double L_dot;
  double a,b,c;
  
  psi = make_vector(NUM + 1);
  PSI = make_vector(NUM + 1);
  
  psi[1] = 0.0;
  L_dot = 0.0;
  
  for(i = 1; i <= NUM; i++ ){
    
    L_dot += kappa[i] * v[i] * l[i];

  }
  
  for( i = 2; i <= NUM; i++ ){
    
    psi[i] = ( L_dot / NUM ) - V[i] * sin(phi[i] / 2.0) - V[i-1] * sin(phi[i - 1] / 2.0) + ( ( L / NUM ) - l[i] ) * omega(NUM);

  }
  
  PSI[1] = psi[1];
  
  for( i = 2; i <= NUM; i++ ){
    
    PSI[i] = PSI[i-1] + psi[i];

  }

  a = 0.0;
  b = 0.0;

  for( i = 1; i <= NUM; i++ ){
    
    a += PSI[i] / cos(phi[i] / 2.0);
    b += 1.0 / cos(phi[i] / 2.0);

  }
  c = -a / b;

  for( i = 1; i <= NUM; i++ ){
    
    W[i] = ( PSI[i] + c ) / cos(phi[i] / 2.0);

  }
  connect(W);

  free(PSI);
  free(psi);

}
