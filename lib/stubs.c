#include <stdio.h>
#include <stdint.h>
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/bigarray.h>

#include "cluster.h"

#define Char_val(x) ((char)Int_val(x))

CAMLprim value
caml_hi(value x) {
    printf("Hi: %s\n", String_val(x));
    return Val_unit;
}

CAMLprim value
caml_mean(value data) {
    int n = Bigarray_val(data)->dim[0];
    printf("N: %d\n", n);
    double m = mean(n, Data_bigarray_val(data));
    return caml_copy_double(m);
}

CAMLprim value
caml_median(value data) {
    int n = Bigarray_val(data)->dim[0];
    printf("N: %d\n", n);
    double m = median(n, Data_bigarray_val(data));
    return caml_copy_double(m);
}

double** copy_to_c_mat(value omat) {
    double **mato;
    double *u = (double *)Data_bigarray_val(omat);
    int n = Bigarray_val(omat)->dim[0];
    int m = Bigarray_val(omat)->dim[1];
    mato = (double **)malloc(n * sizeof(double*));
    for(int i = 0; i < n; i++) {
        mato[i] = (double*) malloc(m * sizeof(double));
    }

    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            mato[i][j] = u[i * n + j];
        }
    }
    return mato;
}

int** copy_to_c_int_mat(value omat) {
    int **mato;
    int *u = (int *)Data_bigarray_val(omat);
    int n = Bigarray_val(omat)->dim[0];
    int m = Bigarray_val(omat)->dim[1];
    mato = (int **)malloc(n * sizeof(int *));
    for(int i = 0; i < n; i++) {
        mato[i] = (int*) malloc(m * sizeof(int));
    }
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            mato[i][j] = u[i + j];
        }
    }
    return mato;
}

double** make_mat(int n, int m) {
    double **mato;
    mato = (double **)malloc(n * sizeof(double*));
    for(int i = 0; i < n; i++) {
        mato[i] = (double*) malloc(m * sizeof(double));
    }
    return mato;
}

void copy_c_to_o(value o, int n, int m, double **c) {
    double *oa = (double *)Data_bigarray_val(o);
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            oa[i * n + j * m] = c[i][j];
        }
    }
}

CAMLprim value
caml_pca(value v_data, value v_nrows, value v_ncols) {
    CAMLparam3(v_data, v_nrows, v_ncols);
    CAMLlocal4(vu, vv, vw, out);
    int nrows = Int_val(v_nrows);
    int ncols = Int_val(v_ncols);
    double **u = copy_to_c_mat(v_data);
    double **v = copy_to_c_mat(v_data);
    double *w = (double *)malloc(nrows * sizeof(double*));

    int res = pca(nrows, ncols, u, v, w);
    printf("Post Pca %d\n", res);
    for(int i=0; i<nrows; i++) {
        printf("[C] ");
        for(int j=0; j<ncols; j++) {
            printf("%f ", u[i][j]);
        }
        printf("\n");
    }

    vu = alloc_bigarray_dims(BIGARRAY_FLOAT64 | BIGARRAY_C_LAYOUT,
            2, *u, nrows, ncols);
    vv = alloc_bigarray_dims(BIGARRAY_FLOAT64 | BIGARRAY_C_LAYOUT,
            2, *v, nrows, ncols);
    vw = alloc_bigarray_dims(BIGARRAY_FLOAT64 | BIGARRAY_C_LAYOUT,
            1, w, nrows);
    out = caml_alloc_tuple(4);
    Store_field(out, 0, Val_int(res));
    Store_field(out, 1, vu);
    Store_field(out, 2, vv);
    Store_field(out, 3, vw);
    CAMLreturn(out);
}

CAMLprim value
caml_kcluster(value v_data,
              value v_k,
              value v_mask,
              value v_weight,
              value v_opts) {
    CAMLparam5(v_data, v_k, v_mask, v_weight, v_opts);
    CAMLlocal2(out, v_clusterid);
    double **data = copy_to_c_mat(v_data);
    int **mask = copy_to_c_int_mat(v_mask);
    int nrows = Bigarray_val(v_data)->dim[0];
    int ncols = Bigarray_val(v_data)->dim[1];
    double *weight = (double *)Data_bigarray_val(v_weight);

    int transpose = Int_val(Field(v_opts, 0));
    int npass = Int_val(Field(v_opts, 1));
    char method = Char_val(Field(v_opts, 2));
    char dist = Char_val(Field(v_opts, 3));

    int nclusterids = nrows;
    if (transpose == 1) {
        nclusterids = ncols;
    }
    int *clusterid = (int *)malloc(nclusterids * sizeof(int));
    double error;
    int ifound;
    kcluster(Int_val(v_k),
             nrows,
             ncols,
             data,
             mask,
             weight,
             transpose,
             npass,
             method,
             dist,
             clusterid,
             &error,
             &ifound);
    for(int i=0; i<nclusterids; i++) {
        printf("Error: %f CID: %d\n", error, clusterid[i]); fflush(stdout);
    }
    v_clusterid = alloc_bigarray_dims(BIGARRAY_INT32| BIGARRAY_C_LAYOUT,
                                      1, clusterid, nclusterids);
    out = caml_alloc_tuple(3);
    Store_field(out, 0, v_clusterid);
    Store_field(out, 1, caml_copy_double(error));
    Store_field(out, 2, ifound);
    CAMLreturn(out);
}

/*
void somcluster (int nrows, int ncolumns, double** data, int** mask,
  const double weight[], int transpose, int nxnodes, int nynodes,
  double inittau, int niter, char dist, double*** celldata,
  int clusterid[][2]);
  */
CAMLprim value
caml_somcluster(value v_data,
                value v_mask,
                value v_weight,
                value v_opts) {
    CAMLparam4(v_data, v_mask, v_weight, v_opts);
    CAMLlocal3(out, v_clusterid, v_celldata);
    double **data = copy_to_c_mat(v_data);
    int **mask = copy_to_c_int_mat(v_mask);
    int nrows = Bigarray_val(v_data)->dim[0];
    int ncols = Bigarray_val(v_data)->dim[1];
    double *weight = (double *)Data_bigarray_val(v_weight);
    int transpose = Int_val(Field(v_opts, 0));
    int nxgrid = Int_val(Field(v_opts, 1));
    int nygrid = Int_val(Field(v_opts, 2));
    int inittau = Double_val(Field(v_opts, 3));
    int niter = Int_val(Field(v_opts, 4));
    char dist = Char_val(Field(v_opts, 5));

    int third_dim = ncols;
    if (transpose == 1) {
        third_dim = nrows;
    }
    double ***celldata;
    celldata = (double ***)malloc(nxgrid * sizeof(double*));
    for(int i = 0; i < nxgrid; i++) {
        celldata[i] = (double **) malloc(nygrid * sizeof(double));
        for(int j = 0; j < nygrid; j++) {
            celldata[i][j] = (double*) malloc(third_dim * sizeof(double));
        }
    }
    int **clusterid = (int **) malloc(third_dim * sizeof(int));
    for(int i=0; i<third_dim; i++) {
        //clusterid[i][2] = (int (*)[2]) clusterid[i];
        //clusterid[i] = (int (*)[2]) malloc(2 * sizeof(int));
        clusterid[i] =  (int *) malloc(2 * sizeof(int));
    }
    somcluster(nrows,
               ncols,
               data,
               mask,
               weight,
               transpose,
               nxgrid,
               nygrid,
               inittau,
               niter,
               dist,
               celldata,
               clusterid);
    v_celldata = alloc_bigarray_dims(BIGARRAY_FLOAT64 | BIGARRAY_C_LAYOUT,
            3, celldata, nxgrid, nygrid, third_dim);
    v_clusterid = alloc_bigarray_dims(BIGARRAY_INT32| BIGARRAY_C_LAYOUT,
                                      2, clusterid, third_dim);
    out = caml_alloc_tuple(2);
    Store_field(out, 0, v_celldata);
    Store_field(out, 1, v_clusterid);
    CAMLreturn(out);
}

CAMLprim value
hack(value mat) {
    //CAMLparam1(mat);
    double *u = (double *)Data_bigarray_val(mat);
    int n = Bigarray_val(mat)->dim[0];
    int m = Bigarray_val(mat)->dim[1];

    double **mato;
    mato = (double **)malloc(n * sizeof(double*));
    for(int i = 0; i < n; i++) {
        mato[i] = (double*) malloc(m * sizeof(double));
    }

    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            mato[i][j] = u[i * n + j * m];
            //printf("MATO: %f\n", mato[i][j]);
        }
    }

    printf("Yo: %f\n", u[0]); fflush(stdout);
    for (int i=0; i < n*m; i++) {
        printf("X %f\n", u[i]);
    }
    //printf("Dub: %f\n", u[0][0]); fflush(stdout);
    //printf("HACK N: %d %d\n", n, m); fflush(stdout);
    //CAMLreturn(Val_unit);
    return Val_unit;
}

CAMLprim value
bbb(value n, value m) {
    CAMLparam2(n, m);
    CAMLlocal1(out);
    int nrows = Int_val(n);
    int ncols = Int_val(m);
    out = caml_ba_alloc_dims(
            BIGARRAY_FLOAT64 | BIGARRAY_C_LAYOUT,
            2, NULL, nrows, ncols);
    double **data = (double **)Data_bigarray_val(out);
    printf("DATA\n"); fflush(stdout);

    for(int i=0; i<nrows; i++) {
        printf("[c] "); fflush(stdout);
        for(int j=0; j<ncols; j++) {
            printf("%f ", data[i][j]);
        }
        printf("\n");
    }

    CAMLreturn(out);
}
