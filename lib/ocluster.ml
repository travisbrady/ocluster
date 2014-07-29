open Printf
module B=Bigarray.Array1
module B2=Bigarray.Array2
module B3=Bigarray.Array3

type ta = (float, Bigarray.float64_elt, Bigarray.c_layout) B.t
type tan = (int32, Bigarray.int32_elt, Bigarray.c_layout) B.t
type tai = (int, Bigarray.int_elt, Bigarray.c_layout) B.t
type ta2 = (float, Bigarray.float64_elt, Bigarray.c_layout) B2.t
type tai2 = (int, Bigarray.int_elt, Bigarray.c_layout) B2.t
type ta3 = (float, Bigarray.float64_elt, Bigarray.c_layout) B3.t

external hi : string -> unit = "caml_hi"
external mean : ta -> float = "caml_mean"
external median: ta -> float = "caml_median"
(*external _pca : ta2 -> int -> int -> ta2 -> ta2 -> int = "caml_pca"*)
external _pca : ta2 -> int -> int -> (int * ta2 * ta2 * ta) = "caml_pca"
external hack : ta2 -> unit = "hack"
external bbb : int -> int -> ta2 = "bbb"

type kcluster_opts = (int * int * char * char)
(* data k mask weight *)
external _kcluster : ta2 -> int -> tai2 -> ta -> kcluster_opts -> (tan * float * int) = "caml_kcluster"

type somcluster_opts = (int * int * int * float * int * char)
external _somcluster : ta2 -> tai2 -> ta -> somcluster_opts -> (ta3 * tai2) = "caml_somcluster"

let int_of_transpose = function
    | false -> 0
    | true -> 1

let char_of_distance = function
    | `Euclidean -> 'e'
    | `City_block -> 'b'
    | `Correlation -> 'c'
    | `Abs_val_of_correlation -> 'a'
    | `Uncentered_correlation -> 'u'
    | `Abs_uncentered_correlation -> 'x'
    | `Spearman -> 's'
    | `Kendall -> 'k'

let somcluster 
    data 
    ?(transpose=false) 
    ?(inittau=0.02)
    ?(niter=1) 
    ?(nxgrid=2) 
    ?(nygrid=1) 
    ?(dist=`Euclidean)
    mask
    weight =
        let opts = (int_of_transpose transpose, nxgrid, nygrid, inittau, niter, char_of_distance dist) in
        _somcluster data mask weight opts
    
let kmeans data ?(transpose=false) ?(npass=100) ?(dist=`Euclidean) k mask weight =
    let opts = (int_of_transpose transpose, npass, 'a', char_of_distance dist) in
    _kcluster data k mask weight opts

let demean mat nrows ncols =
    for j = 0 to ncols-1 do
        let sum = ref 0.0 in
        for i = 0 to nrows-1 do
            sum := !sum +. mat.{i, j}
        done;
        let mean = !sum /. (float_of_int nrows) in
        for i = 0 to nrows-1 do
            mat.{i, j} <- (mat.{i, j} -. mean)
        done;
    done;
    ()

let pca arr =
    let nrows = B2.dim1 arr in
    let ncols = B2.dim2 arr in

    demean arr nrows ncols;
    _pca arr nrows ncols

