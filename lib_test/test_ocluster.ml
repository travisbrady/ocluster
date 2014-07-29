open Printf
module B = Bigarray.Array1
module B2 = Bigarray.Array2

let print_ba v n m =
    for i = 0 to n-1 do
        printf "[O] ";
        for j = 0 to m-1 do
            printf "%f " v.{i, j}
        done;
        printf "\n"
    done

let test_pca () =
    let data = B2.of_array Bigarray.float64 Bigarray.c_layout 
    [|
        [| 3.1; 1.2 |];
        [| 1.4; 1.3 |];
        [| 1.1; 1.5 |];
        [| 2.0; 1.5 |];
        [| 1.7; 1.9 |];
        [| 1.7; 1.9 |];
        [| 5.7; 5.9 |];
        [| 5.7; 5.9 |];
        [| 3.1; 3.3 |];
        [| 5.4; 5.3 |];
        [| 5.1; 5.5 |];
        [| 5.0; 5.5 |];
        [| 5.1; 5.2 |];
    |] in

    let n = B2.dim1 data in
    let m = B2.dim2 data in
    let res,u,v,w = Ocluster.pca data in
    print_ba u n m;
    ()

let test_kcluster () =
    let k = 3 in
    let weight = B.of_array Bigarray.float64 Bigarray.c_layout [|1.0; 1.0; 1.0; 1.0; 1.0|] in
    let data = B2.of_array Bigarray.float64 Bigarray.c_layout
            [|
                [| 1.1; 2.2; 3.3; 4.4; 5.5|];
                [| 3.1; 3.2; 1.3; 2.4; 1.5|];
                [| 4.1; 2.2; 0.3; 5.4; 0.5|];
                [|12.1; 2.0; 0.0; 5.0; 0.0|]
            |] 
    in
    let mask = B2.of_array Bigarray.int Bigarray.c_layout
        [|
            [| 1; 1; 1; 1; 1|];
            [| 1; 1; 1; 1; 1|];
            [| 1; 1; 1; 1; 1|];
            [| 1; 1; 1; 1; 1|]
        |]
    in
    let clusterid, error, nfound = Ocluster.kmeans data k mask weight in
    printf "Error: %f\n" error;

    let nc = B.dim clusterid in
    for i = 0 to nc-1 do
        printf "Clust: %ld\n" clusterid.{i}
    done


let test_somcluster () =
    let weight = B.of_array Bigarray.float64 Bigarray.c_layout 
        [|1.0; 1.0; 1.0; 1.0; 1.0|] in
    let data = B2.of_array Bigarray.float64 Bigarray.c_layout 
        [|
            [|  1.1; 2.2; 3.3; 4.4; 5.5|];
            [|  3.1; 3.2; 1.3; 2.4; 1.5|];
            [|  4.1; 2.2; 0.3; 5.4; 0.5|];
            [| 12.1; 2.0; 0.0; 5.0; 0.0|]
        |] in
    let mask = B2.create Bigarray.int Bigarray.c_layout 4 5 in
    B2.fill mask 1;
    let celldata,clusterid = Ocluster.somcluster 
                                data 
                                mask 
                                weight 
    in
    (*
    printf "[test_somcluster] %d\n" clusterid.{0,0};
    printf "[test_somcluster] %d\n" (B.dim (B2.slice_left 0))
    *)
    ()

let () =
    let data = B.of_array Bigarray.float64 Bigarray.c_layout 
        [|1.0; 2.0; 3.0; 5.0; 9.0; 15.0; 22.0|] in
    let m = Ocluster.mean data in
    (*
    printf "Mean: %f\n" m;
    printf "Median: %f\n" (Ocluster.median data);
    *)

    let distmat = B2.of_array Bigarray.float64 Bigarray.c_layout
        [|
            [|0.0; 1.0; 2.0|];
            [|1.0; 0.0; 1.0|];
            [|2.0; 1.0; 0.0|]
        |]
    in
    (*
    let vec = B.of_array Bigarray.float64 Bigarray.c_layout [|2.0; 1.0; 0.0|] in
    Ocluster.hack vec
    *)
    test_pca ();
    test_kcluster ();
    test_somcluster ();
    ()

