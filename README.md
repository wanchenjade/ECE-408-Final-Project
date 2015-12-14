# Phylogenetic Tree Reconstruction

The project is successfully compiled under the following system configuration:
  - Ubuntu 14.04.3
  - CUDA Toolkit 7.5
  - g++ 4.8.5

In our project, we parallelize two popular heuristic methods:
  - Unweighted Pair Group Method with Arithmetic Mean (UPGMA).
  - Neighbor Joining (NJ).

## Tested results:
 - NVIDIA Geforce GTX 970 vs Intel Core i7-4820K @4.5GHz
   + **34.3x speedup** for UPGMA with 10000 taxa.
   + **58.25x speedup** for NJ with 10000 taxa.
   
## Folder structure:
  + UPGMA:
    + cpu: sequential code for UPGMA
    + gpu: GPU-accelerated code for UPGMA
  + NJ:
    + cpu: sequential code for NJ
    + gpu: GPU-accelerated code for NJ

## Output format:
  - The tree is printed in Newick format.
  - More detail of Newick format can be consulted here:
    http://evolution.genetics.washington.edu/phylip/newicktree.html

## How to visualize the printed tree:
Compare2Trees: pairwise comparison of phylogenies

More detail available here: http://www.mas.ncl.ac.uk/~ntmwn/compare2trees/

## How to compile:
  - To compile sequential UPGMA:
    
    ```
    cd UPGMA/cpu
    make
    ```
  - To compile GPU-accelerated UPGMA
  
    ```
    cd UPGMA/gpu
    make
    ```
  - To compile sequential NJ:
    
    ```
    cd NJ/cpu
    make
    ```
  - To compile GPU-accelerated NJ:
    
    ```
    cd NJ/gpu
    make
    ```

## How to test:
###To test a simple test case:
- Enter appropriate folder, open appropriate cpp or cu file.
- In main function, look for '#if ...' and change it to '#if 1'.
- Compile.
- Run the executable.
  
**For example:**

**Example 1:** we want to compare the result of sequential and GPU-accelerated versions of UPGMA

**Sequential version:**
  ```
  * cd UPGMA/cpu
  * Open upgma.cpp
  * In 'main' function, look for '#if ...' and change it to '#if 1'.
  * make
  * Run the executable: ./upgma
  ```
  The tree should have the same shape as the tree on this page (Source tab)
  http://www.southampton.ac.uk/~re1u06/teaching/upgma/
  
  The result on my computer is:
  ((((A0: 4.000000, A3: 4.000000): 4.250000, ((A1: 0.500000, A5: 0.500000): 5.750000, A6: 6.250000): 2.000000): 6.250000, A2: 14.500000): 2.500000, A4: 17.000000)

  Convention: A0 = A, A1 = B, A2 = C, A3 = D, A4 = E, A5 = F, A6 = G
  
  **GPU accelerated version:**
  ```
  * cd UPGMA/gpu
  * Open upgma.cu
  * In 'main' function, look for '#if ...' and change it to '#if 1'.
  * make
  * Run the executable: ./upgma
  ```
  The tree should have the same shape as the tree of the sequential version and the tree on the example page.

  The result on my computer is: 
  ((((A0: 4.000000, A3: 4.000000): 4.250000, ((A1: 0.500000, A5: 0.500000): 5.750000, A6: 6.250000): 2.000000): 6.250000, A2: 14.500000): 2.500000, A4: 17.000000)

**Example 2:** we want to compare the result of sequential and GPU-accelerated versions of NJ

**Sequential version:**
```
* cd NJ/cpu
* Open nj.cpp
* In 'main' function, look for '#if ...' and change it to '#if 1'.
* make
* Run the executable: ./nj
```
The tree should be the same as the tree on Wiki 
https://en.wikipedia.org/wiki/Neighbor_joining

The result on my computer is (((A0: 2, A1: 3): 3, A2: 4): 2, A3: 2, A4: 1)
Convention: a = A0, b = A1, c = A2, d = A3, e = A4, u,v are internal nodes, doesn't have name in Newick format

**GPU accelerated version:**
```
* cd NJ/gpu
* Open nj.cu
* In 'main' function, look for '#if ...' and change it to '#if 1'.
* make
* Run the executable: ./nj
```
The tree should have the same shape as the tree of the sequential version and the tree on the example page.        

The result on my computer is (((A0: 2, A1: 3): 3, A2: 4): 2, A4: 1, A3: 2)
**Note**: The result on GPU flips A3 and A4 order.

### To test for bigger test case:
Since we don't have the code to generate a distance matrix from a real dataset, we use mock data to test
- Enter appropriate folder, open appropriate cpp or cu file.
- In main function, look for '#if ...' and change it to '#if 0'.
- Compile
- Run the executable: ./excecutable number

**Example 1:** we want to compare the result of sequential and GPU-accelerated versions of UPGMA for 1000 taxa

**Sequential version:**
```
* cd UPGMA/cpu
* Open upgma.cpp
* In 'main' function, look for '#if ...' and change it to '#if 0'.
* make
* Run the executable: ./upgma 1000
```

**GPU accelerated version:**
```
* cd UPGMA/gpu
* Open upgma.cu
* In 'main' function, look for '#if ...' and change it to '#if 0'.
* make
* Run the executable: ./upgma 1000
```        
The tree should have the same shape as the tree of the sequential version. You can use the method described in **How to visualize the printed tree** to test.

**Example 2:** we want to compare the result of sequential and GPU-accelerated versions of NJ for 1000 taxa

**Sequential version:**
```        
* cd NJ/cpu
* Open nj.cpp
* In 'main' function, look for '#if ...' and change it to '#if 0'.
* make
* Run the executable: ./nj 1000
```

**GPU accelerated version:**
```
* cd NJ/gpu
* Open nj.cu
* In 'main' function, look for '#if ...' and change it to '#if 0'.
* make
* Run the executable: ./nj 1000
```        
The tree should have the same shape as the tree of the sequential version. You can use the method described in **How to visualize the printed tree** to test.
