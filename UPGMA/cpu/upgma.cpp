#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>
#include <iomanip>
#include <sys/time.h>

using std::swap;
using std::cout;
using std::vector;

inline double seconds() {
  struct timeval tp;
  struct timezone tzp;
  gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

struct Node {
  int id;
  int num_nodes;          // number of nodes in the subtree
  Node *left;             // left subtree
  Node *right;            // right subtree
  float total_length;     // total length of the subtree
  float branch_length[2]; // lengths of left and right subtrees
  Node(int _id, int _num_nodes, float _length, Node *_left, Node *_right,
       float length1, float length2)
      : id(_id), num_nodes(_num_nodes), left(_left), right(_right),
        total_length(_length) {
    branch_length[0] = length1;
    branch_length[1] = length2;
  }
};

class UPGMA {
public:
  UPGMA(float *_mat, int _num_seqs) : mat{_mat}, num_seqs{_num_seqs} {
    vector<Node *> nodes(num_seqs);
    for (int i = 0; i < num_seqs; ++i) {
      nodes[i] = new Node(i, 1, 0.0f, nullptr, nullptr, 0.0f, 0.0f);
    }
    for (int remain = num_seqs; remain >= 2; --remain) {
      int idx = getMinIdx();
      int idx1 = idx / num_seqs;
      int idx2 = idx % num_seqs;
      if (idx1 > idx2) {
        swap(idx1, idx2);
      }

      float length = mat[idx1 * num_seqs + idx2];
      root = new Node(-1, nodes[idx1]->num_nodes + nodes[idx2]->num_nodes,
                      length / 2, nodes[idx1], nodes[idx2],
                      length / 2 - nodes[idx1]->total_length,
                      length / 2 - nodes[idx2]->total_length);
      update(idx1, idx2, nodes[idx1]->num_nodes, nodes[idx2]->num_nodes);
      nodes[idx1] = root;
      nodes[idx2] = nullptr;
    }
  }

  ~UPGMA() { cleanup(root); }

  void print() {
    print(root);
    cout << "\n";
  }

private:
  float *mat;
  int num_seqs;
  Node *root;
  int getMinIdx() {
    int n = num_seqs * num_seqs;
    float val = INFINITY;
    int idx = -1;
    for (int i = 0; i < n; ++i) {
      if (mat[i] < val) {
        idx = i;
        val = mat[i];
      }
    }
    return idx;
  }

  void update(int idx1, int idx2, int num_nodes1, int num_nodes2) {
    int total_nodes = num_nodes1 + num_nodes2;
    for (int i = 0; i < num_seqs; ++i) {
      if (i == idx2) {
        mat[num_seqs * idx1 + i] = INFINITY;
        mat[num_seqs * i + idx1] = INFINITY;
        continue;
      }
      float val = mat[num_seqs * idx1 + i];
      if (isinf(val)) {
        continue;
      }
      float new_val =
          (val * num_nodes1 + mat[num_seqs * idx2 + i] * num_nodes2) /
          total_nodes;
      mat[num_seqs * idx1 + i] = new_val;
      mat[num_seqs * idx2 + i] = INFINITY;
      mat[num_seqs * i + idx1] = new_val;
      mat[num_seqs * i + idx2] = INFINITY;
    }
  }

  void cleanup(Node *node) {
    if (node == nullptr) {
      return;
    }
    cleanup(node->left);
    cleanup(node->right);
    delete node;
  }

  void print(Node *node) {
    // Reach the leaf
    if (node->left == nullptr && node->right == nullptr) {
      cout << "A" + std::to_string(node->id);
      return;
    }
    cout << "(";
    print(node->left);
    cout << ": " << std::fixed << node->branch_length[0] << ", ";
    print(node->right);
    cout << ": " << std::fixed << node->branch_length[1] << ")";
  }
};

int main(int argc, char *argv[]) {
#if 0
  // This is the test case
  // The tree should have the same shape as the tree on this page (Source tab)
  // http://www.southampton.ac.uk/~re1u06/teaching/upgma/
  // Convention: A0 = A, A1 = B, A2 = C, A3 = D, A4 = E, A5 = F, A6 = G
  const int num_seqs = 7;
  float a[num_seqs][num_seqs]{
      {INFINITY, 19.0f, 27.0f, 8.0f, 33.0f, 18.0f, 13.0f},
      {19.0f, INFINITY, 31.0f, 18.0f, 36.0f, 1.0f, 13.0f},
      {27.0f, 31.0f, INFINITY, 26.0f, 41.0f, 32.0f, 29.0f},
      {8.0f, 18.0f, 26.0f, INFINITY, 31.0f, 17.0f, 14.0f},
      {33.0f, 36.0f, 41.0f, 31.0f, INFINITY, 35.0f, 28.0f},
      {18.0f, 1.0f, 32.0f, 17.0f, 35.0f, INFINITY, 12.0f},
      {13.0f, 13.0f, 29.0f, 14.0f, 28.0f, 12.0f, INFINITY}};
  UPGMA upgma((float *)a, num_seqs);
  upgma.print();
#else
  // This is mock data to test with large matrix
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " number\n";
    exit(-1);
  }
  const int num_seqs = atoi(argv[1]);
  float *a = new float[num_seqs * num_seqs];
  srand(0);
  for (int i = 0; i < num_seqs; ++i) {
    for (int j = 0; j < i; ++j) {
      a[i * num_seqs + j] = rand() / (float)RAND_MAX;
      a[j * num_seqs + i] = a[i * num_seqs + j];
    }
    a[i * num_seqs + i] = INFINITY;
  }

  double start = seconds();
  UPGMA upgma(a, num_seqs);
  double elapsed = seconds() - start;
  upgma.print();
  //cout << "Time to reconstruct the tree: " << elapsed << "\n";
  delete a;
#endif
  return 0;
}
