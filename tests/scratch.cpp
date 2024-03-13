#include <iostream>
#include <vector>

#include <torch/torch.h>

auto scratch_1() -> int {
  auto row_1 = std::vector<int>{1, 1, 1};
  auto row_2 = std::vector<int>{2, 2, 2};
  row_1.insert(row_1.end(), row_2.begin(), row_2.end());
  auto tensor = torch::from_blob(row_1.data(), {2, 3}, torch::kInt32);
  std::cout << tensor << '\n';
  return 0;
}

auto scratch_2() -> int {
  auto indices = std::vector<int>{1, 2, 2, 1, 1, 3, 3, 1};
  auto const tensor =
      torch::from_blob(
          indices.data(), {static_cast<long>(indices.size() / 2), 2},
          torch::kInt32)
          .transpose_(0, 1);
  std::cout << tensor << '\n';
  return 0;
}
