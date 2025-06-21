// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.

// template <class Op>
// Buffer binary_op(const Buffer &lhs, const Buffer &rhs,
//                  const std::vector<std::size_t> &lhs_shape,
//                  const std::vector<std::size_t> &rhs_shape,
//                  const std::vector<std::size_t> &out_shape,
//                  const std::string &dtype) {
//   std::size_t out_size = 1;
//   for (auto s : out_shape)
//     out_size *= s;

//   Buffer out(out_size, dtype);
// }

// Buffer add(const Buffer &lhs, const Buffer &rhs,
//            const std::vector<std::size_t> &lhs_shape,
//            const std::vector<std::size_t> &rhs_shape,
//            const std::vector<std::size_t> &out_shape,
//            const std::string &dtype) {
//   return binary_op<AddOp>(lhs, rhs, lhs_shape, rhs_shape, out_shape, dtype);
// }
