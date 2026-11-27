// Adapted from oMLX (633d6cf62971262149c2b2477361487958da3915), Apache-2.0.
// The upstream code includes Apple MLX Metal routines, copyright (c) 2024-25 Apple Inc., MIT.
// See THIRD_PARTY_NOTICES.md.
#if os(macOS)
// swiftlint:disable file_length line_length
extension Inkling2BandedPrefill {
    static let metalSource = #"""
    #include <metal_simdgroup>
    #include <metal_simdgroup_matrix>
    #include <metal_stdlib>

    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

    #ifndef OMLX_NAX_TILES
    #define OMLX_NAX_TILES

    #define OMLX_NAX_CONST static constant constexpr const
    #define OMLX_NAX_UNROLL _Pragma("clang loop unroll(full)")
    #define OMLX_NAX_NO_UNROLL _Pragma("clang loop unroll(disable)")

    namespace omlx_nax {

    using namespace metal;

    template <typename T>
    struct pointer_element {};
    template <typename T>
    struct pointer_element<thread T*> {
      using type = remove_cv_t<T>;
    };
    template <typename T>
    struct pointer_element<device T*> {
      using type = remove_cv_t<T>;
    };
    template <typename T>
    struct pointer_element<constant T*> {
      using type = remove_cv_t<T>;
    };
    template <typename T>
    struct pointer_element<threadgroup T*> {
      using type = remove_cv_t<T>;
    };
    template <typename T>
    using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

    template <typename T, T v>
    struct integral_constant {
      static constexpr constant T value = v;
      using value_type = T;
      using type = integral_constant;

      METAL_FUNC constexpr operator value_type() const thread noexcept {
        return value;
      }
    };

    template <bool B>
    using bool_constant = integral_constant<bool, B>;
    using true_type = bool_constant<true>;
    using false_type = bool_constant<false>;

    template <int val>
    using Int = integral_constant<int, val>;

    #define OMLX_NAX_CONST_BINOP(__op__, __operator__)          \
      template <typename T, T tv, typename U, U uv>             \
      METAL_FUNC constexpr auto __operator__(                   \
          integral_constant<T, tv>, integral_constant<U, uv>) { \
        constexpr auto res = tv __op__ uv;                      \
        using res_t = metal::remove_addrspace_t<decltype(res)>; \
        return integral_constant<res_t, res>{};                 \
      }

    OMLX_NAX_CONST_BINOP(+, operator+);
    OMLX_NAX_CONST_BINOP(-, operator-);
    OMLX_NAX_CONST_BINOP(*, operator*);
    OMLX_NAX_CONST_BINOP(/, operator/);

    #undef OMLX_NAX_CONST_BINOP

    template <typename F>
    void dispatch_bool(bool v, F f) {
      if (v) {
        f(true_type{});
      } else {
        f(false_type{});
      }
    }

    template <int start, int stop, int step, typename F>
    constexpr void const_for_loop(F f) {
      if constexpr (start < stop) {
        constexpr auto idx = Int<start>{};
        f(idx);
        const_for_loop<start + step, stop, step, F>(f);
      }
    }

    struct BaseNAXFrag {
      OMLX_NAX_CONST short kFragRows = 16;
      OMLX_NAX_CONST short kFragCols = 16;

      OMLX_NAX_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;

      OMLX_NAX_CONST short kElemRows = 2;
      OMLX_NAX_CONST short kElemCols = 4;

      OMLX_NAX_CONST short kElemRowsJump = 8;

      template <typename U>
      using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

      METAL_FUNC static short2 get_coord() {
        const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
        const short qid = simd_lane_id >> 2;
        const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
        const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
        return short2{fn, fm};
      }

      template <
          typename T,
          typename SrcPtrType,
          typename StrX,
          typename StrY,
          typename OffX = Int<0>,
          typename OffY = Int<0>>
      METAL_FUNC static constexpr void load(
          thread dtype_frag_t<T>& dst,
          SrcPtrType src,
          StrX str_x,
          StrY str_y,
          OffX off_x = {},
          OffY off_y = {}) {
        const short2 sc = get_coord();
        src += sc.y * str_x + sc.x * str_y;

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemRows; i++) {
          const auto r = off_x + i * kElemRowsJump;
          const auto c = off_y;

          if constexpr (metal::is_same_v<StrY, Int<1>>) {
            OMLX_NAX_UNROLL
            for (short j = 0; j < kElemCols; j++) {
              dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
            }
          } else {
            OMLX_NAX_UNROLL
            for (short j = 0; j < kElemCols; j++) {
              dst[i * kElemCols + j] =
                  static_cast<T>(src[r * str_x + (c + j) * str_y]);
            }
          }
        }
      }

      template <
          typename T,
          typename SrcPtrType,
          typename StrX,
          typename StrY,
          typename LimX,
          typename OffX = Int<0>,
          typename OffY = Int<0>>
      METAL_FUNC static constexpr void load_rows(
          thread dtype_frag_t<T>& dst,
          SrcPtrType src,
          StrX str_x,
          StrY str_y,
          LimX lim_x,
          OffX off_x = {},
          OffY off_y = {}) {
        const short2 sc = get_coord();
        src += sc.y * str_x + sc.x * str_y;
        auto lx = lim_x - sc.y;

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemRows; i++) {
          const auto r = off_x + i * kElemRowsJump;
          const auto c = off_y;

          if (r < lx) {
            if constexpr (metal::is_same_v<StrY, Int<1>>) {
              OMLX_NAX_UNROLL
              for (short j = 0; j < kElemCols; j++) {
                dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j)]);
              }
            } else {
              OMLX_NAX_UNROLL
              for (short j = 0; j < kElemCols; j++) {
                dst[i * kElemCols + j] =
                    static_cast<T>(src[r * str_x + (c + j) * str_y]);
              }
            }

          } else {
            OMLX_NAX_UNROLL
            for (short j = 0; j < kElemCols; j++) {
              dst[i * kElemCols + j] = T(0);
            }
          }
        }
      }

      template <
          typename T,
          typename SrcPtrType,
          typename StrX,
          typename StrY,
          typename LimX,
          typename LimY,
          typename OffX = Int<0>,
          typename OffY = Int<0>>
      METAL_FUNC static constexpr void load_safe(
          thread dtype_frag_t<T>& dst,
          SrcPtrType src,
          StrX str_x,
          StrY str_y,
          LimX lim_x,
          LimY lim_y,
          OffX off_x = {},
          OffY off_y = {}) {
        const short2 sc = get_coord();
        src += sc.y * str_x + sc.x * str_y;
        auto lx = lim_x - sc.y;
        auto ly = lim_y - sc.x;

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemRows; i++) {
          const auto r = off_x + i * kElemRowsJump;
          const auto c = off_y;
          OMLX_NAX_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            if ((r < lx) && ((c + j) < ly)) {
              dst[i * kElemCols + j] =
                  static_cast<T>(src[r * str_x + (c + j) * str_y]);
            } else {
              dst[i * kElemCols + j] = T(0);
            }
          }
        }
      }

      template <
          typename T,
          typename DstPtrType,
          typename StrX,
          typename StrY,
          typename OffX = Int<0>,
          typename OffY = Int<0>>
      METAL_FUNC static constexpr void store(
          const thread dtype_frag_t<T>& src,
          DstPtrType dst,
          StrX str_x,
          StrY str_y,
          OffX off_x = {},
          OffY off_y = {}) {
        using U = pointer_element_t<DstPtrType>;

        const short2 sc = get_coord();
        dst += sc.y * str_x + sc.x * str_y;

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemRows; i++) {
          const auto r = off_x + i * kElemRowsJump;
          const auto c = off_y;

          if constexpr (metal::is_same_v<StrY, Int<1>>) {
            OMLX_NAX_UNROLL
            for (short j = 0; j < kElemCols; j++) {
              dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
            }
          } else {
            OMLX_NAX_UNROLL
            for (short j = 0; j < kElemCols; j++) {
              dst[r * str_x + (c + j) * str_y] =
                  static_cast<U>(src[i * kElemCols + j]);
            }
          }
        }
      }

      template <
          typename T,
          typename DstPtrType,
          typename StrX,
          typename StrY,
          typename LimX,
          typename OffX = Int<0>,
          typename OffY = Int<0>>
      METAL_FUNC static constexpr void store_rows(
          const thread dtype_frag_t<T>& src,
          DstPtrType dst,
          StrX str_x,
          StrY str_y,
          LimX lim_x,
          OffX off_x = {},
          OffY off_y = {}) {
        using U = pointer_element_t<DstPtrType>;

        const short2 sc = get_coord();
        dst += sc.y * str_x + sc.x * str_y;
        auto lx = lim_x - sc.y;

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemRows; i++) {
          const auto r = off_x + i * kElemRowsJump;
          const auto c = off_y;

          if (r < lx) {
            if constexpr (metal::is_same_v<StrY, Int<1>>) {
              OMLX_NAX_UNROLL
              for (short j = 0; j < kElemCols; j++) {
                dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
              }
            } else {
              OMLX_NAX_UNROLL
              for (short j = 0; j < kElemCols; j++) {
                dst[r * str_x + (c + j) * str_y] =
                    static_cast<U>(src[i * kElemCols + j]);
              }
            }
          }
        }
      }

      template <
          typename T,
          typename DstPtrType,
          typename StrX,
          typename StrY,
          typename LimX,
          typename LimY,
          typename OffX = Int<0>,
          typename OffY = Int<0>>
      METAL_FUNC static constexpr void store_safe(
          const thread dtype_frag_t<T>& src,
          DstPtrType dst,
          StrX str_x,
          StrY str_y,
          LimX lim_x,
          LimY lim_y,
          OffX off_x = {},
          OffY off_y = {}) {
        using U = pointer_element_t<DstPtrType>;

        const short2 sc = get_coord();
        dst += sc.y * str_x + sc.x * str_y;
        auto lx = lim_x - sc.y;
        auto ly = lim_y - sc.x;

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemRows; i++) {
          const auto r = off_x + i * kElemRowsJump;
          const auto c = off_y;

          OMLX_NAX_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            if (r < lx && (c + j) < ly) {
              dst[r * str_x + (c + j) * str_y] =
                  static_cast<U>(src[i * kElemCols + j]);
            }
          }
        }
      }

      template <typename Op, typename T>
      METAL_FUNC static constexpr void row_reduce(
          thread const dtype_frag_t<T>& inp_vals,
          thread T* reduced_vals) {
        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemRows; i++) {
          T thr_reduce = Op::apply(
              Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
              Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));

          T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
          qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

          T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
          sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

          reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
        }
      }

      template <typename Op, typename T>
      METAL_FUNC static constexpr void row_bin_op(
          thread dtype_frag_t<T>& inp_vals,
          thread T* row_vals) {
        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemRows; i++) {
          OMLX_NAX_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            inp_vals[i * kElemCols + j] =
                Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
          }
        }
      }

      template <
          typename CType,
          typename AType,
          typename BType,
          bool transpose_a = false,
          bool transpose_b = false>
      METAL_FUNC static constexpr void mma(
          thread dtype_frag_t<CType>& Cn0,
          thread dtype_frag_t<CType>& Cn1,
          const thread dtype_frag_t<AType>& A,
          metal::bool_constant<transpose_a>,
          const thread dtype_frag_t<BType>& Bn0,
          const thread dtype_frag_t<BType>& Bn1,
          metal::bool_constant<transpose_b>) {
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16,
            32,
            16,
            transpose_a,
            transpose_b,
            true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

        auto ct_a =
            gemm_op
                .template get_left_input_cooperative_tensor<AType, BType, CType>();
        auto ct_b =
            gemm_op
                .template get_right_input_cooperative_tensor<AType, BType, CType>();

        auto ct_c = gemm_op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(ct_a)>,
            metal::remove_addrspace_t<decltype(ct_b)>,
            CType>();

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_a[i] = A[i];
        }

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_b[i] = Bn0[i];
          ct_b[kElemsPerFrag + i] = Bn1[i];
        }

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_c[i] = Cn0[i];
          ct_c[kElemsPerFrag + i] = Cn1[i];
        }

        gemm_op.run(ct_a, ct_b, ct_c);

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) {
          Cn0[i] = ct_c[i];
          Cn1[i] = ct_c[kElemsPerFrag + i];
        }
      }

      template <
          typename CType,
          typename AType,
          typename BType,
          bool transpose_a = false,
          bool transpose_b = false>
      METAL_FUNC static constexpr void mma(
          thread dtype_frag_t<CType>& Cm0,
          thread dtype_frag_t<CType>& Cm1,
          const thread dtype_frag_t<AType>& Am0,
          const thread dtype_frag_t<AType>& Am1,
          metal::bool_constant<transpose_a>,
          const thread dtype_frag_t<BType>& B,
          metal::bool_constant<transpose_b>) {
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16,
            32,
            16,
            transpose_a,
            transpose_b,
            true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

        auto ct_a =
            gemm_op
                .template get_left_input_cooperative_tensor<AType, BType, CType>();
        auto ct_b =
            gemm_op
                .template get_right_input_cooperative_tensor<AType, BType, CType>();

        auto ct_c = gemm_op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(ct_a)>,
            metal::remove_addrspace_t<decltype(ct_b)>,
            CType>();

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_a[i] = Am0[i];
          ct_a[kElemsPerFrag + i] = Am1[i];
        }

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_b[i] = B[i];
        }

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_c[i] = Cm0[i];
          ct_c[kElemsPerFrag + i] = Cm1[i];
        }

        gemm_op.run(ct_a, ct_b, ct_c);

        OMLX_NAX_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) {
          Cm0[i] = ct_c[i];
          Cm1[i] = ct_c[kElemsPerFrag + i];
        }
      }
    };

    template <
        typename T,
        short kTileRows_,
        short kTileCols_,
        class NAXFrag_ = BaseNAXFrag>
    struct NAXTile {
      using NAXFrag_t = NAXFrag_;
      using elem_type = T;

      OMLX_NAX_CONST short kFragRows = NAXFrag_t::kFragRows;
      OMLX_NAX_CONST short kFragCols = NAXFrag_t::kFragCols;
      OMLX_NAX_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;

      OMLX_NAX_CONST short kTileRows = kTileRows_;
      OMLX_NAX_CONST short kTileCols = kTileCols_;

      OMLX_NAX_CONST short kRows = kTileRows * kFragRows;
      OMLX_NAX_CONST short kCols = kTileCols * kFragCols;

      OMLX_NAX_CONST short kNumFrags = kTileRows * kTileCols;
      OMLX_NAX_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;

      OMLX_NAX_CONST short kFragThrRows = NAXFrag_t::kElemRows;
      OMLX_NAX_CONST short kFragThrCols = NAXFrag_t::kElemCols;
      OMLX_NAX_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;

      OMLX_NAX_CONST short kRowsPerThread = kTileRows * NAXFrag_t::kElemRows;
      OMLX_NAX_CONST short kColsPerThread = kTileCols * NAXFrag_t::kElemCols;

      typedef typename NAXFrag_t::template dtype_frag_t<T> frag_type;

      frag_type val_frags[kNumFrags];

      METAL_FUNC NAXTile() thread {}

      METAL_FUNC constexpr void clear() thread {
        OMLX_NAX_UNROLL
        for (short i = 0; i < kNumFrags; ++i) {
          val_frags[i] = frag_type(0);
        }
      }

      METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j)
          thread {
        return val_frags[i * kTileCols + j];
      }

      METAL_FUNC constexpr const thread frag_type& frag_at(
          const short i,
          const short j) const thread {
        return val_frags[i * kTileCols + j];
      }

      template <int i, int j>
      METAL_FUNC constexpr thread frag_type& frag_at() thread {
        return val_frags[i * kTileCols + j];
      }

      template <int i, int j>
      METAL_FUNC constexpr const thread frag_type& frag_at() const thread {
        return val_frags[i * kTileCols + j];
      }

      template <bool transpose>
      METAL_FUNC constexpr thread frag_type& frag_at(
          const short i,
          const short j,
          metal::bool_constant<transpose>) thread {
        if constexpr (transpose) {
          return frag_at(j, i);
        } else {
          return frag_at(i, j);
        }
      }

      template <bool transpose>
      METAL_FUNC constexpr const thread frag_type& frag_at(
          const short i,
          const short j,
          metal::bool_constant<transpose>) const thread {
        if constexpr (transpose) {
          return frag_at(j, i);
        } else {
          return frag_at(i, j);
        }
      }

      METAL_FUNC thread elem_type* elems() thread {
        return reinterpret_cast<thread elem_type*>(val_frags);
      }

      METAL_FUNC const thread elem_type* elems() const thread {
        return reinterpret_cast<const thread elem_type*>(val_frags);
      }

      template <typename Op>
      METAL_FUNC void row_reduce(
          thread metal::vec<T, kRowsPerThread>& vals) const thread {
        auto vptr = (thread T*)(&vals);
        OMLX_NAX_UNROLL
        for (short i = 0; i < kTileRows; ++i) {
          OMLX_NAX_UNROLL
          for (short j = 0; j < kTileCols; ++j) {
            NAXFrag_t::template row_reduce<Op>(
                frag_at(i, j), &vptr[i * kFragThrRows]);
          }
        }
      }

      template <typename Op>
      METAL_FUNC void row_bin_op(
          thread metal::vec<T, kRowsPerThread>& vals) thread {
        auto vptr = (thread T*)(&vals);
        OMLX_NAX_UNROLL
        for (short i = 0; i < kTileRows; ++i) {
          OMLX_NAX_UNROLL
          for (short j = 0; j < kTileCols; ++j) {
            NAXFrag_t::template row_bin_op<Op>(
                frag_at(i, j), &vptr[i * kFragThrRows]);
          }
        }
      }

      template <typename U, int str_x, int str_y>
      METAL_FUNC void load(const threadgroup U* src) thread {
        const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
          const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
            NAXFrag_t::load(
                frag_at<idx_row.value, idx_col.value>(),
                src,
                Int<str_x>{},
                Int<str_y>{},
                idx_row * Int<kFragRows>{},
                idx_col * Int<kFragCols>{});
          });
        });
      }

      template <typename U>
      METAL_FUNC void load(const device U* src, const int ld) thread {
        const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
          const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
            NAXFrag_t::load(
                frag_at<idx_row.value, idx_col.value>(),
                src,
                ld,
                Int<1>{},
                idx_row * Int<kFragRows>{},
                idx_col * Int<kFragCols>{});
          });
        });
      }

      template <typename U>
      METAL_FUNC void store(device U* dst, const int ld) const thread {
        const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
          const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
            NAXFrag_t::store(
                frag_at<idx_row.value, idx_col.value>(),
                dst,
                ld,
                Int<1>{},
                idx_row * Int<kFragRows>{},
                idx_col * Int<kFragCols>{});
          });
        });
      }

      template <typename U>
      METAL_FUNC void
      load_rows(const device U* src, const int ld, const short n_rows) thread {
        const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
          const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
            NAXFrag_t::load_rows(
                frag_at<idx_row.value, idx_col.value>(),
                src,
                ld,
                Int<1>{},
                n_rows,
                idx_row * Int<kFragRows>{},
                idx_col * Int<kFragCols>{});
          });
        });
      }

      template <typename U>
      METAL_FUNC void load_safe(
          const device U* src,
          const int ld,
          const short2 src_tile_dims) thread {
        const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
          const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
            NAXFrag_t::load_safe(
                frag_at<idx_row.value, idx_col.value>(),
                src,
                ld,
                Int<1>{},
                src_tile_dims.y,
                src_tile_dims.x,
                idx_row * Int<kFragRows>{},
                idx_col * Int<kFragCols>{});
          });
        });
      }

      template <typename U>
      METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows)
          const thread {
        const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
          const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
            NAXFrag_t::store_rows(
                frag_at<idx_row.value, idx_col.value>(),
                dst,
                ld,
                Int<1>{},
                n_rows,
                idx_row * Int<kFragRows>{},
                idx_col * Int<kFragCols>{});
          });
        });
      }

      template <typename U>
      METAL_FUNC void store_safe(
          device U* dst,
          const int ld,
          const short2 dst_tile_dims) const thread {
        const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
          const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
            NAXFrag_t::store_safe(
                frag_at<idx_row.value, idx_col.value>(),
                dst,
                ld,
                Int<1>{},
                dst_tile_dims.y,
                dst_tile_dims.x,
                idx_row * Int<kFragRows>{},
                idx_col * Int<kFragCols>{});
          });
        });
      }
    };

    template <
        class CTile,
        class ATile,
        class BTile,
        bool transpose_a,
        bool transpose_b>
    METAL_FUNC void tile_matmad_nax(
        thread CTile& C,
        thread ATile& A,
        metal::bool_constant<transpose_a>,
        thread BTile& B,
        metal::bool_constant<transpose_b>) {
      constexpr short TMa = transpose_a ? ATile::kTileCols : ATile::kTileRows;
      constexpr short TM = CTile::kTileRows;
      static_assert(TMa == TM, "MXU tile matmul: M dimensions do not match");

      constexpr short TNb = transpose_b ? BTile::kTileRows : BTile::kTileCols;
      constexpr short TN = CTile::kTileCols;
      static_assert(TNb == TN, "MXU tile matmul: N dimensions do not match");

      constexpr short TKa = transpose_a ? ATile::kTileRows : ATile::kTileCols;
      constexpr short TK = transpose_b ? BTile::kTileCols : BTile::kTileRows;
      static_assert(TKa == TK, "MXU tile matmul: K dimensions do not match");

      constexpr auto ta = metal::bool_constant<transpose_a>{};
      constexpr auto tb = metal::bool_constant<transpose_b>{};

      if constexpr (TN == 1 && TM % 2 == 0) {
        OMLX_NAX_UNROLL
        for (short mm = 0; mm < TM; mm += 2) {
          OMLX_NAX_UNROLL
          for (short nn = 0; nn < TN; ++nn) {
            OMLX_NAX_UNROLL
            for (short kk = 0; kk < TK; ++kk) {
              CTile::NAXFrag_t::mma(
                  C.frag_at(mm, nn),
                  C.frag_at(mm + 1, nn),
                  A.frag_at(mm, kk, ta),
                  A.frag_at(mm + 1, kk, ta),
                  metal::bool_constant<transpose_a>{},
                  B.frag_at(kk, nn, tb),
                  metal::bool_constant<transpose_b>{});
            }
          }
        }
      } else if constexpr (TN % 2 == 0) {
        OMLX_NAX_UNROLL
        for (short mm = 0; mm < TM; ++mm) {
          OMLX_NAX_UNROLL
          for (short nn = 0; nn < TN; nn += 2) {
            OMLX_NAX_UNROLL
            for (short kk = 0; kk < TK; ++kk) {
              CTile::NAXFrag_t::mma(
                  C.frag_at(mm, nn),
                  C.frag_at(mm, nn + 1),
                  A.frag_at(mm, kk, ta),
                  metal::bool_constant<transpose_a>{},
                  B.frag_at(kk, nn, tb),
                  B.frag_at(kk, nn + 1, tb),
                  metal::bool_constant<transpose_b>{});
            }
          }
        }
      }
    }

    } // namespace omlx_nax

    #endif // OMLX_NAX_TILES
    namespace omlx_nax {

    // Scalar parameters of one attention call (MLX's AttnParams without the
    // strides, which the kernel reads from the inputs' own stride vectors).
    struct AttnParams {
      int B;
      int H;
      int D;
      int qL;
      int kL;
      int gqa_factor;
      float scale;
      int NQ;
      int NK;
      int NQ_aligned;
      int NK_aligned;
      int qL_rem;
      int kL_rem;
      int qL_off;
      // Key blocks [kb_begin, kb_end) of this dispatch (key-range passes).
      int kb_begin;
      int kb_end;
      // Extent of the relative bias band (has_band), in key positions.
      int band;
      int window;
    };

    struct MaxOp {
      template <typename T>
      METAL_FUNC static constexpr T apply(T x, T y) {
        return metal::max(x, y);
      }
    };

    struct SumOp {
      template <typename T>
      METAL_FUNC static constexpr T apply(T x, T y) {
        return x + y;
      }
    };

    struct MulOp {
      template <typename T>
      METAL_FUNC static constexpr T apply(T x, T y) {
        return x * y;
      }
    };

    struct ExpSubOp {
      template <typename T>
      METAL_FUNC static constexpr T apply(T x, T y) {
        return fast::exp2(x - y);
      }
    };

    // MLX's attention_nax with a value head dim BDV <= BD. The plumbing
    // differs: strides come from the inputs, the output is written as
    // [B, qL, H, BDV] rows (MLX's SDPA output layout), function constants are
    // template arguments, and the mask column stride is honoured. Beyond that:
    //
    // * Key-range passes. One dispatch covers key blocks [kb_begin, kb_end).
    //   Unless FIRST, the online-softmax state of every query row (fp32 O
    //   accumulator, running max, running sum) is loaded from Sin; unless LAST it
    //   is stored to Sout instead of normalising and writing the output. Every
    //   row thus runs exactly the operations of one full dispatch, in the same
    //   order (bit-identical output). Within a dispatch the threadgroups in
    //   flight only stream one key slice of their KV head, which stays on chip;
    //   over one long dispatch they drift apart and, once the KV head no longer
    //   fits in the caches, each re-streams it from DRAM.
    // * The query tile stays in registers instead of being re-read from
    //   memory for every key block, and each row max is reduced over all of the
    //   row's score fragments before the cross-lane shuffles (max is exact, so
    //   the grouping does not matter).
    // * WN = 2 splits the head dims over simdgroup pairs, as MLX's
    //   attention_nax_dsplit does for 256-wide heads: each simdgroup of a pair
    //   computes Q @ K.T over half of the query/key dims and P @ V for half of
    //   the value dims, the pair adds its partial scores through threadgroup
    //   memory, and both run the softmax on the full score tile. Half the O
    //   accumulator and query registers per thread; the scores become the sum
    //   of two fp32 partial dot products (a summation-order change).
    template <
        typename T,
        int BQ,
        int BK,
        int BD,
        int BDV,
        int WM,
        int WN,
        bool align_Q,
        bool align_K,
        bool has_mask,
        bool do_causal,
        bool has_sinks,
        bool has_band,
        bool row_sinks,
        bool FIRST,
        bool LAST,
        typename MaskType,
        typename AccumType,
        typename StridePtr,
        typename MaskPtr,
        typename SinkPtr,
        typename StatePtr>
    METAL_FUNC void attention_nax_bdv(
        const device T* Q,
        const device T* K,
        const device T* V,
        device T* O,
        const device AttnParams* params,
        StridePtr q_str,
        StridePtr k_str,
        StridePtr v_str,
        StridePtr m_str,
        MaskPtr mask,
        SinkPtr sinks,
        StatePtr Sin,
        device float* Sout,
        threadgroup AccumType* xchg,
        uint simd_group_id,
        uint simd_lane_id,
        uint3 tid) {
      ulong3 tidl{tid.x, tid.y, tid.z};

      const int64_t Q_strides[3] = {q_str[0], q_str[1], q_str[2]};
      const int64_t K_strides[3] = {k_str[0], k_str[1], k_str[2]};
      const int64_t V_strides[3] = {v_str[0], v_str[1], v_str[2]};
      const int64_t O_strides[3] = {
          int64_t(params->qL) * params->H * BDV, BDV, int64_t(params->H) * BDV};

      Q += tidl.z * Q_strides[0] + // Batch
          tidl.y * Q_strides[1] + // Head
          tidl.x * BQ * Q_strides[2]; // Sequence

      ulong kv_head_idx = int(tid.y) / params->gqa_factor;
      K += tidl.z * K_strides[0] + // Batch
          kv_head_idx * K_strides[1]; // Head

      V += tidl.z * V_strides[0] + // Batch
          kv_head_idx * V_strides[1]; // Head

      O += tidl.z * O_strides[0] + // Batch
          tidl.y * O_strides[1] + // Head
          tidl.x * BQ * O_strides[2]; // Sequence

      if (has_mask) {
        mask += tidl.z * m_str[0] + // Batch
            tidl.y * m_str[1]; // Head
      }

      const metal::uniform<float> scale2 =
          make_uniform(params->scale) * make_uniform(1.44269504089f);

      // Prepare MMA tiles
      constexpr short kU = 16;

      // WM groups of 16 query rows; each group's WN simdgroups split the head
      // dims (WN = 2: MLX's attention_nax_dsplit scheme; see the note above).
      static_assert(BQ == WM * kU, "One 16-row fragment per row group");
      static_assert(WN == 1 || WN == 2, "Head dims split over 1 or 2 simdgroups");

      // Q seq frags per warp
      constexpr int TQ = 1;
      // HeadDim frags of this simdgroup
      constexpr int TD = BD / kU / WN;
      // Value head dim frags of this simdgroup
      constexpr int TDV = BDV / kU / WN;
      // KV seq frags per warp
      constexpr short TK = BK / kU;

      static_assert(TD * kU * WN == BD, "The head dim must split evenly");
      static_assert(TDV % 2 == 0, "P@V accumulates output fragments in pairs");
      using otile_t = NAXTile<AccumType, TQ, TDV>;
      otile_t Otile;

      Otile.clear();

      // Prepare mma tile offsets: rows of this row group, columns of this
      // simdgroup's share of the head dims.
      const short row_group = simd_group_id / WN;
      const short d_part = simd_group_id % WN;
      const short tm = kU * TQ * row_group;
      Q += tm * int(Q_strides[2]) + d_part * (BD / WN);
      K += d_part * (BD / WN);
      V += d_part * (BDV / WN);
      O += d_part * (BDV / WN);

      const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
      const short sm = simd_coord.y;
      const short sn = simd_coord.x;

      // Init row reduction variables
      constexpr short kRowsPT = otile_t::kRowsPerThread;

      metal::vec<AccumType, kRowsPT> max_score;
      metal::vec<AccumType, kRowsPT> sum_score{0};

      // Online-softmax state rows of this simdgroup in the pass buffers:
      // [B, H, NQ * BQ] rows of BDV + 2 floats (O accumulator, max, sum).
      constexpr int kSW = BDV + 2;
      const int64_t srow0 = (int64_t(tid.z) * params->H + tid.y) *
              (int64_t(params->NQ) * BQ) +
          int64_t(tid.x) * BQ + tm;

      if constexpr (FIRST) {
        // Init to -Inf
        OMLX_NAX_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
          max_score[i] = Limits<AccumType>::finite_min;
        }

        if (has_sinks) {
          OMLX_NAX_UNROLL
          for (short i = 0; i < kRowsPT; ++i) {
            int64_t sink_at = tidl.y;
            if (row_sinks) {
              // One sink per query row: [B, qL, H]. Rows past a query tail
              // read the last row's and are never output.
              const int row = min(
                  int(tid.x) * BQ + tm + sm + i * otile_t::kFragRowsJump,
                  params->qL - 1);
              sink_at = (int64_t(tid.z) * params->qL + row) * params->H + tid.y;
            }
            max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[sink_at]);
            sum_score[i] = 1;
          }
        }
      } else {
        // Resume the previous pass (rows past a query tail are never output).
        Otile.load(Sin + srow0 * kSW + d_part * (BDV / WN), kSW);
        OMLX_NAX_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
          const int64_t r = srow0 + sm + i * otile_t::kFragRowsJump;
          max_score[i] = Sin[r * kSW + BDV];
          sum_score[i] = Sin[r * kSW + BDV + 1];
        }
      }

      int kb_lim = params->NK;
      int kb_min_causal = params->NK;

      if (do_causal) {
        int q_max = (tid.x + 1) * BQ + params->qL_off;
        kb_lim = (q_max + BK - 1) / BK;
        kb_lim = min(params->NK, kb_lim);

        int q_min = tid.x * BQ + params->qL_off;
        q_min = max(0, q_min);
        kb_min_causal = (q_min / BK);
      }

      const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
      const bool is_last_q = is_last_bq;

      const short lim_rows_q = params->qL_rem - tm;
      const short lim_rows_k = params->kL_rem;

      // This dispatch's key blocks.
      const int kb_lo = params->kb_begin;
      const int kb_hi = min(kb_lim, params->kb_end);
      const int kb_start = params->window > 0
          ? max(kb_lo, max(0, int(tid.x) * BQ + params->qL_off - params->window + 1) / BK)
          : kb_lo;
      K += int64_t(kb_start) * BK * K_strides[2];
      V += int64_t(kb_start) * BK * V_strides[2];

      // Query fragments, loaded once and kept in registers.
      NAXTile<T, TQ, TD> Qreg;
      if (!align_Q && is_last_q) {
        Qreg.load_rows(Q, int(Q_strides[2]), lim_rows_q);
      } else {
        Qreg.load(Q, int(Q_strides[2]));
      }

      // Loop over KV seq length
      for (int kb = kb_start; kb < kb_hi; kb++) {
        const int is_last_k = (kb == (params->NK_aligned));

        // Do S = Q @ K.T
        using stile_t = NAXTile<AccumType, TQ, TK>;
        stile_t Stile;

        Stile.clear();

        OMLX_NAX_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          OMLX_NAX_UNROLL
          for (short ik = 0; ik < TK; ik += 2) {
            auto qk_step = [&](short id) {
              NAXTile<T, 1, 1> Qtile;
              NAXTile<T, 2, 1> Ktile;

              const int K_load_off = ik * kU * int(K_strides[2]) + id * kU;

              Qtile.frag_at(0, 0) = Qreg.frag_at(iq, id);

              if (!align_K && is_last_k) {
                Ktile.load_rows(
                    K + K_load_off, int(K_strides[2]), lim_rows_k - ik * kU);
              } else {
                Ktile.load(K + K_load_off, int(K_strides[2]));
              }

              stile_t::NAXFrag_t::mma(
                  Stile.frag_at(iq, ik),
                  Stile.frag_at(iq, ik + 1),
                  Qtile.frag_at(0, 0),
                  metal::false_type{},
                  Ktile.frag_at(0, 0),
                  Ktile.frag_at(1, 0),
                  metal::true_type{});
            };
            if constexpr (WN == 2) {
              // The head-dim split kernel keeps static fragment indices.
              OMLX_NAX_UNROLL
              for (short id = 0; id < TD; id++) {
                qk_step(id);
              }
            } else {
    #pragma clang loop unroll_count(4)
              for (short id = 0; id < TD; id++) {
                qk_step(id);
              }
            }

            if constexpr (WN == 2) {
              // Add the peer's partial scores (its half of the head dims).
              constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
              threadgroup AccumType* slot =
                  xchg + (row_group * WN + d_part) * (2 * kEPF * 32);
              const threadgroup AccumType* peer =
                  xchg + (row_group * WN + 1 - d_part) * (2 * kEPF * 32);
              thread auto& s0 = Stile.frag_at(iq, ik);
              thread auto& s1 = Stile.frag_at(iq, ik + 1);
              const short base = short(simd_lane_id) * (2 * kEPF);
              OMLX_NAX_UNROLL
              for (short i = 0; i < kEPF; i++) {
                slot[base + i] = s0[i];
                slot[base + kEPF + i] = s1[i];
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              OMLX_NAX_UNROLL
              for (short i = 0; i < kEPF; i++) {
                s0[i] += peer[base + i];
                s1[i] += peer[base + kEPF + i];
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
          }
        }

        // Scale S
        OMLX_NAX_UNROLL
        for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
          Stile.elems()[ii] *= float(scale2);
        }

        // Mask out length sequence
        if (!align_K && is_last_k) {
          constexpr auto neg_inf = Limits<AccumType>::finite_min;

          OMLX_NAX_UNROLL
          for (short iq = 0; iq < TQ; iq++) {
            OMLX_NAX_UNROLL
            for (short ik = 0; ik < TK; ik++) {
              const short col_pos = ik * kU + sn;

              thread auto& fg = Stile.frag_at(iq, ik);

              OMLX_NAX_UNROLL
              for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
                OMLX_NAX_UNROLL
                for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
                  const auto loc = ii * stile_t::kFragThrCols + jj;
                  fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
                }
              }
            }
          }
        }

        // Mask out if causal
        if (do_causal && kb >= kb_min_causal) {
          constexpr auto neg_inf = Limits<AccumType>::finite_min;

          const int base_row = tid.x * BQ + params->qL_off + tm;
          const int base_col = kb * BK;

          OMLX_NAX_UNROLL
          for (short iq = 0; iq < TQ; iq++) {
            OMLX_NAX_UNROLL
            for (short ik = 0; ik < TK; ik++) {
              thread auto& fg = Stile.frag_at(iq, ik);

              OMLX_NAX_UNROLL
              for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
                OMLX_NAX_UNROLL
                for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
                  const auto r =
                      base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
                  const auto c = base_col + ik * kU + jj + sn;
                  const auto loc = ii * stile_t::kFragThrCols + jj;
                  fg[loc] = (r < c) ? neg_inf : fg[loc];
                }
              }
            }
          }
        }

        // Relative bias band: a query at key position p adds rel[p - c] to its
        // score of key c for 0 <= p - c < band. ``mask`` holds rel as
        // [B, qL, H, band], read in place of a [qL, kL] additive mask. The
        // first left_pad keys of a batch row are padding (one int per row
        // follows the parameter struct).
        if (has_band) {
          constexpr auto neg_inf = Limits<AccumType>::finite_min;

          const int base_row = tid.x * BQ + tm;
          const int base_col = kb * BK;
          const int left_pad =
              reinterpret_cast<const device int*>(params + 1)[tid.z];

          if (params->window > 0 ||
              base_col + BK + params->band > base_row + params->qL_off + 1 ||
              base_col < left_pad) {
            OMLX_NAX_UNROLL
            for (short iq = 0; iq < TQ; iq++) {
              OMLX_NAX_UNROLL
              for (short ik = 0; ik < TK; ik++) {
                thread auto& fg = Stile.frag_at(iq, ik);

                OMLX_NAX_UNROLL
                for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
                  OMLX_NAX_UNROLL
                  for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
                    const int r =
                        base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
                    const int c = base_col + ik * kU + jj + sn;
                    const int d = r + params->qL_off - c;
                    const int64_t at =
                        ((int64_t(tid.z) * params->qL + min(r, params->qL - 1)) *
                                params->H +
                            tid.y) *
                            params->band +
                        clamp(d, 0, params->band - 1);
                    const auto loc = ii * stile_t::kFragThrCols + jj;
                    const AccumType bias = M_LOG2E_F * AccumType(mask[at]);
                    fg[loc] += (d >= 0 && d < params->band) ? bias : AccumType(0);
                    fg[loc] = (c < left_pad || (params->window > 0 && d >= params->window))
                        ? neg_inf : fg[loc];
                  }
                }
              }
            }
          }
        }

        // Other masking as needed
        if (has_mask) {
          constexpr auto neg_inf = Limits<AccumType>::finite_min;

          const int base_row = tid.x * BQ + tm;
          const int base_col = kb * BK;

          constexpr bool is_bool = is_same_v<MaskType, bool>;
          using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
          using mtile_t = NAXTile<melem_t, TQ, TK>;
          using mfrag_t = typename mtile_t::frag_type;

          if (base_row + BQ <= params->qL && base_col + BK <= params->kL) {
            for (short iq = 0; iq < TQ; iq++) {
              OMLX_NAX_UNROLL
              for (short ik = 0; ik < TK; ik++) {
                const int row_pos = base_row + iq * kU;
                const int col_pos = base_col + ik * kU;

                mfrag_t mfrag;
                mtile_t::NAXFrag_t::load(
                    mfrag,
                    mask,
                    int64_t(m_str[2]),
                    int64_t(m_str[3]),
                    row_pos,
                    col_pos);

                thread auto& fg = Stile.frag_at(iq, ik);

                OMLX_NAX_UNROLL
                for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
                  if constexpr (is_bool) {
                    fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
                  } else {
                    fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
                  }
                }
              }
            }
          } else {
            OMLX_NAX_UNROLL
            for (short iq = 0; iq < TQ; iq++) {
              OMLX_NAX_UNROLL
              for (short ik = 0; ik < TK; ik++) {
                const int row_pos = base_row + iq * kU;
                const int col_pos = base_col + ik * kU;

                mfrag_t mfrag;
                mtile_t::NAXFrag_t::load_safe(
                    mfrag,
                    mask,
                    int64_t(m_str[2]),
                    int64_t(m_str[3]),
                    params->qL,
                    params->kL,
                    row_pos,
                    col_pos);

                thread auto& fg = Stile.frag_at(iq, ik);

                OMLX_NAX_UNROLL
                for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
                  if constexpr (is_bool) {
                    fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
                  } else {
                    fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
                  }
                }
              }
            }
          }
        }

        // Do softmax

        // Temp variables
        metal::vec<AccumType, kRowsPT> new_max;
        metal::vec<AccumType, kRowsPT> factor;
        OMLX_NAX_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
          new_max[i] = max_score[i];
        }

        // Row max: all of a row's fragments first, then across lanes (the
        // grouping is free, max is exact).
        OMLX_NAX_UNROLL
        for (short iq = 0; iq < TQ; ++iq) {
          OMLX_NAX_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ++ii) {
            const short loc0 = ii * stile_t::kFragThrCols;
            AccumType m = Stile.frag_at(iq, 0)[loc0];
            OMLX_NAX_UNROLL
            for (short ik = 0; ik < TK; ++ik) {
              OMLX_NAX_UNROLL
              for (short jj = 0; jj < stile_t::kFragThrCols; ++jj) {
                m = metal::max(m, Stile.frag_at(iq, ik)[loc0 + jj]);
              }
            }
            m = metal::max(m, simd_shuffle_xor(m, ushort(1)));
            m = metal::max(m, simd_shuffle_xor(m, ushort(8)));
            const short r = iq * stile_t::kFragThrRows + ii;
            new_max[r] = metal::max(new_max[r], m);
          }
        }

        // exp(Si - rowmax(Si))
        Stile.template row_bin_op<ExpSubOp>(new_max);

        // Factor exp(rowmax(Si) - rowmax(Si-1))
        OMLX_NAX_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
          factor[i] = fast::exp2(max_score[i] - new_max[i]);
          max_score[i] = new_max[i];
        }

        // Row Sum
        OMLX_NAX_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
          sum_score[i] = sum_score[i] * factor[i];
        }

        Stile.template row_reduce<SumOp>(sum_score);

        // Update O
        Otile.template row_bin_op<MulOp>(factor);

        simdgroup_barrier(mem_flags::mem_none);

        // Do O = P @ V
        OMLX_NAX_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          OMLX_NAX_UNROLL
          for (short id = 0; id < TDV; id += 2) {
            if constexpr (BDV == 128 && WN == 1) {
              if (id == 4) {
                threadgroup_barrier(mem_flags::mem_none);
              }
            }

            OMLX_NAX_UNROLL
            for (short ik = 0; ik < TK; ik++) {
              NAXTile<T, 1, 2> Vtile;

              const int V_load_off = ik * kU * int(V_strides[2]) + id * kU;

              if (!align_K && is_last_k) {
                Vtile.load_rows(
                    V + V_load_off, int(V_strides[2]), lim_rows_k - ik * kU);
              } else {
                Vtile.load(V + V_load_off, int(V_strides[2]));
              }

              otile_t::NAXFrag_t::mma(
                  Otile.frag_at(iq, id),
                  Otile.frag_at(iq, id + 1),
                  Stile.frag_at(iq, ik),
                  metal::false_type{},
                  Vtile.frag_at(0, 0),
                  Vtile.frag_at(0, 1),
                  metal::false_type{});
            }
          }
        }

        // Prepare for next iteration
        K += BK * int(K_strides[2]);
        V += BK * int(V_strides[2]);
      }

      threadgroup_barrier(mem_flags::mem_none);

      if constexpr (!LAST) {
        // Hand the row state to the next pass.
        Otile.store(Sout + srow0 * kSW + d_part * (BDV / WN), kSW);
        if (sn == 0 && d_part == 0) {
          OMLX_NAX_UNROLL
          for (short i = 0; i < kRowsPT; ++i) {
            const int64_t r = srow0 + sm + i * otile_t::kFragRowsJump;
            Sout[r * kSW + BDV] = max_score[i];
            Sout[r * kSW + BDV + 1] = sum_score[i];
          }
        }
        return;
      }

      // Normalize output

      metal::vec<AccumType, kRowsPT> rcp;
      OMLX_NAX_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        rcp[i] = 1.f / sum_score[i];
      }

      Otile.template row_bin_op<MulOp>(rcp);

      // Store results
      O += tm * int(O_strides[2]);

      if (!align_Q && is_last_q) {
        if (lim_rows_q <= 0)
          return;

        Otile.store_rows(O, int(O_strides[2]), lim_rows_q);
      } else {
        Otile.store(O, int(O_strides[2]));
      }
    }

    } // namespace omlx_nax
    """#
}
#endif
