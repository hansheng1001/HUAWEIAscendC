#include "kernel_operator.h"

#if 0
extern "C" __global__ __aicore__ void cross(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
#endif

using namespace AscendC;
// constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BUFFER_NUM = 1;

class CrossNorm {
public:
  __aicore__ inline CrossNorm() {}
  __aicore__ inline void
  Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t dataType, uint32_t xDataNums,
       uint32_t tensorNums, uint32_t tensorLen, uint32_t tensorElementStride,
       uint32_t tensorContinueElemnetNums, uint32_t updateTimesPerTensor,
       uint32_t tensorNumsPerTile, uint32_t tensorNumsLastTile,
       uint32_t dataLenPerTile, uint32_t dataLenLastTile,
       uint32_t tensorLastUpDataLen, uint32_t tileNum, uint32_t avaiUBDataLen) {

    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

    this->dataType = dataType;

    this->xDataNums = xDataNums;
    this->tensorNums = tensorNums;
    this->tensorLen = tensorLen;

    this->tensorElementStride = tensorElementStride;
    this->tensorContinueElemnetNums = tensorContinueElemnetNums;

    this->updateTimesPerTensor = updateTimesPerTensor;

    this->tensorNumsPerTile = tensorNumsPerTile;
    this->tensorNumsLastTile = tensorNumsLastTile;

    this->dataLenPerTile = dataLenPerTile;
    this->dataLenLastTile = dataLenLastTile;

    this->tensorLastUpDataLen = tensorLastUpDataLen;

    this->tileNum = tileNum;
    this->avaiUBDataLen = avaiUBDataLen;

    x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1, this->xDataNums);
    x2Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x2, this->xDataNums);

    yGm.SetGlobalBuffer((__gm__ DTYPE_X1 *)y, this->xDataNums);

    // this->updateTensorIndex = 0;

    pipe.InitBuffer(inQueueIN, BUFFER_NUM,
                    this->avaiUBDataLen * 4 * sizeof(DTYPE_X1));
    pipe.InitBuffer(outQueueOUT, BUFFER_NUM,
                    this->avaiUBDataLen * sizeof(DTYPE_X1));

    if (2 == this->dataType) {

      pipe.InitBuffer(comHalfX, 4 * this->avaiUBDataLen * sizeof(half));

      pipe.InitBuffer(comHalfY, this->avaiUBDataLen * sizeof(half));
    }

    if (1 == this->dataType) {

      pipe.InitBuffer(comFloatX, 4 * this->avaiUBDataLen * sizeof(float));

      pipe.InitBuffer(comFloatY, this->avaiUBDataLen * sizeof(float));
    }
  }
  __aicore__ inline void Process() {
    int32_t loopCount = this->tileNum * BUFFER_NUM;

    // 都以C的索引为基础进行处理
    // i是C的indexNums
    // j是C的updateIndex
    for (int32_t i = 0; i < loopCount; i++) {
      for (int32_t j = 0; j < this->updateTimesPerTensor; j++) {
        CopyInStr1(i, j);
        ComputeStr1(i, j);
        CopyOut(i, j);
      }
    }

    // for (int32_t i = 0; i < loopCount; i++) {
    //   for (int32_t j = 0; j < this->updateTimesPerTensor; j++) {
    //     CopyInStr2(i, j);
    //     ComputeStr2(i, j);
    //     CopyOut(i, j);
    //   }
    // }
  }

private:
  __aicore__ inline void CopyLTensorIn(int32_t tensorIndex,
                                       int32_t perTensorUpdateIndex,
                                       LocalTensor<DTYPE_X1> &lTensor,
                                       GlobalTensor<DTYPE_X1> &xGm) {

    // 分为1个tensor是否可以全部UB
    // 和tensor内部元素是否连续进行讨论
    if (BUFFER_NUM == 1) {
      // 1.先按照1个Tensor是否能够全部放入到UB中进行划分--能够全部放入
      if (1 == this->updateTimesPerTensor) {
        // 1.1 tensor内部连续的情况,可以使用DataCopy进行数据搬运
        if (1 == this->tensorElementStride) {

          uint32_t baseOffset = tensorIndex * this->tensorLen;

          // 由于每次只处理1个Tensor,所以每次只搬运1个tensorLength的长度的数据
          if (0 == (this->tensorLen * sizeof(DTYPE_X1) % 32)) {
            DataCopy(lTensor[0], xGm[baseOffset], this->tensorLen);

          } else {
            uint32_t dLen = this->tensorLen / (32 / sizeof(DTYPE_X1)) *
                            (32 / sizeof(DTYPE_X1));
            uint32_t restLen = this->tensorLen % (32 / sizeof(DTYPE_X1));

            if (0 != dLen) {
              DataCopy(lTensor[0], xGm[baseOffset], dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              lTensor.SetValue(dLen + i, xGm.GetValue(baseOffset + dLen + i));
            }
          }

        } else {

          uint32_t tensorBaseAddr =
              tensorIndex * this->tensorContinueElemnetNums;

          for (uint32_t i = 0, offset = 0;
               i < (this->tensorLen / this->tensorContinueElemnetNums);
               i++, offset += this->tensorElementStride) {

            uint32_t base = this->tensorContinueElemnetNums * i;

            if (0 ==
                (this->tensorContinueElemnetNums * sizeof(DTYPE_X1) % 32)) {
              DataCopy(lTensor[base], xGm[tensorBaseAddr + offset],
                       this->tensorContinueElemnetNums);

            } else {
              for (uint32_t i = 0; i < this->tensorContinueElemnetNums; i++) {
                lTensor.SetValue(base + i,
                                 xGm.GetValue(tensorBaseAddr + offset + i));
              }
            }
          }
        }

      } else { // 2.不能够全部放入
        // 2.1 tensor内部数据连续
        if (1 == this->tensorElementStride) {
          if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {

            uint32_t baseOffset = tensorIndex * this->tensorLen +
                                  perTensorUpdateIndex * this->dataLenPerTile;

            DataCopy(lTensor[0], xGm[baseOffset], this->dataLenPerTile);

          } else {

            uint32_t baseOffset = tensorIndex * this->tensorLen +
                                  perTensorUpdateIndex * this->dataLenPerTile;

            if (0 == (this->tensorLastUpDataLen * sizeof(DTYPE_X1) % 32)) {
              DataCopy(lTensor[0], xGm[baseOffset], this->tensorLastUpDataLen);
            } else {
              uint32_t dLen = this->tensorLastUpDataLen /
                              (32 / sizeof(DTYPE_X1)) * (32 / sizeof(DTYPE_X1));
              uint32_t restLen =
                  this->tensorLastUpDataLen % (32 / sizeof(DTYPE_X1));

              if (0 != dLen) {
                DataCopy(lTensor[0], xGm[baseOffset], dLen);
              }

              for (uint32_t i = 0; i < restLen; i++) {
                lTensor.SetValue(dLen + i, xGm.GetValue(baseOffset + dLen + i));
              }
            }
          }

        } else {
          if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {

            uint32_t tensorBaseAddr =
                tensorIndex * this->tensorContinueElemnetNums +
                perTensorUpdateIndex * this->dataLenPerTile /
                    this->tensorContinueElemnetNums * this->tensorElementStride;

            for (uint32_t i = 0;
                 i < this->dataLenPerTile / this->tensorContinueElemnetNums;
                 i++, tensorBaseAddr += this->tensorElementStride) {
              for (uint32_t j = 0; j < this->tensorContinueElemnetNums; j++) {
                lTensor.SetValue(i * this->tensorContinueElemnetNums + j,
                                 xGm.GetValue(tensorBaseAddr + j));
              }
            }

          } else {

            uint32_t tensorBaseAddr =
                tensorIndex * this->tensorContinueElemnetNums +
                perTensorUpdateIndex * this->dataLenPerTile /
                    this->tensorContinueElemnetNums * this->tensorElementStride;

            for (uint32_t i = 0; i < this->tensorLastUpDataLen /
                                         this->tensorContinueElemnetNums;
                 i++, tensorBaseAddr += this->tensorElementStride) {
              for (uint32_t j = 0; j < this->tensorContinueElemnetNums; j++) {
                lTensor.SetValue(i * this->tensorContinueElemnetNums + j,
                                 xGm.GetValue(tensorBaseAddr + j));
              }
            }
          }
        }
      }
    }
  }

  __aicore__ inline void CopyInStr1(int32_t tensorIndex,
                                    int32_t perTensorUpdateIndex) {
    LocalTensor<DTYPE_X1> inLocal = inQueueIN.AllocTensor<DTYPE_X1>();

    LocalTensor<DTYPE_X1> x1Local = inLocal;
    LocalTensor<DTYPE_X1> x2Local = inLocal[this->avaiUBDataLen];
    // 以C为基础,把x1拷贝进来
    uint32_t tensorIndexX1 = (tensorIndex + 1) % 3;
    CopyLTensorIn(tensorIndexX1, perTensorUpdateIndex, x1Local, x1Gm);

    // 把x2拷贝进来
    uint32_t tensorIndexX2 = (tensorIndex + 2) % 3;
    CopyLTensorIn(tensorIndexX2, perTensorUpdateIndex, x2Local, x2Gm);

    //拷贝后面的部分过来
    x1Local = inLocal[2 * this->avaiUBDataLen];
    x2Local = inLocal[3 * this->avaiUBDataLen];

    // 以C为基础,把x1拷贝进来
    tensorIndexX1 = (tensorIndex + 2) % 3;
    CopyLTensorIn(tensorIndexX1, perTensorUpdateIndex, x1Local, x1Gm);

    // 把x2拷贝进来
    tensorIndexX2 = (tensorIndex + 1) % 3;
    CopyLTensorIn(tensorIndexX2, perTensorUpdateIndex, x2Local, x2Gm);

    inQueueIN.EnQue(inLocal);
  }

  __aicore__ inline void ComputeStr1(int32_t tensorIndex,
                                     int32_t perTensorUpdateIndex) {
    LocalTensor<DTYPE_X1> inLocal = inQueueIN.DeQue<DTYPE_X1>();

    LocalTensor<DTYPE_X1> x1Local = inLocal;
    LocalTensor<DTYPE_X1> x2Local = inLocal[this->avaiUBDataLen];

    LocalTensor<DTYPE_X1> x3Local = inLocal[2 * this->avaiUBDataLen];
    LocalTensor<DTYPE_X1> x4Local = inLocal[3 * this->avaiUBDataLen];

    LocalTensor<DTYPE_X1> yLocal = outQueueOUT.AllocTensor<DTYPE_X1>();

    if (2 == this->dataType) {

      // TBuf<QuePosition::VECCALC> comHalfInput, comHalfX1, comHalfX2,
      // comHalfY;

      LocalTensor<half> xHalf = comHalfX.Get<half>();

      LocalTensor<half> x1Half = xHalf;
      LocalTensor<half> x2Half = xHalf[this->avaiUBDataLen];

      LocalTensor<half> x3Half = xHalf[2 * this->avaiUBDataLen];
      LocalTensor<half> x4Half = xHalf[3 * this->avaiUBDataLen];

      LocalTensor<half> yHalf = comHalfY.Get<half>();

      LocalTensor<int8_t> x1_tensor1 = x1Local.ReinterpretCast<int8_t>();
      LocalTensor<int8_t> x2_tensor1 = x2Local.ReinterpretCast<int8_t>();

      LocalTensor<int8_t> x3_tensor1 = x3Local.ReinterpretCast<int8_t>();
      LocalTensor<int8_t> x4_tensor1 = x4Local.ReinterpretCast<int8_t>();

      LocalTensor<int8_t> y_tensor1 = yLocal.ReinterpretCast<int8_t>();

      if (1 == this->updateTimesPerTensor) {

        Cast(x1Half, x1_tensor1, RoundMode::CAST_NONE, this->tensorLen);
        Cast(x2Half, x2_tensor1, RoundMode::CAST_NONE, this->tensorLen);

        Cast(x3Half, x3_tensor1, RoundMode::CAST_NONE, this->tensorLen);
        Cast(x4Half, x3_tensor1, RoundMode::CAST_NONE, this->tensorLen);

        Cast(yHalf, y_tensor1, RoundMode::CAST_NONE, this->tensorLen);

        Mul(yHalf, x1Half, x2Half, this->tensorLen);

        Mul(x1Half, x3Half, x4Half, this->tensorLen);
        Sub(yHalf, yHalf, x1Half, this->tensorLen);

        // LocalTensor<int8_t> yLocal = yLocal.ReinterpretCast<int8_t>();
        Cast(y_tensor1, yHalf, RoundMode::CAST_NONE, this->tensorLen);
      } else {
        if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {
          // Mul(yLocal, x1Local, x2Local, this->dataLenPerTile);

          Cast(x1Half, x1_tensor1, RoundMode::CAST_NONE, this->dataLenPerTile);
          Cast(x2Half, x2_tensor1, RoundMode::CAST_NONE, this->dataLenPerTile);

          Cast(x3Half, x3_tensor1, RoundMode::CAST_NONE, this->dataLenPerTile);
          Cast(x4Half, x3_tensor1, RoundMode::CAST_NONE, this->dataLenPerTile);

          Cast(yHalf, y_tensor1, RoundMode::CAST_NONE, this->dataLenPerTile);

          Mul(yHalf, x1Half, x2Half, this->dataLenPerTile);

          Mul(x1Half, x3Half, x4Half, this->dataLenPerTile);
          Sub(yHalf, yHalf, x1Half, this->dataLenPerTile);

          // LocalTensor<int8_t> yLocal = yLocal.ReinterpretCast<int8_t>();
          Cast(y_tensor1, yHalf, RoundMode::CAST_NONE, this->dataLenPerTile);
        } else {

          Cast(x1Half, x1_tensor1, RoundMode::CAST_NONE,
               this->tensorLastUpDataLen);
          Cast(x2Half, x2_tensor1, RoundMode::CAST_NONE,
               this->tensorLastUpDataLen);

          Cast(x3Half, x3_tensor1, RoundMode::CAST_NONE,
               this->tensorLastUpDataLen);
          Cast(x4Half, x4_tensor1, RoundMode::CAST_NONE,
               this->tensorLastUpDataLen);

          Cast(yHalf, y_tensor1, RoundMode::CAST_NONE,
               this->tensorLastUpDataLen);

          Mul(yHalf, x1Half, x2Half, this->tensorLastUpDataLen);

          Mul(x1Half, x3Half, x4Half, this->tensorLastUpDataLen);
          Sub(yHalf, yHalf, x1Half, this->tensorLastUpDataLen);

          // LocalTensor<int8_t> yLocal = yLocal.ReinterpretCast<int8_t>();
          Cast(y_tensor1, yHalf, RoundMode::CAST_NONE,
               this->tensorLastUpDataLen);
        }
      }
    } else if (1 == this->dataType) {

      // TBuf<QuePosition::VECCALC> comHalfInput, comHalfX1, comHalfX2,
      // comHalfY;

      LocalTensor<float> xHalf = comFloatX.Get<float>();

      LocalTensor<float> x1Half = xHalf;
      LocalTensor<float> x2Half = xHalf[this->avaiUBDataLen];

      LocalTensor<float> x3Half = xHalf[2 * this->avaiUBDataLen];
      LocalTensor<float> x4Half = xHalf[3 * this->avaiUBDataLen];

      LocalTensor<float> yHalf = comFloatY.Get<float>();

      LocalTensor<half> x1_tensor1 = x1Local.ReinterpretCast<half>();
      LocalTensor<half> x2_tensor1 = x2Local.ReinterpretCast<half>();

      LocalTensor<half> x3_tensor1 = x3Local.ReinterpretCast<half>();
      LocalTensor<half> x4_tensor1 = x4Local.ReinterpretCast<half>();

      LocalTensor<half> y_tensor1 = yLocal.ReinterpretCast<half>();

      if (1 == this->updateTimesPerTensor) {

        Cast(x1Half, x1_tensor1, RoundMode::CAST_NONE, this->tensorLen);
        Cast(x2Half, x2_tensor1, RoundMode::CAST_NONE, this->tensorLen);

        Cast(x3Half, x3_tensor1, RoundMode::CAST_NONE, this->tensorLen);
        Cast(x4Half, x4_tensor1, RoundMode::CAST_NONE, this->tensorLen);

        Cast(yHalf, y_tensor1, RoundMode::CAST_NONE, this->tensorLen);

        Mul(yHalf, x1Half, x2Half, this->tensorLen);

        Mul(x1Half, x3Half, x4Half, this->tensorLen);
        Sub(yHalf, yHalf, x1Half, this->tensorLen);

        // LocalTensor<int8_t> yLocal = yLocal.ReinterpretCast<int8_t>();
        Cast(y_tensor1, yHalf, RoundMode::CAST_NONE, this->tensorLen);
      } else {
        if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {
          // Mul(yLocal, x1Local, x2Local, this->dataLenPerTile);

          Cast(x1Half, x1_tensor1, RoundMode::CAST_NONE, this->dataLenPerTile);
          Cast(x2Half, x2_tensor1, RoundMode::CAST_NONE, this->dataLenPerTile);

          Cast(x3Half, x3_tensor1, RoundMode::CAST_NONE, this->dataLenPerTile);
          Cast(x4Half, x4_tensor1, RoundMode::CAST_NONE, this->dataLenPerTile);

          Cast(yHalf, y_tensor1, RoundMode::CAST_NONE, this->dataLenPerTile);

          Mul(yHalf, x1Half, x2Half, this->dataLenPerTile);

          Mul(x1Half, x3Half, x4Half, this->dataLenPerTile);
          Sub(yHalf, yHalf, x1Half, this->dataLenPerTile);

          // LocalTensor<int8_t> yLocal = yLocal.ReinterpretCast<int8_t>();
          Cast(y_tensor1, yHalf, RoundMode::CAST_NONE, this->dataLenPerTile);
        } else {

          Cast(x1Half, x1_tensor1, RoundMode::CAST_NONE,
               this->tensorLastUpDataLen);
          Cast(x2Half, x2_tensor1, RoundMode::CAST_NONE,
               this->tensorLastUpDataLen);

          Cast(x3Half, x3_tensor1, RoundMode::CAST_NONE,
               this->tensorLastUpDataLen);
          Cast(x4Half, x4_tensor1, RoundMode::CAST_NONE,
               this->tensorLastUpDataLen);

          Cast(yHalf, y_tensor1, RoundMode::CAST_NONE,
               this->tensorLastUpDataLen);

          Mul(yHalf, x1Half, x2Half, this->tensorLastUpDataLen);

          Mul(x1Half, x3Half, x4Half, this->tensorLastUpDataLen);
          Sub(yHalf, yHalf, x1Half, this->tensorLastUpDataLen);

          // LocalTensor<int8_t> yLocal = yLocal.ReinterpretCast<int8_t>();
          Cast(y_tensor1, yHalf, RoundMode::CAST_NONE,
               this->tensorLastUpDataLen);
        }
      }
    } else {

      if (1 == this->updateTimesPerTensor) {
        Mul(yLocal, x1Local, x2Local, this->tensorLen);

        Mul(x1Local, x3Local, x4Local, this->tensorLen);
        Sub(yLocal, yLocal, x1Local, this->tensorLen);
      } else {
        if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {
          Mul(yLocal, x1Local, x2Local, this->dataLenPerTile);

          Mul(x1Local, x3Local, x4Local, this->dataLenPerTile);
          Sub(yLocal, yLocal, x1Local, this->dataLenPerTile);
        } else {
          Mul(yLocal, x1Local, x2Local, this->tensorLastUpDataLen);

          Mul(x1Local, x3Local, x4Local, this->tensorLastUpDataLen);
          Sub(yLocal, yLocal, x1Local, this->tensorLastUpDataLen);
        }
      }
    }

    outQueueOUT.EnQue<DTYPE_X1>(yLocal);

    inQueueIN.FreeTensor(inLocal);
  }

  __aicore__ inline void CopyOut(int32_t tensorIndex,
                                 int32_t perTensorUpdateIndex) {
    LocalTensor<DTYPE_X1> yLocal = outQueueOUT.DeQue<DTYPE_X1>();

    // 分为1个tensor是否可以全部UB
    // 和tensor内部元素是否连续进行讨论
    if (BUFFER_NUM == 1) {
      // 1.先按照1个Tensor是否能够全部放入到UB中进行划分--能够全部放入
      if (1 == this->updateTimesPerTensor) {
        // 1.1 tensor内部连续的情况,可以使用DataCopy进行数据搬运
        if (1 == this->tensorElementStride) {

          uint32_t baseOffset = tensorIndex * this->tensorLen;

          // 由于每次只处理1个Tensor,所以每次只搬运1个tensorLength的长度的数据
          if (0 == (this->tensorLen * sizeof(DTYPE_X1) % 32)) {
            DataCopy(yGm[baseOffset], yLocal[0], this->tensorLen);

          } else {
            uint32_t dLen = this->tensorLen / (32 / sizeof(DTYPE_X1)) *
                            (32 / sizeof(DTYPE_X1));
            uint32_t restLen = this->tensorLen % (32 / sizeof(DTYPE_X1));

            if (0 != dLen) {
              DataCopy(yGm[baseOffset], yLocal[0], dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              yGm.SetValue(baseOffset + dLen + i, yLocal.GetValue(dLen + i));
            }
          }

        } else {

          uint32_t tensorBaseAddr =
              tensorIndex * this->tensorContinueElemnetNums;

          for (uint32_t i = 0, offset = 0;
               i < (this->tensorLen / this->tensorContinueElemnetNums);
               i++, offset += this->tensorElementStride) {

            uint32_t base = this->tensorContinueElemnetNums * i;

            if (0 ==
                (this->tensorContinueElemnetNums * sizeof(DTYPE_X1) % 32)) {
              DataCopy(yGm[tensorBaseAddr + offset], yLocal[base],
                       this->tensorContinueElemnetNums);

            } else {
              for (uint32_t i = 0; i < this->tensorContinueElemnetNums; i++) {
                yGm.SetValue(tensorBaseAddr + offset + i,
                             yLocal.GetValue(base + i));
              }
            }
          }
        }

      } else { // 2.不能够全部放入
        // 2.1 tensor内部数据连续
        if (1 == this->tensorElementStride) {
          if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {

            uint32_t baseOffset = tensorIndex * this->tensorLen +
                                  perTensorUpdateIndex * this->dataLenPerTile;

            DataCopy(yGm[baseOffset], yLocal[0], this->dataLenPerTile);

          } else {

            uint32_t baseOffset = tensorIndex * this->tensorLen +
                                  perTensorUpdateIndex * this->dataLenPerTile;

            if (0 == (this->tensorLastUpDataLen * sizeof(DTYPE_X1) % 32)) {
              DataCopy(yGm[baseOffset], yLocal[0], this->tensorLastUpDataLen);
            } else {
              uint32_t dLen = this->tensorLastUpDataLen /
                              (32 / sizeof(DTYPE_X1)) * (32 / sizeof(DTYPE_X1));
              uint32_t restLen =
                  this->tensorLastUpDataLen % (32 / sizeof(DTYPE_X1));

              if (0 != dLen) {
                DataCopy(yGm[baseOffset], yLocal[0], dLen);
              }

              for (uint32_t i = 0; i < restLen; i++) {
                yGm.SetValue(baseOffset + dLen + i, yLocal.GetValue(dLen + i));
              }
            }
          }

        } else {
          if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {

            uint32_t tensorBaseAddr =
                tensorIndex * this->tensorContinueElemnetNums +
                perTensorUpdateIndex * this->dataLenPerTile /
                    this->tensorContinueElemnetNums * this->tensorElementStride;

            for (uint32_t i = 0;
                 i < this->dataLenPerTile / this->tensorContinueElemnetNums;
                 i++, tensorBaseAddr += this->tensorElementStride) {
              for (uint32_t j = 0; j < this->tensorContinueElemnetNums; j++) {
                yGm.SetValue(
                    tensorBaseAddr + j,
                    yLocal.GetValue(i * this->tensorContinueElemnetNums + j));
              }
            }

          } else {

            uint32_t tensorBaseAddr =
                tensorIndex * this->tensorContinueElemnetNums +
                perTensorUpdateIndex * this->dataLenPerTile /
                    this->tensorContinueElemnetNums * this->tensorElementStride;

            for (uint32_t i = 0; i < this->tensorLastUpDataLen /
                                         this->tensorContinueElemnetNums;
                 i++, tensorBaseAddr += this->tensorElementStride) {
              for (uint32_t j = 0; j < this->tensorContinueElemnetNums; j++) {
                yGm.SetValue(
                    tensorBaseAddr + j,
                    yLocal.GetValue(i * this->tensorContinueElemnetNums + j));
              }
            }
          }
        }
      }
    }

    outQueueOUT.FreeTensor(yLocal);
  }

private:
  TPipe pipe;
  // TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY, inQueueZ;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIN;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;

  TBuf<QuePosition::VECCALC> comHalfX, comHalfY;
  TBuf<QuePosition::VECCALC> comFloatX, comFloatY;

  // GlobalTensor<float> xGm;
  // GlobalTensor<float> yGm;
  // GlobalTensor<float> outGm;

  //   GlobalTensor<DTYPE_VALUE> valueGm;
  //   GlobalTensor<DTYPE_INPUT_DATA> input_dataGm;
  //   GlobalTensor<DTYPE_X2> x2Gm;

  GlobalTensor<DTYPE_X1> x1Gm;
  GlobalTensor<DTYPE_X1> x2Gm;

  GlobalTensor<DTYPE_X1> yGm;

  uint32_t dataType;

  uint32_t xDataNums;
  uint32_t tensorNums;
  uint32_t tensorLen;

  uint32_t tensorElementStride;
  uint32_t tensorContinueElemnetNums;

  uint32_t updateTimesPerTensor;

  uint32_t tensorNumsPerTile;
  uint32_t tensorNumsLastTile;

  uint32_t dataLenPerTile;
  uint32_t dataLenLastTile;

  uint32_t tensorLastUpDataLen;

  uint32_t tileNum;
  uint32_t avaiUBDataLen;

  //   float sqtVar;
};

extern "C" __global__ __aicore__ void cross(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                            GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  CrossNorm op;

  op.Init(x1, x2, y, tiling_data.dataType, tiling_data.xDataNums,
          tiling_data.tensorNums, tiling_data.tensorLen,
          tiling_data.tensorElementStride,
          tiling_data.tensorContinueElemnetNums,
          tiling_data.updateTimesPerTensor, tiling_data.tensorNumsPerTile,
          tiling_data.tensorNumsLastTile, tiling_data.dataLenPerTile,
          tiling_data.dataLenLastTile, tiling_data.tensorLastUpDataLen,
          tiling_data.tileNum, tiling_data.avaiUBDataLen);
  op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void cross_do(uint32_t blockDim, void *l2ctrl, void *stream, GM_ADDR x1,
              GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  cross<<<blockDim, l2ctrl, stream>>>(x1, x2, y, workspace, tiling);

  aclrtSynchronizeStream(stream);

  // 这个问题需要解决呢？
  // GET_TILING_DATA(tiling_data, tiling);

  //   std::cout << "reduction" << tiling_data.reduction
  //             << ", BlockNum=" << GetBlockNum() << "TILING_KEY_IS(1)"
  //             << TILING_KEY_IS(1) << std::endl;
}
#endif