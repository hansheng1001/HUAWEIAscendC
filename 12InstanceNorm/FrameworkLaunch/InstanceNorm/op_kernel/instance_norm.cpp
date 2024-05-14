#include "kernel_operator.h"

// #include <cmath>

#if 0
extern "C" __global__ __aicore__ void instance_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
#endif

using namespace AscendC;
// constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BUFFER_NUM = 1;

class KernelInstanceNorm {
public:
  __aicore__ inline KernelInstanceNorm() {}
  __aicore__ inline void
  Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
       GM_ADDR variance, float epsilon, uint32_t dataType, uint32_t xDataNums,
       uint32_t tensorNums, uint32_t tensorLen, uint32_t tensorElementStride,
       uint32_t updateTimesPerTensor, uint32_t tensorNumsPerTile,
       uint32_t tensorNumsLastTile, uint32_t dataLenPerTile,
       uint32_t dataLenLastTile, uint32_t tensorLastUpDataLen, uint32_t tileNum,
       uint32_t avaiUBDataLen) {

    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

    // if (epsilon)
    this->epsilon = epsilon;

    this->dataType = dataType;

    this->xDataNums = xDataNums;
    this->tensorNums = tensorNums;
    this->tensorLen = tensorLen;

    this->tensorElementStride = tensorElementStride;
    this->updateTimesPerTensor = updateTimesPerTensor;

    this->tensorNumsPerTile = tensorNumsPerTile;
    this->tensorNumsLastTile = tensorNumsLastTile;

    this->dataLenPerTile = dataLenPerTile;
    this->dataLenLastTile = dataLenLastTile;

    this->tensorLastUpDataLen = tensorLastUpDataLen;

    this->tileNum = tileNum;
    this->avaiUBDataLen = avaiUBDataLen;

    xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, this->xDataNums);
    gGm.SetGlobalBuffer((__gm__ DTYPE_X *)gamma, this->xDataNums);
    bGm.SetGlobalBuffer((__gm__ DTYPE_X *)beta, this->xDataNums);

    yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y, this->xDataNums);

    mGm.SetGlobalBuffer((__gm__ DTYPE_X *)mean, this->tensorNums);
    vGm.SetGlobalBuffer((__gm__ DTYPE_X *)variance, this->tensorNums);

    // this->updateTensorIndex = 0;

    this->tensorMeanSum = 0.0f;
    this->tensorMean = 0.0f;

    this->tensorVarianceSum = 0.0f;
    this->tensorVariance = 0.0f;

    pipe.InitBuffer(inQueueIN, BUFFER_NUM,
                    this->avaiUBDataLen * 3 * sizeof(DTYPE_X));
    pipe.InitBuffer(outQueueOUT, BUFFER_NUM,
                    this->avaiUBDataLen * sizeof(DTYPE_X));

    pipe.InitBuffer(w1Buf, BUFFER_NUM, this->avaiUBDataLen * sizeof(DTYPE_X));
    pipe.InitBuffer(w2Buf, BUFFER_NUM, this->avaiUBDataLen * sizeof(DTYPE_X));
  }
  __aicore__ inline void Process() {
    int32_t loopCount = this->tileNum * BUFFER_NUM;

    // 1.每个Tensor可以放入到UB中,可以1个流水处理完mean/variance/instancnorm
    if (1 == this->updateTimesPerTensor) {
      // 这是i其实就是正在处理的Tensor的索引,因为每个tile只处于1个Tensor
      for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i, 0);
        Compute(i, 0);
        CopyOut(i, 0);
      }
    } else { // 2.每个Tensor不能够全部放入到UB中, 需要多次更新,
             // 分为3个流水,分别处理3个过程
      for (int32_t i = 0; i < loopCount; i++) {
        // 2.1 处理mean
        for (int32_t j = 0; j < this->updateTimesPerTensor; j++) {
          CopyXIn(i, j);
          ComputeMean(i, j);
        }

        // 2.2 处理variance
        for (int32_t j = 0; j < this->updateTimesPerTensor; j++) {
          CopyXIn(i, j);
          ComputeVariance(i, j);
        }

        // 2.2 处理instance
        for (int32_t j = 0; j < this->updateTimesPerTensor; j++) {
          CopyIn(i, j);
          ComputeInstance(i, j);
          CopyOut(i, j);
        }
      }
    }
  }

private:
  __aicore__ inline void CopyIn(int32_t tensorIndex,
                                int32_t perTensorUpdateIndex) {
    LocalTensor<DTYPE_X> inLocal = inQueueIN.AllocTensor<DTYPE_X>();

    // 分为1个tensor是否可以全部UB
    // 和tensor内部元素是否连续进行讨论
    if (BUFFER_NUM == 1) {
      // 1.先按照1个Tensor是否能够全部放入到UB中进行划分--能够全部放入
      if (1 == this->updateTimesPerTensor) {
        // 1.1 tensor内部连续的情况,可以使用DataCopy进行数据搬运
        if (1 == this->tensorElementStride) {

          uint32_t baseOffset = tensorIndex * this->tensorLen;

          // 由于每次只处理1个Tensor,所以每次只搬运1个tensorLength的长度的数据
          if (0 == (this->tensorLen * sizeof(DTYPE_X) % 32)) {
            DataCopy(inLocal[0], xGm[baseOffset], this->tensorLen);
            DataCopy(inLocal[this->avaiUBDataLen], gGm[baseOffset],
                     this->tensorLen);
            DataCopy(inLocal[2 * this->avaiUBDataLen], bGm[baseOffset],
                     this->tensorLen);

          } else {
            uint32_t dLen = this->tensorLen / (32 / sizeof(DTYPE_X)) *
                            (32 / sizeof(DTYPE_X));
            uint32_t restLen = this->tensorLen % (32 / sizeof(DTYPE_X));

            if (0 != dLen) {
              DataCopy(inLocal[0], xGm[baseOffset], dLen);
              DataCopy(inLocal[this->avaiUBDataLen], gGm[baseOffset], dLen);
              DataCopy(inLocal[2 * this->avaiUBDataLen], bGm[baseOffset], dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              inLocal.SetValue(dLen + i, xGm.GetValue(baseOffset + dLen + i));
              inLocal.SetValue(this->avaiUBDataLen + dLen + i,
                               gGm.GetValue(baseOffset + dLen + i));
              inLocal.SetValue(2 * this->avaiUBDataLen + dLen + i,
                               bGm.GetValue(baseOffset + dLen + i));
            }
          }

        } else {
          // 1.2 如果不连续
          // 求出需要拷贝的Tensor的起始数据位置
          uint32_t tensorBaseAddr =
              (tensorIndex / this->tensorElementStride) *
                  (this->tensorElementStride * this->tensorLen) +
              (tensorIndex % this->tensorElementStride);

          for (uint32_t i = 0, offset = 0; i < this->tensorLen;
               i++, offset += this->tensorElementStride) {

            inLocal.SetValue(i, xGm.GetValue(tensorBaseAddr + offset));
            inLocal.SetValue(this->avaiUBDataLen + i,
                             gGm.GetValue(tensorBaseAddr + offset));
            inLocal.SetValue(2 * this->avaiUBDataLen + i,
                             bGm.GetValue(tensorBaseAddr + offset));
          }
        }
      } else { // 2.不能够全部放入
        // 2.1 tensor内部数据连续
        if (1 == this->tensorElementStride) {
          if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {

            uint32_t baseOffset = tensorIndex * this->tensorLen +
                                  perTensorUpdateIndex * this->dataLenPerTile;

            // 长度是肯定能够被32Byte整除的
            DataCopy(inLocal[0], xGm[baseOffset], this->dataLenPerTile);
            DataCopy(inLocal[this->avaiUBDataLen], gGm[baseOffset],
                     this->dataLenPerTile);
            DataCopy(inLocal[2 * this->avaiUBDataLen], bGm[baseOffset],
                     this->dataLenPerTile);

          } else {

            uint32_t baseOffset = tensorIndex * this->tensorLen +
                                  perTensorUpdateIndex * this->dataLenPerTile;

            if (0 == (this->tensorLastUpDataLen * sizeof(DTYPE_X) % 32)) {
              DataCopy(inLocal[0], xGm[baseOffset], this->tensorLastUpDataLen);
              DataCopy(inLocal[this->avaiUBDataLen], gGm[baseOffset],
                       this->tensorLastUpDataLen);
              DataCopy(inLocal[2 * this->avaiUBDataLen], bGm[baseOffset],
                       this->tensorLastUpDataLen);

            } else {
              uint32_t dLen = this->tensorLastUpDataLen /
                              (32 / sizeof(DTYPE_X)) * (32 / sizeof(DTYPE_X));
              uint32_t restLen =
                  this->tensorLastUpDataLen % (32 / sizeof(DTYPE_X));

              if (0 != dLen) {
                DataCopy(inLocal[0], xGm[baseOffset], dLen);
                DataCopy(inLocal[this->avaiUBDataLen], gGm[baseOffset], dLen);
                DataCopy(inLocal[2 * this->avaiUBDataLen], bGm[baseOffset],
                         dLen);
              }

              for (uint32_t i = 0; i < restLen; i++) {
                inLocal.SetValue(dLen + i, xGm.GetValue(baseOffset + dLen + i));
                inLocal.SetValue(this->avaiUBDataLen + dLen + i,
                                 gGm.GetValue(baseOffset + dLen + i));
                inLocal.SetValue(2 * this->avaiUBDataLen + dLen + i,
                                 bGm.GetValue(baseOffset + dLen + i));
              }
            }
          }

        } else {
          if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {
            // 2.2 如果不连续
            // 求出需要拷贝数据的起始数据位置
            uint32_t tensorBaseAddr =
                (tensorIndex / this->tensorElementStride) *
                    (this->tensorElementStride * this->tensorLen) +
                (tensorIndex % this->tensorElementStride) +
                perTensorUpdateIndex * this->dataLenPerTile;

            for (uint32_t i = 0, offset = 0; i < this->dataLenPerTile;
                 i++, offset += this->tensorElementStride) {

              inLocal.SetValue(i, xGm.GetValue(tensorBaseAddr + offset));
              inLocal.SetValue(this->avaiUBDataLen + i,
                               gGm.GetValue(tensorBaseAddr + offset));
              inLocal.SetValue(2 * this->avaiUBDataLen + i,
                               bGm.GetValue(tensorBaseAddr + offset));
            }

          } else {

            // 2.3 如果不连续
            // 求出需要拷贝数据的起始数据位置
            uint32_t tensorBaseAddr =
                (tensorIndex / this->tensorElementStride) *
                    (this->tensorElementStride * this->tensorLen) +
                (tensorIndex % this->tensorElementStride) +
                perTensorUpdateIndex * this->dataLenPerTile;

            for (uint32_t i = 0, offset = 0; i < this->tensorLastUpDataLen;
                 i++, offset += this->tensorElementStride) {

              inLocal.SetValue(i, xGm.GetValue(tensorBaseAddr + offset));
              inLocal.SetValue(this->avaiUBDataLen + i,
                               gGm.GetValue(tensorBaseAddr + offset));
              inLocal.SetValue(2 * this->avaiUBDataLen + i,
                               bGm.GetValue(tensorBaseAddr + offset));
            }
          }
        }
      }
    }

    inQueueIN.EnQue(inLocal);
  }

  __aicore__ inline void CopyXIn(int32_t tensorIndex,
                                 int32_t perTensorUpdateIndex) {
    LocalTensor<DTYPE_X> inLocal = inQueueIN.AllocTensor<DTYPE_X>();

    // 分为1个tensor是否可以全部UB
    // 和tensor内部元素是否连续进行讨论
    if (BUFFER_NUM == 1) {
      // 1.先按照1个Tensor是否能够全部放入到UB中进行划分--能够全部放入
      if (1 == this->updateTimesPerTensor) {
        // 1.1 tensor内部连续的情况,可以使用DataCopy进行数据搬运
        if (1 == this->tensorElementStride) {

          uint32_t baseOffset = tensorIndex * this->tensorLen;

          // 由于每次只处理1个Tensor,所以每次只搬运1个tensorLength的长度的数据
          if (0 == (this->tensorLen * sizeof(DTYPE_X) % 32)) {
            DataCopy(inLocal[0], xGm[baseOffset], this->tensorLen);

          } else {
            uint32_t dLen = this->tensorLen / (32 / sizeof(DTYPE_X)) *
                            (32 / sizeof(DTYPE_X));
            uint32_t restLen = this->tensorLen % (32 / sizeof(DTYPE_X));

            if (0 != dLen) {
              DataCopy(inLocal[0], xGm[baseOffset], dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              inLocal.SetValue(dLen + i, xGm.GetValue(baseOffset + dLen + i));
            }
          }

        } else {
          // 1.2 如果不连续
          // 求出需要拷贝的Tensor的起始数据位置
          uint32_t tensorBaseAddr =
              (tensorIndex / this->tensorElementStride) *
                  (this->tensorElementStride * this->tensorLen) +
              (tensorIndex % this->tensorElementStride);

          for (uint32_t i = 0, offset = 0; i < this->tensorLen;
               i++, offset += this->tensorElementStride) {

            inLocal.SetValue(i, xGm.GetValue(tensorBaseAddr + offset));
          }
        }
      } else { // 2.不能够全部放入
        // 2.1 tensor内部数据连续
        if (1 == this->tensorElementStride) {
          if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {

            uint32_t baseOffset = tensorIndex * this->tensorLen +
                                  perTensorUpdateIndex * this->dataLenPerTile;

            DataCopy(inLocal[0], xGm[baseOffset], this->dataLenPerTile);

          } else {

            uint32_t baseOffset = tensorIndex * this->tensorLen +
                                  perTensorUpdateIndex * this->dataLenPerTile;

            if (0 == (this->tensorLastUpDataLen * sizeof(DTYPE_X) % 32)) {
              DataCopy(inLocal[0], xGm[baseOffset], this->tensorLastUpDataLen);
            } else {
              uint32_t dLen = this->tensorLastUpDataLen /
                              (32 / sizeof(DTYPE_X)) * (32 / sizeof(DTYPE_X));
              uint32_t restLen =
                  this->tensorLastUpDataLen % (32 / sizeof(DTYPE_X));

              if (0 != dLen) {
                DataCopy(inLocal[0], xGm[baseOffset], dLen);
              }

              for (uint32_t i = 0; i < restLen; i++) {
                inLocal.SetValue(dLen + i, xGm.GetValue(baseOffset + dLen + i));
              }
            }
          }

        } else {
          if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {
            // 2.2 如果不连续
            // 求出需要拷贝数据的起始数据位置
            uint32_t tensorBaseAddr =
                (tensorIndex / this->tensorElementStride) *
                    (this->tensorElementStride * this->tensorLen) +
                (tensorIndex % this->tensorElementStride) +
                perTensorUpdateIndex * this->dataLenPerTile;

            for (uint32_t i = 0, offset = 0; i < this->dataLenPerTile;
                 i++, offset += this->tensorElementStride) {

              inLocal.SetValue(i, xGm.GetValue(tensorBaseAddr + offset));
            }

          } else {

            // 2.3 如果不连续
            // 求出需要拷贝数据的起始数据位置
            uint32_t tensorBaseAddr =
                (tensorIndex / this->tensorElementStride) *
                    (this->tensorElementStride * this->tensorLen) +
                (tensorIndex % this->tensorElementStride) +
                perTensorUpdateIndex * this->dataLenPerTile;

            for (uint32_t i = 0, offset = 0; i < this->tensorLastUpDataLen;
                 i++, offset += this->tensorElementStride) {

              inLocal.SetValue(i, xGm.GetValue(tensorBaseAddr + offset));
            }
          }
        }
      }
    }

    inQueueIN.EnQue(inLocal);
  }

  // 只能够针对1个Tensor能够放入UB的情况,其他的情况不能够进行连续的流水
  __aicore__ inline void Compute(int32_t tensorIndex,
                                 int32_t perTensorUpdateIndex) {
    LocalTensor<DTYPE_X> inLocal = inQueueIN.DeQue<DTYPE_X>();

    LocalTensor<DTYPE_X> xLocal = inLocal;

    LocalTensor<DTYPE_X> gLocal = inLocal[this->avaiUBDataLen];
    LocalTensor<DTYPE_X> bLocal = inLocal[2 * this->avaiUBDataLen];

    LocalTensor<DTYPE_X> yLocal = outQueueOUT.AllocTensor<DTYPE_X>();

    LocalTensor<DTYPE_X> w1Local = w1Buf.Get<DTYPE_X>();
    LocalTensor<DTYPE_X> w2Local = w2Buf.Get<DTYPE_X>();

    // 1.求出mean
    if (0 == ((this->tensorLen * sizeof(DTYPE_X)) % 64)) {
      // 针对不是2**n时, ReduceSum有问题
      ReduceSum(yLocal, xLocal, w1Local, this->tensorLen);
      this->tensorMeanSum += (float)yLocal.GetValue(0);
    } else {
      for (uint32_t i = 0; i < this->tensorLen; i++) {
        this->tensorMeanSum += (float)xLocal.GetValue(i);
      }
    }

    this->tensorMean = this->tensorMeanSum / this->tensorLen;
    mGm.SetValue(tensorIndex, (DTYPE_X)this->tensorMean);
    this->tensorMeanSum = 0.0f; // 重新初始化

    // 2.求出variance
    float negMean = -1.0f * this->tensorMean;
    Adds(yLocal, xLocal, (DTYPE_X)negMean, this->tensorLen);
    Mul(w1Local, yLocal, yLocal, this->tensorLen);
    if (0 == ((this->tensorLen * sizeof(DTYPE_X)) % 64)) {
      // 针对不是2**n时, ReduceSum有问题
      ReduceSum(yLocal, w1Local, w2Local, this->tensorLen);
      this->tensorVarianceSum += (float)yLocal.GetValue(0);
    } else {
      for (uint32_t i = 0; i < this->tensorLen; i++) {
        this->tensorVarianceSum += (float)w1Local.GetValue(i);
      }
    }

    this->tensorVariance = this->tensorVarianceSum / this->tensorLen;
    vGm.SetValue(tensorIndex, (DTYPE_X)this->tensorVariance);
    this->tensorVarianceSum = 0.0f; // 重新初始化

    // 3.求instance
    // float verseSqtVar = 1.0f / sqrt(this->tensorVariance + this->epsilon);
    float verseSqtVar = 1.0f / sqrt(this->tensorVariance);

    Adds(yLocal, xLocal, (DTYPE_X)negMean, this->tensorLen);
    Muls(yLocal, yLocal, (DTYPE_X)verseSqtVar, this->tensorLen);
    Mul(yLocal, yLocal, gLocal, this->tensorLen);
    Add(yLocal, yLocal, bLocal, this->tensorLen);

    outQueueOUT.EnQue<DTYPE_X>(yLocal);
    // outQueueOUT.EnQue<DTYPE_X>(xLocal);

    inQueueIN.FreeTensor(inLocal);
  }

  //只针对1个Tensor不能够放入UB的情况
  __aicore__ inline void ComputeMean(int32_t tensorIndex,
                                     int32_t perTensorUpdateIndex) {
    LocalTensor<DTYPE_X> inLocal = inQueueIN.DeQue<DTYPE_X>();

    LocalTensor<DTYPE_X> xLocal = inLocal;
    LocalTensor<DTYPE_X> yLocal = inLocal[this->avaiUBDataLen];
    LocalTensor<DTYPE_X> wLocal = inLocal[2 * this->avaiUBDataLen];

    // 求出mean
    if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {
      // if (0 == ((this->dataLenPerTile * sizeof(DTYPE_X)) % 64)) {
      //   // 针对不是2**n时, ReduceSum有问题
      //   ReduceSum(yLocal, xLocal, wLocal, this->dataLenPerTile);
      //   this->tensorMeanSum += (float)yLocal.GetValue(0);
      // } else {
      //   for (uint32_t i = 0; i < this->dataLenPerTile; i++) {
      //     this->tensorMeanSum += (float)xLocal.GetValue(i);
      //   }
      // }

      // 在不能够全部放入UB中,this->dataLenPerTile肯定是32Byte的倍数
      ReduceSum(yLocal, xLocal, wLocal, this->dataLenPerTile);
      this->tensorMeanSum += (float)yLocal.GetValue(0);
    } else {

      if (0 == ((this->tensorLastUpDataLen * sizeof(DTYPE_X)) % 64)) {
        // 针对不是2**n时, ReduceSum有问题
        ReduceSum(yLocal, xLocal, wLocal, this->tensorLastUpDataLen);
        this->tensorMeanSum += (float)yLocal.GetValue(0);
      } else {
        for (uint32_t i = 0; i < this->tensorLastUpDataLen; i++) {
          this->tensorMeanSum += (float)xLocal.GetValue(i);
        }
      }

      this->tensorMean = this->tensorMeanSum / this->tensorLen;
      mGm.SetValue(tensorIndex, (DTYPE_X)this->tensorMean);

      this->tensorMeanSum = 0.0f; // 重新初始化
    }

    inQueueIN.FreeTensor(inLocal);
  }

  //只针对1个Tensor不能够放入UB的情况
  __aicore__ inline void ComputeVariance(int32_t tensorIndex,
                                         int32_t perTensorUpdateIndex) {
    LocalTensor<DTYPE_X> inLocal = inQueueIN.DeQue<DTYPE_X>();

    LocalTensor<DTYPE_X> xLocal = inLocal;
    LocalTensor<DTYPE_X> yLocal = inLocal[this->avaiUBDataLen];
    LocalTensor<DTYPE_X> wLocal = inLocal[2 * this->avaiUBDataLen];

    LocalTensor<DTYPE_X> w2Local = w2Buf.Get<DTYPE_X>();

    float negMean = -0.1f * (float)mGm.GetValue(tensorIndex);

    // 求出mean
    if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {
      Adds(yLocal, xLocal, (DTYPE_X)negMean, this->dataLenPerTile);
      Mul(wLocal, yLocal, yLocal, this->dataLenPerTile);
      // if (0 == ((this->dataLenPerTile * sizeof(DTYPE_X)) % 32)) {
      //   // 针对不是2**n时, ReduceSum有问题
      //   ReduceSum(yLocal, wLocal, w2Local, this->dataLenPerTile);
      //   this->tensorVarianceSum += (float)yLocal.GetValue(0);
      // } else {
      //   for (uint32_t i = 0; i < this->dataLenPerTile; i++) {
      //     this->tensorVarianceSum += (float)wLocal.GetValue(i);
      //   }
      // }

      ReduceSum(yLocal, wLocal, w2Local, this->dataLenPerTile);
      this->tensorVarianceSum += (float)yLocal.GetValue(0);
    } else {

      Adds(yLocal, xLocal, (DTYPE_X)negMean, this->tensorLastUpDataLen);
      Mul(wLocal, yLocal, yLocal, this->tensorLastUpDataLen);
      if (0 == ((this->tensorLastUpDataLen * sizeof(DTYPE_X)) % 64)) {
        // 针对不是2**n时, ReduceSum有问题
        ReduceSum(yLocal, wLocal, w2Local, this->tensorLastUpDataLen);
        this->tensorVarianceSum += (float)yLocal.GetValue(0);
      } else {
        for (uint32_t i = 0; i < this->tensorLastUpDataLen; i++) {
          this->tensorVarianceSum += (float)wLocal.GetValue(i);
        }
      }

      this->tensorVariance = this->tensorVarianceSum / this->tensorLen;
      vGm.SetValue(tensorIndex, (DTYPE_X)this->tensorVariance);
      this->tensorVarianceSum = 0.0f; // 重新初始化
    }

    inQueueIN.FreeTensor(inLocal);
  }

  //只针对1个Tensor不能够放入UB的情况
  __aicore__ inline void ComputeInstance(int32_t tensorIndex,
                                         int32_t perTensorUpdateIndex) {
    LocalTensor<DTYPE_X> inLocal = inQueueIN.DeQue<DTYPE_X>();

    LocalTensor<DTYPE_X> xLocal = inLocal;
    LocalTensor<DTYPE_X> gLocal = inLocal[this->avaiUBDataLen];
    LocalTensor<DTYPE_X> bLocal = inLocal[2 * this->avaiUBDataLen];

    LocalTensor<DTYPE_X> yLocal = outQueueOUT.AllocTensor<DTYPE_X>();

    float ngeMean = -1.0f * (float)mGm.GetValue(tensorIndex);
    // float verseSqtVar =
    //     1.0f / sqrt((float)vGm.GetValue(tensorIndex) + this->epsilon);

    float verseSqtVar = 1.0f / sqrt((float)vGm.GetValue(tensorIndex));

    // 3.求instance
    if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {
      Adds(yLocal, xLocal, (DTYPE_X)ngeMean, this->dataLenPerTile);
      Muls(yLocal, yLocal, (DTYPE_X)verseSqtVar, this->dataLenPerTile);
      Mul(yLocal, yLocal, gLocal, this->dataLenPerTile);
      Add(yLocal, yLocal, bLocal, this->dataLenPerTile);
    } else {
      Adds(yLocal, xLocal, (DTYPE_X)ngeMean, this->tensorLastUpDataLen);
      Muls(yLocal, yLocal, (DTYPE_X)verseSqtVar, this->tensorLastUpDataLen);
      Mul(yLocal, yLocal, gLocal, this->tensorLastUpDataLen);
      Add(yLocal, yLocal, bLocal, this->tensorLastUpDataLen);
    }

    outQueueOUT.EnQue<DTYPE_X>(yLocal);

    inQueueIN.FreeTensor(inLocal);
  }

  __aicore__ inline void CopyOut(int32_t tensorIndex,
                                 int32_t perTensorUpdateIndex) {
    LocalTensor<DTYPE_X> yLocal = outQueueOUT.DeQue<DTYPE_X>();

    if (BUFFER_NUM == 1) {
      // 1.先按照1个Tensor是否能够全部放入到UB中进行划分--能够全部放入
      if (1 == this->updateTimesPerTensor) {
        // 1.1 tensor内部连续的情况,可以使用DataCopy进行数据搬运
        if (1 == this->tensorElementStride) {

          uint32_t baseOffset = tensorIndex * this->tensorLen;

          // 由于每次只处理1个Tensor,所以每次只搬运1个tensorLength的长度的数据
          if (0 == (this->tensorLen * sizeof(DTYPE_X) % 32)) {
            DataCopy(yGm[baseOffset], yLocal[0], this->tensorLen);

          } else {
            // uint32_t dLen = this->tensorLen / (32 / sizeof(DTYPE_X)) *
            //                 (32 / sizeof(DTYPE_X));
            // uint32_t restLen = this->tensorLen % (32 / sizeof(DTYPE_X));

            // if (0 != dLen) {
            //   DataCopy(yGm[baseOffset], yLocal[0], dLen);
            // }

            // for (uint32_t i = 0; i < restLen; i++) {
            //   yGm.SetValue(baseOffset + dLen + i, yLocal.GetValue(dLen + i));
            // }

            for (uint32_t i = 0; i < this->tensorLen; i++) {
              yGm.SetValue(baseOffset + i, yLocal.GetValue(i));
            }
          }

        } else {
          // 1.2 如果不连续
          // 求出需要拷贝的Tensor的起始数据位置
          uint32_t tensorBaseAddr =
              (tensorIndex / this->tensorElementStride) *
                  (this->tensorElementStride * this->tensorLen) +
              (tensorIndex % this->tensorElementStride);

          for (uint32_t i = 0, offset = 0; i < this->tensorLen;
               i++, offset += this->tensorElementStride) {

            yGm.SetValue(tensorBaseAddr + offset, yLocal.GetValue(i));
          }
        }
      } else { // 2.不能够全部放入
        // 2.1 tensor内部数据连续
        if (1 == this->tensorElementStride) {
          if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {

            uint32_t baseOffset = tensorIndex * this->tensorLen +
                                  perTensorUpdateIndex * this->dataLenPerTile;

            if ((0 == (baseOffset * sizeof(DTYPE_X) % 32))) {
              // 肯定是32Byte的倍数
              DataCopy(yGm[baseOffset], yLocal[0], this->dataLenPerTile);
            } else {

              for (uint32_t i = 0; i < this->dataLenPerTile; i++) {
                yGm.SetValue(baseOffset + i, yLocal.GetValue(i));
              }
            }

            // if (0 == (this->dataLenPerTile * sizeof(DTYPE_X) % 32)) {
            //   DataCopy(yGm[baseOffset], yLocal[0], this->dataLenPerTile);

            // } else {
            //   uint32_t dLen = this->dataLenPerTile / (32 / sizeof(DTYPE_X)) *
            //                   (32 / sizeof(DTYPE_X));
            //   uint32_t restLen = this->dataLenPerTile % (32 /
            //   sizeof(DTYPE_X));

            //   if (0 != dLen) {
            //     DataCopy(yGm[baseOffset], yLocal[0], dLen);
            //   }

            //   for (uint32_t i = 0; i < restLen; i++) {
            //     yGm.SetValue(baseOffset + dLen + i, yLocal.GetValue(dLen +
            //     i));
            //   }
            // }

          } else {

            uint32_t baseOffset = tensorIndex * this->tensorLen +
                                  perTensorUpdateIndex * this->dataLenPerTile;

            // if (0 == (this->tensorLastUpDataLen * sizeof(DTYPE_X) % 32)) {
            //   DataCopy(yGm[baseOffset], yLocal[0],
            //   this->tensorLastUpDataLen);

            // } else {
            //   uint32_t dLen = this->tensorLastUpDataLen /
            //                   (32 / sizeof(DTYPE_X)) * (32 /
            //                   sizeof(DTYPE_X));
            //   uint32_t restLen =
            //       this->tensorLastUpDataLen % (32 / sizeof(DTYPE_X));

            //   if (0 != dLen) {
            //     DataCopy(yGm[baseOffset], yLocal[0], dLen);
            //   }

            //   for (uint32_t i = 0; i < restLen; i++) {
            //     yGm.SetValue(baseOffset + dLen + i, yLocal.GetValue(dLen +
            //     i));
            //   }
            // }

            if ((0 == (baseOffset * sizeof(DTYPE_X) % 32)) &&
                (0 == (this->tensorLastUpDataLen * sizeof(DTYPE_X) % 32))) {
              DataCopy(yGm[baseOffset], yLocal[0], this->tensorLastUpDataLen);

            } else {

              if ((0 == (baseOffset * sizeof(DTYPE_X) % 32))) {
                uint32_t dLen = this->tensorLastUpDataLen /
                                (32 / sizeof(DTYPE_X)) * (32 / sizeof(DTYPE_X));
                uint32_t restLen =
                    this->tensorLastUpDataLen % (32 / sizeof(DTYPE_X));

                if (0 != dLen) {
                  DataCopy(yGm[baseOffset], yLocal[0], dLen);
                }

                for (uint32_t i = 0; i < restLen; i++) {
                  yGm.SetValue(baseOffset + dLen + i,
                               yLocal.GetValue(dLen + i));
                }
              } else {
                for (uint32_t i = 0; i < this->tensorLastUpDataLen; i++) {
                  yGm.SetValue(baseOffset + i, yLocal.GetValue(i));
                }
              }
            }
          }
        } else {
          if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {
            // 2.2 如果不连续
            // 求出需要拷贝数据的起始数据位置
            uint32_t tensorBaseAddr =
                (tensorIndex / this->tensorElementStride) *
                    (this->tensorElementStride * this->tensorLen) +
                (tensorIndex % this->tensorElementStride) +
                perTensorUpdateIndex * this->dataLenPerTile;

            for (uint32_t i = 0, offset = 0; i < this->dataLenPerTile;
                 i++, offset += this->tensorElementStride) {

              yGm.SetValue(tensorBaseAddr + offset, yLocal.GetValue(i));
            }

          } else {

            // 2.3 如果不连续
            // 求出需要拷贝数据的起始数据位置
            uint32_t tensorBaseAddr =
                (tensorIndex / this->tensorElementStride) *
                    (this->tensorElementStride * this->tensorLen) +
                (tensorIndex % this->tensorElementStride) +
                perTensorUpdateIndex * this->dataLenPerTile;

            for (uint32_t i = 0, offset = 0; i < this->tensorLastUpDataLen;
                 i++, offset += this->tensorElementStride) {

              yGm.SetValue(tensorBaseAddr + offset, yLocal.GetValue(i));
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

  TBuf<QuePosition::VECCALC> w1Buf, w2Buf;
  // GlobalTensor<float> xGm;
  // GlobalTensor<float> yGm;
  // GlobalTensor<float> outGm;

  //   GlobalTensor<DTYPE_VALUE> valueGm;
  //   GlobalTensor<DTYPE_INPUT_DATA> input_dataGm;
  //   GlobalTensor<DTYPE_X2> x2Gm;

  GlobalTensor<DTYPE_X> xGm;
  GlobalTensor<DTYPE_X> gGm;
  GlobalTensor<DTYPE_X> bGm;

  GlobalTensor<DTYPE_X> yGm;

  GlobalTensor<DTYPE_X> mGm;
  GlobalTensor<DTYPE_X> vGm;

  float epsilon;

  uint32_t dataType;

  uint32_t xDataNums;
  uint32_t tensorNums;
  uint32_t tensorLen;

  uint32_t tensorElementStride;
  uint32_t updateTimesPerTensor;

  uint32_t tensorNumsPerTile;
  uint32_t tensorNumsLastTile;

  uint32_t dataLenPerTile;
  uint32_t dataLenLastTile;

  uint32_t tensorLastUpDataLen;

  uint32_t tileNum;
  uint32_t avaiUBDataLen;

  float tensorMeanSum;
  float tensorMean;

  float tensorVarianceSum;
  float tensorVariance;
  //   float sqtVar;
};

extern "C" __global__ __aicore__ void
instance_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
              GM_ADDR variance, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  KernelInstanceNorm op;

  op.Init(x, gamma, beta, y, mean, variance, tiling_data.epsilon,
          tiling_data.dataType, tiling_data.xDataNums, tiling_data.tensorNums,
          tiling_data.tensorLen, tiling_data.tensorElementStride,
          tiling_data.updateTimesPerTensor, tiling_data.tensorNumsPerTile,
          tiling_data.tensorNumsLastTile, tiling_data.dataLenPerTile,
          tiling_data.dataLenLastTile, tiling_data.tensorLastUpDataLen,
          tiling_data.tileNum, tiling_data.avaiUBDataLen);
  op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void instance_norm_do(uint32_t blockDim, void *l2ctrl, void *stream, GM_ADDR x,
                      GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
                      GM_ADDR variance, GM_ADDR workspace, GM_ADDR tiling) {
  instance_norm<<<blockDim, l2ctrl, stream>>>(x, gamma, beta, y, mean, variance,
                                              workspace, tiling);

  aclrtSynchronizeStream(stream);

  // 这个问题需要解决呢？
  // GET_TILING_DATA(tiling_data, tiling);

  //   std::cout << "reduction" << tiling_data.reduction
  //             << ", BlockNum=" << GetBlockNum() << "TILING_KEY_IS(1)"
  //             << TILING_KEY_IS(1) << std::endl;
}
#endif