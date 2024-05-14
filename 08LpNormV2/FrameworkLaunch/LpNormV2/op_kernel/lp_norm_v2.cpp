#include "kernel_operator.h"
#include <cmath>
#include <limits>

#if 0
extern "C" __global__ __aicore__ void lp_norm_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
#endif

using namespace AscendC;
// constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BUFFER_NUM = 1;

class KernelLpNormV2 {
public:
  __aicore__ inline KernelLpNormV2() {}
  __aicore__ inline void
  Init(GM_ADDR x, GM_ADDR y, float p, float epsilon, uint32_t dataType,
       uint32_t xDataNums, uint32_t tensorNums, uint32_t tensorLen,
       uint32_t tensorElementStride, uint32_t updateTimesPerTensor,
       uint32_t tensorNumsPerTile, uint32_t tensorNumsLastTile,
       uint32_t dataLenPerTile, uint32_t dataLenLastTile,
       uint32_t tensorLastUpDataLen, uint32_t tileNum, uint32_t avaiUBDataLen) {

    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

    // if (epsilon)
    this->p = p;
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

    this->sum = 0.0f;

    // 最小元素初始值赋值个最大值
    this->minElement = 9999.f;

    // 最大元素初始值赋值个最小值
    this->maxElement = -9999.f;

    this->attrSqrt = 0.0f;

    xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, this->xDataNums);

    yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y, this->tensorNums);

    // this->updateTensorIndex = 0;

    // this->tensorMeanSum = 0.0f;
    // this->tensorMean = 0.0f;

    // this->tensorVarianceSum = 0.0f;
    // this->tensorVariance = 0.0f;

    pipe.InitBuffer(inQueueIN, BUFFER_NUM,
                    this->avaiUBDataLen * sizeof(DTYPE_X));

    pipe.InitBuffer(outQueueOUT, BUFFER_NUM,
                    2 * this->avaiUBDataLen *
                        sizeof(float)); // *从这个fp32Queue拿空间

    if ((p - 0.0f < 0.01f) && (0.0f - p < 0.01f)) {
      pipe.InitBuffer(attrOne, 1, this->avaiUBDataLen * sizeof(float));
      pipe.InitBuffer(attrZero, 1, this->avaiUBDataLen * sizeof(float));

      pipe.InitBuffer(selMask, 1, this->avaiUBDataLen * sizeof(uint8_t));
    }
  }
  __aicore__ inline void Process() {
    int32_t loopCount = this->tileNum * BUFFER_NUM;

    for (int32_t i = 0; i < loopCount; i++) {
      // 2.1 处理mean
      for (int32_t j = 0; j < this->updateTimesPerTensor; j++) {
        CopyIn(i, j);
        Compute(i, j);
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

  // template <typename T>
  // __aicore__ inline void Quickpower(const LocalTensor<float> &dstTensor,
  //                                   const LocalTensor<float> &srcTensor,
  //                                   const float scalarValue,
  //                                   const uint32_t calCount) {
  //   Ln(srcTensor, srcTensor, calCount);
  //   Muls(srcTensor, srcTensor, scalarValue, calCount);
  //   Exp(dstTensor, srcTensor, calCount);
  // }

  __aicore__ inline void Compute(int32_t tensorIndex,
                                 int32_t perTensorUpdateIndex) {
    LocalTensor<DTYPE_X> inLocal = inQueueIN.DeQue<DTYPE_X>();
    LocalTensor<DTYPE_X> xLocal = inLocal;

    LocalTensor<float> calcLocal = outQueueOUT.AllocTensor<float>();
    LocalTensor<float> fp32xLocal = calcLocal;

    LocalTensor<float> yLocal = calcLocal[this->avaiUBDataLen];

    if (this->dataType == 1) { // *fp16->float
      LocalTensor<half> interxLocal = xLocal.ReinterpretCast<half>();
      AscendC::Cast(fp32xLocal, interxLocal, RoundMode::CAST_NONE,
                    this->avaiUBDataLen);
    } else { // *fp32->float
      LocalTensor<float> interxLocal = xLocal.ReinterpretCast<float>();
      // AscendC::Cast(fp32xLocal, interxLocal, RoundMode::CAST_NONE,
      //               this->tileLength);
      fp32xLocal = interxLocal;
    }

    LocalTensor<float> oneLocal;
    LocalTensor<float> zeroLocal;
    LocalTensor<uint8_t> selMaskLocal;
    if ((p - 0.0f < 0.01f) && (0.0f - p < 0.01f)) {

      oneLocal = attrOne.Get<float>();
      zeroLocal = attrZero.Get<float>();
      selMaskLocal = selMask.Get<uint8_t>();
    }

    //每次就可以对1个Tensor进行更新
    if (1 == this->updateTimesPerTensor) {

      if (((p - 2.0f) < 0.01f) && (2.0f - p < 0.01f)) {

        // Abs(xLocal, xLocal, this->lasttileLength);
        Mul(fp32xLocal, fp32xLocal, fp32xLocal, this->tensorLen);

        if (0 == ((this->tensorLen * sizeof(float)) % 256)) {
          ReduceSum(fp32xLocal, fp32xLocal, yLocal, this->tensorLen);
          this->sum += fp32xLocal.GetValue(0);
        } else {
          for (uint32_t i = 0; i < this->tensorLen; i++) {
            float t = fp32xLocal.GetValue(i);
            this->sum += t;
          }
        }

        this->attrSqrt = sqrt(this->sum);
        yGm.SetValue(tensorIndex, (DTYPE_X)this->attrSqrt);

        this->sum = 0.0f;

      } else if ((p - 0.0f < 0.01f) && (0.0f - p < 0.01f)) {

        // 还有些问题,需要求出所有非0元素的个数
        Duplicate(oneLocal, (float)1.0, this->tensorLen);
        Duplicate(zeroLocal, (float)0.0, this->tensorLen);

        Abs(fp32xLocal, fp32xLocal, this->tensorLen);

        if (0 == ((this->tensorLen * sizeof(float)) % 256)) {
          Compare(selMaskLocal, fp32xLocal, zeroLocal, CMPMODE::GT,
                  this->tensorLen);
          Select(fp32xLocal, selMaskLocal, oneLocal, zeroLocal,
                 SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tensorLen);

          ReduceSum(fp32xLocal, fp32xLocal, yLocal, this->tensorLen);
          this->sum += fp32xLocal.GetValue(0);
        } else {
          for (uint32_t i = 0; i < this->tensorLen; i++) {
            float t = fp32xLocal.GetValue(i);
            if (t > 0.f) {
              this->sum += 1.0f;
            }
          }
        }

        this->sum = this->tensorLen > this->sum ? this->sum : this->tensorLen;
        yGm.SetValue(tensorIndex, (DTYPE_X)this->sum);
        this->sum = 0.0f;

      } else if ((p - 1.0f < 0.01f) && (1.0f - p < 0.01f)) {

        Abs(fp32xLocal, fp32xLocal, this->tensorLen);
        // ReduceSum(fp32xLocal, fp32xLocal, yLocal, this->tensorLen);

        // this->sum += fp32xLocal.GetValue(0);

        if (0 == ((this->tensorLen * sizeof(float)) % 256)) {
          // 针对不是2**n时, ReduceSum有问题
          ReduceSum(fp32xLocal, fp32xLocal, yLocal, this->tensorLen);
          this->sum += fp32xLocal.GetValue(0);
        } else {
          // this->sum = fp32xLocal.GetValue(0);
          for (uint32_t i = 0; i < this->tensorLen; i++) {
            float t = fp32xLocal.GetValue(i);
            this->sum += t;
          }
        }

        yGm.SetValue(tensorIndex, (DTYPE_X)this->sum);
        this->sum = 0.0f;

      } else if ((p - 9999.f < 0.01f) && (9999.f - p < 0.01f)) {
        Abs(fp32xLocal, fp32xLocal, this->tensorLen);

        if (0 == ((this->tensorLen * sizeof(float)) % 256)) {
          // 针对不是2**n时, ReduceSum有问题
          ReduceMax(fp32xLocal, fp32xLocal, yLocal, this->tensorLen);
          this->maxElement = fp32xLocal.GetValue(0);
        } else {
          // this->sum = fp32xLocal.GetValue(0);
          for (uint32_t i = 0; i < this->tensorLen; i++) {

            float t = fp32xLocal.GetValue(i);
            if (t > this->maxElement) {
              this->maxElement = t;
            }
          }
        }

        yGm.SetValue(tensorIndex, (DTYPE_X)this->maxElement);
        this->maxElement = -9999.f;

      } else if ((p - (-9999.f) < 0.01f) && ((-9999.f) - p < 0.01f)) {
        // 这个有问题,这儿取出来都是0,需要先把0剔除出去
        Abs(fp32xLocal, fp32xLocal, this->tensorLen);

        if (0 == ((this->tensorLen * sizeof(float)) % 256)) {
          // 针对不是2**n时, ReduceSum有问题
          ReduceMin(fp32xLocal, fp32xLocal, yLocal, this->tensorLen);
          this->minElement = fp32xLocal.GetValue(0);
        } else {
          // this->sum = fp32xLocal.GetValue(0);
          for (uint32_t i = 0; i < this->tensorLen; i++) {

            float t = fp32xLocal.GetValue(i);
            if (t < this->minElement) {
              this->minElement = t;
            }
          }
        }

        yGm.SetValue(tensorIndex, (DTYPE_X)this->minElement);
        this->minElement = 9999.f;
      } else {
        Abs(fp32xLocal, fp32xLocal, this->tensorLen);

        // Quickpower(fp32xLocal, fp32xLocal, p, this->tensorLen);

        Ln(fp32xLocal, fp32xLocal, this->tensorLen);
        Muls(fp32xLocal, fp32xLocal, this->p, this->tensorLen);
        Exp(fp32xLocal, fp32xLocal, this->tensorLen);

        if (0 == ((this->tensorLen * sizeof(float)) % 256)) {
          ReduceSum(fp32xLocal, fp32xLocal, yLocal, this->tensorLen);
          this->sum += fp32xLocal.GetValue(0);
        } else {
          for (uint32_t i = 0; i < this->tensorLen; i++) {
            float t = fp32xLocal.GetValue(i);
            this->sum += t;
          }
        }

        yLocal.SetValue(0, this->sum);
        // Quickpower(yLocal, yLocal, ((float)1.0) / p, 1);

        // Ln(fp32xLocal, fp32xLocal, this->tensorLen);
        // Muls(fp32xLocal, fp32xLocal, ((float)1.0) / this->p,
        // this->tensorLen); Exp(fp32xLocal, fp32xLocal, this->tensorLen);

        Ln(yLocal, yLocal, 1);
        Muls(yLocal, yLocal, ((float)1.0) / this->p, 1);
        Exp(yLocal, yLocal, 1);

        yGm.SetValue(tensorIndex, yLocal.GetValue(0));

        this->sum = 0.0f;
      }

    } else {
      if (perTensorUpdateIndex != (this->updateTimesPerTensor - 1)) {
        if (((p - 2.0f) < 0.01f) && (2.0f - p < 0.01f)) {

          // Abs(xLocal, xLocal, this->lasttileLength);
          Mul(fp32xLocal, fp32xLocal, fp32xLocal, this->dataLenPerTile);

          if (0 == ((this->dataLenPerTile * sizeof(float)) % 256)) {
            ReduceSum(fp32xLocal, fp32xLocal, yLocal, this->dataLenPerTile);
            this->sum += fp32xLocal.GetValue(0);
          } else {
            for (uint32_t i = 0; i < this->dataLenPerTile; i++) {
              float t = fp32xLocal.GetValue(i);
              this->sum += t;
            }
          }
        } else if ((p - 0.0f < 0.01f) && (0.0f - p < 0.01f)) {

          // 还有些问题,需要求出所有非0元素的个数
          Duplicate(oneLocal, (float)1.0, this->dataLenPerTile);
          Duplicate(zeroLocal, (float)0.0, this->dataLenPerTile);

          Abs(fp32xLocal, fp32xLocal, this->dataLenPerTile);

          if (0 == ((this->dataLenPerTile * sizeof(float)) % 256)) {
            Compare(selMaskLocal, fp32xLocal, zeroLocal, CMPMODE::GT,
                    this->dataLenPerTile);
            Select(fp32xLocal, selMaskLocal, oneLocal, zeroLocal,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, this->dataLenPerTile);

            ReduceSum(fp32xLocal, fp32xLocal, yLocal, this->dataLenPerTile);
            this->sum += fp32xLocal.GetValue(0);
          } else {
            for (uint32_t i = 0; i < this->dataLenPerTile; i++) {
              float t = fp32xLocal.GetValue(i);
              if (t > 0.f) {
                this->sum += 1.0f;
              }
            }
          }
        } else if ((p - 1.0f < 0.01f) && (1.0f - p < 0.01f)) {

          Abs(fp32xLocal, fp32xLocal, this->dataLenPerTile);
          // ReduceSum(fp32xLocal, fp32xLocal, yLocal, this->dataLenPerTile);

          // this->sum += fp32xLocal.GetValue(0);

          if (0 == ((this->dataLenPerTile * sizeof(float)) % 256)) {
            // 针对不是2**n时, ReduceSum有问题
            ReduceSum(fp32xLocal, fp32xLocal, yLocal, this->dataLenPerTile);
            this->sum += fp32xLocal.GetValue(0);
          } else {
            // this->sum = fp32xLocal.GetValue(0);
            for (uint32_t i = 0; i < this->dataLenPerTile; i++) {
              float t = fp32xLocal.GetValue(i);
              this->sum += t;
            }
          }

        } else if ((p - 9999.f < 0.01f) && (9999.f - p < 0.01f)) {
          Abs(fp32xLocal, fp32xLocal, this->dataLenPerTile);

          if (0 == ((this->dataLenPerTile * sizeof(float)) % 256)) {
            // 针对不是2**n时, ReduceSum有问题
            ReduceMax(fp32xLocal, fp32xLocal, yLocal, this->dataLenPerTile);
            this->maxElement = fp32xLocal.GetValue(0);
          } else {
            // this->sum = fp32xLocal.GetValue(0);
            for (uint32_t i = 0; i < this->dataLenPerTile; i++) {

              float t = fp32xLocal.GetValue(i);
              if (t > this->maxElement) {
                this->maxElement = t;
              }
            }
          }

        } else if ((p - (-9999.f) < 0.01f) && ((-9999.f) - p < 0.01f)) {
          // 这个有问题,这儿取出来都是0,需要先把0剔除出去
          Abs(fp32xLocal, fp32xLocal, this->dataLenPerTile);

          if (0 == ((this->dataLenPerTile * sizeof(float)) % 256)) {
            // 针对不是2**n时, ReduceSum有问题
            ReduceMin(fp32xLocal, fp32xLocal, yLocal, this->dataLenPerTile);
            this->minElement = fp32xLocal.GetValue(0);
          } else {
            // this->sum = fp32xLocal.GetValue(0);
            for (uint32_t i = 0; i < this->dataLenPerTile; i++) {

              float t = fp32xLocal.GetValue(i);
              if (t < this->minElement) {
                this->minElement = t;
              }
            }
          }
        } else {
          Abs(fp32xLocal, fp32xLocal, this->dataLenPerTile);
          // Quickpower(fp32xLocal, fp32xLocal, p, this->dataLenPerTile);

          Ln(fp32xLocal, fp32xLocal, this->dataLenPerTile);
          Muls(fp32xLocal, fp32xLocal, this->p, this->dataLenPerTile);
          Exp(fp32xLocal, fp32xLocal, this->dataLenPerTile);

          if (0 == ((this->dataLenPerTile * sizeof(float)) % 256)) {
            ReduceSum(fp32xLocal, fp32xLocal, yLocal, this->dataLenPerTile);
            this->sum += fp32xLocal.GetValue(0);
          } else {
            for (uint32_t i = 0; i < this->dataLenPerTile; i++) {
              float t = fp32xLocal.GetValue(i);
              this->sum += t;
            }
          }
        }

      } else {

        if (((p - 2.0f) < 0.01f) && (2.0f - p < 0.01f)) {

          // Abs(xLocal, xLocal, this->lasttileLength);
          Mul(fp32xLocal, fp32xLocal, fp32xLocal, this->tensorLastUpDataLen);

          if (0 == ((this->tensorLastUpDataLen * sizeof(float)) % 256)) {
            ReduceSum(fp32xLocal, fp32xLocal, yLocal,
                      this->tensorLastUpDataLen);
            this->sum += fp32xLocal.GetValue(0);
          } else {
            for (uint32_t i = 0; i < this->tensorLastUpDataLen; i++) {
              float t = fp32xLocal.GetValue(i);
              this->sum += t;
            }
          }

          this->attrSqrt = sqrt(this->sum);
          yGm.SetValue(tensorIndex, (DTYPE_X)this->attrSqrt);

          this->sum = 0.0f;

        } else if ((p - 0.0f < 0.01f) && (0.0f - p < 0.01f)) {

          // 还有些问题,需要求出所有非0元素的个数
          Duplicate(oneLocal, (float)1.0, this->tensorLastUpDataLen);
          Duplicate(zeroLocal, (float)0.0, this->tensorLastUpDataLen);

          Abs(fp32xLocal, fp32xLocal, this->tensorLastUpDataLen);

          if (0 == ((this->tensorLastUpDataLen * sizeof(float)) % 256)) {
            Compare(selMaskLocal, fp32xLocal, zeroLocal, CMPMODE::GT,
                    this->tensorLastUpDataLen);
            Select(fp32xLocal, selMaskLocal, oneLocal, zeroLocal,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tensorLastUpDataLen);

            ReduceSum(fp32xLocal, fp32xLocal, yLocal,
                      this->tensorLastUpDataLen);
            this->sum += fp32xLocal.GetValue(0);
          } else {
            for (uint32_t i = 0; i < this->tensorLastUpDataLen; i++) {
              float t = fp32xLocal.GetValue(i);
              if (t > 0.f) {
                this->sum += 1.f;
              }
            }
          }

          this->sum = this->tensorLastUpDataLen > this->sum
                          ? this->sum
                          : this->tensorLastUpDataLen;
          yGm.SetValue(tensorIndex, (DTYPE_X)this->sum);
          this->sum = 0.0f;

        } else if ((p - 1.0f < 0.01f) && (1.0f - p < 0.01f)) {

          Abs(fp32xLocal, fp32xLocal, this->tensorLastUpDataLen);
          // ReduceSum(fp32xLocal, fp32xLocal, yLocal,
          // this->tensorLastUpDataLen);

          // this->sum += fp32xLocal.GetValue(0);

          if (0 == ((this->tensorLastUpDataLen * sizeof(float)) % 256)) {
            // 针对不是2**n时, ReduceSum有问题
            ReduceSum(fp32xLocal, fp32xLocal, yLocal,
                      this->tensorLastUpDataLen);
            this->sum += fp32xLocal.GetValue(0);
          } else {
            // this->sum = fp32xLocal.GetValue(0);
            for (uint32_t i = 0; i < this->tensorLastUpDataLen; i++) {
              float t = fp32xLocal.GetValue(i);
              this->sum += t;
            }
          }

          yGm.SetValue(tensorIndex, (DTYPE_X)this->sum);
          this->sum = 0.0f;

        } else if ((p - 9999.f < 0.01f) && (9999.f - p < 0.01f)) {
          Abs(fp32xLocal, fp32xLocal, this->tensorLastUpDataLen);

          if (0 == ((this->tensorLastUpDataLen * sizeof(float)) % 256)) {
            // 针对不是2**n时, ReduceSum有问题
            ReduceMax(fp32xLocal, fp32xLocal, yLocal,
                      this->tensorLastUpDataLen);
            this->maxElement = fp32xLocal.GetValue(0);
          } else {
            // this->sum = fp32xLocal.GetValue(0);
            for (uint32_t i = 0; i < this->tensorLastUpDataLen; i++) {

              float t = fp32xLocal.GetValue(i);
              if (t > this->maxElement) {
                this->maxElement = t;
              }
            }
          }

          yGm.SetValue(tensorIndex, (DTYPE_X)this->maxElement);
          this->maxElement = -9999.f;

        } else if ((p - (-9999.f) < 0.01f) && ((-9999.f) - p < 0.01f)) {
          // 这个有问题,这儿取出来都是0,需要先把0剔除出去
          Abs(fp32xLocal, fp32xLocal, this->tensorLastUpDataLen);

          if (0 == ((this->tensorLastUpDataLen * sizeof(float)) % 256)) {
            // 针对不是2**n时, ReduceSum有问题
            ReduceMin(fp32xLocal, fp32xLocal, yLocal,
                      this->tensorLastUpDataLen);
            this->minElement = fp32xLocal.GetValue(0);
          } else {
            // this->sum = fp32xLocal.GetValue(0);
            for (uint32_t i = 0; i < this->tensorLastUpDataLen; i++) {

              float t = fp32xLocal.GetValue(i);
              if (t < this->minElement) {
                this->minElement = t;
              }
            }
          }

          yGm.SetValue(tensorIndex, (DTYPE_X)this->minElement);
          this->minElement = 9999.f;
        } else {
          Abs(fp32xLocal, fp32xLocal, this->tensorLastUpDataLen);
          // Quickpower(fp32xLocal, fp32xLocal, p, this->tensorLastUpDataLen);

          Ln(fp32xLocal, fp32xLocal, this->tensorLastUpDataLen);
          Muls(fp32xLocal, fp32xLocal, this->p, this->tensorLastUpDataLen);
          Exp(fp32xLocal, fp32xLocal, this->tensorLastUpDataLen);

          if (0 == ((this->tensorLastUpDataLen * sizeof(float)) % 256)) {
            ReduceSum(fp32xLocal, fp32xLocal, yLocal,
                      this->tensorLastUpDataLen);
            this->sum += fp32xLocal.GetValue(0);
          } else {
            for (uint32_t i = 0; i < this->tensorLastUpDataLen; i++) {
              float t = fp32xLocal.GetValue(i);
              this->sum += t;
            }
          }

          yLocal.SetValue(0, this->sum);
          // Quickpower(yLocal, yLocal, ((float)1.0) / p, 1);

          Ln(yLocal, yLocal, 1);
          Muls(yLocal, yLocal, ((float)1.0) / this->p, 1);
          Exp(yLocal, yLocal, 1);

          yGm.SetValue(tensorIndex, yLocal.GetValue(0));

          this->sum = 0.0f;
        }
      }
    }

    // outQueueIN.EnQue(yLocal);

    outQueueOUT.FreeTensor(calcLocal);

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
            //   yGm.SetValue(baseOffset + dLen + i, yLocal.GetValue(dLen +
            //   i));
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
            //   uint32_t dLen = this->dataLenPerTile / (32 /
            //   sizeof(DTYPE_X))
            //   *
            //                   (32 / sizeof(DTYPE_X));
            //   uint32_t restLen = this->dataLenPerTile % (32 /
            //   sizeof(DTYPE_X));

            //   if (0 != dLen) {
            //     DataCopy(yGm[baseOffset], yLocal[0], dLen);
            //   }

            //   for (uint32_t i = 0; i < restLen; i++) {
            //     yGm.SetValue(baseOffset + dLen + i, yLocal.GetValue(dLen
            //     + i));
            //   }
            // }

          } else {

            uint32_t baseOffset = tensorIndex * this->tensorLen +
                                  perTensorUpdateIndex * this->dataLenPerTile;

            // if (0 == (this->tensorLastUpDataLen * sizeof(DTYPE_X) % 32))
            // {
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
            //     yGm.SetValue(baseOffset + dLen + i, yLocal.GetValue(dLen
            //     + i));
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
  // TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIN;
  // TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;

  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIN;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;

  TBuf<QuePosition::VECCALC> attrOne, attrZero;
  TBuf<QuePosition::VECCALC> selMask;

  float sum;
  float attrSqrt;

  float minElement;
  float maxElement;
  float p;

  // TBuf<QuePosition::VECCALC> w1Buf, w2Buf;
  // GlobalTensor<float> xGm;
  // GlobalTensor<float> yGm;
  // GlobalTensor<float> outGm;

  //   GlobalTensor<DTYPE_VALUE> valueGm;
  //   GlobalTensor<DTYPE_INPUT_DATA> input_dataGm;
  //   GlobalTensor<DTYPE_X2> x2Gm;

  GlobalTensor<DTYPE_X> xGm;

  GlobalTensor<DTYPE_X> yGm;

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

  // float tensorMeanSum;
  // float tensorMean;

  // float tensorVarianceSum;
  // float tensorVariance;
  //   float sqtVar;
};

extern "C" __global__ __aicore__ void
lp_norm_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  KernelLpNormV2 op;

  op.Init(x, y, tiling_data.p, tiling_data.epsilon, tiling_data.dataType,
          tiling_data.xDataNums, tiling_data.tensorNums, tiling_data.tensorLen,
          tiling_data.tensorElementStride, tiling_data.updateTimesPerTensor,
          tiling_data.tensorNumsPerTile, tiling_data.tensorNumsLastTile,
          tiling_data.dataLenPerTile, tiling_data.dataLenLastTile,
          tiling_data.tensorLastUpDataLen, tiling_data.tileNum,
          tiling_data.avaiUBDataLen);
  op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void lp_norm_v2_do(uint32_t blockDim, void *l2ctrl, void *stream, GM_ADDR x,
                   GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  lp_norm_v2<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);

  aclrtSynchronizeStream(stream);

  // 这个问题需要解决呢？
  // GET_TILING_DATA(tiling_data, tiling);

  //   std::cout << "reduction" << tiling_data.reduction
  //             << ", BlockNum=" << GetBlockNum() << "TILING_KEY_IS(1)"
  //             << TILING_KEY_IS(1) << std::endl;
}
#endif
