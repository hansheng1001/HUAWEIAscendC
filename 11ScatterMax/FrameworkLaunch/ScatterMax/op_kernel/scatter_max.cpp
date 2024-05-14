#include "kernel_operator.h"

#if 0
extern "C" __global__ __aicore__ void scatter_max(GM_ADDR x, GM_ADDR index, GM_ADDR update, GM_ADDR x_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
#endif

using namespace AscendC;
// constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BUFFER_NUM = 1;

class KernelScatterMax {
public:
  __aicore__ inline KernelScatterMax() {}
  __aicore__ inline void
  Init(GM_ADDR x, GM_ADDR index, GM_ADDR update, GM_ADDR y, uint32_t dataType,
       int32_t varDataNums, uint32_t updateTensorNums,
       uint32_t perUpdateTensorLen, uint32_t updateTimesPerTensor,
       uint32_t updateTensorNumsPerTile, uint32_t updateTensorNumsLastTile,
       uint32_t dataLenPerTile, uint32_t dataLenLastTile,
       uint32_t lastUpPerTensordataLen, uint32_t tileNum,
       uint32_t avaiUBDataLen) {

    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

    this->dataType = dataType;

    this->varDataNums = varDataNums;
    this->updateTensorNums = updateTensorNums;

    this->perUpdateTensorLen = perUpdateTensorLen;

    this->updateTimesPerTensor = updateTimesPerTensor;

    this->updateTensorNumsPerTile = updateTensorNumsPerTile;
    this->updateTensorNumsLastTile = updateTensorNumsLastTile;

    this->dataLenPerTile = dataLenPerTile;
    this->dataLenLastTile = dataLenLastTile;

    this->lastUpPerTensordataLen = lastUpPerTensordataLen;

    this->tileNum = tileNum;
    this->avaiUBDataLen = avaiUBDataLen;

    xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, this->varDataNums);

    iGm.SetGlobalBuffer((__gm__ int32_t *)index, this->updateTensorNums);

    uGm.SetGlobalBuffer((__gm__ DTYPE_X *)update,
                        this->updateTensorNums * this->perUpdateTensorLen);

    yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y, this->varDataNums);

    // 用于每个Tensor的更新不止1次的情况,用在CopyIn函数中
    this->processIGmIndex = 0;
    this->updateTensorIndex = iGm.GetValue(0);

    // 用于每个Tensor的更新不止1次的情况,用在CopyOut函数中
    this->processIGmIndexCopyOut = 0;
    this->updateTensorIndexCopyOut = iGm.GetValue(0);

    pipe.InitBuffer(inQueueIN, BUFFER_NUM,
                    this->avaiUBDataLen * 2 * sizeof(DTYPE_X));
    pipe.InitBuffer(outQueueOUT, BUFFER_NUM,
                    this->avaiUBDataLen * sizeof(DTYPE_X));

    if (2 == this->dataType) {
      pipe.InitBuffer(xHalfBuf, 1, this->avaiUBDataLen * sizeof(half));
      pipe.InitBuffer(uHalfBuf, 1, this->avaiUBDataLen * sizeof(half));
      pipe.InitBuffer(yHalfBuf, 1, this->avaiUBDataLen * sizeof(half));
    }
  }
  __aicore__ inline void Process() {
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
  }

private:
  __aicore__ inline void CopyIn(int32_t progress) {
    LocalTensor<DTYPE_X> inLocal = inQueueIN.AllocTensor<DTYPE_X>();

    if (BUFFER_NUM == 1) {
      if (1 == this->updateTimesPerTensor) {
        if (progress == this->tileNum * BUFFER_NUM - 1) {
          // 1.复制带更新的x到inLocal[0]处
          for (uint32_t j = 0; j < this->updateTensorNumsLastTile; j++) {
            uint32_t index =
                iGm.GetValue(this->updateTensorNumsPerTile * progress + j);

            // 1.2.将每x都拷贝到相应位置
            if (0 == (this->perUpdateTensorLen * sizeof(DTYPE_X) % 32)) {
              DataCopy(inLocal[j * this->perUpdateTensorLen],
                       xGm[index * this->perUpdateTensorLen],
                       this->perUpdateTensorLen);
            } else {
              uint32_t dLen = this->perUpdateTensorLen /
                              (32 / sizeof(DTYPE_X)) * (32 / sizeof(DTYPE_X));
              uint32_t restLen =
                  this->perUpdateTensorLen % (32 / sizeof(DTYPE_X));

              if (0 != dLen) {
                DataCopy(inLocal[j * this->perUpdateTensorLen],
                         xGm[index * this->perUpdateTensorLen], dLen);
              }

              for (uint32_t i = 0; i < restLen; i++) {
                inLocal.SetValue(
                    j * this->perUpdateTensorLen + dLen + i,
                    xGm.GetValue(index * this->perUpdateTensorLen + dLen + i));
              }
            }
          }

          // 2.复制update到inLocal[this->avaiUBDataLen]处
          if (0 == (this->dataLenLastTile * sizeof(DTYPE_X) % 32)) {
            DataCopy(inLocal[this->avaiUBDataLen],
                     uGm[progress * this->dataLenPerTile],
                     this->dataLenLastTile);
          } else {
            uint32_t dLen = this->dataLenLastTile / (32 / sizeof(DTYPE_X)) *
                            (32 / sizeof(DTYPE_X));
            uint32_t restLen = this->dataLenLastTile % (32 / sizeof(DTYPE_X));

            if (0 != dLen) {
              DataCopy(inLocal[this->avaiUBDataLen],
                       uGm[progress * this->dataLenPerTile], dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              inLocal.SetValue(
                  this->avaiUBDataLen + dLen + i,
                  uGm.GetValue(progress * this->dataLenPerTile + dLen + i));
            }
          }
        } else {

          // 1.复制待更新的x到inLocal[0]处
          for (uint32_t j = 0; j < this->updateTensorNumsPerTile; j++) {
            uint32_t index =
                iGm.GetValue(this->updateTensorNumsPerTile * progress + j);

            // 1.2.将每x都拷贝到相应位置
            if (0 == (this->perUpdateTensorLen * sizeof(DTYPE_X) % 32)) {
              DataCopy(inLocal[j * this->perUpdateTensorLen],
                       xGm[index * this->perUpdateTensorLen],
                       this->perUpdateTensorLen);
            } else {
              uint32_t dLen = this->perUpdateTensorLen /
                              (32 / sizeof(DTYPE_X)) * (32 / sizeof(DTYPE_X));
              uint32_t restLen =
                  this->perUpdateTensorLen % (32 / sizeof(DTYPE_X));

              if (0 != dLen) {
                DataCopy(inLocal[j * this->perUpdateTensorLen],
                         xGm[index * this->perUpdateTensorLen], dLen);
              }

              for (uint32_t i = 0; i < restLen; i++) {
                inLocal.SetValue(
                    j * this->perUpdateTensorLen + dLen + i,
                    xGm.GetValue(index * this->perUpdateTensorLen + dLen + i));
              }
            }
          }

          // 2.复制update到inLocal[this->avaiUBDataLen]处
          // 肯定是能够被32整除的--不一定
          if (0 == (this->dataLenPerTile * sizeof(DTYPE_X) % 32)) {
            DataCopy(inLocal[this->avaiUBDataLen],
                     uGm[progress * this->dataLenPerTile],
                     this->dataLenPerTile);
          } else {
            uint32_t dLen = this->dataLenPerTile / (32 / sizeof(DTYPE_X)) *
                            (32 / sizeof(DTYPE_X));
            uint32_t restLen = this->dataLenPerTile % (32 / sizeof(DTYPE_X));

            if (0 != dLen) {
              DataCopy(inLocal[this->avaiUBDataLen],
                       uGm[progress * this->dataLenPerTile], dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              inLocal.SetValue(
                  this->avaiUBDataLen + dLen + i,
                  uGm.GetValue(progress * this->dataLenPerTile + dLen + i));
            }
          }
        }
      } else {

        uint32_t IndexT = progress % this->updateTimesPerTensor;

        // 这是每个tensor的最后一部分更新
        if ((this->updateTimesPerTensor - 1) == IndexT) {

          // 1.2.将Tensor的最后一部分
          if (0 == (this->lastUpPerTensordataLen * sizeof(DTYPE_X) % 32)) {
            DataCopy(inLocal[0],
                     xGm[this->updateTensorIndex * this->perUpdateTensorLen +
                         IndexT * this->dataLenPerTile],
                     this->lastUpPerTensordataLen);
          } else {
            uint32_t dLen = this->lastUpPerTensordataLen /
                            (32 / sizeof(DTYPE_X)) * (32 / sizeof(DTYPE_X));
            uint32_t restLen =
                this->lastUpPerTensordataLen % (32 / sizeof(DTYPE_X));

            if (0 != dLen) {
              DataCopy(inLocal[0],
                       xGm[this->updateTensorIndex * this->perUpdateTensorLen +
                           IndexT * this->dataLenPerTile],
                       dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              inLocal.SetValue(dLen + i,
                               xGm.GetValue(this->updateTensorIndex *
                                                this->perUpdateTensorLen +
                                            IndexT * this->dataLenPerTile +
                                            dLen + i));
            }
          }

          // 2.复制update到inLocal[this->avaiUBDataLen]处
          if (0 == (this->lastUpPerTensordataLen * sizeof(DTYPE_X) % 32)) {
            DataCopy(inLocal[this->avaiUBDataLen],
                     uGm[this->processIGmIndex * this->perUpdateTensorLen +
                         IndexT * this->dataLenPerTile],
                     this->lastUpPerTensordataLen);
          } else {
            uint32_t dLen = this->lastUpPerTensordataLen /
                            (32 / sizeof(DTYPE_X)) * (32 / sizeof(DTYPE_X));
            uint32_t restLen =
                this->lastUpPerTensordataLen % (32 / sizeof(DTYPE_X));

            if (0 != dLen) {
              DataCopy(inLocal[this->avaiUBDataLen],
                       uGm[this->processIGmIndex * this->perUpdateTensorLen +
                           IndexT * this->dataLenPerTile],
                       dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              inLocal.SetValue(this->avaiUBDataLen + dLen + i,
                               uGm.GetValue(this->processIGmIndex *
                                                this->perUpdateTensorLen +
                                            IndexT * this->dataLenPerTile +
                                            dLen + i));
            }
          }

          // 更新索引
          this->processIGmIndex++;
          this->updateTensorIndex = iGm.GetValue(this->processIGmIndex);
        } else { // 不是tensor的最后一部分更新

          // 1.将x的一部分都拷贝到相应位置
          DataCopy(inLocal[0],
                   xGm[this->updateTensorIndex * this->perUpdateTensorLen +
                       IndexT * this->dataLenPerTile],
                   this->dataLenPerTile);

          // 2.复制update到inLocal[this->avaiUBDataLen]处
          // 肯定是能够被32整除的
          DataCopy(inLocal[this->avaiUBDataLen],
                   uGm[this->processIGmIndex * this->perUpdateTensorLen +
                       IndexT * this->dataLenPerTile],
                   this->dataLenPerTile);
        }
      }
    }

    inQueueIN.EnQue(inLocal);
  }

  __aicore__ inline void Compute(int32_t progress) {
    LocalTensor<DTYPE_X> inLocal = inQueueIN.DeQue<DTYPE_X>();

    LocalTensor<DTYPE_X> xLocal = inLocal;

    LocalTensor<DTYPE_X> uLocal = inLocal[this->avaiUBDataLen];

    LocalTensor<DTYPE_X> yLocal = outQueueOUT.AllocTensor<DTYPE_X>();

    LocalTensor<half> xHalf, uHalf, yHalf;
    LocalTensor<int8_t> x_tensor, u_tensor, y_tensor;
    if (2 == this->dataType) {
      xHalf = xHalfBuf.Get<half>();
      uHalf = uHalfBuf.Get<half>();
      yHalf = yHalfBuf.Get<half>();

      x_tensor = xLocal.ReinterpretCast<int8_t>();
      u_tensor = uLocal.ReinterpretCast<int8_t>();
      y_tensor = yLocal.ReinterpretCast<int8_t>();
    }

    if (BUFFER_NUM == 1) {
      if (1 == this->updateTimesPerTensor) {
        if (progress == this->tileNum * BUFFER_NUM - 1) {
          if (2 != this->dataType) {
            Max(yLocal, xLocal, uLocal, this->dataLenLastTile);
          } else {
            Cast(xHalf, x_tensor, RoundMode::CAST_NONE, this->dataLenLastTile);
            Cast(uHalf, u_tensor, RoundMode::CAST_NONE, this->dataLenLastTile);

            Max(yHalf, xHalf, uHalf, this->dataLenLastTile);

            Cast(y_tensor, yHalf, RoundMode::CAST_NONE, this->dataLenLastTile);
          }
        } else {
          if (2 != this->dataType) {
            Max(yLocal, xLocal, uLocal, this->dataLenPerTile);
          } else {
            Cast(xHalf, x_tensor, RoundMode::CAST_NONE, this->dataLenPerTile);
            Cast(uHalf, u_tensor, RoundMode::CAST_NONE, this->dataLenPerTile);

            Max(yHalf, xHalf, uHalf, this->dataLenPerTile);

            Cast(y_tensor, yHalf, RoundMode::CAST_NONE, this->dataLenPerTile);
          }
        }
      } else {

        uint32_t IndexT = progress % this->updateTimesPerTensor;

        if ((this->updateTimesPerTensor - 1) == IndexT) {
          if (2 != this->dataType) {
            Max(yLocal, xLocal, uLocal, this->lastUpPerTensordataLen);
          } else {
            Cast(xHalf, x_tensor, RoundMode::CAST_NONE,
                 this->lastUpPerTensordataLen);
            Cast(uHalf, u_tensor, RoundMode::CAST_NONE,
                 this->lastUpPerTensordataLen);

            Max(yHalf, xHalf, uHalf, this->lastUpPerTensordataLen);

            Cast(y_tensor, yHalf, RoundMode::CAST_NONE,
                 this->lastUpPerTensordataLen);
          }
        } else {
          if (2 != this->dataType) {
            Max(yLocal, xLocal, uLocal, this->dataLenPerTile);
          } else {
            Cast(xHalf, x_tensor, RoundMode::CAST_NONE, this->dataLenPerTile);
            Cast(uHalf, u_tensor, RoundMode::CAST_NONE, this->dataLenPerTile);

            Max(yHalf, xHalf, uHalf, this->dataLenPerTile);

            Cast(y_tensor, yHalf, RoundMode::CAST_NONE, this->dataLenPerTile);
          }
        }
      }
    }

    outQueueOUT.EnQue<DTYPE_X>(yLocal);
    // outQueueOUT.EnQue<DTYPE_X>(xLocal);

    inQueueIN.FreeTensor(inLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress) {
    LocalTensor<DTYPE_X> yLocal = outQueueOUT.DeQue<DTYPE_X>();

    if (BUFFER_NUM == 1) {
      if (1 == this->updateTimesPerTensor) {
        if (progress == this->tileNum * BUFFER_NUM - 1) {
          // 1.复制带更新的y到yGm处
          for (uint32_t j = 0; j < this->updateTensorNumsLastTile; j++) {
            uint32_t index =
                iGm.GetValue(this->updateTensorNumsPerTile * progress + j);

            // 1.2.将每x都拷贝到相应位置
            if (0 == (this->perUpdateTensorLen * sizeof(DTYPE_X) % 32)) {
              DataCopy(yGm[index * this->perUpdateTensorLen],
                       yLocal[j * this->perUpdateTensorLen],
                       this->perUpdateTensorLen);
            } else {
              uint32_t dLen = this->perUpdateTensorLen /
                              (32 / sizeof(DTYPE_X)) * (32 / sizeof(DTYPE_X));
              uint32_t restLen =
                  this->perUpdateTensorLen % (32 / sizeof(DTYPE_X));

              if (0 != dLen) {
                DataCopy(yGm[index * this->perUpdateTensorLen],
                         yLocal[j * this->perUpdateTensorLen], dLen);
              }

              for (uint32_t i = 0; i < restLen; i++) {
                yGm.SetValue(
                    index * this->perUpdateTensorLen + dLen + i,
                    yLocal.GetValue(j * this->perUpdateTensorLen + dLen + i));
              }
            }
          }

        } else {

          // 1.复制待更新的x到inLocal[0]处
          for (uint32_t j = 0; j < this->updateTensorNumsPerTile; j++) {
            uint32_t index =
                iGm.GetValue(this->updateTensorNumsPerTile * progress + j);

            // 1.2.将每x都拷贝到相应位置
            if (0 == (this->perUpdateTensorLen * sizeof(DTYPE_X) % 32)) {
              DataCopy(yGm[index * this->perUpdateTensorLen],
                       yLocal[j * this->perUpdateTensorLen],
                       this->perUpdateTensorLen);
            } else {
              uint32_t dLen = this->perUpdateTensorLen /
                              (32 / sizeof(DTYPE_X)) * (32 / sizeof(DTYPE_X));
              uint32_t restLen =
                  this->perUpdateTensorLen % (32 / sizeof(DTYPE_X));

              if (0 != dLen) {
                DataCopy(yGm[index * this->perUpdateTensorLen],
                         yLocal[j * this->perUpdateTensorLen], dLen);
              }

              for (uint32_t i = 0; i < restLen; i++) {
                yGm.SetValue(
                    index * this->perUpdateTensorLen + dLen + i,
                    yLocal.GetValue(j * this->perUpdateTensorLen + dLen + i));
              }
            }
          }
        }
      } else {

        uint32_t IndexT = progress % this->updateTimesPerTensor;

        // 这是每个tensor的最后一部分更新
        if ((this->updateTimesPerTensor - 1) == IndexT) {

          // 1.2.将Tensor的最后一部分
          if (0 == (this->lastUpPerTensordataLen * sizeof(DTYPE_X) % 32)) {
            DataCopy(
                yGm[this->updateTensorIndexCopyOut * this->perUpdateTensorLen +
                    IndexT * this->dataLenPerTile],
                yLocal[0], this->lastUpPerTensordataLen);
          } else {
            uint32_t dLen = this->lastUpPerTensordataLen /
                            (32 / sizeof(DTYPE_X)) * (32 / sizeof(DTYPE_X));
            uint32_t restLen =
                this->lastUpPerTensordataLen % (32 / sizeof(DTYPE_X));

            if (0 != dLen) {
              DataCopy(yGm[this->updateTensorIndexCopyOut *
                               this->perUpdateTensorLen +
                           IndexT * this->dataLenPerTile],
                       yLocal[0], dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              yGm.SetValue(this->updateTensorIndexCopyOut *
                                   this->perUpdateTensorLen +
                               IndexT * this->dataLenPerTile + dLen + i,
                           yLocal.GetValue(dLen + i));
            }
          }

          // 更新索引-- 需要注意
          this->processIGmIndexCopyOut++;
          this->updateTensorIndexCopyOut =
              iGm.GetValue(this->processIGmIndexCopyOut);
        } else { // 不是tensor的最后一部分更新

          // 1.将x的一部分都拷贝到相应位置
          DataCopy(
              yGm[this->updateTensorIndexCopyOut * this->perUpdateTensorLen +
                  IndexT * this->dataLenPerTile],
              yLocal[0], this->dataLenPerTile);
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

  TBuf<QuePosition::VECCALC> xHalfBuf, uHalfBuf, yHalfBuf;

  // TBuf<QuePosition::VECCALC> iBuf, uBuf;
  // GlobalTensor<float> xGm;
  // GlobalTensor<float> yGm;
  // GlobalTensor<float> outGm;

  //   GlobalTensor<DTYPE_VALUE> valueGm;
  //   GlobalTensor<DTYPE_INPUT_DATA> input_dataGm;
  //   GlobalTensor<DTYPE_X2> x2Gm;

  GlobalTensor<DTYPE_X> xGm;
  GlobalTensor<DTYPE_INDEX> iGm;
  GlobalTensor<DTYPE_X> uGm;

  GlobalTensor<DTYPE_X> yGm;

  // float sum; //用于存放sum
  // float sum; //用于存放sum
  // uint32_t dataTotalLength;

  uint32_t dataType;

  int32_t varDataNums;

  uint32_t updateTensorNums;
  uint32_t perUpdateTensorLen;

  uint32_t updateTimesPerTensor;

  uint32_t updateTensorNumsPerTile;
  uint32_t updateTensorNumsLastTile;

  uint32_t dataLenPerTile;
  uint32_t dataLenLastTile;

  uint32_t lastUpPerTensordataLen;

  uint32_t tileNum;
  uint32_t avaiUBDataLen;

  uint32_t processIGmIndex;
  uint32_t updateTensorIndex;

  uint32_t processIGmIndexCopyOut;
  uint32_t updateTensorIndexCopyOut;
};

extern "C" __global__ __aicore__ void scatter_max(GM_ADDR x, GM_ADDR index,
                                                  GM_ADDR update, GM_ADDR x_ref,
                                                  GM_ADDR workspace,
                                                  GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  KernelScatterMax op;

  op.Init(x, index, update, x_ref, tiling_data.dataType,
          tiling_data.varDataNums, tiling_data.updateTensorNums,
          tiling_data.perUpdateTensorLen, tiling_data.updateTimesPerTensor,
          tiling_data.updateTensorNumsPerTile,
          tiling_data.updateTensorNumsLastTile, tiling_data.dataLenPerTile,
          tiling_data.dataLenLastTile, tiling_data.lastUpPerTensordataLen,
          tiling_data.tileNum, tiling_data.avaiUBDataLen);
  op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void scatter_max_do(uint32_t blockDim, void *l2ctrl, void *stream, GM_ADDR x,
                    GM_ADDR index, GM_ADDR update, GM_ADDR x_ref,
                    GM_ADDR workspace, GM_ADDR tiling) {
  scatter_max<<<blockDim, l2ctrl, stream>>>(x, index, update, x_ref, workspace,
                                            tiling);

  aclrtSynchronizeStream(stream);

  // 这个问题需要解决呢？
  // GET_TILING_DATA(tiling_data, tiling);

  //   std::cout << "reduction" << tiling_data.reduction
  //             << ", BlockNum=" << GetBlockNum() << "TILING_KEY_IS(1)"
  //             << TILING_KEY_IS(1) << std::endl;
}
#endif