#include "kernel_operator.h"

#if 0
extern "C" __global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
#endif

using namespace AscendC;
// constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t el_offset = 256;

class KernelLessEqual {
public:
  __aicore__ inline KernelLessEqual() {}
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                              uint32_t blockLength, uint32_t tileNum,
                              uint32_t tileLength, uint32_t lasttileLength,
                              uint32_t formerNum, uint32_t formerLength,
                              uint32_t formertileNum, uint32_t formertileLength,
                              uint32_t formerlasttileLength, uint32_t tailNum,
                              uint32_t tailLength, uint32_t tailtileNum,
                              uint32_t tailtileLength,
                              uint32_t taillasttileLength, uint32_t tilingKey) {
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

    // this->value = valueGm.GetValue(0);
    // this->posAttr = 1.702;  //-1.702
    // this->negAttr = -1.702; //-1.702
    // this->halfAttr = 0.851; // 1.702/2=0.851
    // this->one = 1.0;

    if (tilingKey == 1) {
      this->blockLength = blockLength;
      this->tileNum =
          tileNum ASSERT(tileNum != 0 && "tile num can not be zero!");
      this->tileLength = tileLength / BUFFER_NUM;
      this->lasttileLength = lasttileLength;
      //   this->lasttileLength = lasttileLength / BUFFER_NUM;

      x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1 +
                               this->blockLength * GetBlockIdx(),
                           this->blockLength);

      x2Gm.SetGlobalBuffer((__gm__ DTYPE_X2 *)x2 +
                               this->blockLength * GetBlockIdx(),
                           this->blockLength);

      yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y +
                              this->blockLength * GetBlockIdx(),
                          this->blockLength);
    }

    if (tilingKey == 2) {
      this->formerNum = formerNum;
      this->formerLength = formerLength;
      this->formertileNum = formertileNum;
      this->formertileLength = formertileLength;
      this->formerlasttileLength = formerlasttileLength;

      this->tailNum = tailNum;
      this->tailLength = tailLength;
      this->tailtileNum = tailtileNum;
      this->tailtileLength = tailtileLength;
      this->taillasttileLength = taillasttileLength;

      if (GetBlockIdx() < this->formerNum) { //分到大块核的处理
        this->tileLength = this->formertileLength / BUFFER_NUM;
        this->lasttileLength = this->formerlasttileLength;
        // this->lasttileLength = this->formerlasttileLength / BUFFER_NUM;
        this->tileNum = this->formertileNum * BUFFER_NUM;

        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1 +
                                 this->formerLength * GetBlockIdx(),
                             this->formerLength);

        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X2 *)x2 +
                                 this->formerLength * GetBlockIdx(),
                             this->formerLength);

        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y +
                                this->formerLength * GetBlockIdx(),
                            this->formerLength);

      } else { //分到小块核的处理，需要处理的数据量比大核少alignNum个
        this->tileLength = this->tailtileLength / BUFFER_NUM;
        // this->lasttileLength = this->taillasttileLength;
        this->lasttileLength = this->taillasttileLength / BUFFER_NUM;
        this->tileNum = this->tailtileNum * BUFFER_NUM;

        x1Gm.SetGlobalBuffer(
            (__gm__ DTYPE_X1 *)x1 + this->formerLength * this->formerNum +
                this->tailLength * (GetBlockIdx() - this->formerNum),
            this->tailLength);

        x2Gm.SetGlobalBuffer(
            (__gm__ DTYPE_X2 *)x2 + this->formerLength * this->formerNum +
                this->tailLength * (GetBlockIdx() - this->formerNum),
            this->tailLength);

        yGm.SetGlobalBuffer(
            (__gm__ DTYPE_Y *)y + this->formerLength * this->formerNum +
                this->tailLength * (GetBlockIdx() - this->formerNum),
            this->tailLength);
      }
    }

    this->tileLengthtemp = this->tileLength + el_offset;
    pipe.InitBuffer(inQueueIN, BUFFER_NUM,
                    this->tileLengthtemp * 2 * sizeof(DTYPE_X1));
    pipe.InitBuffer(outQueueOUT, BUFFER_NUM,
                    this->tileLengthtemp * sizeof(DTYPE_Y));
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
    LocalTensor<DTYPE_X1> inLocal = inQueueIN.AllocTensor<DTYPE_X1>();

    if (BUFFER_NUM == 1) {
      if (progress == this->tileNum - 1) {

        if (progress == 0) {
          //如果只有一包，则搬运的起始地址为0，tileLength为实际分块的数据量
          // DataCopy(inLocal[0], x1Gm[progress * this->tileLength],
          //          this->tileLength);

          // DataCopy(inLocal[this->tileLength], x2Gm[progress *
          // this->tileLength],
          //          this->tileLength);

          if (0 == (this->lasttileLength * sizeof(DTYPE_X1)) % 32) {
            DataCopy(inLocal[el_offset], x1Gm[progress * this->tileLength],
                     this->lasttileLength);

            DataCopy(inLocal[this->tileLengthtemp + el_offset],
                     x2Gm[progress * this->tileLength], this->lasttileLength);
          } else {

            uint32_t dLen = this->lasttileLength / (32 / sizeof(DTYPE_X1)) *
                            (32 / sizeof(DTYPE_X1));
            uint32_t restLen = this->lasttileLength % (32 / sizeof(DTYPE_X1));

            if (0 != dLen) {
              DataCopy(inLocal[el_offset], x1Gm[progress * this->tileLength],
                       dLen);
              DataCopy(inLocal[this->tileLengthtemp + el_offset],
                       x2Gm[progress * this->tileLength], dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              inLocal.SetValue(el_offset + dLen + i, x1Gm.GetValue(dLen + i));
              inLocal.SetValue(this->tileLengthtemp + el_offset + dLen + i,
                               x2Gm.GetValue(dLen + i));
            }
          }

        } else {
          //将最后一个分块的起始地址向前移动tileLength-lasttileLength

          DataCopy(
              inLocal[el_offset],
              x1Gm[(progress - 1) * this->tileLength + this->lasttileLength],
              this->tileLength);

          DataCopy(
              inLocal[this->tileLengthtemp + el_offset],
              x2Gm[(progress - 1) * this->tileLength + this->lasttileLength],
              this->tileLength);
        }

      } else {

        DataCopy(inLocal[el_offset], x1Gm[progress * this->tileLength],
                 this->tileLength);

        DataCopy(inLocal[this->tileLengthtemp + el_offset],
                 x2Gm[progress * this->tileLength], this->tileLength);
      }
    }

    if (BUFFER_NUM == 2) {
      //开启double
      // buffer时，由于将输入数据分成了相等的2部分，分块大小为不开启double
      // buffer的一半， 所以需要对最后两个分块数据的起始地址做处理

      if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
          (progress == (this->tileNum * BUFFER_NUM - 1))) {

        // 只有一个分块
        if (progress < 2) {

          DataCopy(inLocal[0], x1Gm[progress * this->tileLength],
                   this->tileLength);

          DataCopy(inLocal[this->tileLength], x2Gm[progress * this->tileLength],
                   this->tileLength);

        } else {

          //分块大小变为tileLength的一半
          //倒数第2个分块数据的起始地址向前移动（tileLength-lasttileLength)，最后一个分块的起始地址以此为基础进行移动

          DataCopy(
              inLocal[0],
              x1Gm[(progress - 2) * (this->tileLength) + this->lasttileLength],
              this->tileLength);

          DataCopy(
              inLocal[this->tileLength],
              x2Gm[(progress - 2) * (this->tileLength) + this->lasttileLength],
              this->tileLength);
        }
      } else {
        DataCopy(inLocal[0], x1Gm[progress * this->tileLength],
                 this->tileLength);

        DataCopy(inLocal[this->tileLength], x2Gm[progress * this->tileLength],
                 this->tileLength);
      }
    }

    inQueueIN.EnQue(inLocal);
  }

  __aicore__ inline void Compute(int32_t progress) {
    LocalTensor<DTYPE_X1> inLocal = inQueueIN.DeQue<DTYPE_X1>();

    LocalTensor<DTYPE_X1> x1Local = inLocal;
    LocalTensor<DTYPE_X1> x2Local = inLocal[this->tileLengthtemp];

    LocalTensor<DTYPE_Y> yLocal = outQueueOUT.AllocTensor<DTYPE_Y>();

    // Compare(yLocal, x1Local, x2Local, CMPMODE::LT, this->tileLength);

    // for (uint32_t i = 0; i < this->tileLengthtemp; i++) {
    //   // float tx1 = (float)(half)x1Local.GetValue(i);
    //   // float tx2 = (float)(half)x2Local.GetValue(i);

    //   float tx1 = x1Local.GetValue(i);
    //   float tx2 = x2Local.GetValue(i);

    //   if (tx1 > tx2) {
    //     yLocal.SetValue(i, false);
    //   } else {
    //     yLocal.SetValue(i, true);
    //   }
    // }

    if (progress != (this->tileNum * BUFFER_NUM - 1)) {
      for (uint32_t i = 0; i < this->tileLength + el_offset; i++) {

        float tx1 = x1Local.GetValue(i);
        float tx2 = x2Local.GetValue(i);

        if (tx1 < tx2) {
          yLocal.SetValue(i, false);
        } else {
          yLocal.SetValue(i, true);
        }
      }

    } else {

      // tileLength=3328; iasttileLength=768; el_offset=256
      // case 1: 执行测试用例2不成功, 有位置随机报错
      // for (uint32_t i = 0; i < this->lasttileLength + el_offset; i++){

      // case 2: 执行测试用例2成功
      for (uint32_t i = 0; i < this->tileLength + el_offset; i++) {

        float tx1 = x1Local.GetValue(i);
        float tx2 = x2Local.GetValue(i);
        if (tx1 < tx2) {
          yLocal.SetValue(i, false);
        } else {
          yLocal.SetValue(i, true);
        }
      }
    }

    outQueueOUT.EnQue<DTYPE_Y>(yLocal);

    inQueueIN.FreeTensor(inLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress) {
    LocalTensor<DTYPE_Y> yLocal = outQueueOUT.DeQue<DTYPE_Y>();

    if (BUFFER_NUM == 1) {
      if (progress == this->tileNum - 1) {

        if (progress == 0) {
          //如果只有一包，则搬运的起始地址为0，tileLength为实际分块的数据量
          // DataCopy(yGm[0], yLocal, this->tileLength);

          if (0 == ((this->lasttileLength * sizeof(DTYPE_Y)) % 32)) {

            DataCopy(yGm[0], yLocal[el_offset], this->lasttileLength);
          } else {

            uint32_t dLen = this->lasttileLength / (32 / sizeof(DTYPE_Y)) *
                            (32 / sizeof(DTYPE_Y));
            uint32_t restLen = this->lasttileLength % (32 / sizeof(DTYPE_Y));

            if (0 != dLen) {
              DataCopy(yGm[0], yLocal[el_offset], dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              yGm.SetValue(dLen + i, yLocal.GetValue(el_offset + dLen + i));
            }
          }

        } else {
          //将最后一个分块的起始地址向前移动tileLength-lasttileLength
          DataCopy(
              yGm[(progress - 1) * this->tileLength + this->lasttileLength],
              yLocal[el_offset], this->tileLength);
        }

      } else {
        DataCopy(yGm[progress * this->tileLength], yLocal[el_offset],
                 this->tileLength);
      }
    }

    if (BUFFER_NUM == 2) {
      //开启double
      // buffer时，由于将输入数据分成了相等的2部分，分块大小为不开启double
      // buffer的一半， 所以需要对最后两个分块数据的起始地址做处理
      if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
          (progress == (this->tileNum * BUFFER_NUM - 1))) {

        if (progress < 2) {
          DataCopy(yGm[progress * (this->tileLength)], yLocal,
                   (this->tileLength));
        } else {
          //分块大小变为tileLength的一半
          //倒数第2个分块数据的起始地址向前移动（tileLength-lasttileLength)，最后一个分块的起始地址以此为基础进行移动
          DataCopy(
              yGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
              yLocal, (this->tileLength));
        }
      } else {
        DataCopy(yGm[progress * (this->tileLength)], yLocal,
                 (this->tileLength));
      }
    }

    outQueueOUT.FreeTensor(yLocal);
  }

private:
  TPipe pipe;
  // TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY, inQueueZ;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIN;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;
  // GlobalTensor<float> xGm;
  // GlobalTensor<float> yGm;
  // GlobalTensor<float> outGm;

  //   GlobalTensor<DTYPE_VALUE> valueGm;
  //   GlobalTensor<DTYPE_INPUT_DATA> input_dataGm;
  //   GlobalTensor<DTYPE_X2> x2Gm;

  GlobalTensor<DTYPE_X1> x1Gm;
  GlobalTensor<DTYPE_X2> x2Gm;
  GlobalTensor<DTYPE_Y> yGm;

  //   DTYPE_X posAttr;  // 1.702
  //   DTYPE_X negAttr;  //-1.702
  //   DTYPE_X halfAttr; // 1.702/2=0.851
  //   DTYPE_X one;
  // float sum; //用于存放sum
  // float sum; //用于存放sum
  // uint32_t dataTotalLength;

  uint32_t tileLengthtemp;

  uint32_t reduction;
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
  uint32_t lasttileLength;

  uint32_t formerNum;
  uint32_t formerLength;
  uint32_t formertileNum;
  uint32_t formertileLength;
  uint32_t formerlasttileLength;

  uint32_t tailNum;
  uint32_t tailLength;
  uint32_t tailtileNum;
  uint32_t tailtileLength;
  uint32_t taillasttileLength;
};

extern "C" __global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2,
                                                 GM_ADDR y, GM_ADDR workspace,
                                                 GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  KernelLessEqual op;

  uint32_t tilingKey = 1;
  if (TILING_KEY_IS(1)) {
    tilingKey = 1;
  } else if (TILING_KEY_IS(2)) {
    tilingKey = 2;
  } else {
    tilingKey = 1;
  }

  op.Init(
      x1, x2, y, tiling_data.blockLength, tiling_data.tileNum,
      tiling_data.tileLength, tiling_data.lasttileLength, tiling_data.formerNum,
      tiling_data.formerLength, tiling_data.formertileNum,
      tiling_data.formertileLength, tiling_data.formerlasttileLength,
      tiling_data.tailNum, tiling_data.tailLength, tiling_data.tailtileNum,
      tiling_data.tailtileLength, tiling_data.taillasttileLength, tilingKey);
  op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void less_equal_do(uint32_t blockDim, void *l2ctrl, void *stream, GM_ADDR x1,
                   GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  less_equal<<<blockDim, l2ctrl, stream>>>(x1, x2, y, workspace, tiling);

  aclrtSynchronizeStream(stream);

  // 这个问题需要解决呢？
  // GET_TILING_DATA(tiling_data, tiling);

  //   std::cout << "reduction" << tiling_data.reduction
  //             << ", BlockNum=" << GetBlockNum() << "TILING_KEY_IS(1)"
  //             << TILING_KEY_IS(1) << std::endl;
}
#endif