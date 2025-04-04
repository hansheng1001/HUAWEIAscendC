#include "kernel_operator.h"

#if 0
extern "C" __global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
#endif

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
// constexpr int32_t BUFFER_NUM = 1;

class KernelFastGelu {
public:
  __aicore__ inline KernelFastGelu() {}
  __aicore__ inline void
  Init(GM_ADDR x, GM_ADDR y, uint32_t blockLength, uint32_t tileNum,
       uint32_t tileLength, uint32_t lasttileLength, uint32_t formerNum,
       uint32_t formerLength, uint32_t formertileNum, uint32_t formertileLength,
       uint32_t formerlasttileLength, uint32_t tailNum, uint32_t tailLength,
       uint32_t tailtileNum, uint32_t tailtileLength,
       uint32_t taillasttileLength, uint32_t tilingKey) {
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

    // this->value = valueGm.GetValue(0);
    this->negAttr = -1.702; //-1.702
    this->halfAttr = 0.851; // 1.702/2=0.851
    this->one = 1.0;

    if (tilingKey == 1) {
      this->blockLength = blockLength;
      this->tileNum =
          tileNum ASSERT(tileNum != 0 && "tile num can not be zero!");
      this->tileLength = tileLength / BUFFER_NUM;
      this->lasttileLength = lasttileLength;
      //   this->lasttileLength = lasttileLength / BUFFER_NUM;

      xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x +
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

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x +
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

        xGm.SetGlobalBuffer(
            (__gm__ DTYPE_X *)x + this->formerLength * this->formerNum +
                this->tailLength * (GetBlockIdx() - this->formerNum),
            this->tailLength);

        yGm.SetGlobalBuffer(
            (__gm__ DTYPE_Y *)y + this->formerLength * this->formerNum +
                this->tailLength * (GetBlockIdx() - this->formerNum),
            this->tailLength);
      }
    }

    pipe.InitBuffer(inQueueIN, BUFFER_NUM,
                    this->tileLength * 3 * sizeof(DTYPE_X));
    pipe.InitBuffer(outQueueOUT, BUFFER_NUM,
                    this->tileLength * sizeof(DTYPE_Y));
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
      if (progress == this->tileNum - 1) {

        if (progress == 0) {
          //如果只有一包，则搬运的起始地址为0，tileLength为实际分块的数据量
          if (0 == (this->lasttileLength * sizeof(DTYPE_X) % 32)) {
            DataCopy(inLocal[0], xGm[progress * this->tileLength],
                     this->lasttileLength);
          } else {

            uint32_t dLen = this->lasttileLength / (32 / sizeof(DTYPE_X)) *
                            (32 / sizeof(DTYPE_X));
            uint32_t restLen = this->lasttileLength % (32 / sizeof(DTYPE_X));

            if (0 != dLen) {
              DataCopy(inLocal[0], xGm[progress * this->tileLength], dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              inLocal.SetValue(dLen + i, xGm.GetValue(dLen + i));
            }
          }

        } else {
          //将最后一个分块的起始地址向前移动tileLength-lasttileLength

          DataCopy(
              inLocal[0],
              xGm[(progress - 1) * this->tileLength + this->lasttileLength],
              this->tileLength);
        }

      } else {

        DataCopy(inLocal[0], xGm[progress * this->tileLength],
                 this->tileLength);
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

          DataCopy(inLocal[0], xGm[progress * this->tileLength],
                   this->tileLength);

        } else {

          //分块大小变为tileLength的一半
          //倒数第2个分块数据的起始地址向前移动（tileLength-lasttileLength)，最后一个分块的起始地址以此为基础进行移动

          DataCopy(
              inLocal[0],
              xGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
              this->tileLength);
        }
      } else {
        DataCopy(inLocal[0], xGm[progress * this->tileLength],
                 this->tileLength);
      }
    }

    inQueueIN.EnQue(inLocal);
  }

  __aicore__ inline void Compute(int32_t progress) {
    LocalTensor<DTYPE_X> inLocal = inQueueIN.DeQue<DTYPE_X>();

    LocalTensor<DTYPE_X> xLocal = inLocal;

    LocalTensor<DTYPE_X> divDownLocal = inLocal[this->tileLength];
    LocalTensor<DTYPE_X> divUpLocal = inLocal[2 * this->tileLength];

    LocalTensor<DTYPE_Y> yLocal = outQueueOUT.AllocTensor<DTYPE_Y>();

#if 1
    // 方案1:精度有问题
    // 1.求分母
    // 1.1 计算|x|
    Abs(divDownLocal, xLocal, this->tileLength);
    // 1.2 计算-1.702*|x|
    Muls(divDownLocal, divDownLocal, this->negAttr, this->tileLength);
    // 1.2 计算exp(-1.702*|x|)
    Exp(divDownLocal, divDownLocal, this->tileLength);
    // 1.3 计算1.0+exp(-1.702*|x|)
    Adds(divDownLocal, divDownLocal, this->one, this->tileLength);

    // 2.求分子
    // 2.1 计算|x|
    Abs(divUpLocal, xLocal, this->tileLength);
    // 2.2 计算x-|x|
    Sub(divUpLocal, xLocal, divUpLocal, this->tileLength);
    // 2.3 计算0.851*(x-|x|)
    Muls(divUpLocal, divUpLocal, this->halfAttr, this->tileLength);
    // 2.4 计算exp(0.851*(x-|x|))
    Exp(divUpLocal, divUpLocal, this->tileLength);

    // 2.5 计算x*exp(0.851*(x-|x|))
    Mul(divUpLocal, xLocal, divUpLocal, this->tileLength);

    // 3.计算最终值
    Div(yLocal, divUpLocal, divDownLocal, this->tileLength);
#endif

#if 0
    // 方案2:
    // 1.求分母
    // 1.1 计算|x|
    Abs(divDownLocal, xLocal, this->tileLength);
    // 1.2 计算-1.702*|x|
    Muls(divDownLocal, divDownLocal, this->negAttr, this->tileLength);
    // 1.2 计算exp(-1.702*|x|)
    Exp(divDownLocal, divDownLocal, this->tileLength);
    // 1.3 计算1.0+exp(-1.702*|x|)
    Adds(divDownLocal, divDownLocal, this->one, this->tileLength);

    // 2.求分子
    // 2.1 计算|x|
    Abs(divUpLocal, xLocal, this->tileLength);
    // 2.2 计算x-|x|
    Sub(divUpLocal, xLocal, divUpLocal, this->tileLength);
    // 2.3 计算0.851*(x-|x|)
    Muls(divUpLocal, divUpLocal, this->halfAttr, this->tileLength);
    // 2.4 计算exp(0.851*(x-|x|))
    Exp(divUpLocal, divUpLocal, this->tileLength);

    // 3.先计算除法
    Div(yLocal, divUpLocal, divDownLocal, this->tileLength);

    // 4 计算最终值
    Mul(yLocal, yLocal, xLocal this->tileLength);
#endif

    outQueueOUT.EnQue<DTYPE_Y>(yLocal);

    inQueueIN.FreeTensor(inLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress) {
    LocalTensor<DTYPE_Y> yLocal = outQueueOUT.DeQue<DTYPE_Y>();

    if (BUFFER_NUM == 1) {
      if (progress == this->tileNum - 1) {

#if 1
        if (progress == 0) {
          //如果只有一包，则搬运的起始地址为0，tileLength为实际分块的数据量
          // DataCopy(yGm[0], yLocal, this->tileLength);
          if (0 == ((this->lasttileLength * sizeof(DTYPE_Y)) % 32)) {
            // DataCopy(zGm[0], zLocal, this->tileLength);

            DataCopy(yGm[0], yLocal, this->lasttileLength);
          } else {
            // 不能使用copyData,只能够使用DataCopyPad
            // DataCopyExtParams dataCopyParams{
            //     1, (uint32_t)(this->lasttileLength * sizeof(DTYPE_Z)), 0, 0,
            //     0};

            // // 当只有1个tile的时候,tileLength是32byte对齐的长度,
            // // 而lasttileLength存放的是真正的数据长度
            // DataCopyParams dataCopyParams{
            //     1, (uint16_t)(this->lasttileLength * sizeof(DTYPE_Z)), 0, 0};
            // // // dataCopyParams.blockCount = 1;
            // // // dataCopyParams.blockLen = this->tileLength;

            // DataCopyPad(zGm[0], zLocal, dataCopyParams);

            // DataCopy(
            //     zGm[(progress - 1) * this->tileLength +
            //     this->lasttileLength], zLocal, this->tileLength);

            // DataCopy(zGm[0], zLocal, this->tileLength);

            uint32_t dLen = this->lasttileLength / (32 / sizeof(DTYPE_Y)) *
                            (32 / sizeof(DTYPE_Y));
            uint32_t restLen = this->lasttileLength % (32 / sizeof(DTYPE_Y));

            if (0 != dLen) {
              DataCopy(yGm[0], yLocal, dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              yGm.SetValue(dLen + i, yLocal.GetValue(dLen + i));
            }
          }

        } else {
          //将最后一个分块的起始地址向前移动tileLength-lasttileLength
          DataCopy(
              yGm[(progress - 1) * this->tileLength + this->lasttileLength],
              yLocal, this->tileLength);
        }
#endif

      } else {
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
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

  GlobalTensor<DTYPE_X> xGm;
  GlobalTensor<DTYPE_Y> yGm;

  DTYPE_X negAttr;  //-1.702
  DTYPE_X halfAttr; // 1.702/2=0.851
  DTYPE_X one;
  // float sum; //用于存放sum
  // float sum; //用于存放sum
  // uint32_t dataTotalLength;

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

extern "C" __global__ __aicore__ void
fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  KernelFastGelu op;

  uint32_t tilingKey = 1;
  if (TILING_KEY_IS(1)) {
    tilingKey = 1;
  } else if (TILING_KEY_IS(2)) {
    tilingKey = 2;
  } else {
    tilingKey = 1;
  }

  op.Init(
      x, y, tiling_data.blockLength, tiling_data.tileNum,
      tiling_data.tileLength, tiling_data.lasttileLength, tiling_data.formerNum,
      tiling_data.formerLength, tiling_data.formertileNum,
      tiling_data.formertileLength, tiling_data.formerlasttileLength,
      tiling_data.tailNum, tiling_data.tailLength, tiling_data.tailtileNum,
      tiling_data.tailtileLength, tiling_data.taillasttileLength, tilingKey);
  op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void fast_gelu_do(uint32_t blockDim, void *l2ctrl, void *stream, GM_ADDR x,
                  GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  fast_gelu<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);

  aclrtSynchronizeStream(stream);

  // 这个问题需要解决呢？
  // GET_TILING_DATA(tiling_data, tiling);

  //   std::cout << "reduction" << tiling_data.reduction
  //             << ", BlockNum=" << GetBlockNum() << "TILING_KEY_IS(1)"
  //             << TILING_KEY_IS(1) << std::endl;
}
#endif