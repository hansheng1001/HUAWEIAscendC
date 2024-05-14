#include "kernel_operator.h"

#if 0
extern "C" __global__ __aicore__ void spence(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
#endif

// #include <cmath>

#if 0
#include "mconf.h"
#ifdef UNK
/* 用于有理函数逼近的系数 A 和 B */
static double A[8] = {
    4.65128586073990045278E-5, 7.31589045238094711071E-3,
    1.33847639578309018650E-1, 8.79691311754530315341E-1,
    2.71149851196553469920E0,  4.25697156008121755724E0,
    3.29771340985225106936E0,  1.00000000000000000126E0,
};
static double B[8] = {
    6.90990488912553276999E-4, 2.54043763932544379113E-2,
    2.82974860602568089943E-1, 1.41172597751831069617E0,
    3.63800533345137075418E0,  5.03278880143316990390E0,
    3.54771340985225096217E0,  9.99999999999999998740E-1,
};
#endif

#ifdef ANSIPROT
/* 声明需要用到的其他数学函数 */
extern double fabs(double);
extern double log(double);
extern double polevl(double, void *, int);
#else
double fabs(), log(), polevl();
#endif

#endif

using namespace AscendC;
// constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t el_offset = 256;

constexpr float MINLOGF = -103.278929903431851103; /* log(2^-149) */

constexpr float SQRTHF = 0.707106781186547524;

class KernelSpence {
public:
  __aicore__ float frexpf(float x, int *pw2) {
    union {
      float y;
      unsigned short i[2];
    } u;
    int i, k;
    short *q;

    u.y = x;

    q = (short *)&u.i[1];

    /* find the exponent (power of 2) */

    i = (*q >> 7) & 0xff;
    if (i == 0) {
      if (u.y == 0.0f) {
        *pw2 = 0;
        return (0.0f);
      }

      do {
        u.y *= 2.0f;
        i -= 1;
        k = (*q >> 7) & 0xff;
      } while (k == 0);
      i = i + k;
    }

    i -= 0x7e;
    *pw2 = i;
    *q &= 0x807f; /* strip all exponent bits */
    *q |= 0x3f00; /* mantissa between 0.5 and 1 */
    return (u.y);
  }

  __aicore__ float logf(float xx) {
    // register float y;
    float y;
    float x, z, fe;
    int e;

    x = xx;
    fe = 0.0f;
    /* Test for domain */
    if (x <= 0.0f) {

      return (MINLOGF);
    }

    x = frexpf(x, &e);
    if (x < SQRTHF) {
      e -= 1;
      x = x + x - 1.0f; /*  2x - 1  */
    } else {
      x = x - 1.0f;
    }
    z = x * x;
    /* 3.4e-9 */
    /*
    p = logfcof;
    y = *p++ * x;
    for( i=0; i<8; i++ )
            {
            y += *p++;
            y *= x;
            }
    y *= z;
    */

    y = ((((((((7.0376836292E-2f * x - 1.1514610310E-1f) * x +
               1.1676998740E-1f) *
                  x -
              1.2420140846E-1f) *
                 x +
             1.4249322787E-1f) *
                x -
            1.6668057665E-1f) *
               x +
           2.0000714765E-1f) *
              x -
          2.4999993993E-1f) *
             x +
         3.3333331174E-1f) *
        x * z;

    if (e) {
      fe = e;
      y += -2.12194440e-4f * fe;
    }

    y += -0.5f * z; /* y - 0.5 x^2 */
    z = x + y;      /* ... + x  */

    if (e)
      z += 0.693359375f * fe;

    return (z);
  }

  __aicore__ float polevlA(float x, int N) {
    float ans;
    int i;
    float *p;

    float A[8] = {
        4.65128586073990045278E-5, 7.31589045238094711071E-3,
        1.33847639578309018650E-1, 8.79691311754530315341E-1,
        2.71149851196553469920E0,  4.25697156008121755724E0,
        3.29771340985225106936E0,  1.00000000000000000126E0,
    };

    p = A;
    ans = *p++;
    i = N;

    do
      ans = ans * x + *p++;
    while (--i);

    return ans;
  }

  __aicore__ float polevlB(float x, int N) {
    float ans;
    int i;
    float *p;

    float B[8] = {
        6.90990488912553276999E-4, 2.54043763932544379113E-2,
        2.82974860602568089943E-1, 1.41172597751831069617E0,
        3.63800533345137075418E0,  5.03278880143316990390E0,
        3.54771340985225096217E0,  9.99999999999999998740E-1,
    };

    p = B;
    ans = *p++;
    i = N;

    do
      ans = ans * x + *p++;
    while (--i);

    return ans;
  }

  // extern double PI, MACHEP;

  // extern float PI;

#define PIFS 1.64493406684822643647f

  __aicore__ float mySpence(float xx) {
    float x, w, y, z;
    int flag;

    x = xx;
    if (x < 0.0f) {
      // mtherr("spencef", DOMAIN);
      return (0.0f);
    }

    if (x == 1.0f)
      return (0.0f);

    if (x == 0.0f)
      return (PIFS);

    flag = 0;

    if (x > 2.0f) {
      x = 1.0f / x;
      flag |= 2;
    }

    if (x > 1.5f) {
      w = (1.0f / x) - 1.0f;
      flag |= 2;
    }

    else if (x < 0.5f) {
      w = -x;
      flag |= 1;
    }

    else
      w = x - 1.0f;

    y = -w * polevlA(w, 7) / polevlB(w, 7);

    if (flag & 1)
      y = PIFS - logf(x) * logf(1.0f - x) - y;

    if (flag & 2) {
      z = logf(x);
      y = -0.5f * z * z - y;
    }

    return (y);
  }

  __aicore__ inline KernelSpence() {}
  __aicore__ inline void
  Init(GM_ADDR x, GM_ADDR y, uint32_t blockLength, uint32_t tileNum,
       uint32_t tileLength, uint32_t lasttileLength, uint32_t formerNum,
       uint32_t formerLength, uint32_t formertileNum, uint32_t formertileLength,
       uint32_t formerlasttileLength, uint32_t tailNum, uint32_t tailLength,
       uint32_t tailtileNum, uint32_t tailtileLength,
       uint32_t taillasttileLength, uint32_t tilingKey) {
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

    // this->value = valueGm.GetValue(0);

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
    this->tileLengthtemp = this->tileLength + el_offset;
    pipe.InitBuffer(inQueueIN, BUFFER_NUM,
                    this->tileLengthtemp * 3 * sizeof(DTYPE_X));
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
    LocalTensor<DTYPE_X> inLocal = inQueueIN.AllocTensor<DTYPE_X>();

    if (BUFFER_NUM == 1) {
      if (progress == this->tileNum - 1) {

        if (progress == 0) {
          //如果只有一包，则搬运的起始地址为0，tileLength为实际分块的数据量
          // DataCopy(inLocal[el_offset], xGm[progress * this->tileLength],
          //          this->tileLength);

          if (0 == (this->lasttileLength * sizeof(DTYPE_X)) % 32) {
            DataCopy(inLocal[el_offset], xGm[progress * this->tileLength],
                     this->lasttileLength);

          } else {

            uint32_t dLen = this->lasttileLength / (32 / sizeof(DTYPE_X)) *
                            (32 / sizeof(DTYPE_X));
            uint32_t restLen = this->lasttileLength % (32 / sizeof(DTYPE_X));

            if (0 != dLen) {
              DataCopy(inLocal[el_offset], xGm[progress * this->tileLength],
                       dLen);
            }

            for (uint32_t i = 0; i < restLen; i++) {
              inLocal.SetValue(el_offset + dLen + i, xGm.GetValue(dLen + i));
            }
          }

        } else {
          //将最后一个分块的起始地址向前移动tileLength-lasttileLength

          DataCopy(
              inLocal[el_offset],
              xGm[(progress - 1) * this->tileLength + this->lasttileLength],
              this->tileLength);
        }

      } else {

        DataCopy(inLocal[el_offset], xGm[progress * this->tileLength],
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

    // LocalTensor<DTYPE_X> divDownLocal = inLocal[this->tileLength];
    // LocalTensor<DTYPE_X> divUpLocal = inLocal[2 * this->tileLength];

    LocalTensor<DTYPE_Y> yLocal = outQueueOUT.AllocTensor<DTYPE_Y>();

    for (uint32_t i = 0; i < this->tileLengthtemp; i++) {
      // uint32_t index = progress*

      float tX = (float)xLocal.GetValue(i);
      float tY = mySpence(tX);

      yLocal.SetValue(i, (DTYPE_Y)tY);
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
          // DataCopy(yGm[0], yLocal[el_offset], this->tileLength);
          if (0 == ((this->lasttileLength * sizeof(DTYPE_Y)) % 32)) {
            // DataCopy(zGm[0], zLocal, this->tileLength);

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

  GlobalTensor<DTYPE_X> xGm;
  GlobalTensor<DTYPE_Y> yGm;

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

extern "C" __global__ __aicore__ void
spence(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  KernelSpence op;

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
void spence_do(uint32_t blockDim, void *l2ctrl, void *stream, GM_ADDR x,
               GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  spence<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);

  aclrtSynchronizeStream(stream);

  // 这个问题需要解决呢？
  // GET_TILING_DATA(tiling_data, tiling);

  //   std::cout << "reduction" << tiling_data.reduction
  //             << ", BlockNum=" << GetBlockNum() << "TILING_KEY_IS(1)"
  //             << TILING_KEY_IS(1) << std::endl;
}
#endif