
#include "lp_norm_v2_tiling.h"
#include "register/op_def_registry.h"

#if 0
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  LpNormV2TilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  context->SetBlockDim(8);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}
#endif

#include "tiling/platform/platform_ascendc.h"
#include <cmath>

namespace optiling {
const uint32_t BLOCK_SIZE = 256;
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  LpNormV2TilingData tiling;
  uint32_t sizeofdatatype;
  uint32_t totalLengthAligned;

  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

  // 获取平台版本号
  auto socVersion = ascendcPlatform.GetSocVersion();

  // 获取UB的大小
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

  // 获取aiv个数
  auto aivNum = ascendcPlatform.GetCoreNumAiv();

  // 获取aic个数
  auto aicNum = ascendcPlatform.GetCoreNumAic();

  const float *p = context->GetAttrs()->GetFloat(0);
  if (true == std::isinf(*p)) {
    if (true == std::signbit(*p)) {
      tiling.set_p(-9999.f);
    } else {
      tiling.set_p(9999.f);
    }
  } else {
    tiling.set_p(*p);
  }
  // tiling.set_p(*p);
  std::cout << "p = " << *p << std::endl;

  const float *epsilon = context->GetAttrs()->GetFloat(1);
  tiling.set_epsilon(*epsilon);
  std::cout << "epsilon = " << *epsilon << std::endl;

  // 1.求出带进行InstanceNorm的轴信息
  // 1.1 获取输入数据的维度形状
  // std::vector<int> axis;

  // 4.获取有多少个Tensor需要更新
  // 4.1获取x中有多少个数据,注意这不是Tensor个数
  uint32_t xDataNums = context->GetInputTensor(0)->GetShapeSize();
  tiling.set_xDataNums(xDataNums);
  std::cout << "xDataNums = " << xDataNums << std::endl;

  const gert::StorageShape *x_shape = context->GetInputShape(0);

  // const char *data_format = context->GetAttrs()->GetStr(0);

  // const gert::TypedContinuousVector<int64_t> *pAxis =
  //     context->GetAttrs()->GetListInt(0);

  // //  auto pAxis =   reinterpret_cast<ContinuousVector *>pAxisType
  // auto axis = pAxis->GetData();

  // axis = *pAxis; //使用解索引的方式

  // const gert::TypedContinuousVector<int64_t> axis =
  //     context->GetAttrs()->GetListInt(0);

  // axis = *pAxis; //使用解索引的方式

  const gert::ContinuousVector *axesCV =
      context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(1);
  size_t axisLen = 0;
  int64_t *axis = {0};
  if (axesCV && axesCV->GetSize() && axesCV->GetData()) {
    axisLen = axesCV->GetSize();
    axis = (int64_t *)axesCV->GetData();
    if (axis && axisLen == 1 && (axis[0] > 5)) {
      axisLen = 0;
    }
  }

  //进行正确性检查
  // int axisLen = pAxis->GetSize();

  std::cout << "axisLen: " << axisLen << std::endl;
  // if (0 == axisLen) {
  //   std::cout << "Input is error, the dim(dim = " << axisLen
  //             << " ) of Input must >= 3. " << std::endl;
  //   return ge::GRAPH_SUCCESS;
  // }

  uint32_t tensorLen = 1;
  uint32_t tensorElementStride = 1;

  if (0 != axisLen) {

    // 打印轴信息
    for (int i = 0; i < axisLen; i++) {
      std::cout << "axis[" << i << "]: " << axis[i] << std::endl;
    }

    if ((axis[axisLen - 1] - axis[0]) != (axisLen - 1)) {
      std::cout << "Input is error, the dim(dim = " << axisLen
                << " ) of Input must be continuous. " << std::endl;
      return ge::GRAPH_SUCCESS;
    }

    // 2.求出待更新的Tensor的长度
    // uint32_t tensorLen = 1;
    for (int i = axis[0]; i <= axis[axisLen - 1]; i++) {
      // totalLength *= x1_shape->GetStorageShape().GetDim(i);

      uint32_t lt = x_shape->GetStorageShape().GetDim(i);
      tensorLen *= lt;
    }

    // 3.求出带带更新的Tensor元素之间的间隔
    // uint32_t tensorElementStride = 1;
    for (int i = axis[axisLen - 1] + 1;
         i < x_shape->GetStorageShape().GetDimNum(); i++) {
      // totalLength *= x1_shape->GetStorageShape().GetDim(i);

      uint32_t lt = x_shape->GetStorageShape().GetDim(i);
      tensorElementStride *= lt;
    }

  } else {

    std::cout << "the input axis is null." << std::endl;
    tensorLen = xDataNums;
    tensorElementStride = 1;
  }

  tiling.set_tensorLen(tensorLen);
  std::cout << "tensorLen = " << tensorLen << std::endl;

  tiling.set_tensorElementStride(tensorElementStride);
  std::cout << "tensorElementStride = " << tensorElementStride << std::endl;

  // 4.2获取有多少个Tensor需要更新
  uint32_t tensorNums = xDataNums / tensorLen;
  tiling.set_tensorNums(tensorNums);
  std::cout << "tensorNums = " << tensorNums << std::endl;

  // 5.计算出每个tile能够计算的Tensor个数。
  // 5.1获取数据类型
  auto dt = context->GetInputTensor(0)->GetDataType();

  tiling.set_dataType(dt); //设置数据类型

  //这个地方应该需要有多个判断, 需要对不同的数据类型进行处理
  switch (dt) {
    // DT_FLOAT
  case 0:
    sizeofdatatype = 4;
    break;

    // DT_FLOAT16
  case 1:
    sizeofdatatype = 2;
    break;

    // DT_INT8 = 2, // int8 type
  case 2:
    sizeofdatatype = 1;
    break;

    // DT_INT32 = 3, // int32 type
  case 3:
    sizeofdatatype = 4;
    break;

  default:
    std::cout << "Not supportted this Datatype" << std::endl;
    return ge::GRAPH_SUCCESS;
    // break;
  }

  // 5.2计算每个tile能够处理的数据块的个数
  // 每个block能够处理的数据个数
  uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;

  uint32_t ub_block_num_real = 0;
  //   if (2 != dt) {

  //     ub_block_num_real =
  //         ((ub_size) / BLOCK_SIZE / (sizeofdatatype * 4)) * 13 /
  //         20; // ub_block在Ascend
  //         C中不能全部被用来作为输入输出，给了13/20系数。
  //   } else {
  //     ub_block_num_real =
  //         ((ub_size) / BLOCK_SIZE / (sizeofdatatype * (4 + 4 * 2))) * 13 /
  //         20; // ub_block在Ascend
  //         C中不能全部被用来作为输入输出，给了13/20系数。
  //   }

  // DT_FLOAT
  if (0 == dt) {
    ub_block_num_real =
        ((ub_size) / BLOCK_SIZE / (sizeofdatatype * 6)) * 13 /
        20; // ub_block在Ascend C中不能全部被用来作为输入输出，给了13/20系数。
  } else {
    ub_block_num_real =
        ((ub_size) / BLOCK_SIZE / (sizeofdatatype * 10)) * 13 /
        20; // ub_block在Ascend C中不能全部被用来作为输入输出，给了13/20系数。
  }

  // uint32_t ub_block_num = 5; //为测试方便，验证代码流程
  uint32_t ub_block_num = ub_block_num_real; //为测试方便，验证代码流程
  if (ub_block_num % 2 != 0) {
    ub_block_num = ub_block_num - 1;
  }

  std::cout << "ub_size:" << ub_size << ", ub_block_num:" << ub_block_num
            << ", aivNum:" << aivNum << std::endl;

  // 5.4 计算每个tile能够更新的Tensor的个数
  // = 每个tile能够处理的最大数据量/每个Tensor的数据长度
  uint32_t tensorNumsPerTile = (ub_block_num * ALIGN_NUM) / tensorLen;
  // uint32_t updateTensorNumsPerTile = 1;

  // 正常情况下每个向量都是能够转入到UB中的,也有可能出现1个Tensor不能够全部装入
  // UB的情况,那就需要每次更新Tensor的一部分。那就涉及到多次更新
  uint32_t updateTimesPerTensor = 1;

  // 不考虑1个UB的长度都不能够装下1个带更新Tensor的情况
  if (0 == tensorNumsPerTile) {
    std::cout << "Not supportted perUpdateVetorLen > available UB size "
              << std::endl;
    // return ge::GRAPH_SUCCESS;

    tensorNumsPerTile = 1;
    updateTimesPerTensor =
        (tensorLen + ub_block_num * ALIGN_NUM - 1) / (ub_block_num * ALIGN_NUM);
  }

  // 每个tile只计算1个Tensor比较简单,特别是对于求和的时候。虽然效率低下
  if (tensorNumsPerTile > 1) {
    tensorNumsPerTile = 1;
  }

  tiling.set_tensorNumsPerTile(tensorNumsPerTile);
  tiling.set_updateTimesPerTensor(updateTimesPerTensor);

  std::cout << "tensorNumsPerTile: " << tensorNumsPerTile << std::endl;
  std::cout << "updateTimesPerTensor: " << updateTimesPerTensor << std::endl;

  // 4.计算tileNum数
  uint32_t tileNum = (tensorNums + tensorNumsPerTile - 1) / tensorNumsPerTile;

  uint32_t dataLenPerTile = 0;
  uint32_t tensorLastUpDataLen = 0;

  uint32_t avaiUBDataLen = ub_block_num * ALIGN_NUM;
  tiling.set_avaiUBDataLen(avaiUBDataLen);

  std::cout << "avaiUBDataLen: " << avaiUBDataLen << std::endl;

  uint32_t tensorNumsLastTile = 0;
  uint32_t dataLenLastTile = 0;

  if (1 == updateTimesPerTensor) {

    // 每次处理的数据长度
    dataLenPerTile = tensorNumsPerTile * tensorLen;

    tensorNumsLastTile = tensorNums - (tileNum - 1) * tensorNumsPerTile;

    dataLenLastTile = tensorNumsLastTile * tensorLen;
  } else {
    dataLenPerTile = avaiUBDataLen;
    tensorLastUpDataLen =
        tensorLen - (updateTimesPerTensor - 1) * dataLenPerTile;
  }

  tiling.set_dataLenPerTile(dataLenPerTile);
  std::cout << "dataLenPerTile: " << dataLenPerTile << std::endl;

  tiling.set_dataLenLastTile(dataLenLastTile);
  std::cout << "dataLenLastTile: " << dataLenLastTile << std::endl;

  tiling.set_tensorNumsLastTile(tensorNumsLastTile);
  std::cout << "tensorNumsLastTile: " << tensorNumsLastTile << std::endl;

  tiling.set_tensorLastUpDataLen(tensorLastUpDataLen);
  std::cout << "tensorLastUpDataLen: " << tensorLastUpDataLen << std::endl;

  // 目前只有1个核,也就是只能分为1个block
  context->SetBlockDim(1);

  tiling.set_tileNum(tileNum);
  std::cout << "tileNum: " << tileNum << std::endl;

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;

  return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *x1_shape = context->GetInputShape(0);
  gert::Shape *y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class LpNormV2 : public OpDef {
public:
  explicit LpNormV2(const char *name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("p").AttrType(OPTIONAL).Float(2);
    this->Attr("axes").AttrType(OPTIONAL).ListInt({});
    this->Attr("keepdim").AttrType(OPTIONAL).Bool(false);
    this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-12);

    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend310b");
  }
};

OP_ADD(LpNormV2);
} // namespace ops
