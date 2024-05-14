
#include "instance_norm_tiling.h"
#include "register/op_def_registry.h"

#if 0
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  InstanceNormTilingData tiling;
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

// 目前为了简单每次只有1个Tensor参加更新。
// 未去考虑Tensor内部元素之间存在连续与不连续元素的情况

// 总体思路:目前只考虑连续维度的InstanceNorm.将这些连续维度的数据看成待更新的Tensor。
// 但是这些Tensor内部元素之间不一定是连续的。
// 是可以考虑0级算子进行计算的,但是由于提供的Block大小太死板,实现起来比较复杂(最小的block只能是32Byte。
// 而大Block似乎也只能够是256或者比他小。因为从按位的Mask我们可以看出来。),所以采用2级算子和SetValue()/
// GetValue()来实现。按照1个Tensor是否能够完全装入到UB中和Tensor内部元素是否连续来进行分情况实现。
// 1.如果1个Tensor能够完全装入到UB中,那么可以使用1个流水完成mean/variance/InstanceNorm3个过程;
// 2.如果1个Tensor不能够完全装入到UB中,那么需要使用3个流水,分别处理mean/variance/InstanceNorm3
// 3.如果1个Tensor内部的元素连续,则可以使用DataCopy进行数据搬运
// 4.如果1个Tensor内部的元素不连续,则需要使用GetValue/SetValue进行数据搬运

// host端伪代码:1.获取需要进行InstanceNorm的轴信息;2.计算带更新Tensor的长度;
// 3.计算带更新Tensor中元素之间的间隔;4.计算带更新Tensor的个数.
// 5.再分为能不能够装入UB进行讨论,分别求出tilelength和lasttileLength。

namespace optiling {
const uint32_t BLOCK_SIZE = 256;
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  InstanceNormTilingData tiling;
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

  const float *epsilon = context->GetAttrs()->GetFloat(0);
  tiling.set_epsilon(*epsilon);
  std::cout << "epsilon = " << *epsilon << std::endl;

  // 1.求出带进行InstanceNorm的轴信息
  // 1.1 获取输入数据的维度形状
  std::vector<int> axis;

  const gert::StorageShape *x_shape = context->GetInputShape(0);

  // const char *data_format = context->GetAttrs()->GetStr(0);

  const std::string data_format = context->GetAttrs()->GetStr(0);
  std::cout << "data_format = " << data_format << std::endl;

  if (data_format == "NDHWC") {
    axis = {1, 2, 3};
  } else if (data_format == "NCDHW") {
    axis = {2, 3, 4};
  } else if (data_format == "NHWC") {
    axis = {1, 2};
  } else if (data_format == "NCHW") {
    axis = {2, 3};
  } else if (data_format == "ND") {
    axis.clear();
    for (int i = 2; i < x_shape->GetStorageShape().GetDimNum(); i++) {
      axis.push_back(i);
    }
  }

  //进行正确性检查
  int axisLen = axis.size();
  if (0 == axisLen) {
    std::cout << "Input is error, the dim(dim = " << axisLen
              << " ) of Input must >= 3. " << std::endl;
    return ge::GRAPH_SUCCESS;
  }

  // 打印轴信息
  for (int i = 0; i < axisLen; i++) {
    std::cout << "axis[" << i << "]: " << axis[i] << std::endl;
  }

  // 2.求出待更新的Tensor的长度
  uint32_t tensorLen = 1;
  for (int i = axis[0]; i <= axis[axisLen - 1]; i++) {
    // totalLength *= x1_shape->GetStorageShape().GetDim(i);

    uint32_t lt = x_shape->GetStorageShape().GetDim(i);
    tensorLen *= lt;
  }
  tiling.set_tensorLen(tensorLen);

  std::cout << "tensorLen = " << tensorLen << std::endl;

  // 3.求出带带更新的Tensor元素之间的间隔
  uint32_t tensorElementStride = 1;
  for (int i = axis[axisLen - 1] + 1;
       i < x_shape->GetStorageShape().GetDimNum(); i++) {
    // totalLength *= x1_shape->GetStorageShape().GetDim(i);

    uint32_t lt = x_shape->GetStorageShape().GetDim(i);
    tensorElementStride *= lt;
  }
  tiling.set_tensorElementStride(tensorElementStride);

  std::cout << "tensorElementStride = " << tensorElementStride << std::endl;

  // 4.获取有多少个Tensor需要更新
  // 4.1获取x中有多少个数据,注意这不是Tensor个数
  uint32_t xDataNums = context->GetInputTensor(0)->GetShapeSize();
  tiling.set_xDataNums(xDataNums);
  std::cout << "xDataNums = " << xDataNums << std::endl;

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

  ub_block_num_real =
      ((ub_size) / BLOCK_SIZE / (sizeofdatatype * 6)) * 13 /
      20; // ub_block在Ascend C中不能全部被用来作为输入输出，给了13/20系数。

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
class InstanceNorm : public OpDef {
public:
  explicit InstanceNorm(const char *name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("gamma")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("beta")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("mean")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("variance")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");
    this->Attr("epsilon").AttrType(OPTIONAL).Float(0.0);

    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend310b");
  }
};

OP_ADD(InstanceNorm);
} // namespace ops
