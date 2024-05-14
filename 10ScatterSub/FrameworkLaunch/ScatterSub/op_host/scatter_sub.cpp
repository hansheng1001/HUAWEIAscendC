
#include "register/op_def_registry.h"
#include "scatter_sub_tiling.h"

#if 0
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  ScatterSubTilingData tiling;
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

namespace optiling {
const uint32_t BLOCK_SIZE = 256;
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  ScatterSubTilingData tiling;
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

  // 获取输入数据的总长度,指的是数据长度,不是字节长度?
  //   uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();

  // 采用根据indexs长度进行分解tileNum的方式。这样对于稀疏矩阵的更新能够减少很多的计算量。

  // 获取input_var中有多少个数据,注意这不是Tensor个数
  uint32_t varDataNums = context->GetInputTensor(0)->GetShapeSize();
  tiling.set_varDataNums(varDataNums);

  std::cout << "varDataNums = " << varDataNums << std::endl;

  // 0.对var、index和update的维度是否满足要求进行检测。
  // 就是判断var.shape ==  index.shape +
  // update.shape[1:]---这个不进行,默认所有的都是满足的

  // 1.求出每个待更新Tensor的数据量。直接从update中去获取
  const gert::StorageShape *x_shape = context->GetInputShape(2);
  int32_t perUpdateTensorLen = 1;
  for (int i = 1; i < x_shape->GetStorageShape().GetDimNum(); i++) {
    // totalLength *= x1_shape->GetStorageShape().GetDim(i);

    int32_t lt = x_shape->GetStorageShape().GetDim(i);
    perUpdateTensorLen *= lt;
  }
  tiling.set_perUpdateTensorLen(perUpdateTensorLen);

  std::cout << "perUpdateTensorLen = " << perUpdateTensorLen << std::endl;

  // 2.算出需要更新的TensorNums,直接从index中去获取
  // const gert::StorageShape *index_shape = context->GetInputShape(1);

  // int32_t updateTensorNums = 1;
  // for (int i = 0; i < index_shape->GetStorageShape().GetDimNum(); i++) {
  //   // totalLength *= x1_shape->GetStorageShape().GetDim(i);

  //   int32_t lt = index_shape->GetStorageShape().GetDim(i);
  //   updateTensorNums *= lt;
  // }
  uint32_t updateTensorNums = context->GetInputTensor(1)->GetShapeSize();
  tiling.set_updateTensorNums(updateTensorNums);

  std::cout << "updateTensorNums = " << updateTensorNums << std::endl;

  // 3.计算出每个tile能够计算的数据量。
  // 3.1获取数据类型
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

  // 3.2计算每个tile能够处理的数据块的个数
  // 每个block能够处理的数据个数
  uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;

  uint32_t ub_block_num_real = 0;
  if (2 != dt) {

    ub_block_num_real =
        ((ub_size) / BLOCK_SIZE / (sizeofdatatype * 3)) * 13 /
        20; // ub_block在Ascend C中不能全部被用来作为输入输出，给了13/20系数。
  } else {
    ub_block_num_real =
        ((ub_size) / BLOCK_SIZE / (sizeofdatatype * (3 + 3 * 2))) * 13 /
        20; // ub_block在Ascend C中不能全部被用来作为输入输出，给了13/20系数。
  }

  // uint32_t ub_block_num = 5; //为测试方便，验证代码流程
  uint32_t ub_block_num = ub_block_num_real; //为测试方便，验证代码流程
  if (ub_block_num % 2 != 0) {
    ub_block_num = ub_block_num - 1;
  }

  std::cout << "ub_size:" << ub_size << ", ub_block_num:" << ub_block_num
            << ", aivNum:" << aivNum << std::endl;

  // 3.4 计算每个tile能够更新的Tensor的个数
  // = 每个tile能够处理的最大数据量/每个Tensor的数据长度
  uint32_t updateTensorNumsPerTile =
      (ub_block_num * ALIGN_NUM) / perUpdateTensorLen;
  // uint32_t updateTensorNumsPerTile = 1;

  // 正常情况下每个向量都是能够转入到UB中的,也有可能出现1个Tensor不能够全部装入
  // UB的情况,那就需要每次更新Tensor的一部分。那就涉及到多次更新
  uint32_t updateTimesPerTensor = 1;

  // 不考虑1个UB的长度都不能够装下1个带更新Tensor的情况
  if (0 == updateTensorNumsPerTile) {
    std::cout << "Not supportted perUpdateVetorLen > available UB size "
              << std::endl;
    // return ge::GRAPH_SUCCESS;

    updateTensorNumsPerTile = 1;
    updateTimesPerTensor = (perUpdateTensorLen + ub_block_num * ALIGN_NUM - 1) /
                           (ub_block_num * ALIGN_NUM);
  }

  // 虽然会降低效率,但是可以防止同1次更新时存在相同索引会覆盖前面索引更新的bug
  // 当存在相同索引时需要进行依次更新
  if (updateTensorNumsPerTile > 1) {
    updateTensorNumsPerTile = 1;
  }

  tiling.set_updateTensorNumsPerTile(updateTensorNumsPerTile);
  tiling.set_updateTimesPerTensor(updateTimesPerTensor);

  std::cout << "updateTensorNumsPerTile: " << updateTensorNumsPerTile
            << std::endl;
  std::cout << "updateTimesPerTensor: " << updateTimesPerTensor << std::endl;

  // 4.计算tileNum数
  uint32_t tileNum = (updateTensorNums + updateTensorNumsPerTile - 1) /
                     updateTensorNumsPerTile;

  uint32_t dataLenPerTile = 0;
  uint32_t lastUpPerTensordataLen = 0;

  uint32_t avaiUBDataLen = ub_block_num * ALIGN_NUM;
  tiling.set_avaiUBDataLen(avaiUBDataLen);

  std::cout << "avaiUBDataLen: " << avaiUBDataLen << std::endl;

  uint32_t updateTensorNumsLastTile = 0;
  uint32_t dataLenLastTile = 0;

  if (1 == updateTimesPerTensor) {

    // 每次处理的数据长度
    dataLenPerTile = updateTensorNumsPerTile * perUpdateTensorLen;

    updateTensorNumsLastTile =
        updateTensorNums - (tileNum - 1) * updateTensorNumsPerTile;

    dataLenLastTile = updateTensorNumsLastTile * perUpdateTensorLen;
  } else {
    dataLenPerTile = avaiUBDataLen;
    lastUpPerTensordataLen =
        perUpdateTensorLen - (updateTimesPerTensor - 1) * dataLenPerTile;

    // 更新需要的tile数
    tileNum = tileNum * updateTimesPerTensor;
  }

  tiling.set_dataLenLastTile(dataLenLastTile);
  std::cout << "dataLenLastTile: " << dataLenLastTile << std::endl;

  tiling.set_updateTensorNumsLastTile(updateTensorNumsLastTile);
  std::cout << "updateTensorNumsLastTile: " << updateTensorNumsLastTile
            << std::endl;

  // 目前只有1个核,也就是只能分为1个block
  context->SetBlockDim(1);

  tiling.set_tileNum(tileNum);
  tiling.set_dataLenPerTile(dataLenPerTile);
  tiling.set_lastUpPerTensordataLen(lastUpPerTensordataLen);

  std::cout << "tileNum: " << tileNum << std::endl;
  std::cout << "dataLenPerTile: " << dataLenPerTile << std::endl;
  std::cout << "lastUpPerTensordataLen: " << lastUpPerTensordataLen
            << std::endl;

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
class ScatterSub : public OpDef {
public:
  explicit ScatterSub(const char *name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("index")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("update")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);

    // this->Input("x")
    //     .ParamType(REQUIRED)
    //     .DataType({ge::DT_INT8})
    //     .Format({ge::FORMAT_ND})
    //     .UnknownShapeFormat({ge::FORMAT_ND});
    // this->Input("index")
    //     .ParamType(REQUIRED)
    //     .DataType({ge::DT_INT32})
    //     .Format({ge::FORMAT_ND})
    //     .UnknownShapeFormat({ge::FORMAT_ND});
    // this->Input("update")
    //     .ParamType(REQUIRED)
    //     .DataType({ge::DT_INT8})
    //     .Format({ge::FORMAT_ND})
    //     .UnknownShapeFormat({ge::FORMAT_ND});
    // this->Output("x")
    //     .ParamType(REQUIRED)
    //     .DataType({ge::DT_INT8})
    //     .Format({ge::FORMAT_ND})
    //     .UnknownShapeFormat({ge::FORMAT_ND});
    // this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);

    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend310b");
  }
};

OP_ADD(ScatterSub);
} // namespace ops
