
#include "register/tilingdata_base.h"

#if 0
namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterMaxTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterMax, ScatterMaxTilingData)
}
#endif

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterMaxTilingData)

TILING_DATA_FIELD_DEF(uint32_t, varDataNums);

TILING_DATA_FIELD_DEF(uint32_t, updateTensorNums);
TILING_DATA_FIELD_DEF(uint32_t, perUpdateTensorLen);

TILING_DATA_FIELD_DEF(uint32_t, dataType);

TILING_DATA_FIELD_DEF(uint32_t, updateTimesPerTensor);

TILING_DATA_FIELD_DEF(uint32_t, updateTensorNumsPerTile);
TILING_DATA_FIELD_DEF(uint32_t, updateTensorNumsLastTile);

TILING_DATA_FIELD_DEF(uint32_t, dataLenPerTile);
TILING_DATA_FIELD_DEF(uint32_t, dataLenLastTile);

TILING_DATA_FIELD_DEF(uint32_t, lastUpPerTensordataLen);

TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(uint32_t, avaiUBDataLen);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterMax, ScatterMaxTilingData)
} // namespace optiling