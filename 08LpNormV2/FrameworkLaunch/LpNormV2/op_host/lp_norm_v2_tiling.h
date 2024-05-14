
#include "register/tilingdata_base.h"

#if 0
namespace optiling {
BEGIN_TILING_DATA_DEF(LpNormV2TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LpNormV2, LpNormV2TilingData)
}
#endif

namespace optiling {
BEGIN_TILING_DATA_DEF(LpNormV2TilingData)
TILING_DATA_FIELD_DEF(float, p);

TILING_DATA_FIELD_DEF(float, epsilon);

TILING_DATA_FIELD_DEF(uint32_t, dataType);

TILING_DATA_FIELD_DEF(uint32_t, xDataNums);
TILING_DATA_FIELD_DEF(uint32_t, tensorNums);
TILING_DATA_FIELD_DEF(uint32_t, tensorLen);

TILING_DATA_FIELD_DEF(uint32_t, tensorElementStride);
TILING_DATA_FIELD_DEF(uint32_t, updateTimesPerTensor);

TILING_DATA_FIELD_DEF(uint32_t, tensorNumsPerTile);
TILING_DATA_FIELD_DEF(uint32_t, tensorNumsLastTile);

TILING_DATA_FIELD_DEF(uint32_t, dataLenPerTile);
TILING_DATA_FIELD_DEF(uint32_t, dataLenLastTile);

TILING_DATA_FIELD_DEF(uint32_t, tensorLastUpDataLen);

TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(uint32_t, avaiUBDataLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LpNormV2, LpNormV2TilingData)
} // namespace optiling
