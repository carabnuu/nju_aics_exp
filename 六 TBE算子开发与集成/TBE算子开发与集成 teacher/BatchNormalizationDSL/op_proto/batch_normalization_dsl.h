/**
 * Copyright (C)  2020-2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_BATCH_NORMALIZATION_DSL_H
#define GE_OP_BATCH_NORMALIZATION_DSL_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(BatchNormalizationDSL)
    .INPUT(x, TensorType({DT_FLOAT,DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT,DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT,DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT,DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT,DT_FLOAT16}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT,DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT,DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .REQUIRED_ATTR(is_training, Bool)
    .OP_END_FACTORY_REG(BatchNormalizationDSL)
}
#endif //GE_OP_BATCH_NORMALIZATION_DSL_H
