//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"

#define CNTK_ONNX_MODEL_VERSION 1
#define STRINGIFY(s) #s
#define MACRO_TO_STRING(x) STRINGIFY(x)
const std::string CNTK_ONNX_PRODUCER_NAME = "CNTK";

namespace ONNXIR
{
    class Model;
}

namespace CNTK
{
    class CNTKToONNX
    {
    public:
        static std::unique_ptr<ONNXIR::Model> CreateModel(const FunctionPtr& src);
    };
}