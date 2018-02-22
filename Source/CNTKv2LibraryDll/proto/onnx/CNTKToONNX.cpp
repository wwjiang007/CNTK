//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKToONNX.h"
#include "proto/onnx/core/model.h"
#include "proto/onnx/core/graph.h"
#include "proto/onnx/core/status.h"

#include "Utils.h"
#include "Operators.h"
#include "BlockFunction.h"
#include <vector>
#include <tuple>
#include <numeric>

using namespace CNTK::ONNX;
using namespace CNTK;

// TODO: hardcoded sequnce length
const int SequenceLen = 20;
onnx::TypeProto TensorShapeProtoToTypeProto(const onnx::TensorShapeProto* inputShape)
{
    onnx::TypeProto newShape;
    int inputRank = inputShape->dim_size();
    for (int index = 0; index < inputRank; index++)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(inputShape->dim(index).dim_value());

    return newShape;
}

bool HasSequenceAxis(Variable operand)
{
    return (operand.DynamicAxes().size() - (operand.HasBatchAxis() ? 1 : 0)) > 0;
}

//
// Helper function to reduce the rank of a shape.
//
onnx::TypeProto ReduceRank(const onnx::TensorShapeProto* inputShape, int reductionRank, bool rightReduction)
{
    assert(inputShape != nullptr);

    int inputRank = inputShape->dim_size();
    assert(inputRank > reductionRank);

    onnx::TypeProto newShape;
    int64_t reduceDim = 1;

    if (rightReduction)
    {
        for (int index = 0; index < (inputRank - reductionRank); index++)
            newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(inputShape->dim(index).dim_value());

        for (int index = (inputRank - reductionRank); index < inputRank; index++)
            reduceDim *= inputShape->dim(index).dim_value();

        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(reduceDim);
    }
    else
    {
        for (int index = 0; index < reductionRank; index++)
            reduceDim *= inputShape->dim(index).dim_value();

        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(reduceDim);

        for (int index = reductionRank; index < inputRank; index++)
            newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(inputShape->dim(index).dim_value());
    }

    return newShape;
}

namespace CNTK
{

    class CNTKToONNXHelper
    {
    public:
        //
        // Copy the entire CNTK graph to ONNX graph.
        //
        static void Copy(const FunctionPtr& src, ONNXIR::Graph* dst);

    private:
        //
        // Recursively create ONNX nodes corresponding to each CNTK node.
        //
        static ONNXIR::Node* CreateNode(const FunctionPtr& src,
            ONNXIR::Graph* graph,
            std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
            std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::unordered_map<Variable, Variable>& compositeOutputsMap);

        static ONNXIR::Node *AddArgMaxNode(const ONNXIR::NodeArg &nodeArg, ONNXIR::Graph* graph);
        static ONNXIR::Node *AddCastNode(const ONNXIR::NodeArg &nodeArg, ONNXIR::Graph* graph);
        static ONNXIR::Node *AddReshapeNodeToCNTKFunction(const FunctionPtr &src, ONNXIR::Node* node, const std::vector<int> &shape, ONNXIR::Graph* graph);
        static ONNXIR::Node* CreateLSTMRecurrenceNode(ONNXIR::Graph* graph, const std::string &nodeName);

        static ONNXIR::Node* CreateOptimizedRNNStackNode(const FunctionPtr& src,
                ONNXIR::Graph* graph,
                std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
                std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
                const std::unordered_map<Variable, Variable>& compositeOutputsMap);
        static ONNXIR::Node* CreateLSTMNode(const FunctionPtr& src,
            ONNXIR::Graph* graph,
            std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
            std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::unordered_map<Variable, Variable>& compositeOutputsMap);
        static void PrepareInput(const Variable &X, std::vector<ONNXIR::NodeArg> &nodeInputs);
        static void PrepareInitialState(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::vector<Variable> &initialVariables, int batchSize, int cellSize, 
            const std::string &uid, std::vector<ONNXIR::NodeArg> &nodeInputs);

        //static void PrepareWeightNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
        //    const std::vector<Variable> &Ws, std::vector<ONNXIR::NodeArg> &nodeInputs, int r, int c);
        static void PrepareWeightNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::vector<Variable> &Ws, float *stabilizerConstants, std::vector<ONNXIR::NodeArg> &nodeInputs);
        static void PrepareBiasNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::vector<Variable> &Ws, std::vector<ONNXIR::NodeArg> &nodeInputs);

        static ONNXIR::Node* CreateConstantNode(
            ONNXIR::Graph* graph, const Variable &variable, ONNXIR::NodeArg &inputArg, onnx::TypeProto &inputArgType);

        static bool BreadthFirstTraverseGetIndixedNodes(const FunctionPtr& src,
            std::map<std::vector<int>, std::string> &nodeIndices,
            std::map<std::string, Variable> variablesMap);
        //
        // Traverse the entire graph and collect variable mapping between graph inside and outside the block.
        //
        static void TraverseGraph(const FunctionPtr& src,
            std::set<FunctionPtr>& visited,
            std::unordered_map<Variable, Variable>& compositeOutputsMap);

        //
        // Copy the content of NDArrayView to TensorProto, and do the needed
        // convergence.
        //
        static void CopyTensor(const NDArrayViewPtr src, onnx::TensorProto& dst, onnx::TypeProto *inputArgType = nullptr);
        static void FillTensorWithScalar(const std::vector<NDArrayViewPtr>& src, onnx::TensorProto& dst, const std::vector<int> dstShape);

        static void CopyTensorsWithCNTKToONNXLSTMWeightLayoutConversion(const std::vector<NDArrayViewPtr> &src, float *stabilizerConstants,
            onnx::TensorProto& dst, onnx::TypeProto *inputArgType /*=nullptr*/);
        //
        // Copy supported attributes from CNTK node to corresponding ONNX node.
        //
        static void CopyAttributes(const FunctionPtr& src, ONNXIR::Node* node);

        //
        // Convert Axis object to actual tensor index.
        //
        static int ToIndex(const Axis& axis);

        //
        // Convert NDShape and various std::vector types to TensorShape
        //
        static onnx::TypeProto ToTypeProto(const NDShape& shape, int dynamicAxisCount);
        static onnx::TypeProto ToTypeProto(const NDShape& shape, bool hasBatchAxis = false, bool hasSequenceAxis = false);
        static onnx::TypeProto ToTypeProto(const std::vector<bool>& shape);
        static onnx::TypeProto ToTypeProto(const std::vector<int>& shape, bool doReverseVec = true);
        static onnx::TypeProto ToTypeProto(const std::vector<Axis>& axes);

        //
        // Convert TypeProto, NDShape and various std::vector types to std::vector
        //
        static std::vector<int64_t> ToINTS(const onnx::TypeProto& shape);
        static std::vector<int64_t> ToINTS(const NDShape& shape, bool hasBatchAxis = false);
        static std::vector<int64_t> ToINTS(const std::vector<bool>& shape);
        static std::vector<int64_t> ToINTS(const std::vector<int>& shape, bool doReverseVec = true);
        static std::vector<int64_t> ToINTS(const std::vector<Axis>& axes);

        static std::vector<float> INTSToVecFloat(const std::vector<int64_t> &ints);
        static std::vector<int64_t> ConvertPermutationCNTKToONNX(const std::vector<Axis> &axes, bool hasBatchAxis);

        //
        // Convert data types from CNTK to ONNX.
        //
        static void UpdateONNXType(DataType dataType, onnx::TypeProto& type);

        //
        // Map CNTK OP names to ONNX OP Names.
        //
        static std::string ToOPName(const FunctionPtr& src);

        static bool OpInputsHasBatchAxis(const FunctionPtr& src);

        //
        // Which input to ignore during converting a CNTK block to a primitive OP in ONNX.
        //
        static bool FilterInput(const FunctionPtr& src, const CNTK::Variable& input, size_t inputIndex);

        //
        // Converts axis (in CNTK C++ API sense) to index in ONNX sense
        //
        static int64_t ConvertAxisToOnnx(const Axis &axis, const Variable &operand);

        //
        // Converts axes (in CNTK C++ API sense) to index in ONNX sense
        //
        static std::vector<int64_t> ConvertAxesToOnnx(const std::vector<Axis> &axes, const Variable &operand);

        //
        // Given input tersors of a CNTK elementwise operation, figure out
        // input shapes for ONNX operation.
        // It also returns whether broadcast is required and the axis for broadcast.
        // Due to the fact that ONNX only allows braodcast of right-hand-side,
        // inputs may need to be swapped. In this case the last bool is true.
        static std::tuple<std::pair<std::vector<int>, std::vector<int>>, bool, int, bool> AdjustForBroadcastShape(
            const Variable &input1, const Variable &input2);

        static std::tuple<std::vector<int>, bool, int, bool > CalculateBroadcastAxis(
            const std::vector<int> &dims1, const std::vector<int> &dims2);

        //
        // Argument orders between CNTK and ONNX aren't always the same.
        //
        static std::vector<ONNXIR::NodeArg> MapInputsOrderToONNX(const FunctionPtr& src, const std::vector<ONNXIR::NodeArg>& inputs);

        static std::vector<int> GetVariableONNXShape(const Variable &operand);

        //
        // Add current CNTK node to ONNX graph.
        //
        static ONNXIR::Node* AddNode(const FunctionPtr& src, ONNXIR::Graph* graph, const std::vector<ONNXIR::NodeArg>& inputs, const std::vector<ONNXIR::NodeArg>& outputs);

        //
        // Get ONNX 'pads' attribute value based on CNTK node's autoPadding attribute value.
        //
        static std::pair<std::vector<int>, std::vector<int> > GetONNXPadsAttributeFromCNTKNode(
            const NDShape &inputShape,
            const std::vector<bool>& cntkAutoPadding, const NDShape& kernelShape, bool ceilOutDim);

        //
        // Adds attributes 'auto_pad' or 'pads' to saved node (typically convolution or pooling).
        //
        static void PutAutopadOrPadAttrInNode(ONNXIR::Node* node, const NDShape &inputShape,
            const std::vector<bool>& autoPadding,
            const NDShape& kernelShape, bool ceilOutDim = false);

        //
        // A helper function, to reverse any iterable container and return a copy
        // of the reversed container.
        //
        template<typename ItrType>
        static ItrType reverse(ItrType v)
        {
            std::reverse(std::begin(v), std::end(v));
            return v;
        }

        template<class T, class V>
        static inline std::vector<V> Cast(const std::vector<T>& v)
        {
            std::vector<V> result;
            result.reserve(v.size());
            for (auto d : v)
                result.push_back((V)d);
            return result;
        }
    };
}

void UpdateSequenceDim(ONNXIR::Graph* graph, int sequenceCount)
{
    const std::vector<const ONNXIR::NodeArg*> &inputs = graph->GetInputs();
    for (int i = 0; i < inputs.size(); i++)
    {
        ONNXIR::NodeArg* a = const_cast<ONNXIR::NodeArg *>(inputs[i]);
        if (a->Name().find("Input") != -1)
        {
            TensorShapeProto* shape = const_cast<TensorShapeProto*>(a->Shape());
            shape->mutable_dim(0)->set_dim_value(sequenceCount);
        }
    }

    //const std::vector<const ONNXIR::NodeArg*> &outputs = graph->GetOutputs();
    //for (int i = 0; i < outputs.size(); i++)
    //{
    //    ONNXIR::NodeArg* a = const_cast<ONNXIR::NodeArg *>(outputs[i]);
    //    if (a->Name().find("Output") != -1)
    //    {
    //        TensorShapeProto* shape = const_cast<TensorShapeProto*>(a->Shape());
    //        shape->mutable_dim(0)->set_dim_value(sequenceCount);
    //    }
    //}
}

std::unique_ptr<ONNXIR::Model> CNTKToONNX::CreateModel(const FunctionPtr& src)
{
    std::unique_ptr<ONNXIR::Model> model(new ONNXIR::Model("CNTKGraph", true));
    auto dstGraph = model->MainGraph();
    CNTKToONNXHelper::Copy(src, dstGraph);
    ONNXIR::Common::Status status = dstGraph->Resolve();
    if (!status.Ok())
        LogicError("%s", status.ErrorMessage().c_str());


    // TODO: experiment code update Sequence count for RNN
    // UpdateSequenceDim(dstGraph, 3);

    model->SetModelversion(static_cast<ONNXIR::VERSION>(CNTK_ONNX_MODEL_VERSION)); // This is the default. Should be surfaced as graph's 'save' API input.
    model->SetProducerVersion(CNTK_ONNX_PRODUCER_VERSION);
    model->SetProducerName(CNTK_ONNX_PRODUCER_NAME);
    return model;
}

void CNTKToONNXHelper::Copy(const FunctionPtr& src, ONNXIR::Graph* dst)
{
    std::set<FunctionPtr> visited;
    std::unordered_map<Variable, Variable> compositeOutputsMap;
    std::unordered_map<FunctionPtr, ONNXIR::Node*> functionNodes;
    std::unordered_map<Variable, ONNXIR::Node*> variableNodes;

    //
    // Traverse the graph and collect some information.
    //
    TraverseGraph(src, visited, compositeOutputsMap);

    //
    // Iterate through each node in CNTK graph and create an equivalent node
    // in ONNX graph.
    //
    CreateNode(src, dst, functionNodes, variableNodes, compositeOutputsMap);
}

template<typename DType>
void AppendCNTKBiasWeightToONNXTensor(DType *data, const NDShape &shape, onnx::TensorProto& dst)
{
    auto totalSize = shape.TotalSize();
    int cell_size = shape[0] / 4;
    for (size_t targetIndex = 0; targetIndex < totalSize; targetIndex++)
    {
        int row = targetIndex;

        // TODO: specific to LSTM. icfo (CNTK) to iofc(ONNX)
        int block = row / cell_size;
        if (block == 1)
        {
            // c
            row += 2 * cell_size;
        }
        else if (block == 3)
        {
            // o
            row -= 2 * cell_size;
        }

        // soruce is collmn major
        int src_index = row;
        if (typeid(DType) == typeid(float))
            *(dst.mutable_float_data()->Add()) = (float)data[src_index];
        else if (typeid(DType) == typeid(double))
            *(dst.mutable_double_data()->Add()) = (double)data[src_index];
        else
            NOT_IMPLEMENTED;
    }

    // ONNX requires bias being 8 * cell_size with separated Wb and Rb for each gate.
    // CNTK only have bias applied to input side. put zeros for hidden side. 
    // It is numerically equivalent.
    for (size_t targetIndex = 0; targetIndex < totalSize; targetIndex++)
    {
        if (typeid(DType) == typeid(float))
            *(dst.mutable_float_data()->Add()) = 0;
        else if (typeid(DType) == typeid(double))
            *(dst.mutable_double_data()->Add()) = 0;
        else
            NOT_IMPLEMENTED;
    }
}

template<typename DType>
void AppendCNTKWeightToONNXTensor(DType *data, const NDShape &shape, onnx::TensorProto& dst, float stabilizer)
{
    if (shape.Rank() == 1)
    {
        AppendCNTKBiasWeightToONNXTensor(data, shape, dst);
        return;
    }

    auto totalSize = shape.TotalSize();
    for (size_t targetIndex = 0; targetIndex < totalSize; targetIndex++)
    {
        int cell_size = shape[0] / 4;
        int input_size = shape[1];

        bool rowMajor = true;
        int row, col;
        if (rowMajor)
        {
            // row major layout
            row = targetIndex / input_size;
            col = targetIndex % input_size;
        }
        else
        {
            row = targetIndex % (cell_size * 4);
            col = targetIndex / (cell_size * 4);
        }

        // TODO: specific to LSTM. icfo (CNTK) to iofc(ONNX)
        int block = row / cell_size;
        if (block == 1)
        {
            // c
            row += 2 * cell_size;
        }
        else if (block == 3)
        {
            // o
            row -= 2 * cell_size;
        }

        // soruce is collum major
        int src_index = 4 * cell_size * col + row;
        if (typeid(DType) == typeid(float))
            *(dst.mutable_float_data()->Add()) = (float)(data[src_index] * stabilizer);
        else if(typeid(DType) == typeid(double))
            *(dst.mutable_double_data()->Add()) = (double)(data[src_index] * stabilizer);
        else 
            NOT_IMPLEMENTED;
    }
}

void CNTKToONNXHelper::CopyTensorsWithCNTKToONNXLSTMWeightLayoutConversion(const std::vector<NDArrayViewPtr> &src, float *stabilizerConstants,
    onnx::TensorProto& dst, onnx::TypeProto *inputArgType /*=nullptr*/)
{
    // TODO: all NDArrayViewPtr shall have the same shape and data types.
    if (src.empty())
    {
        // TODO: error
        return;
    }
    auto dataType = src[0]->GetDataType();
    switch (dataType)
    {
    case DataType::Float:
        dst.set_data_type(onnx::TensorProto_DataType_FLOAT);
        break;
    case DataType::Double:
        dst.set_data_type(onnx::TensorProto_DataType_DOUBLE);
        break;
    default:
        NOT_IMPLEMENTED;
    }

    for (int i = 0; i < src.size(); i++)
    {
        auto srcTemp = src[i]->DeepClone();
        auto srcShape = srcTemp->Shape();

        // This is our own copy so move it to the CPU.
        srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());

        float stabilizer = stabilizerConstants != nullptr ? stabilizerConstants[i] : 1;

        switch (dataType)
        {
        case DataType::Float:
        {
            auto data = srcTemp->DataBuffer<float>();
            AppendCNTKWeightToONNXTensor(data, srcShape, dst, stabilizer);
            break;
        }
        case DataType::Double:
        {
            auto data = srcTemp->DataBuffer<double>();
            AppendCNTKWeightToONNXTensor(data, srcShape, dst, stabilizer);
            break;
        }
        default:
            NOT_IMPLEMENTED;
        }
    }

    // use 
    if (inputArgType != nullptr)
    {
        std::vector<int64_t> dimensions = CNTKToONNXHelper::ToINTS(*inputArgType);
        for (auto dim : dimensions)
            *(dst.mutable_dims()->Add()) = dim;
    }
    else
    {
        if (src.size() > 1)
            *(dst.mutable_dims()->Add()) = src.size();
        auto dimensions = CNTKToONNXHelper::reverse(src[0]->Shape().Dimensions());
        for (auto dim : dimensions)
            *(dst.mutable_dims()->Add()) = dim;
    }
}

void CNTKToONNXHelper::CopyTensor(const NDArrayViewPtr src, onnx::TensorProto& dst, onnx::TypeProto *inputArgType /*=nullptr*/)
{
    auto dataType = src->GetDataType();
    auto srcTemp = src->DeepClone();
    auto srcShape = srcTemp->Shape();
    auto totalSize = srcShape.TotalSize();

    // This is our own copy so move it to the CPU.
    srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());

    switch (dataType)
    {
    case DataType::Float:
    {
        dst.set_data_type(onnx::TensorProto_DataType_FLOAT);
        auto data = srcTemp->DataBuffer<float>();
        for (size_t index = 0; index < totalSize; index++)
        { 
            *(dst.mutable_float_data()->Add()) = data[index];
        }

        break;
    }
    case DataType::Double:
    {
        dst.set_data_type(onnx::TensorProto_DataType_DOUBLE);
        auto data = srcTemp->DataBuffer<double>();
        for (size_t index = 0; index < totalSize; index++)
            *(dst.mutable_double_data()->Add()) = data[index];

        break;
    }
    default:
        NOT_IMPLEMENTED;
    }

    // use 
    if (inputArgType != nullptr)
    {
        std::vector<int64_t> dimensions = CNTKToONNXHelper::ToINTS(*inputArgType);
        for (auto dim : dimensions)
            *(dst.mutable_dims()->Add()) = dim;
    }
    else
    {
        auto dimensions = CNTKToONNXHelper::reverse(srcShape.Dimensions());
        for (auto dim : dimensions)
            *(dst.mutable_dims()->Add()) = dim;
    }
}

int CNTKToONNXHelper::ToIndex(const Axis& axis)
{
    if ((axis == Axis::AllAxes()) || (axis == Axis::AllStaticAxes()))
        LogicError("AllAxes and AllStaticAxes are currently not supported.");

    if (axis.IsSequenceAxis())
        LogicError("Sequence axis are currently not supported.");

    if (axis.IsBatchAxis())
        return 0;

    return axis.StaticAxisIndex() + 1;
}

onnx::TypeProto CNTKToONNXHelper::ToTypeProto(const NDShape& shape, int dynamicAxisCount)
{
    onnx::TypeProto newShape;
    if (shape.HasUnboundDimension())
        LogicError("Inferred and FreeDimension aren't currently supported.");

    for (int i = 0; i < dynamicAxisCount; i++)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    auto dimensions = reverse(shape.Dimensions());
    for (auto dimension : dimensions)
    {
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);
    }

    return newShape;
}

onnx::TypeProto CNTKToONNXHelper::ToTypeProto(const NDShape& shape, bool hasBatchAxis, bool hasSequenceAxis)
{
    onnx::TypeProto newShape;
    if (shape.HasUnboundDimension())
        LogicError("Inferred and FreeDimension aren't currently supported.");

    if (hasBatchAxis)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    if (hasSequenceAxis)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    auto dimensions = reverse(shape.Dimensions());
    for (auto dimension : dimensions)
    {
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);
    }

    return newShape;
}

onnx::TypeProto CNTKToONNXHelper::ToTypeProto(const std::vector<bool>& shape)
{
    onnx::TypeProto newShape;
    auto dimensions = reverse(shape);
    for (auto dimension : dimensions)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension ? 1 : 0);

    return newShape;
}

onnx::TypeProto CNTKToONNXHelper::ToTypeProto(const std::vector<int>& shape,
    bool doReverseVec /* = true*/)
{
    onnx::TypeProto newShape;
    std::vector<int> dimensions(shape);
    if (doReverseVec)
        dimensions = reverse(dimensions);
    for (auto dimension : dimensions)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);

    return newShape;
}

onnx::TypeProto CNTKToONNXHelper::ToTypeProto(const std::vector<Axis>& axes)
{
    std::vector<int> axesValue;
    for (auto axis : axes)
    {
        axesValue.push_back(ToIndex(axis));
    }
    std::sort(axesValue.begin(), axesValue.end());

    onnx::TypeProto newShape;
    for (auto dimension : axesValue)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);

    return newShape;
}

// this method is to undo an idempotent convertion in sanitize_permutation:
// Find the permutation such that when it is applied to the reverse
// of an input gives the reverse of perm applied to the input
// Example:
// input is[a, b, c, d], perm is[3, 0, 2, 1], perm of input is[d, a, c, b]
// we are looking for[2, 1, 3, 0] because when we apply it to[d, c, b, a]
// the result is[b, c, a, d] which is the revese of[d, a, c, b]
std::vector<int64_t> CNTKToONNXHelper::ConvertPermutationCNTKToONNX(const std::vector<Axis> &axes, bool hasBatchAxis)
{
    std::vector<int64_t> permutation(axes.size());
    for (int i = 0; i < axes.size(); i++)
    {
        int indexToONNXPermTable = axes.size() - i - 1;
        int axisIndexInCNTK = axes[i].StaticAxisIndex();
        int axisIndexInONNX = axes.size() - axisIndexInCNTK - 1;
        permutation[indexToONNXPermTable] = axisIndexInONNX;
    }
    if (hasBatchAxis)
    {
        for (int i = 0; i < permutation.size(); i++)
            permutation[i]++;
        permutation.insert(permutation.begin(), 0);
    }
    return permutation;
}

std::vector<float> CNTKToONNXHelper::INTSToVecFloat(const std::vector<int64_t> &ints)
{
    std::vector<float> vecFloat(ints.size());
    for (int i = 0; i < ints.size(); i++)
    {
        vecFloat[i] = (float)ints[i];
    }

    return vecFloat;
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const onnx::TypeProto& shape)
{
    std::vector<int64_t> newShape;

    for (int i = 0; i < shape.tensor_type().shape().dim_size(); i++)
        newShape.push_back((int64_t)shape.tensor_type().shape().dim(i).dim_value());

    return newShape;
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const NDShape& shape, bool hasBatchAxis)
{
    return ToINTS(ToTypeProto(shape, hasBatchAxis));
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const std::vector<bool>& shape)
{
    return ToINTS(ToTypeProto(shape));
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const std::vector<int>& shape,
    bool doReverseVec /* = true*/)
{
    return ToINTS(ToTypeProto(shape, doReverseVec));
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const std::vector<Axis>& axes)
{
    return ToINTS(ToTypeProto(axes));
}

void CNTKToONNXHelper::UpdateONNXType(DataType dataType, onnx::TypeProto& type)
{
    switch (dataType)
    {
    case DataType::Float:
        type.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_FLOAT);
        break;
    case DataType::Double:
        type.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_DOUBLE);
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

std::string CNTKToONNXHelper::ToOPName(const FunctionPtr& src)
{
    auto lookup = Operators::CntkToONNXLookup();
    assert(lookup.count(src->OpName()) != 0);

    std::string opName = ToString(src->OpName());
    if (lookup.count(src->OpName()) == 1)
    {
        auto attributesMap = lookup.find(src->OpName())->second.map;
        opName = attributesMap[src->OpName()];
    }
    else
    {
        // Some nodes map one to many.
        if (src->OpName() == L"Convolution")
        {
            auto transpose = (bool)src->Attributes()[L"transpose"].Value<bool>();
            if (transpose)
                opName = "ConvTranspose";
            else
                opName = "Conv";
        }
        else if (src->OpName() == L"Pooling")
        {
            PoolingType poolingType = (PoolingType)src->Attributes()[L"poolingType"].Value<size_t>();
            if (poolingType == PoolingType::Max)
                opName = "MaxPool";
            else
                opName = "AveragePool";
        }
        else if (src->OpName() == L"ReduceElements")
        {
            wstring cntkAttributeOpName = (wstring)src->Attributes()[PrimitiveFunction::AttributeNameReductionOpName].Value<wstring>();

            const AttributesMapping& attributeMap = Operators::FindAttributeMap(src->OpName(), cntkAttributeOpName);

            opName = attributeMap.map.at(cntkAttributeOpName);
        }
    }

    return opName;
}

// whether this op has any input with batch axis
bool CNTKToONNXHelper::OpInputsHasBatchAxis(const FunctionPtr& src)
{
    std::vector<Variable> inputs = src->Inputs();
    for (std::vector<Variable>::const_iterator it = inputs.cbegin(); it != inputs.cend(); it++)
    {
        if ((*it).HasBatchAxis())
            return true;
    }
    return false;
}

bool CNTKToONNXHelper::FilterInput(const FunctionPtr& src, const CNTK::Variable& input, size_t inputIndex)
{
    // In CNTK block functions, they expose all constants inside the block. For block functions that
    // map directly to ONNX OP, we don't care about constanst inside the block.
    if (input.IsConstant())
        return !Operators::IsValidInputs(src->OpName(), inputIndex);
    return false;
}

/*
CNTK python static axis is zero based. Free/Inferred axis is not static.
ONNX batch axis, if exists, is 0. in this case static axes start from 1.
CNTK cpp get static axis in a dis-normalized form (e.g. -axis - 1)
In general CNTK node attribute contains axis in this dis-normalized form.
This function converts dis-normalized form to ONNX form.
*/
int64_t CNTKToONNXHelper::ConvertAxisToOnnx(const Axis &axis, const Variable &operand)
{
    if (axis.IsBatchAxis())
    {
        if (operand.DynamicAxes().size() == 1)
            return 0;
        else if (operand.DynamicAxes().size() == 2)
            return 1;
        else
            LogicError("Inconsitant Axis in ConvertAxisToOnnx");
    }
    else if (axis.IsSequenceAxis())
    {
        return 0;
    }

    NDShape inputShape = operand.Shape();
    Axis normalizedAxis = NormalizeStaticAxis(const_cast<Axis &>(axis), inputShape.Rank());
    int64_t ax = inputShape.Rank() - normalizedAxis.StaticAxisIndex() - 1;
    ax += operand.DynamicAxes().size();
    return ax;
}

std::vector<int64_t> CNTKToONNXHelper::ConvertAxesToOnnx(const std::vector<Axis> &axes, const Variable &operand)
{
    if (std::any_of(axes.cbegin(), axes.cend(), [](const Axis &axis) {return axis== Axis::AllStaticAxes(); }))
    {
        std::vector<int64_t> onnxAxes;
        for (int i = 0; i < operand.Shape().Rank(); i++)
        {
            onnxAxes.push_back(i + operand.DynamicAxes().size());
        }
        return onnxAxes;
    }

    std::vector<int64_t> onnxAxes(axes.size());
    for (int i = 0; i < axes.size(); i++)
    {
        onnxAxes[i] = ConvertAxisToOnnx(axes[i], operand);
    }
    return onnxAxes;
}

/*
ONNX specifies braodcast for elementwise ops in following manners
shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
shape(A) = (2, 3, 4, 5), shape(B) = (5,)
shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

CNTK handles braodcast implicitely as numpy does. For example with above example 4,
the shape of the shall be:
(1, 3, 4, 1) or (3, 4, 1)

more general cases:
same rank:
case1: [a, b, c] + [1, b, 1] - broadcast
case1: [a, b, c] + [a, b, 1] - broadcast
case1: [a, b, c] + [1, b, c] - broadcast
case2: [1, b, 1] + [a, b, c] - swap to become case1 then broadcast
case2: [a, b, 1] + [a, b, c] - swap to become case1 then broadcast
case2: [1, b, c] + [a, b, c] - swap to become case1 then broadcast
case3: [a2, b, c2] + [a, b, c]: cannot broadcast

different ranks:
[a, b, c] + [b, 1]: reshape input[1] to [1, b, 1] to become case 1
[a, b, c] + [b, c]: reshape input[1] to [1, b, c] to become case 1

[b, 1] + [a, b, c]: reshape input[0] to [1, b, 1] to become case 2
[b, c] + [a, b, c]: reshape input[0] to [1, b, c] to become case 2

[a2, b, c2] + [b, c]: reshape input[1] to [1, b, c] to become case 3 (cannot broadcast)
[b, c2] + [a, b, c]: reshape input[0] to [1, b, c2] to become case 3 (cannot broadcast)

Note that there is an addition batch dimension at the front of the shape in ONNX.

*/
std::tuple<std::pair<std::vector<int>, std::vector<int>>, bool, int, bool> CNTKToONNXHelper::AdjustForBroadcastShape(
    const Variable &input1, const Variable &input2)
{
    bool broadcast;
    int axis = 0;
    NDShape shape1 = input1.Shape(), shape2 = input2.Shape();
    bool swapInput = false;

    bool hasAnyBatchAxis = input1.HasBatchAxis() || input2.HasBatchAxis();
    bool hasAnySequenceAxis = HasSequenceAxis(input1) || HasSequenceAxis(input2);

    // CNTK and ONNX dimensions are reversed.
    // Reverse the dimension so that broadcast and axis calculation is in ONNX sense.
    std::vector<int> dims1(reverse(Cast<size_t, int>(shape1.Dimensions())));
    std::vector<int> dims2(reverse(Cast<size_t, int>(shape2.Dimensions())));

    if ((shape1.TotalSize() > 1 && shape2.TotalSize() == 1) || (shape1.TotalSize() == 1 && shape2.TotalSize() > 1))
    {
        broadcast = true;
        swapInput = (shape1.TotalSize() == 1 && shape2.TotalSize() > 1);

        if (swapInput)
            std::swap(dims1, dims2);
        
        if (hasAnySequenceAxis)
            dims1.insert(dims1.begin(), 1);
        if (hasAnyBatchAxis)
            dims1.insert(dims1.begin(), 1);

        return make_tuple(std::pair<std::vector<int>, std::vector<int>>(dims1, dims2), broadcast, axis, swapInput);
    }

    if (shape1.Rank() < shape2.Rank())
    {
        // This is a case of [b, c] + [a, b, c].
        // Need to swap the inputs to fit into ONNX spec - only right-hand-side argument will be broadcasted.
        std::swap(dims1, dims2);
        swapInput = true;
    }

    if (dims1.size() > dims2.size())
    {
        // This is a case like [a, b, c] + [b, 1]. Make it [a, b, c] + [1, b, 1].
        dims2.insert(dims2.begin(), dims1.size() - dims2.size(), 1);
    }

    // Append batch dimension if needed.
    if (hasAnySequenceAxis)
    {
        dims1.insert(dims1.begin(), 1);
        dims2.insert(dims2.begin(), 1);
    }
    if (hasAnyBatchAxis)
    {
        dims1.insert(dims1.begin(), 1);
        dims2.insert(dims2.begin(), 1);
    }

    bool swapInputDueToDims;
    std::tie<std::vector<int>, bool, int>(dims2, broadcast, axis, swapInputDueToDims) = CalculateBroadcastAxis(dims1, dims2);

    if (broadcast && swapInput && swapInputDueToDims)
    {
        LogicError("Shapes of elementwise binary operation are not compatible.");
    }

    return make_tuple(std::pair<std::vector<int>, std::vector<int>>(dims1, dims2), broadcast, axis, swapInput || swapInputDueToDims);
}

/*
For example with:
case1: [a, b, c] + [ b, 1] - broadcast
broadcast shape = [b], broadcast = true, axis = 1
*/
std::tuple<std::vector<int>, bool, int, bool> CNTKToONNXHelper::CalculateBroadcastAxis(
    const std::vector<int> &dims1, const std::vector<int> &dims2)
{
    bool swapInput = false;
    // this method assumes dims1.size() == dims2.size(), which is granted by caller AdjustForBroadcastShape.
    bool broadCast = false;
    int axis_start = -1;
    int axis_stop = dims2.size();
    for (int i = 0; i < dims2.size(); i++)
    {
        if (dims1[i] != dims2[i])
        {
            if (dims1[i] == 1)
                swapInput = true;

            broadCast = true;
            if (axis_start != -1)
            {
                axis_stop = i;
                break;
            }
        }
        else
            if (dims2[i] != 1 && axis_start == -1)
            {
                axis_start = i;
            }
    }

    if (!broadCast)
    {
        return make_tuple(dims2, broadCast, axis_start, swapInput);
    }

    axis_start = axis_start > 0 ? axis_start : 0;

    const std::vector<int> broadcaseInputDims = swapInput ? dims1 : dims2;
    // sanity check;
    for (int i = 0; i < broadcaseInputDims.size(); i++)
    {
        if ((i < axis_start || i >= axis_stop) && broadcaseInputDims[i] != 1)
        {
            LogicError("dimension %d cannot be broadcasted", i);
        }
        else if (i >= axis_start && i < axis_stop && dims1[i] != dims2[i])
        {
            LogicError("dimension %d cannot be broadcasted", i);
        }
    }
    std::vector<int> dimensions;
    for (int i = axis_start; i < axis_stop; i++)
    {
        dimensions.push_back(broadcaseInputDims[i]);
    }

    return make_tuple(dimensions, broadCast, axis_start, swapInput);
}

bool CNTKToONNXHelper::BreadthFirstTraverseGetIndixedNodes(const FunctionPtr& src, 
    std::map<std::vector<int>, std::string> &nodeIndices,
    std::map<std::string, Variable> foundVariablesMap)
{
    std::vector<std::pair<std::vector<int>, Variable>> front, nextFront;
    std::vector<Variable> inputs = src->Inputs();
    for (int i = 0; i < inputs.size(); i++)
    {
        front.push_back({ {i}, inputs[i] });
    }

    std::unordered_set<Variable> visited;
    while (!front.empty() && !nodeIndices.empty())
    {
        for (std::vector<std::pair<std::vector<int>, Variable>>::iterator itFront = front.begin(); itFront != front.end(); itFront++)
        {
            Variable &input = itFront->second;
            std::vector<int> &inputIndices = itFront->first;
            std::map<std::vector<int>, std::string>::const_iterator itFoundNode = nodeIndices.find(inputIndices);
            if (itFoundNode != nodeIndices.end())
            {
                foundVariablesMap.insert({ itFoundNode->second, input });
                nodeIndices.erase(itFoundNode);
            }
            if (input.IsOutput())
            {
                std::vector<Variable> nodeInputs = input.Owner()->Inputs();
                for (int j = 0; j < nodeInputs.size(); j++)
                {
                    if (visited.find(nodeInputs[j]) == visited.end())
                    {
                        inputIndices.push_back(j);
                        nextFront.push_back(std::pair<std::vector<int>, Variable>(inputIndices, nodeInputs[j]));
                        inputIndices.pop_back();
                        visited.insert(nodeInputs[j]);
                    }
                }
            }
        }
        front = nextFront;
        nextFront.clear();
    }

    return nodeIndices.empty();
}

ONNXIR::Node* CNTKToONNXHelper::CreateLSTMRecurrenceNode(ONNXIR::Graph* graph, const std::string &nodeName)
{
    // TODO: completed implentation
    std::vector<ONNXIR::NodeArg> orderedInputs;
    std::vector<ONNXIR::NodeArg> outputs;
    // inputs: X(T), W(T), R(T), B(o, T), sequence_lens(o, tensor(int32)), initial_h(o, T), initial_c(o, T), P(o, T)
    // outputs: Y(o, T), Y_h, Y_c
    
    ONNXIR::Node *node = graph->AddNode(nodeName, "LSTM", "", orderedInputs, outputs);
    return node;
}

ONNXIR::Node* CNTKToONNXHelper::CreateOptimizedRNNStackNode(const FunctionPtr& src,
    ONNXIR::Graph* graph,
    std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
    std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    std::vector<Parameter> parameters = src->Parameters();

    std::string rnn_name = ToString(src->Name()); 
    auto input_var = src->Inputs()[0]; 
    // int hidden_size = src->RootFunction()->Attributes()[L"hiddenSize"].Value<int>(); 
    int num_layers = src->RootFunction()->Attributes()[L"numLayers"].Value<int>();
    // bool bidirectional = src->RootFunction()->Attributes()[L"bidirectional"].Value<bool>();
    std::wstring recurrent_op = src->RootFunction()->Attributes()[L"recurrentOp"].Value<std::wstring>();
    
    int input_size = 1;
    if (input_var.Shape().Rank() != 0)
        input_size = input_var.Shape()[0];

    // int num_gates = 1;

    if (recurrent_op == L"lstm")
    {
        ONNXIR::Node *node = nullptr;
        for (int layer = 0; layer < num_layers; layer++)
        {
            if (layer == 0)
                node = CreateLSTMRecurrenceNode(graph, "");
            else
                node = CreateLSTMRecurrenceNode(graph, "");
        }
    }
    else if (recurrent_op == L"rnnReLU")
    {
    }
    else if (recurrent_op == L"rnnTanh")
    {

    }
    else
    {
        LogicError("Unsupported recurrent_op value %S", recurrent_op.c_str());
    }
    return nullptr;
}

template <typename FunctionType>
void TraverseGraphWithPrePostActions(FunctionPtr cntkFunction, std::unordered_set<FunctionPtr>& visitedFunctions,
    FunctionType preFunctor, FunctionType postFunctor)
{
    visitedFunctions.insert(cntkFunction);
    preFunctor(cntkFunction);

    std::vector<Variable> functionInputs = cntkFunction->Inputs();
    for (const auto& input : functionInputs)
    {
        if (input.IsOutput() && visitedFunctions.find(input.Owner()) == visitedFunctions.end())
        {
            const auto& inputFunction = input.Owner();
            TraverseGraphWithPrePostActions(inputFunction, visitedFunctions, preFunctor, postFunctor);
        }
    }

    postFunctor(cntkFunction);
}

ONNXIR::Node* CNTKToONNXHelper::CreateConstantNode(
    ONNXIR::Graph* graph, const Variable &variable, ONNXIR::NodeArg &inputArg, onnx::TypeProto &inputArgType)
{
    std::string inputName = inputArg.Name();

    std::vector<ONNXIR::NodeArg> varInputs;
    std::vector<ONNXIR::NodeArg> varOutputs;

    varOutputs.push_back({ inputArg });
    ONNXIR::Node* variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);
    auto srcTensor = variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value();

    onnx::TensorProto dstTensor;
    CopyTensor(srcTensor, dstTensor, &inputArgType);

    variableNode->AddAttribute("value", dstTensor);
    return variableNode;
}

void TraceLSTMPathes(const FunctionPtr& src,
    std::vector<FunctionPtr>& ht_it_path,
    std::vector<FunctionPtr>& ht_bit_path,
    std::vector<FunctionPtr>& ht_ft_path,
    std::vector<FunctionPtr>& ht_ot_path,
    bool &goBackwards)
{
    // src has to be an LSTM node. 
    std::vector<Variable> inputVars = src->Inputs();
    int pastValueInputCount = (int)std::count_if(inputVars.begin(), inputVars.end(),
        [](Variable& input) {return input.Owner() != nullptr && input.Owner()->OpName() == L"PastValue"; });
    int futureValueInputCount = (int)std::count_if(inputVars.begin(), inputVars.end(),
        [](Variable& input) {return input.Owner() != nullptr && input.Owner()->OpName() == L"FutureValue"; });

    if (pastValueInputCount == 2 && futureValueInputCount == 0)
    {
        goBackwards = false;
    }
    else if (pastValueInputCount == 0 && futureValueInputCount == 2)
    {
        goBackwards = true;
    }
    else
    {
        CNTK::LogicError("Node %s (%s) is not a valid LSTM node", ToString(src->Name()).c_str(), ToString(src->Uid()).c_str());
    }

    std::unordered_set<FunctionPtr> visitedFunctions;
    for (std::vector<Variable>::const_iterator it = inputVars.begin(); it != inputVars.end(); it++)
    {
        visitedFunctions.insert(it->Owner());
    }

    std::vector<std::vector<FunctionPtr>> pathesToPlusSlice;
    std::vector<FunctionPtr> currentPath;


    TraverseGraphWithPrePostActions(src->BlockRoot(),
        visitedFunctions,
        (std::function<void(const FunctionPtr&)>)[&pathesToPlusSlice, &currentPath](const FunctionPtr& function)
    {
        currentPath.push_back(function);
        if (function->OpName() == L"Slice")
        {
            FunctionPtr functionSource = function->Inputs()[0].Owner();
            if (functionSource->OpName() == L"Plus")
            {
                pathesToPlusSlice.push_back(currentPath);
            }
        }
    },
        (std::function<void(const FunctionPtr&)>)[&currentPath](const FunctionPtr& function)
    {
        currentPath.pop_back();
    });

    if (pathesToPlusSlice.size() != 4)
    {
        CNTK::LogicError("pathesToPlusSlice.size() != 4");
    }

    std::sort(pathesToPlusSlice.begin(), pathesToPlusSlice.end(),
        [](const std::vector<FunctionPtr>& path1, const std::vector<FunctionPtr>& path2)
    {
        FunctionPtr slice1 = *path1.rbegin();
        FunctionPtr slice2 = *path2.rbegin();
        int beginIndex1 = slice1->Attributes()[PrimitiveFunction::AttributeNameBeginIndex].Value<int>();
        int beginIndex2 = slice2->Attributes()[PrimitiveFunction::AttributeNameBeginIndex].Value<int>();
        return beginIndex1 < beginIndex2;
    });

    ht_it_path = pathesToPlusSlice[0];
    ht_bit_path = pathesToPlusSlice[1];
    ht_ft_path = pathesToPlusSlice[2];
    ht_ot_path = pathesToPlusSlice[3];
}

bool IsSupportedRNNActivation(const std::wstring &cntkOpName)
{
    static std::vector<std::wstring> supportedRNNActivations(
        {
            L"ReLU",
            L"Tanh",
            L"StableSigmoid"
        });
    return std::find(supportedRNNActivations.cbegin(), supportedRNNActivations.cend(), cntkOpName) !=
        supportedRNNActivations.cend();
}

std::string FindActivation(const std::vector<FunctionPtr> &path, int nth)
{
    int count = 0;
    for (std::vector<FunctionPtr>::const_iterator it = path.begin(); it != path.end(); it++)
    {
        std::wstring opName = (*it)->OpName();
        if (IsSupportedRNNActivation(opName))
        {
            if (count == nth)
            {
                std::unordered_multimap<std::wstring, AttributesMapping>::const_iterator itLookup = Operators::CntkToONNXLookup().find(opName);
                if (itLookup == Operators::CntkToONNXLookup().cend())
                    CNTK::LogicError("Invalid activation (%s)", ToString(opName).c_str());

                std::unordered_map<std::wstring, std::string>::const_iterator itMap = (*itLookup).second.map.find(opName);
                if (itMap == (*itLookup).second.map.cend())
                    CNTK::LogicError("Invalid activation (%s)", ToString(opName).c_str());
                return itMap->second;
            }
            count++;
        }
    }
    return "";
}

void CNTKToONNXHelper::PrepareInput(const Variable &X, std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    Variable input;
    wstring opName = X.Owner() ? X.Owner()->OpName() : L"";
    if (X.BlockFunctionVariableMapping() != Variable() && opName == L"LayerNormalization")
    {
        input = X.BlockFunctionVariableMapping();
    }
    else
    {
        input = X;
    }


    std::string inputName = ToString(input.Uid());
    onnx::TypeProto inputArgType = ToTypeProto(input.Shape(), (int)(input.DynamicAxes().size()));
    
    // TODO: figure out how to handdle sequence dimension.
    if (ToString(input.Uid()).find("Input") != -1 && !input.IsConstant() && !input.IsOutput())
        (*inputArgType.mutable_tensor_type()->mutable_shape()->mutable_dim())[0].set_dim_value(SequenceLen);

    UpdateONNXType(input.GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(inputName, &inputArgType);
    nodeInputs.push_back(inputArg);
}

void CNTKToONNXHelper::PrepareInitialState(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::vector<Variable> &initialVariables, int batchSize, int cellSize, 
    const std::string &uid, std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    std::vector<int> shape({ (int)initialVariables.size(), batchSize , cellSize });
    bool doReverseVec = false;
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(initialVariables[0].GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(uid, &inputArgType);
    std::vector<ONNXIR::NodeArg> varOutputs({ inputArg });
    std::vector<ONNXIR::NodeArg> varInputs;
    std::string inputName = inputArg.Name();
    ONNXIR::Node* variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < initialVariables.size(); i++)
    {
        const Variable &variable = initialVariables[i];
        auto srcTensor = variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value();
        if (srcTensor->Shape().Rank() == 0 || srcTensor->Shape().TotalSize() == 1)
        {
            srcTensors.push_back(srcTensor);
        }
        else
        {
            // TODO:
            NOT_IMPLEMENTED;
        }
    }

    onnx::TensorProto dstTensor;
    FillTensorWithScalar(srcTensors, dstTensor, shape);

    variableNode->AddAttribute("value", dstTensor);
    nodeInputs.push_back(inputArg);

    // TODO:
    variableNodes.emplace(initialVariables[0], variableNode);
}

void CNTKToONNXHelper::PrepareBiasNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::vector<Variable> &Bs, std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    // TODO: sanity check for all variables to have the same shape and data types.
    // NDShape is in reversed order relative CNTK python so doReverseVec need to be true 
    // when converting to ONNX tensor.
    // However with LSTM, CNTK python weight tensor shape is already reversed relative to ONNX.
    // We do not want to reverse again.
    bool doReverseVec = false;

    std::vector<int> shape = Cast<size_t, int>((NDShape({ Bs.size() }).AppendShape(Bs[0].Shape())).Dimensions());
    shape[1] *= 2;
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Bs[0].GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(ToString(Bs[0].Uid()), &inputArgType);
    std::vector<ONNXIR::NodeArg> varOutputs({ inputArg });
    std::vector<ONNXIR::NodeArg> varInputs;
    std::string inputName = inputArg.Name();
    ONNXIR::Node* variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < Bs.size(); i++)
    {
        const Variable &variable = Bs[i];
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;

    CopyTensorsWithCNTKToONNXLSTMWeightLayoutConversion(srcTensors, nullptr, dstTensor, &inputArgType);
    variableNode->AddAttribute("value", dstTensor);
    nodeInputs.push_back(inputArg);

    // TODO:
    variableNodes.emplace(Bs[0], variableNode);
}

void CNTKToONNXHelper::PrepareWeightNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::vector<Variable> &Ws, float *stabilizerConstants, std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    // TODO: sanity check for all variables to have the same shape and data types.
    // NDShape is in reversed order relative CNTK python so doReverseVec need to be true 
    // when converting to ONNX tensor.
    // However with LSTM, CNTK python weight tensor shape is already reversed relative to ONNX.
    // We do not want to reverse again.
    bool doReverseVec = false;

    std::vector<int> shape = Cast<size_t, int>((NDShape({ Ws.size() }).AppendShape(Ws[0].Shape())).Dimensions());
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Ws[0].GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(ToString(Ws[0].Uid()), &inputArgType);
    std::vector<ONNXIR::NodeArg> varOutputs({ inputArg });
    std::vector<ONNXIR::NodeArg> varInputs;
    std::string inputName = inputArg.Name(); 
    ONNXIR::Node* variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < Ws.size(); i++)
    {
        const Variable &variable = Ws[i];
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;
    
    CopyTensorsWithCNTKToONNXLSTMWeightLayoutConversion(srcTensors, stabilizerConstants, dstTensor, &inputArgType);
    variableNode->AddAttribute("value", dstTensor);
    nodeInputs.push_back(inputArg);

    // TODO:
    variableNodes.emplace(Ws[0], variableNode);
}

float GetScaler(Variable variable)
{
    NDArrayViewPtr v = variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value();
    NDArrayViewPtr cpuV = v->DeepClone();
    cpuV->ChangeDevice(DeviceDescriptor::CPUDevice());

    return *((float *)cpuV->DataBuffer<float>());

}

#pragma warning(disable: 4189)
ONNXIR::Node* CNTKToONNXHelper::CreateLSTMNode(const FunctionPtr &src,
    ONNXIR::Graph* graph,
    std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
    std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    // sanity check: 
    std::vector<FunctionPtr> lstms;
    if (src->OpName() == L"LSTM")
    {
        lstms.push_back(src);
    }
    else if (src->OpName() == L"Splice") // src is a Splice op with inputs from two LSTM ops.
    {
        for (auto &input : src->Inputs())
        {
            lstms.push_back(input.Owner());
        }
    }
    else
    {
        // LogicError()
        return nullptr;
    }

    if (lstms.size() == 0 || lstms.size() > 2 ||
        std::any_of(lstms.cbegin(), lstms.cend(), [](const FunctionPtr &f) {return f->OpName() != L"LSTM"; }))
    {
        LogicError("Invalid LSTM node");
    }
    
    // order forward, backward

    std::vector<std::string> activations(lstms.size() * 3);
    std::vector<Variable> Xs(lstms.size()), Ws(lstms.size()), Rs(lstms.size()), Bs(lstms.size()), Yhs(lstms.size()), Ycs(lstms.size()),
        initialHs(lstms.size()), initialCs(lstms.size());
    std::vector<float> stabilizerCoefs(lstms.size());
    std::map<bool, int> directionCount({ {false, 0} ,{true, 0} });

    for (std::vector<FunctionPtr>::const_iterator itLSTMBlock = lstms.cbegin(); itLSTMBlock != lstms.cend(); itLSTMBlock++)
    {
        // src has to be an LSTM node. 
        std::vector<FunctionPtr> ht_it_path;
        std::vector<FunctionPtr> ht_bit_path;
        std::vector<FunctionPtr> ht_ft_path;
        std::vector<FunctionPtr> ht_ot_path;
        bool goBackwards;
        const FunctionPtr& lstm = *itLSTMBlock;
        TraceLSTMPathes(lstm, ht_it_path, ht_bit_path, ht_ft_path, ht_ot_path, goBackwards);
        directionCount[goBackwards]++;

        int directionIndex = lstms.size() == 1 ? 0 : (goBackwards ? 1 : 0);

        string f_activation = FindActivation(ht_ot_path, 0);
        string g_activation = FindActivation(ht_bit_path, 1);
        string h_activation = FindActivation(ht_bit_path, 0);
        activations[directionIndex * 3 + 0] = f_activation;
        activations[directionIndex * 3 + 1] = g_activation;
        activations[directionIndex * 3 + 2] = h_activation;
    
        std::vector<Variable> inputs = lstm->Inputs();
        int inputIndex = inputs.size() == 10 ? 9 : 5;
        Xs[directionIndex] = inputs[inputIndex];
        Ws[directionIndex] = inputs[1];
        Rs[directionIndex] = inputs[2];
        Bs[directionIndex] = inputs[0];

        std::vector<Variable> outputs = lstm->Outputs();
        // TODO: handle return_full_state case
        Yhs[directionIndex] = outputs[0];
        Ycs[directionIndex] = outputs[1];

        if (inputs.size() == 10)
        {
            float steepness = GetScaler(inputs[5]);
            float alpha = GetScaler(inputs[6]);
            initialHs[directionIndex] = inputs[7].Owner()->Inputs()[1];
            initialCs[directionIndex] = inputs[8].Owner()->Inputs()[1];

            stabilizerCoefs[directionIndex] = (log(exp(alpha * steepness) + 1) / steepness);
            // stabilizerCoefs[directionIndex] = 1.0;
        }
        else
        {
            stabilizerCoefs[directionIndex] = 1.0;
        }
    }

    // ensure that if there is one direction, it is not backward.
    // if there two directions, they are forward and backward, and
    // that the inputs (Xs) are the same.
    if (std::any_of(directionCount.begin(), directionCount.end(), [](std::map<bool, int>::value_type &v) {return v.second > 1; }))
    {
        LogicError("LSTM node is invalid because there should be no more than one path in each direction.");
    }
    if (lstms.size() == 2 && Xs[0] != Xs[1])
    {
        LogicError("Bi-directional LSTM node is invalid because the two LSTM nodes do not share one same input.");
    }


    string direction = lstms.size() == 2 ? "bidirectional" : (directionCount[true] == 1 ? "reverse" : "forward");

    // TODO: following commented out attributes are not supported yet. Use default.
    // float clip; // no clip yet
    // std::vector<float> activation_alpha;    // no supported activation need alpha.
    // std::vector<float> activation_beta;    // no supported activation need beta.
    int hidden_size = lstms[0]->Outputs()[0].Shape()[0];
    int input_forget = 0;
    int output_sequence = 1;        // always output for CNTK LSTM
    // Variable P;

    // inputs
    std::vector<ONNXIR::NodeArg> nodeInputs;
    PrepareInput(Xs[0], nodeInputs);
    PrepareWeightNode(graph, variableNodes, Ws, nullptr, nodeInputs);
    PrepareWeightNode(graph, variableNodes, Rs, &stabilizerCoefs[0], nodeInputs);
    
    // TODO: size B in CNTK is 4 * hidden_size. 
    // Not sure why in ONNX it is 2 * 4 * hidden_size.

    // int sequence_lens; optional
    // LotusRT needs sequence_lens
    std::string dummyName = "Dummy_";
    {
        bool hasBias = true;
        if (hasBias)
        {
            PrepareBiasNode(graph, variableNodes, Bs, nodeInputs);
        }
        else
        {
            ONNXIR::NodeArg inputArg(dummyName + "_B", nullptr);
            nodeInputs.push_back(inputArg);
        }

        bool has_sequence_lens = false;
        std::string sequence_lens_inputName = "sequence_lens___";
        if (has_sequence_lens)
        {
            onnx::TypeProto inputArgType = ToTypeProto(std::vector<int>({ 1 }), false);
            inputArgType.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
            ONNXIR::NodeArg inputArg(sequence_lens_inputName, &inputArgType);
            nodeInputs.push_back(inputArg);
        }
        else
        {
            ONNXIR::NodeArg inputArg(dummyName + sequence_lens_inputName, nullptr);
            nodeInputs.push_back(inputArg);
        }

        bool has_initial_h = initialHs[0] != Variable();
        if (has_initial_h)
        {
            // TODO: how to set batchsize
            int batchSize = 1;
            std::string hiddenUid = ToString(Yhs[0].Uid()) + "_initial_h";
            PrepareInitialState(graph, variableNodes, initialHs, batchSize, hidden_size, hiddenUid, nodeInputs);

            std::string cellUid = ToString(Ycs[0].Uid()) + "_initial_c";
            PrepareInitialState(graph, variableNodes, initialCs, batchSize, hidden_size, cellUid, nodeInputs);
        }
        else
        {
            {
                ONNXIR::NodeArg inputArg(dummyName + "initial_h___", nullptr);
                nodeInputs.push_back(inputArg);
            }
            {
                ONNXIR::NodeArg inputArg(dummyName + "initial_c___", nullptr);
                nodeInputs.push_back(inputArg);
            }
        }

        {
            ONNXIR::NodeArg inputArg(dummyName + "P___", nullptr);
            nodeInputs.push_back(inputArg);
        }
    }

    std::vector<ONNXIR::NodeArg> nodeOutputs;
    {
        if (output_sequence == 1)
        {
            // TODO: do these sequence/batch dimensions meta data matter?
            int sequence_len = 1;
            int batchSize = 1;
            std::string nodeName;
            if (lstms.size() == 1)
                nodeName = ToString(Yhs[0].Uid());
            else
                nodeName = ToString(src->Output().Uid());

            auto outputArgType = ToTypeProto(std::vector<int>({ sequence_len, (int)Yhs.size(), batchSize, (int)Yhs[0].Shape()[0] }), false);
            UpdateONNXType(Yhs[0].GetDataType(), outputArgType);
            ONNXIR::NodeArg outputArg(nodeName, &outputArgType);
            nodeOutputs.push_back(outputArg);
        }
        else
        {
            ONNXIR::NodeArg outputArg(dummyName + "Y", nullptr);
            nodeOutputs.push_back(outputArg);
        }

        {
            Variable Yh = Yhs[0];
            std::string nodeName = ToString(Yh.Uid()) + "_h";
            // TODO:
            int batchSize = 1;
            auto outputArgType = ToTypeProto(std::vector<int>({ (int)Yhs.size(), batchSize, (int)Yh.Shape()[0]}), false);
            UpdateONNXType(Yh.GetDataType(), outputArgType);
            ONNXIR::NodeArg outputArg(nodeName, &outputArgType);
            nodeOutputs.push_back(outputArg);
        }
        {
            Variable Yc = Ycs[0];
            std::string nodeName = ToString(Yc.Uid()) + "_c";
            int batchSize = 1;
            auto outputArgType = ToTypeProto(std::vector<int>({ (int)Ycs.size(), batchSize, (int)Yc.Shape()[0] }), false);
            UpdateONNXType(Yc.GetDataType(), outputArgType);
            ONNXIR::NodeArg outputArg(nodeName, &outputArgType);
            nodeOutputs.push_back(outputArg);
        }
    }

    if (Xs[0].Owner().get() != nullptr)
        CreateNode(Xs[0].Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);

    auto nodeName = src->Name().empty() ? ToString(src->Uid()) : ToString(src->Name());
    ONNXIR::Node *lstmNode = graph->AddNode(nodeName, "LSTM", "", nodeInputs, nodeOutputs);

    lstmNode->AddAttribute("activations", activations);
    lstmNode->AddAttribute("direction", direction);
    lstmNode->AddAttribute("hidden_size", (int64_t)hidden_size);
    lstmNode->AddAttribute("output_sequence", (int64_t)output_sequence);
    
    if (lstms.size() == 2)
        NOT_IMPLEMENTED;

    // squeeze direction axis out. This is safe because it is not bi-directional node.
    // TODO: sequence and batch size
    std::vector<int> shape({ SequenceLen, 1, hidden_size });

    ONNXIR::Node *squeezedLSTMNode = AddReshapeNodeToCNTKFunction(src, lstmNode, shape, graph);

    functionNodes.emplace(src, squeezedLSTMNode);
    return squeezedLSTMNode;
}

ONNXIR::Node *CNTKToONNXHelper::AddArgMaxNode(const ONNXIR::NodeArg &nodeArg, ONNXIR::Graph* graph)
{
    // ONNXIR::NodeArg inputArg(nodeArg.Name(), nullptr);
    ONNXIR::NodeArg outputArg(nodeArg.Name() + "argmax_out", nullptr);
    ONNXIR::Node* argMaxNode = graph->AddNode(nodeArg.Name() + string("_argmax"), "ArgMax", "", { nodeArg }, { outputArg });
    argMaxNode->AddAttribute("axis", (int64_t)2);
    argMaxNode->AddAttribute("keepdims", (int64_t)1);    
    return argMaxNode;
}

ONNXIR::Node *CNTKToONNXHelper::AddCastNode(const ONNXIR::NodeArg &nodeArg, ONNXIR::Graph* graph)
{
    // ONNXIR::NodeArg inputArg(nodeArg.Name(), nullptr);
    ONNXIR::NodeArg outputArg(nodeArg.Name() + "cast_out", nullptr);
    ONNXIR::Node* castNode = graph->AddNode(nodeArg.Name() + string("_cast"), "Cast", "", { nodeArg }, { outputArg });
    castNode->AddAttribute("to", "INT32");
    return castNode;
}

// This method is to workaround the fact that ONNX spec of LSTM ops does not allow easy layer stacking.
// Under such circumstance, mapping memory layout from a bidirectional LSTM may need some work.
// For now we simply treat a bidirectional LSTM as two separate LSTMs. We still need to reshape 
// its output to squeeze away the direction dimension.
// TODO: expend this method to handle bidirection LSTMs.
ONNXIR::Node *CNTKToONNXHelper::AddReshapeNodeToCNTKFunction(const FunctionPtr &src, ONNXIR::Node* node, const std::vector<int> &shape, ONNXIR::Graph* graph)
{
    FunctionPtr blockRoot = src->BlockRoot();
    Variable output;
    if (src->OpName() == L"LSTM")
        output = src->Outputs()[0];
    else
        // a bidirection LSTM cast
        NOT_IMPLEMENTED
    
    std::string nodeName = ToString(blockRoot->Uid());
    
    // NodeArg name of the output of the reshaped node
    std::string outputNodeArgName = ToString(output.Uid());

    // We need to name reshape node's output arg with LSTM output name.
    // Thus we need to give LSTM node output a different name.   
    std::vector<ONNXIR::NodeArg>& outputArgs = node->Mutable_OutputDefs();
    TensorShapeProto inputShape(*outputArgs[0].Shape());

    // replace LSTM output arg with one of different name and same shape. 
    std::string lstmToReshapeNodeArgName = outputNodeArgName + "_tmp";
    outputArgs[0] = ONNXIR::NodeArg(lstmToReshapeNodeArgName, nullptr);
    outputArgs[0].SetShape(inputShape);

    // 
    ONNXIR::NodeArg inputArg = ONNXIR::NodeArg(lstmToReshapeNodeArgName, nullptr);
    inputArg.SetShape(inputShape);

    // this is the output NodeArg of the reshaped node. It has to be named 
    // with the original node's output NodeArg so that LotusIR can make a the connection. 
    onnx::TypeProto typeProto = ToTypeProto(shape, false);
    ONNXIR::NodeArg outputArg(outputNodeArgName, &typeProto);

    ONNXIR::Node* reshapeNode = graph->AddNode(nodeName + string("_reshape"), "Reshape", "", { inputArg }, { outputArg });
    reshapeNode->AddAttribute("shape", ToINTS(shape, false));
    return reshapeNode;
}

#include <iostream>
//
// This is the main horsepower, it navigate CNTK graph recursivley while keep track of all visited nodes and variables, 
// and create the corresponding ONNX graph.
//
ONNXIR::Node* CNTKToONNXHelper::CreateNode(const FunctionPtr& src,
    ONNXIR::Graph* graph,
    std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
    std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    auto iter = functionNodes.find(src);
    if (iter != functionNodes.end())
        return iter->second;

    ONNXIR::Node* functionNode = nullptr;
    std::string opName = ToString(src->OpName());

    //if (opName == "Splice")
    //{ 
    // TODO: uncomment this code once bidirectional LSTM is supprted
    //    std::vector<Variable> inputs = src->Inputs();
    //    bool bidiectionalLSTM = inputs.size() == 2 &&
    //        std::all_of(inputs.begin(), inputs.end(), [](Variable &input) {return input.Owner() != nullptr && input.Owner()->OpName() == L"LSTM"; });
    //    if (bidiectionalLSTM)
    //        return CreateLSTMNode(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    //}
    //else 
    if (opName == "LSTM")
    {
        return CreateLSTMNode(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (opName == "OptimizedRNNStack")
    {
        return CreateOptimizedRNNStackNode(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (opName == "Combine")
    {
        for (size_t inputIndex = 0; inputIndex < src->Inputs().size(); ++inputIndex)
        {
            auto input = src->Inputs()[inputIndex];
            CreateNode(input.Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);
        }

        // not a single node, 
        return nullptr;
    }

    //
    // If this block node equivalent to a primitive ONNX OP, then treated as such.
    // And just maps its argument to ONNX node.
    //
    if (src->IsBlock() &&
        (!Operators::IsSupportedCNTKOP(src->OpName()) || Operators::IsLayerCNTKOP(src->OpName())))
    {
        functionNode = CreateNode(src->BlockRoot(), graph, functionNodes, variableNodes, compositeOutputsMap);
        // TODO: this is to workaround LotusRT lossing sequence dimension issue 
        if (false && opName == "Embedding" && src->Uid() == L"Block18870")
        {
            functionNodes.emplace(src, functionNode);
            std::vector<int> shape({1, 1, (int)src->Output().Shape().Dimensions()[0]});
            ONNXIR::Node* reshapedNode = AddReshapeNodeToCNTKFunction(src, functionNode, shape, graph);
            return reshapedNode;
        }
    }
    //
    // For compatibility of other framework that support ONNX, we will limit the list of OPs to the one
    // supported by ONNX https://github.com/onnx/onnx/tree/master/onnx/defs.
    //
    else if (Operators::IsSupportedCNTKOP(src->OpName()))
    {
        std::vector<ONNXIR::NodeArg> inputs;
        std::vector<ONNXIR::NodeArg> outputs;

        for (const auto& output : src->Outputs())
        {
            auto outputArgType = ToTypeProto(output.Shape(), output.HasBatchAxis(), HasSequenceAxis(output));
            UpdateONNXType(output.GetDataType(), outputArgType);

            ONNXIR::NodeArg outputArg(ToString(output.Uid()), &outputArgType);
            outputs.push_back(outputArg);
        }

        for (size_t inputIndex = 0; inputIndex < src->Inputs().size(); ++inputIndex)
        {
            if (src->Uid() == L"Plus20546")
                std::cout << "Plus20546" << std::endl;

            auto input = src->Inputs()[inputIndex];

            if (input.IsPlaceholder())
            {
                input = input.BlockFunctionVariableMapping();
                if (input.IsPlaceholder())
                    LogicError("Node '%S': Placeholder isn't supported currently.", src->AsString().c_str());
            }

            if (FilterInput(src, input, inputIndex))
                continue;

            //
            // Use user defined name if available otherwise use our internel unique name ID.
            //
            std::string inputName = ToString(input.Uid());
            auto inputItr = compositeOutputsMap.find(input);
            if (inputItr != compositeOutputsMap.end())
                inputName = ToString(inputItr->second.Uid());

            bool isConstant = (input.IsParameter() || input.IsConstant()) &&
                !Operators::IgnoreConstantAndParameter(src->OpName(), inputIndex);

            onnx::TypeProto inputArgType;

            bool broadcastSwapped = false;
            if (Operators::SupportBroadcast(src->OpName()))
            {
                std::pair<std::vector<int>, std::vector<int>> adjustedDims;
                bool broadcast = false;
                int axis = 0;
                int index0, index1;
                std::tie<int, int>(index0, index1) = Operators::GetElementWiseInputIndices(src->OpName());

                if (index0 != inputIndex && index1 != inputIndex)
                    continue;

                std::tie<std::pair<std::vector<int>, std::vector<int>>, bool, int, bool>(adjustedDims, broadcast, axis, broadcastSwapped) =
                    AdjustForBroadcastShape(src->Inputs()[index0], src->Inputs()[index1]);
                if (inputIndex == index0)
                    inputArgType = ToTypeProto(adjustedDims.first, false);
                else if (inputIndex == index1)
                    inputArgType = ToTypeProto(adjustedDims.second, false);
            }
            else if (opName == "Splice")
            {
                // for ops like Concat, batch axis may exist in one of the operand
                // CNTK allows the other operand(s) not having batch axis. But ONNX 
                // requires operands to have the same rank
                inputArgType = ToTypeProto(input.Shape(), OpInputsHasBatchAxis(src));
            }
            else if (opName == "Hardmax" || opName == "ImageScaler")
            {
                // ONNX specifies that hardmax, ImageScaler always need a batch axis
                inputArgType = ToTypeProto(input.Shape(), true);
            }
            else
            {
                if (isConstant && opName == "BatchNormalization" && (inputIndex > 0 && inputIndex <= 4)
                    && input.Shape().Rank() == 2)
                    // this is a workaround for brainscript models that have rank = 2 for BN inputs.
                    inputArgType = ToTypeProto(input.Shape().SubShape(0, input.Shape().Rank() - 1));
                else
                    inputArgType = ToTypeProto(input.Shape(), input.HasBatchAxis(), HasSequenceAxis(input));
                if (ToString(input.Uid()).find("Input") != -1 && !isConstant && !input.IsOutput())
                    (*inputArgType.mutable_tensor_type()->mutable_shape()->mutable_dim())[0].set_dim_value(SequenceLen);
            }

            UpdateONNXType(input.GetDataType(), inputArgType);
            ONNXIR::NodeArg inputArg(inputName, &inputArgType);

            inputs.push_back(inputArg);

            if (broadcastSwapped && inputs.size() == 2)
                swap(inputs[0], inputs[1]);
            //
            // Leaf nodes are data entry to the graph and need their own node with only output arg.
            //
            if (isConstant)
            {
                if (variableNodes.find(input) == variableNodes.end())
                {
                    std::vector<ONNXIR::NodeArg> varInputs;
                    std::vector<ONNXIR::NodeArg> varOutputs;

                    varOutputs.push_back({ inputArg });
                    ONNXIR::Node* variableNode = nullptr;
                    if (input.IsParameter() || input.IsConstant())
                    {
                        variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);
                        auto srcTensor = input.IsParameter() ? Parameter(input).Value() : Constant(input).Value();

                        onnx::TensorProto dstTensor;
                        CopyTensor(srcTensor, dstTensor, &inputArgType);

                        variableNode->AddAttribute("value", dstTensor);
                        variableNodes.emplace(input, variableNode);
                    }
                }
            }
            //
            // If this input is output, then it is the ouput of an up stream node. Recursively add all upstream nodes.
            // Pretty much, we are doing DFS.
            //
            else if (input.IsOutput())
                CreateNode(input.Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);
        }

        //
        // Finally add a new node to ONNX graph.
        //
        functionNode = AddNode(src, graph, inputs, outputs);
    }
    else
        LogicError("Node '%S': Unsupported node.", src->AsString().c_str());

    functionNodes.emplace(src, functionNode);
    return functionNode;
}

void CNTKToONNXHelper::TraverseGraph(const FunctionPtr& src,
    std::set<FunctionPtr>& visited,
    std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    auto iter = visited.find(src);
    if (iter != visited.end())
        return;

    std::string opName = ToString(src->OpName());
    if (opName == "PastValue" || opName == "FutureValue")
    {
        return;
    }

    if (opName != "LSTM" &&
        src->IsBlock() && (!Operators::IsSupportedCNTKOP(src->OpName()) || Operators::IsLayerCNTKOP(src->OpName())))
    {
        auto blockSrc = dynamic_cast<BlockFunction*>(src.get());
        for (auto map : blockSrc->CompositeOutputsMap())
            compositeOutputsMap.insert(map);
        TraverseGraph(src->BlockRoot(), visited, compositeOutputsMap);
    }
    else
    {
        for (auto input : src->Inputs())
        {
            if (input.IsPlaceholder() && opName != "LSTM")
            {
                input = input.BlockFunctionVariableMapping();
                if (input.IsPlaceholder())
                    LogicError("Node '%S': Placeholder isn't supported currently.", src->AsString().c_str());
            }

            if (input.IsOutput())
                TraverseGraph(input.Owner(), visited, compositeOutputsMap);
        }
    }

    visited.emplace(src);
}

void CNTKToONNXHelper::CopyAttributes(const FunctionPtr& src, ONNXIR::Node* node)
{
    auto lookup = Operators::CntkToONNXLookup();
    assert(lookup.count(src->OpName()) != 0);

    std::string opName = ToString(src->OpName());
    if (lookup.count(src->OpName()) == 1)
    {
        auto attributesMap = lookup.find(src->OpName())->second.map;
        opName = attributesMap[src->OpName()];

        if (src->OpName() == L"Clip")
        {
            if (src->Inputs().size() != 3)
            {
                LogicError("Clip should have 3 inputs.");
            }
            float minValue = src->Inputs()[1].Value()->AsScalar<float>();
            float maxValue = src->Inputs()[2].Value()->AsScalar<float>();
            node->AddAttribute("min", minValue);
            node->AddAttribute("max", maxValue);
        }
        if (src->OpName() == L"BatchNormalization")
        {
            auto spatial = (int64_t)((bool)src->Attributes()[L"spatial"].Value<bool>() ? 1 : 0);
            auto normalizationTimeConstant = (float)src->Attributes()[L"normalizationTimeConstant"].Value<double>();
            // auto blendTimeConstant = (float)src->Attributes()[L"blendTimeConstant"].Value<double>();
            auto epsilon = (float)src->Attributes()[L"epsilon"].Value<double>();

            //
            // onnx: running_mean = running_mean * momentum + mean * (1 - momentum)
            // cntk: expAvgFactor * MB stats + (1-expAvgFactor) * prev running stats
            //
            auto momentum = 0.0f;
            if (!isfinite(normalizationTimeConstant))
                momentum = 1.0f;
            else if (normalizationTimeConstant > 0)
                momentum = 1.0f + expm1(-48.0f / normalizationTimeConstant);

            node->AddAttribute(attributesMap[L"spatial"], spatial);
            node->AddAttribute("is_test", (int64_t)1);
            node->AddAttribute(attributesMap[L"epsilon"], epsilon);
            node->AddAttribute("momentum", momentum);
        }
        else if (src->OpName() == L"LocalResponseNormalization")
        {
            auto depthRadius = (int64_t)src->Attributes()[L"depthRadius"].Value<size_t>();
            auto bias = (float)src->Attributes()[L"bias"].Value<double>();
            auto alpha = (float)src->Attributes()[L"alpha"].Value<double>();
            auto beta = (float)src->Attributes()[L"beta"].Value<double>();

            node->AddAttribute(attributesMap[L"size"], depthRadius);
            node->AddAttribute(attributesMap[L"bias"], bias);
            node->AddAttribute(attributesMap[L"alpha"], alpha);
            node->AddAttribute(attributesMap[L"beta"], beta);
        }
        else if ((src->OpName() == L"LeakyReLU") || (src->OpName() == L"ELU"))
        {
            auto alpha = 0.01f;
            if (src->Attributes().Contains(L"alpha"))
                alpha = (float)src->Attributes()[L"alpha"].Value<float>();
            node->AddAttribute("alpha", alpha);
        }
        else if (src->OpName() == L"SELU")
        {
            auto alpha = 1.6732f;
            if (src->Attributes().Contains(L"alpha"))
                alpha = (float)src->Attributes()[L"alpha"].Value<double>();

            auto gamma = 1.0507f;
            if (src->Attributes().Contains(L"gamma"))
                gamma = (float)src->Attributes()[L"gamma"].Value<double>();

            node->AddAttribute("alpha", alpha);
            node->AddAttribute("gamma", gamma);
        }
        else if (src->OpName() == L"Dropout")
        {
            auto dropoutRate = (float)src->Attributes()[L"dropoutRate"].Value<double>();
            node->AddAttribute(attributesMap[L"dropoutRate"], dropoutRate);
            node->AddAttribute("is_test", (int64_t)1);
        }
        else if ((src->OpName() == L"RandomDistribution") ||
            (src->OpName() == L"UniformRandom") || (src->OpName() == L"NormalRandom") ||
            (src->OpName() == L"UniformRandomLike") || (src->OpName() == L"NormalRandomLike"))
        {
            auto randomArgs = AsVector<double>(src->Attributes()[L"randomDistributionArgs"].Value<std::vector<DictionaryValue>>());
            auto seed = (int64_t)src->Attributes()[L"rngSeed"].Value<int>();

            if ((src->OpName() == L"UniformRandom") || (src->OpName() == L"UniformRandomLike"))
            {
                node->AddAttribute("low", (float)randomArgs[0]);
                node->AddAttribute("high", (float)randomArgs[1]);
            }
            else
            {
                node->AddAttribute("mean", (float)randomArgs[0]);
                node->AddAttribute("scale", (float)randomArgs[1]);
            }

            node->AddAttribute(attributesMap[L"rngSeed"], seed);
            if ((src->OpName() == L"UniformRandom") || (src->OpName() == L"NormalRandom"))
            {
                auto shape = (NDShape)src->Attributes()[L"newShape"].Value<NDShape>();
                node->AddAttribute(attributesMap[L"newShape"], ToINTS(shape));
            }
        }
        else if ((src->OpName() == L"ReduceL1") || (src->OpName() == L"ReduceL2") || (src->OpName() == L"ReduceSumSquare"))
        {
            auto keepReducedDimensions = (int64_t)((bool)src->Attributes()[L"reductionKeepDimensions"].Value<bool>() ? 1 : 0);
            std::vector<Axis> reductionAxes;
            if (src->Attributes().Contains(L"axisVec"))
                reductionAxes = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
            else if (src->Attributes().Contains(L"axis"))
                reductionAxes.push_back((Axis)(src->Attributes()[L"axis"].Value<Axis>()));

            node->AddAttribute(attributesMap[L"reductionKeepDimensions"], keepReducedDimensions);

            std::vector<int64_t> axes = ConvertAxesToOnnx(reductionAxes, src->Inputs()[0]);
            node->AddAttribute("axes", axes);
        }
        else if (src->OpName() == L"TransposeAxes")
        {
            if (src->Attributes().Contains(L"axisVec"))
            {
                std::vector<Axis> permutation = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
                // CNTK permutation attribute is argsorted. Shall redo argsort (undo) to get the original python/ONNX perm attribute.
                std::vector<int64_t> perm = ConvertPermutationCNTKToONNX(permutation, src->Inputs()[0].HasBatchAxis());
                node->AddAttribute(attributesMap[L"axisVec"], perm);
            }
            else if (src->Attributes().Contains(L"axis1") && src->Attributes().Contains(L"axis2"))
            {
                // swapaxis: permutation is between two axes
                int rank = src->Output().Shape().Rank();
                std::vector<int64_t> perm;
                bool hasBatchAxis = src->Inputs()[0].HasBatchAxis();
                for (int index = 0; index < (hasBatchAxis ? (rank + 1) : rank); index++)
                {
                    perm.push_back(index);
                }

                Axis axis1 = (Axis)(src->Attributes()[L"axis1"].Value<Axis>()).StaticAxisIndex();
                Axis axis2 = (Axis)(src->Attributes()[L"axis2"].Value<Axis>()).StaticAxisIndex();
                int64_t axisIndex1 = ConvertAxisToOnnx(axis1, src->Inputs()[0]);
                int64_t axisIndex2 = ConvertAxisToOnnx(axis2, src->Inputs()[0]);
                std::swap(perm[axisIndex1], perm[axisIndex2]);
                node->AddAttribute(attributesMap[L"axisVec"], perm);
            }
        }
        else if (src->OpName() == L"Reshape")
        {
            // TODO: handle CNTK reshape with begin and end axes.
            auto shapeVec = src->Output().Shape().Dimensions();
            std::vector<int> newShapeVec;
            size_t numInferredDimensions(0);
            for (const auto& axisSize : shapeVec)
            {
                if (axisSize == NDShape::InferredDimension)
                {
                    numInferredDimensions++;
                    if (numInferredDimensions > 1)
                        LogicError("Reshape: Multiple InferredDimension not supported by ONNX.");
                    else
                        newShapeVec.push_back(-1);
                }
                else // REVIEW SPTIWARI: Should we fill 0 for FreeDimension here?
                    newShapeVec.push_back(static_cast<int>(axisSize));
            }
            // Always add a 1 to the shape for batch axis in ONNX tensors.
            if ((src->Inputs().size() > 0) && (src->Inputs()[0].HasBatchAxis()))
                newShapeVec.push_back(1);
            node->AddAttribute(attributesMap[L"shape"], ToINTS(newShapeVec));
        }
        else if (src->OpName() == L"Splice")
        {
            Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
            int64_t axisIndex = ConvertAxisToOnnx(axis, src->Inputs()[0]);
            node->AddAttribute(attributesMap[L"axis"], axisIndex);
        }
        else if (src->OpName() == L"Slice")
        {
            std::vector<int> beginIndex;
            std::vector<int> endIndex;

            if (src->Attributes().Contains(L"axisVec"))
            {
                std::vector<Axis> sliceAxes = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
                node->AddAttribute(attributesMap[L"axes"], ToINTS(sliceAxes));

                beginIndex = AsVector<int>(src->Attributes()[L"beginIndexVec"].Value<std::vector<DictionaryValue>>());
                endIndex = AsVector<int>(src->Attributes()[L"endIndexVec"].Value<std::vector<DictionaryValue>>());
            }
            else if (src->Attributes().Contains(L"axis"))
            {
                Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
                int64_t axisIndex = ConvertAxisToOnnx(axis, src->Inputs()[0]);
                bool workaroundONNXRT = false;
                // this code is to workarund a LotusRT bug that fails
                // to take axes attribute into consideration.
                // we need to convert op attribute to a default ONNX case
                // where axes is not set (or set to ordered indices).
                if (workaroundONNXRT)
                {
                    bool hasBatchAxis = src->Inputs()[0].HasBatchAxis();
                    NDShape inputShape = src->Inputs()[0].Shape();
                    std::vector<int64_t> sliceAxes;
                    int numDims = hasBatchAxis ? (inputShape.Rank() + 1) : inputShape.Rank();
                    for (int onnxAxis = 0; onnxAxis < numDims; onnxAxis++)
                    {
                        sliceAxes.push_back(onnxAxis);
                        if (onnxAxis == 0 && hasBatchAxis)
                        {
                            // batch axis
                            beginIndex.push_back(0);
                            endIndex.push_back(1);
                        }
                        else
                        {
                            if (axisIndex == onnxAxis)
                            {
                                beginIndex.push_back((int)(src->Attributes()[L"beginIndex"].Value<int>()));
                                endIndex.push_back((int)(src->Attributes()[L"endIndex"].Value<int>()));
                            }
                            else
                            {
                                int cntkAxisIndex = numDims - onnxAxis - 1;
                                beginIndex.push_back(0);
                                endIndex.push_back(inputShape[cntkAxisIndex]);
                            }
                        }
                    }
                    node->AddAttribute(attributesMap[L"axes"], sliceAxes);
                }
                else
                {
                    std::vector<int64_t> sliceAxes;
                    sliceAxes.push_back(axisIndex);
                    node->AddAttribute(attributesMap[L"axes"], sliceAxes);

                    beginIndex.push_back((int)(src->Attributes()[L"beginIndex"].Value<int>()));
                    endIndex.push_back((int)(src->Attributes()[L"endIndex"].Value<int>()));
                }
            }

            std::vector<int64_t> beginIndex64 = Cast<int, int64_t>(beginIndex);
            std::vector<int64_t> endIndex64 = Cast<int, int64_t>(endIndex);

            node->AddAttribute(attributesMap[L"beginIndexVec"], beginIndex64);
            node->AddAttribute(attributesMap[L"endIndexVec"], endIndex64);
        }
        if (src->OpName() == L"Pad")
        {
            auto value = (float)src->Attributes()[L"paddingConstantValue"].Value<double>();
            auto mode = (size_t)src->Attributes()[L"paddingMode"].Value<size_t>();
            auto head = ToINTS(AsVector<size_t>(src->Attributes()[L"paddingHead"].Value<std::vector<DictionaryValue>>()));
            auto foot = ToINTS(AsVector<size_t>(src->Attributes()[L"paddingFoot"].Value<std::vector<DictionaryValue>>()));
            if (OpInputsHasBatchAxis(src))
            {
                head.insert(head.begin(), 0);
                foot.insert(foot.begin(), 0);
            }

            head.insert(head.end(), foot.begin(), foot.end());
            string modeStr;
            if (mode == 0)
                modeStr = "constant";
            else if (mode == 1)
                modeStr = "reflect";
            else if (mode == 2)
                NOT_IMPLEMENTED
            else
                LogicError("Invalid 'mode' value encountered in CNTK Pad node.");

            node->AddAttribute("mode", modeStr);
            node->AddAttribute("pads", head);
            if (mode == 0)
                node->AddAttribute("value", value);
        }
        else if (src->OpName() == L"DepthToSpace" || src->OpName() == L"SpaceToDepth")
        {
            size_t blockSize = src->Attributes()[L"blockSize"].Value<size_t>();
            node->AddAttribute("blocksize", static_cast<int64_t>(blockSize));
        }
        else if (src->OpName() == L"Softmax" || src->OpName() == L"LogSoftmax")
        {
            Axis axis = Axis(0);
            if (src->Attributes().Contains(L"axis"))
                axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
            node->AddAttribute(attributesMap[L"axis"], (int64_t)ToIndex(axis));
        }
        else if (Operators::SupportBroadcast(src->OpName()))
        {
            if (src->Uid() == L"Plus20546")
            {
                std::cout << "Plus20546" << std::endl;
            }
            std::pair<std::vector<int>, std::vector<int>> adjustedDims;
            bool broadcast = false, swapInput = false;
            int axis = 0;
            int index0, index1;
            std::tie<int, int>(index0, index1) = Operators::GetElementWiseInputIndices(src->OpName());
            std::tie<std::pair<std::vector<int>, std::vector<int>>, bool, int>(adjustedDims, broadcast, axis, swapInput) =
                AdjustForBroadcastShape(src->Inputs()[index0], src->Inputs()[index1]);


            if (src->Inputs()[1].IsConstant() && src->Inputs()[1].Shape().Rank() == 0 &&
                src->Inputs()[0].DynamicAxes().size() != 0)
            {
                // TODO: move into AdjustForBroadcastShape
                // a scalar with dynamic access elementwise a constant scalar.   
                broadcast = true;
            }

            node->AddAttribute("broadcast", (int64_t)(broadcast ? 1 : 0));
            if (broadcast && axis >= 0)
            {
                // +1 to take into consideration the batch aies
                node->AddAttribute("axis", (int64_t)axis);
            }
        }
        else if (src->OpName() == L"Times")
        {
            size_t outputRank = src->Attributes()[L"outputRank"].Value<size_t>();
            if (outputRank > 1)
                LogicError("Output rank other than 1 is not supported.");
        }
        else if (src->OpName() == L"ROIPooling")
        {
            auto roiOutputShape = (NDShape)src->Attributes()[L"roiOutputShape"].Value<NDShape>();
            auto ints = ToINTS(roiOutputShape, false);
            std::vector<float> pooled_shape = INTSToVecFloat(ints);

            auto spatialScale = (float)src->Attributes()[L"spatialScale"].Value<double>();

            node->AddAttribute("pooled_shape", pooled_shape);
            node->AddAttribute("spatial_scale", spatialScale);
        }
        else if (src->OpName() == L"HardSigmoid")
        {
            float alpha = (float)src->Attributes()[L"alpha"].Value<float>();
            float beta = (float)src->Attributes()[L"beta"].Value<float>();
            node->AddAttribute("alpha", alpha);
            node->AddAttribute("beta", beta);
        }
        else if (src->OpName() == L"Flatten")
        {
            Axis axis(0);
            if (src->Attributes().Contains(L"axis"))
            {
                axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
            }
            int64_t ax = ConvertAxisToOnnx(axis, src->Inputs()[0]);
            node->AddAttribute(attributesMap[L"axis"], ax);
        }
        else if (src->OpName() == L"Squeeze")
        {
            std::vector<Axis> axes;
            if (src->Attributes().Contains(L"axisVec"))
            {
                axes = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
            }
            else if (src->Attributes().Contains(L"axis"))
            {
                axes.push_back((Axis)(src->Attributes()[L"axis"].Value<Axis>()));
            }
            node->AddAttribute("axes", ToINTS(axes));
        }
        else if (src->OpName() == L"Gather")
        {
            if (src->Attributes().Contains(L"axis"))
            {
                Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
                int64_t ax = ConvertAxisToOnnx(axis, src->Inputs()[0]);
                node->AddAttribute(attributesMap[L"axis"], ax);
            }
        }
        else if (src->OpName() == L"ImageScaler")
        {
            float scale = (float)(src->Attributes()[L"Scaler"].Value<float>());
            std::vector<float> biases = AsVector<float>(src->Attributes()[L"Biases"].Value<std::vector<DictionaryValue>>());

            node->AddAttribute("scale", scale);
            node->AddAttribute("bias", biases);
        }
    }
    else
    {
        // Some nodes map one to many.
        if (src->OpName() == L"Convolution")
        {
            auto kernelShape = (NDShape)src->Attributes()[L"kernelShape"].Value<NDShape>();
            auto strides = (NDShape)src->Attributes()[L"strides"].Value<NDShape>();
            auto autoPadding = AsVector<bool>(src->Attributes()[L"autoPadding"].Value<std::vector<DictionaryValue>>());
            auto dilations = (NDShape)src->Attributes()[L"dilation"].Value<NDShape>();
            auto transpose = (bool)src->Attributes()[L"transpose"].Value<bool>();

            //
            // Remove the channel part for ONNX.
            //
            kernelShape = kernelShape.SubShape(0, kernelShape.Rank() - 1);
            strides = strides.SubShape(0, strides.Rank() - 1);
            autoPadding.pop_back();
            dilations = dilations.SubShape(0, dilations.Rank() - 1);

            node->AddAttribute("kernel_shape", ToINTS(kernelShape));
            node->AddAttribute("strides", ToINTS(strides));
            node->AddAttribute("dilations", ToINTS(dilations));
            node->AddAttribute("group", (int64_t)1);

            const NDShape &inputShape = src->Inputs()[1].Shape();

            if (transpose)
            {
                auto outputShape = (NDShape)src->Attributes()[L"outputShape"].Value<NDShape>();
                node->AddAttribute("output_shape", ToINTS(outputShape, src->Inputs()[1].HasBatchAxis()));
            }
            else
            {
                PutAutopadOrPadAttrInNode(node, inputShape, autoPadding, kernelShape);
            }
        }
        else if (src->OpName() == L"Pooling")
        {
            auto kernelShape = (NDShape)src->Attributes()[L"poolingWindowShape"].Value<NDShape>();
            auto strides = (NDShape)src->Attributes()[L"strides"].Value<NDShape>();
            bool ceilOutDim = (bool)src->Attributes()[L"ceilOutDim"].Value<bool>();
            if (strides.Rank() < kernelShape.Rank())
            {
                // TODO: Try removing this branch. May not be needed after batch dimension fix.
                strides = strides.AppendShape(NDShape(std::vector<size_t>(kernelShape.Rank() - strides.Rank(), 1)));
            }
            if ((strides.Rank() - kernelShape.Rank()) == 1)
            {
                // This can happen, for example, because a CNTK node includes strides for the channel axis as well. 
                strides = strides.SubShape(0, strides.Rank() - 1);
            }
            else if ((strides.Rank() - kernelShape.Rank()) > 1)
            {
                // This means that the length of kernel shape and strides is off by two or more which should not happen.
                LogicError("Node '%S': kernel shape and strides dimensionality does not match.", src->AsString().c_str());
            }
            auto autoPadding = AsVector<bool>(src->Attributes()[L"autoPadding"].Value<std::vector<DictionaryValue>>());

            node->AddAttribute("kernel_shape", ToINTS(kernelShape));
            node->AddAttribute("strides", ToINTS(strides));
            const NDShape &inputShape = src->Inputs()[0].Shape();
            PutAutopadOrPadAttrInNode(node, inputShape, autoPadding, kernelShape, ceilOutDim);
        }
        else if (src->OpName() == L"ReduceElements")
        {
            wstring cntkAttributeOpName = (wstring)src->Attributes()[PrimitiveFunction::AttributeNameReductionOpName].Value<wstring>();
            const AttributesMapping& attributeMap = Operators::FindAttributeMap(src->OpName(), cntkAttributeOpName);

            auto keepReducedDimensions = (int64_t)((bool)src->Attributes()[L"reductionKeepDimensions"].Value<bool>() ? 1 : 0);
            node->AddAttribute(attributeMap.map.at(L"reductionKeepDimensions"), keepReducedDimensions);

            if (src->Attributes().Contains(L"axisVec"))
            {
                std::vector<Axis> reductionAxes;
                reductionAxes = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
                std::vector<int64_t> axes = ConvertAxesToOnnx(reductionAxes, src->Inputs()[0]);
                node->AddAttribute("axes", axes);
            }
            else if (src->Attributes().Contains(L"axis"))
            {
                // py axis -> cpp (-axis -1) -> normalize (rank + axis)
                Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
                int64_t ax = ConvertAxisToOnnx(axis, src->Inputs()[0]);

                node->AddAttribute("axis", ax);
            }
        }
    }
}

void CNTKToONNXHelper::PutAutopadOrPadAttrInNode(ONNXIR::Node* node,
    const NDShape &inputShape, const std::vector<bool>& autoPadding,
    const NDShape& kernelShape, bool ceilOutDim)
{
    // Based on the CNTK node choose to put either the auto_pad or pads attribute in the ONNX node.

    // ONNX spec says that if 'pads' attributes is specified then 'VALID'
    // for 'auto_pad' is implied, and 'auto_pad' attribute should not (must not)
    // be explicitly specified/set.
    bool isExplicitPadValueNeeded = std::find(autoPadding.begin(), autoPadding.end(), false) != autoPadding.end();
    if (isExplicitPadValueNeeded && !ceilOutDim)
    {
        auto padsValueVectorsForONNX = GetONNXPadsAttributeFromCNTKNode(inputShape, autoPadding, kernelShape, ceilOutDim);
        auto lowerPads = ToINTS(padsValueVectorsForONNX.first);
        auto upperPads = ToINTS(padsValueVectorsForONNX.second);
        lowerPads.insert(lowerPads.end(), upperPads.cbegin(), upperPads.cend());
        node->AddAttribute("pads", lowerPads);
    }
    else if (ceilOutDim)
        node->AddAttribute("auto_pad", "SAME_LOWER");
    else
        node->AddAttribute("auto_pad", "SAME_UPPER");
}

std::vector<ONNXIR::NodeArg> CNTKToONNXHelper::MapInputsOrderToONNX(const FunctionPtr& src, const std::vector<ONNXIR::NodeArg>& inputs)
{
    if (Operators::HasInputIndexMap(src->OpName()))
    {
        std::vector<ONNXIR::NodeArg> orderedInputs;
        std::map<int, ONNXIR::NodeArg> orderedInputsMap;
        auto map = Operators::ToONNXInputIndexMap(src->OpName());

        for (size_t inputIndex = 0; inputIndex < inputs.size(); ++inputIndex)
        {
            if (map[inputIndex] >= 0)
                orderedInputsMap.insert(std::pair<int, ONNXIR::NodeArg>(map[inputIndex], inputs[inputIndex]));
        }

        for (const auto& item : orderedInputsMap)
            orderedInputs.push_back(item.second);

        return orderedInputs;
    }

    return inputs;
}

ONNXIR::Node* FindByName(ONNXIR::Graph* graph, const std::string &name)
{
    for (ONNXIR::Graph::NodeIterator it = graph->Nodes_begin(); it != graph->Nodes_end(); ++it)
    {
        ONNXIR::Node *node = *it;

        const std::vector<ONNXIR::NodeArg>& outputNodeArgs = node->OutputDefs();
        for (int i = 0; i < outputNodeArgs.size(); i++)
        {
            if (outputNodeArgs[i].Name() == name)
            {
                return node;
            }
        }
    }
    return nullptr;
}

ONNXIR::Node* CNTKToONNXHelper::AddNode(const FunctionPtr& src, ONNXIR::Graph* graph, const std::vector<ONNXIR::NodeArg>& inputs, const std::vector<ONNXIR::NodeArg>& outputs)
{
    ONNXIR::Node* node = nullptr;
    std::vector<ONNXIR::NodeArg> orderedInputs = MapInputsOrderToONNX(src, inputs);
    auto nodeName = src->Name().empty() ? ToString(src->Uid()) : ToString(src->Name());

    if (L"Embedding" == src->OpName())
    {
        ONNXIR::Node* argMax = AddArgMaxNode(orderedInputs[1], graph);
        ONNXIR::Node* int32Cast = AddCastNode(argMax->OutputDefs()[0], graph);

        bool reshapeGather = true;
        if (reshapeGather)
        {
            ONNXIR::NodeArg gatherIndexInputNodeArg(int32Cast->OutputDefs()[0].Name(), nullptr);
            ONNXIR::NodeArg gatherSourceInputNodeArg(orderedInputs[0].Name(), nullptr);
            ONNXIR::NodeArg gatherOutputArg(nodeName + "_gather_tmp", nullptr);
            ONNXIR::Node* gatherNode = graph->AddNode(nodeName + "_tmp", "Gather", "", { gatherSourceInputNodeArg , gatherIndexInputNodeArg }, { gatherOutputArg });

            ONNXIR::NodeArg reshapeInputNodeArg(gatherNode->OutputDefs()[0].Name(), nullptr);
            ONNXIR::Node* reshapedGather = graph->AddNode(nodeName, "Reshape", "", { reshapeInputNodeArg }, outputs);
            int input_size = src->Output().Shape()[0];
            std::vector<int> newShape({ SequenceLen, 1, input_size });
            reshapedGather->AddAttribute("shape", ToINTS(newShape, false));
            return reshapedGather;
        }
        else
        {
            ONNXIR::NodeArg gatherIndexInputNodeArg(int32Cast->OutputDefs()[0].Name(), nullptr);
            ONNXIR::Node* gatherNode = graph->AddNode(nodeName, "Gather", "", { orderedInputs[0] , gatherIndexInputNodeArg }, outputs);
        }
    }
    else if (Operators::SupportBroadcast(src->OpName()))
    {
        // when converting CNTK to ONNX with broadcasting, the boardcasting input at right-hand-side
        // needs to be reshaped. Reshape is not needed if the broadcasting input is a constant. In such case
        // CreateNode already created a constant with the needed shape. 
        // If the broadcasting input is not a constant, a reshape operation needs to be inserted. 
        // The following code does this reshape insertion.
        const TensorShapeProto* input1Shape = orderedInputs[0].Shape();
        const TensorShapeProto* input2Shape = orderedInputs[1].Shape();
        int input1Rank = input1Shape->dim_size();
        int input2Rank = input2Shape->dim_size();
        ONNXIR::Node* inputNode2 = FindByName(graph, orderedInputs[1].Name());
        if (input2Rank != 0 && input2Rank < input1Rank && inputNode2 != nullptr && inputNode2->OpType() != "Constant")
        {
            ONNXIR::NodeArg inputOutput2Arg(orderedInputs[1].Name() + string("_reshape1"), nullptr);
            inputOutput2Arg.SetShape(*input2Shape);

            auto reshapeNode2 = graph->AddNode(nodeName + string("_reshape1"), "Reshape", "", { orderedInputs[1] }, { inputOutput2Arg });

            onnx::TypeProto reshapeTypeProto2 = TensorShapeProtoToTypeProto(input2Shape);

            reshapeNode2->AddAttribute("shape", ToINTS(reshapeTypeProto2));

            node = graph->AddNode(nodeName, ToOPName(src), "", { orderedInputs[0] , inputOutput2Arg }, outputs);
        }
        else
        {
            // TODO: AdjustForBroadcastShape failed to handle case [#, *](2) + (2). It shall return broadcast = true.
            if (src->Uid() == L"Plus20546")
            {
                if (true)
                {
                    // TODO: apply workaround to MatMul by wrapping it with reshape ops. 
                    // This shall be done after code refactoring.
                    // in this case, "Plus20546" comes after matmul which collaped the first 2 axis (sequence and batch)
                    // into one. need to recover it assuming batch size = 1.
                    std::vector<int64_t> shape1 = ToINTS(TensorShapeProtoToTypeProto(input1Shape));
                    std::vector<int64_t> shape2 = ToINTS(TensorShapeProtoToTypeProto(input2Shape));

                    ONNXIR::NodeArg inputOutput2Arg(orderedInputs[1].Name() + string("_reshape2"), nullptr);
                    {
                        auto reshapeNode2 = graph->AddNode(nodeName + string("_reshape2"), "Reshape", "", { orderedInputs[1] }, { inputOutput2Arg });
                        // remove batch and sequence dimensions
                        shape2.erase(shape2.begin());
                        shape2.erase(shape2.begin());
                        reshapeNode2->AddAttribute("shape", shape2);
                    }

                    ONNXIR::NodeArg inputOutput1Arg(orderedInputs[0].Name() + string("_reshape1"), nullptr);
                    {
                        auto reshapeNode1 = graph->AddNode(nodeName + string("_reshape1"), "Reshape", "", { orderedInputs[0] }, { inputOutput1Arg });
                        (const_cast<TensorShapeProto*>(input1Shape))->mutable_dim(0)->set_dim_value(SequenceLen);
                        onnx::TypeProto reshapeTypeProto1 = TensorShapeProtoToTypeProto(input1Shape);
                        reshapeNode1->AddAttribute("shape", ToINTS(reshapeTypeProto1));
                    }

                    node = graph->AddNode(nodeName, ToOPName(src), "", { inputOutput1Arg, inputOutput2Arg }, outputs);
                    node->AddAttribute("broadcast", (int64_t)1);
                }
                else
                    node = graph->AddNode(nodeName, ToOPName(src), "", orderedInputs, outputs);
            }
            else
                node = graph->AddNode(nodeName, ToOPName(src), "", orderedInputs, outputs);
        }
    }
    else
        //
        // CNTK Times OP is way more flexible for ONNX, so depend on the inputs and output shape,
        // we will need to insert some reshapes.
        //
        if (src->OpName() == L"Times")
        {
            auto input1Shape = orderedInputs[0].Shape();
            auto input2Shape = orderedInputs[1].Shape();
            auto outputShape = outputs[0].Shape();

            int input1Rank = input1Shape->dim_size();
            int input2Rank = input2Shape->dim_size();
            int outputRank = outputShape->dim_size();
            int reductionRank = (input1Rank + input2Rank - outputRank) / 2;

            if (reductionRank > 1) // We need to insert reshape.
            {
                auto input1Reshape = ReduceRank(input1Shape, reductionRank, true);
                auto input2Reshape = ReduceRank(input2Shape, reductionRank, false);

                UpdateONNXType(src->Inputs()[1].GetDataType(), input1Reshape);
                UpdateONNXType(src->Inputs()[0].GetDataType(), input2Reshape);

                ONNXIR::NodeArg inputOutput1Arg(orderedInputs[0].Name() + string("_reshape0"), &input1Reshape);
                ONNXIR::NodeArg inputOutput2Arg(orderedInputs[1].Name() + string("_reshape1"), &input2Reshape);

                auto reshapeNode1 = graph->AddNode(nodeName + string("_reshape0"), "Reshape", "", { orderedInputs[0] }, { inputOutput1Arg });
                auto reshapeNode2 = graph->AddNode(nodeName + string("_reshape1"), "Reshape", "", { orderedInputs[1] }, { inputOutput2Arg });

                reshapeNode1->AddAttribute("shape", ToINTS(input1Reshape));
                reshapeNode2->AddAttribute("shape", ToINTS(input2Reshape));

                node = graph->AddNode(nodeName, ToOPName(src), "", { inputOutput1Arg , inputOutput2Arg }, outputs);
            }
            else
                node = graph->AddNode(nodeName, ToOPName(src), "", orderedInputs, outputs);
        }
        else
            node = graph->AddNode(nodeName, ToOPName(src), "", orderedInputs, outputs);

    //
    // Copy and validate attributes.
    //
    CopyAttributes(src, node);

    return node;
}

std::pair<std::vector<int>, std::vector<int> > CNTKToONNXHelper::GetONNXPadsAttributeFromCNTKNode(
    const NDShape &inputShape,
    const std::vector<bool>& cntkAutoPadding, const NDShape& kernelShape, bool ceilOutDim)
{
    // Figure out the value for 'pads' ONNX attribute.

    // Only one of the two ONNX conv attributes, auto_pad and pads, can be specified in the saved model. 
    // It is assumed at this point that we need an explicit padding vector, pads, and not the auto_pad attribute. 
    // The 'auto_pad' atrribute is implied to be 'VALID' by ONNX specification if the 'pads' attribute is specified
    // (padsValueVector) for the dimensions for which cntkAutoPadding is true.
    assert(kernelShape.Rank() == cntkAutoPadding.size());
    std::vector<int> padsValueVectorLower(kernelShape.Rank(), 0);
    std::vector<int> padsValueVectorUpper(kernelShape.Rank(), 0);
    for (size_t i = 0; i < cntkAutoPadding.size(); ++i)
    {
        if (!cntkAutoPadding[i]) continue;
        auto q = kernelShape[i] / 2;
        padsValueVectorLower[i] = kernelShape[i] % 2 ? q : (q - 1);
        padsValueVectorUpper[i] = q;
    }
    return std::make_pair(padsValueVectorLower, padsValueVectorUpper);
}

#pragma warning(disable: 4189)

void CNTKToONNXHelper::FillTensorWithScalar(const std::vector<NDArrayViewPtr> &srcs,
    onnx::TensorProto& dst, const std::vector<int> dstShape)
{
    auto dataType = srcs[0]->GetDataType();
    // the first dimension is for srcs count
    int eachSrcSize = std::accumulate(dstShape.begin() + 1, dstShape.end(), 1, std::multiplies<int>());
    switch (dataType)
    {
    case DataType::Float:
    {
        dst.set_data_type(onnx::TensorProto_DataType_FLOAT);
        for (int i = 0; i < srcs.size(); i++)
        {
            auto srcTemp = srcs[i]->DeepClone();
            srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());
            float scalar = *srcTemp->DataBuffer<float>();

            for (size_t index = 0; index < eachSrcSize; index++)
            {
                *(dst.mutable_float_data()->Add()) = scalar;
            }
        }

        break;
    }
    case DataType::Double:
    {
        dst.set_data_type(onnx::TensorProto_DataType_DOUBLE);
        for (int i = 0; i < srcs.size(); i++)
        {
            auto srcTemp = srcs[i]->DeepClone();
            srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());
            float scalar = *srcTemp->DataBuffer<float>();

            for (size_t index = 0; index < eachSrcSize; index++)
            {
                *(dst.mutable_double_data()->Add()) = scalar;
            }
        }

        break;
    }
    default:
        NOT_IMPLEMENTED;
    }

    for (auto dim : dstShape)
        *(dst.mutable_dims()->Add()) = dim;
}

