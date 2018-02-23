#pragma once
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// originally from CNTK\Tests\EndToEndTests\CNTKv2Library\Common\Common.h
// 

#pragma once
#include <algorithm>
#include "CNTKLibrary.h"
#include <functional>

using namespace CNTK;

std::pair<FunctionPtr, FunctionPtr> LSTMPCell(Variable input, Variable prevOutput, Variable prevCellState,
    Constant &W, Constant &R, Constant &B)
{
    size_t outputDim = prevOutput.Shape()[0];
    int stacked_dim = outputDim;

    FunctionPtr proj4;
    if (B != Variable())
    {
        proj4 = Plus(Plus(B, Times(W, input)), Times(R, prevOutput));
    }
    else
    {
        proj4 = Plus(Times(W, input), Times(R, prevOutput));
    }

    std::vector<Axis> stack_axis({ Axis(-1) });
    FunctionPtr it_proj = Slice(proj4, stack_axis, { 0 * stacked_dim }, { 1 * stacked_dim });
    FunctionPtr bit_proj = Slice(proj4, stack_axis, { 1 * stacked_dim }, { 2 * stacked_dim });
    FunctionPtr ft_proj = Slice(proj4, stack_axis, { 2 * stacked_dim }, { 3 * stacked_dim });
    FunctionPtr ot_proj = Slice(proj4, stack_axis, { 3 * stacked_dim }, { 4 * stacked_dim });

    // Input gate
    auto it = Sigmoid(it_proj);
    auto bit = ElementTimes(it, Tanh(bit_proj));

    auto ft = Sigmoid(ft_proj);
    auto bft = ElementTimes(ft, prevCellState);

    auto ct = Plus(bft, bit);

    auto ot = Sigmoid(ot_proj);
    auto ht = ElementTimes(ot, Tanh(ct));

    auto c = ct;
    auto h = ht;

    return{ h, c };
}


std::tuple<FunctionPtr, FunctionPtr> LSTMPComponent(Variable input,
    const NDShape& outputShape,
    const NDShape& cellShape,
    const std::function<FunctionPtr(const Variable&)>& recurrenceHookH,
    const std::function<FunctionPtr(const Variable&)>& recurrenceHookC,
    Constant &W, Constant &R, Constant &B)
{
    auto dh = PlaceholderVariable(outputShape, input.DynamicAxes());
    auto dc = PlaceholderVariable(cellShape, input.DynamicAxes());

    auto LSTMCell = LSTMPCell(input, dh, dc, W, R, B);

    auto actualDh = recurrenceHookH(LSTMCell.first);
    auto actualDc = recurrenceHookC(LSTMCell.second);

    // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
    LSTMCell.first->ReplacePlaceholders({ { dh, actualDh },{ dc, actualDc } });

    return std::make_tuple(LSTMCell.first, LSTMCell.second);
}

// This is currently unused
inline FunctionPtr SimpleRecurrentLayer(const  Variable& input, const  NDShape& outputDim, const std::function<FunctionPtr(const Variable&)>& recurrenceHook, const DeviceDescriptor& device)
{
    auto dh = PlaceholderVariable(outputDim, input.DynamicAxes());

    unsigned long seed = 1;
    auto createProjectionParam = [device, &seed](size_t outputDim, size_t inputDim) {
        return Parameter(NDArrayView::RandomUniform<float>({ outputDim, inputDim }, -0.5, 0.5, seed++, device));
    };

    auto hProjWeights = createProjectionParam(outputDim[0], outputDim[0]);
    auto inputProjWeights = createProjectionParam(outputDim[0], input.Shape()[0]);

    auto output = Times(hProjWeights, recurrenceHook(dh)) + Times(inputProjWeights, input);
    return output->ReplacePlaceholders({ { dh, output } });
}

#pragma warning(disable: 4189)

FunctionPtr CreateLSTM(const Node *node, const std::vector<Variable> &inputs, const std::string &direction)
{
    int numDirections = direction == "bidirectional" ? 2 : 1;
    std::vector<FunctionPtr> encoderOutputHs;
    for (int dir = 0; dir < numDirections; dir++)
    {
        // TODO: make this work with any input combinations (use empty Variable for those optioned out inputs).
        // Here is a hack code.
        Variable X = inputs[0];
        Variable W = inputs[1 + dir];
        Variable R = inputs[numDirections == 1 ? 2 : (3 + dir)];
        Variable B = Variable();
        if (numDirections == 1 && inputs.size() >= 4)
            B = inputs[3];
        else if (numDirections == 2 && inputs.size() >= 7)
            B = inputs[5 + dir];

        Variable initHVariable = X.GetDataType() == DataType::Double ? Constant::Scalar(0.0) : Constant::Scalar(0.0f);
        if (numDirections == 1 && inputs.size() >= 5)
            initHVariable = inputs[4];
        else if (numDirections == 2 && inputs.size() >= 9)
            initHVariable = inputs[7 + dir];

        Variable initCVariable = X.GetDataType() == DataType::Double ? Constant::Scalar(0.0) : Constant::Scalar(0.0f);
        if (numDirections == 1 && inputs.size() >= 6)
            initCVariable = inputs[5];
        else if (numDirections == 2 && inputs.size() >= 11)
            initCVariable = inputs[9 + dir];

        // TODO: DONT hard code, use init value
        //float init_state_value = 0.1F;
        //Constant initVariable = Constant::Scalar(init_state_value);

        int hiddenDim = W.Shape()[0] / 4;
        int inputDim = W.Shape()[1];

        FunctionPtr encoderOutputH;
        FunctionPtr encoderOutputC;

        bool go_backwards = direction == "reverse" || (numDirections == 2 && dir == 1);

        std::function<FunctionPtr(const Variable&)> futureValueRecurrenceHook;
        if (go_backwards)
            futureValueRecurrenceHook = [initHVariable](const Variable& x) { return FutureValue(x, initHVariable); };
        else
            futureValueRecurrenceHook = [initCVariable](const Variable& x) { return PastValue(x, initCVariable); };

        std::tie<FunctionPtr, FunctionPtr>(encoderOutputH, encoderOutputC) = LSTMPComponent(
            X, { (size_t)hiddenDim }, { (size_t)hiddenDim },
            futureValueRecurrenceHook, futureValueRecurrenceHook, (Constant &)W, (Constant &)R, (Constant &)B);
        encoderOutputHs.push_back(encoderOutputH);
    }
    if (encoderOutputHs.size() == 1)
        return encoderOutputHs[0];
    else
    {
        std::vector<Variable> operands({ encoderOutputHs[0], encoderOutputHs[1] });
        return Splice(operands, Axis(0), ToWString(node->Name()));
    }
}

