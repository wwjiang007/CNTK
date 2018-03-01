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
using namespace ONNXIR;

const std::string LSTMInputBiasNameHint = "_bias_";
const std::string LSTMInputInitialHNameHint = "_initial_h_";
const std::string LSTMInputInitialCNameHint = "_initial_c_";
const std::string LSTMInputPeepholeNameHint = "_peephole_";

const int LSTMInputIndexX = 0;
const int LSTMInputIndexW = 1;
const int LSTMInputIndexH = 2;
const int LSTMInputIndexB = 3;
const int LSTMInputIndexsequence_lens = 4;
const int LSTMInputIndexinitial_h = 5;
const int LSTMInputIndexinitial_c = 6;
const int LSTMInputIndexP = 7;

std::function<FunctionPtr(const Variable&)> ActivationMap(const std::string &activationName)
{
    if (activationName == "Relu")
    {
        return [](const Variable& x) { return ReLU(x); };
    }
    else if (activationName == "Tanh")
    {
        return [](const Variable& x) { return Tanh(x); };
    }
    else if (activationName == "Sigmoid")
    {
        return [](const Variable& x) { return Sigmoid(x); };
    }
    // else if (activationName == "Affine")
    // else if (activationName == "LeakyRelu")
    // else else if (activationName == "ThresholdedRelu")
    // else else if (activationName == "ScaledTanh")
    // else if (activationName == "HardSigmoid")
    else if (activationName == "Elu")
    {
        return [](const Variable& x) { return ELU(x); };
    }
    else if (activationName == "Softsign")
    {
        return [](const Variable& x) { return Softsign(x); };
    }
    else if (activationName == "Softplus")
    {
        return [](const Variable& x) { return Softplus(x); };
    }
    else
    {
        CNTK::LogicError("LSTM does not support activation: %s", activationName.c_str());
    }
}

std::function<FunctionPtr(const Variable&)> ActivationMap(const std::string &activationName,
    float activation_alpha)
{
    if (activationName == "LeakyRelu")
    {
        return [activation_alpha](const Variable& x) { return LeakyReLU(x, activation_alpha); };
    }
    else
    {
        return ActivationMap(activationName);
    }
}

std::function<FunctionPtr(const Variable&)> ActivationMap(const std::string &activationName,
    float activation_alpha, float activation_beta)
{
    if (activationName == "HardSigmoid")
    {
        return [activation_alpha, activation_beta](const Variable& x) { return HardSigmoid(x, activation_alpha, activation_beta); };
    }
    else
    {
        return ActivationMap(activationName, activation_alpha);
    }
}

std::tuple<std::function<FunctionPtr(const Variable&)>, std::function<FunctionPtr(const Variable&)>, std::function<FunctionPtr(const Variable&)>> 
GetActivations(const std::vector<string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta, int dir)
{
    if (activations.size() < (dir + 1) * 3)
        CNTK::LogicError("LSTM activations shall be 3 or 6 of strings");

    // 
    int iofActivationIndex = dir * 3 + 0;
    int cellActivation = dir * 3 + 1;
    int hiddenActivationIndex = dir * 3 + 2;
    
    // ONNX spec is not clear on how activation alpha and beta is set. 
    // Here we assume if they are set, they are set for all activations, regardless whether 
    // an activation needs those values or not.
    bool hasAlpha = activation_alpha.size() == (dir + 1) * 3;
    bool hasBeta = hasAlpha && activation_beta.size() == (dir + 1) * 3;
    std::function<FunctionPtr(const Variable&)> iofActivationOp, cellActivationOp, hiddenActivationOp;
    if (hasBeta)
    {
        iofActivationOp = ActivationMap(activations[iofActivationIndex], activation_alpha[iofActivationIndex], activation_beta[iofActivationIndex]);
        cellActivationOp = ActivationMap(activations[cellActivation], activation_alpha[cellActivation], activation_beta[cellActivation]); 
        hiddenActivationOp = ActivationMap(activations[hiddenActivationIndex], activation_alpha[hiddenActivationIndex], activation_beta[hiddenActivationIndex]);
    }
    else if (hasAlpha)
    {
        iofActivationOp = ActivationMap(activations[iofActivationIndex], activation_alpha[iofActivationIndex]);
        cellActivationOp = ActivationMap(activations[cellActivation], activation_alpha[cellActivation]);
        hiddenActivationOp = ActivationMap(activations[hiddenActivationIndex], activation_alpha[hiddenActivationIndex]);
    }
    else
    {
        iofActivationOp = ActivationMap(activations[iofActivationIndex]);
        cellActivationOp = ActivationMap(activations[cellActivation]);
        hiddenActivationOp = ActivationMap(activations[hiddenActivationIndex]);
    }

    return std::make_tuple(iofActivationOp, cellActivationOp, hiddenActivationOp);

}

std::pair<FunctionPtr, FunctionPtr> LSTMPCell(Variable input, 
    const std::function<FunctionPtr(const Variable&)> &iofActivationOp,
    const std::function<FunctionPtr(const Variable&)> &cellActivationOp,
    const std::function<FunctionPtr(const Variable&)> &hiddenActivationOp,
    Variable prevOutput, Variable prevCellState,
    Constant &W, Constant &R, Constant &B, Constant &Ci, Constant &Cf, Constant &Co)
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

    // CNTK weight and bias are in icfo order. 
    std::vector<Axis> stack_axis({ Axis(-1) });
    FunctionPtr it_proj = Slice(proj4, stack_axis, { 0 * stacked_dim }, { 1 * stacked_dim });
    FunctionPtr bit_proj = Slice(proj4, stack_axis, { 1 * stacked_dim }, { 2 * stacked_dim });
    FunctionPtr ft_proj = Slice(proj4, stack_axis, { 2 * stacked_dim }, { 3 * stacked_dim });
    FunctionPtr ot_proj = Slice(proj4, stack_axis, { 3 * stacked_dim }, { 4 * stacked_dim });

    bool hasPeephole = Ci != Variable();

    // Input gate
    auto it = hasPeephole ? iofActivationOp(it_proj + ElementTimes(Ci, prevCellState)) : Sigmoid(it_proj);
    auto bit = ElementTimes(it, cellActivationOp(bit_proj));

    auto ft = hasPeephole ? iofActivationOp(ft_proj + ElementTimes(Cf, prevCellState)) : Sigmoid(ft_proj);
    auto bft = ElementTimes(ft, prevCellState);

    auto ct = Plus(bft, bit);

    auto ot = hasPeephole ? iofActivationOp(ot_proj + ElementTimes(Co, ct)) : Sigmoid(ot_proj);
    auto ht = ElementTimes(ot, hiddenActivationOp(ct));

    auto c = ct;
    auto h = ht;

    return{ h, c };
}


std::tuple<FunctionPtr, FunctionPtr> LSTMPComponent(Variable input,
    const NDShape& cellShape,
    const std::function<FunctionPtr(const Variable&)> &iofActivationOp,
    const std::function<FunctionPtr(const Variable&)> &cellActivationOp,
    const std::function<FunctionPtr(const Variable&)> &hiddenActivationOp,
    const std::function<FunctionPtr(const Variable&)>& recurrenceHookH,
    const std::function<FunctionPtr(const Variable&)>& recurrenceHookC,
    Constant &W, Constant &R, Constant &B,
    Constant &Ci, Constant &Cf, Constant &Co)
{
    auto dh = PlaceholderVariable(cellShape, input.DynamicAxes());
    auto dc = PlaceholderVariable(cellShape, input.DynamicAxes());

    auto LSTMCell = LSTMPCell(input, 
        iofActivationOp, cellActivationOp, hiddenActivationOp,
        dh, dc, W, R, B, Ci, Cf, Co);

    auto actualDh = recurrenceHookH(LSTMCell.first);
    auto actualDc = recurrenceHookC(LSTMCell.second);

    // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
    LSTMCell.first->ReplacePlaceholders({ { dh, actualDh },{ dc, actualDc } });

    return std::make_tuple(LSTMCell.first, LSTMCell.second);
}

const std::vector<Variable> FindByNameHint(const std::vector<Variable> &inputs, const string &hint)
{
    std::vector<Variable> variables;
    for (auto v : inputs)
    {
        if (ToString(v.Name()).find(hint) != -1)
        {
            variables.push_back(v);
        }
    }
    return variables;
}

Variable GetInitialStateVariable(const std::vector<Variable> &inputs, int numDirections,
    const std::string &nameHint, DataType datatype)
{
    Variable initialVariable = datatype == DataType::Double ? Constant::Scalar(0.0) : Constant::Scalar(0.0f);
    const std::vector<Variable> initialVariables = FindByNameHint(inputs, nameHint);
    if (numDirections == 1 && initialVariables.size() >= 1)
    {
        initialVariable = initialVariables[0];
    }
    else if (numDirections == 2 && initialVariables.size() >= 2)
    {
        initialVariable = initialVariables[1];
    }

    return initialVariable;
}

FunctionPtr CreateLSTM(const ONNXIR::Node *node, const std::vector<Variable> &inputs, const std::string &direction,
    const std::vector<string> &activations, const std::vector<float> &activation_alpha, const std::vector<float> &activation_beta)
{
    int numDirections = direction == "bidirectional" ? 2 : 1;
    std::vector<FunctionPtr> outputHs;
    for (int dir = 0; dir < numDirections; dir++)
    {
        std::function<FunctionPtr(const Variable&)> iofActivationOp, cellActivationOp, hiddenActivationOp;
        std::tie<std::function<FunctionPtr(const Variable&)>, std::function<FunctionPtr(const Variable&)>, std::function<FunctionPtr(const Variable&)>>
            (iofActivationOp, cellActivationOp, hiddenActivationOp) = GetActivations(activations, activation_alpha, activation_beta, dir);

        // the first a few inputs are (in order): X, numDirections * W, numDirections * R
        Variable X = inputs[0];
        Variable W = inputs[1 + dir];
        Variable R = inputs[numDirections == 1 ? 2 : (3 + dir)];
        Variable B = Variable();
        std::vector<Variable> biasVariables = FindByNameHint(inputs, LSTMInputBiasNameHint);
        if (numDirections == 1 && biasVariables.size() >= 1)
            B = biasVariables[0];
        else if (numDirections == 2 && biasVariables.size() == 2)
            B = biasVariables[1];

        Variable initHVariable = GetInitialStateVariable(inputs, numDirections, LSTMInputInitialCNameHint, X.GetDataType());
        Variable initCVariable = GetInitialStateVariable(inputs, numDirections, LSTMInputInitialHNameHint, X.GetDataType());

        std::vector<Variable> peepholeVariables = FindByNameHint(inputs, LSTMInputPeepholeNameHint);
        Variable Ci, Cf, Co;
        if (numDirections == 1 && peepholeVariables.size() >= 3)
        {
            Ci = peepholeVariables[0];
            Co = peepholeVariables[1];
            Cf = peepholeVariables[2];
        }
        else if (numDirections == 2 && inputs.size() == 6)
        {
            Ci = peepholeVariables[3];
            Co = peepholeVariables[4];
            Cf = peepholeVariables[5];
        }

        int hiddenDim = W.Shape()[0] / 4;

        FunctionPtr outputH;
        FunctionPtr outputC;

        bool go_backwards = direction == "reverse" || (numDirections == 2 && dir == 1);

        std::function<FunctionPtr(const Variable&)> futureValueRecurrenceHook;
        if (go_backwards)
            futureValueRecurrenceHook = [initHVariable](const Variable& x) { return FutureValue(x, initHVariable); };
        else
            futureValueRecurrenceHook = [initCVariable](const Variable& x) { return PastValue(x, initCVariable); };

        std::tie<FunctionPtr, FunctionPtr>(outputH, outputC) = LSTMPComponent(
            X, { (size_t)hiddenDim }, iofActivationOp, cellActivationOp, hiddenActivationOp,
            futureValueRecurrenceHook, futureValueRecurrenceHook, (Constant &)W, (Constant &)R, (Constant &)B,
            (Constant &)Ci, (Constant &)Cf, (Constant &)Co);
        outputHs.push_back(outputH);
    }
    if (outputHs.size() == 1)
        return outputHs[0];
    else
    {
        std::vector<Variable> operands({ outputHs[0], outputHs[1] });
        return Splice(operands, Axis(0), ToWString(node->Name()));
    }
}

